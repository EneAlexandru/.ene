"""ffprobe + hashes of the source file -> the `source` object in the header (SPEC §3.2)."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
from fractions import Fraction


def ffprobe(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path],
        capture_output=True, text=True, encoding="utf-8", check=True,
    ).stdout
    return json.loads(out)


def sha256(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fast_hash(path: str) -> str:
    """OpenSubtitles-style hash: size + sum of the LE uint64 words of the first and last 64 KB, mod 2^64."""
    size = os.path.getsize(path)
    chunk = 65536
    h = size
    with open(path, "rb") as f:
        head = f.read(chunk)
        f.seek(max(0, size - chunk))
        tail = f.read(chunk)
    for buf in (head, tail):
        n = len(buf) // 8
        for (v,) in struct.iter_unpack("<Q", buf[: n * 8]):
            h = (h + v) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def source_dict(path: str, full_hash: bool = True) -> dict:
    p = ffprobe(path)
    fmt = p["format"]
    duration = float(fmt.get("duration", 0))
    video = next((s for s in p["streams"] if s["codec_type"] == "video"), None)
    src: dict = {
        "filename": os.path.basename(path),
        "size_bytes": int(fmt.get("size", os.path.getsize(path))),
        "hash": {"fast": fast_hash(path)},
        "container": fmt.get("format_name", "").split(",")[0],
        "duration_ms": int(round(duration * 1000)),
        "bit_rate": int(fmt.get("bit_rate", 0)),
        "audio": [],
        "subtitles": [],
    }
    if full_hash:
        src["hash"]["sha256"] = sha256(path)
    if video:
        fps = Fraction(video.get("r_frame_rate", "0/1"))
        nb = video.get("nb_frames")
        frames = int(nb) if nb and nb.isdigit() else int(round(duration * float(fps))) if fps else None
        src["video"] = {
            "index": video["index"], "codec": video.get("codec_name"), "width": video.get("width"), "height": video.get("height"),
            "fps": str(fps), "frames": frames, "frames_exact": bool(nb and nb.isdigit()),
        }
    for s in p["streams"]:
        lang = (s.get("tags") or {}).get("language", "und")
        if s["codec_type"] == "audio":
            src["audio"].append({"index": s["index"], "lang": lang, "codec": s.get("codec_name"), "channels": s.get("channels"), "sample_rate": int(s.get("sample_rate", 0))})
        elif s["codec_type"] == "subtitle":
            src["subtitles"].append({"index": s["index"], "lang": lang, "codec": s.get("codec_name"), "origin": "embedded"})
    return src


def uid_from_source(src: dict) -> str:
    h = src["hash"].get("sha256") or src["hash"]["fast"]
    return h[:16]
