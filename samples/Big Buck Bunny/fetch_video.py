"""Fetch the Big Buck Bunny source video: the 2013 "sunflower" 1080p/30 fps render (CC BY 3.0, Blender Foundation).

The video is not kept in git. Blender's download server offers it zipped; this script downloads the archive next to
this file, extracts the one video inside, checks its size and sha256 against the values the included `.ene` was built
from, and removes the archive. Standard library only; run from anywhere:

    python "samples/Big Buck Bunny/fetch_video.py"
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
import zipfile

URL = "https://download.blender.org/demo/movies/BBB/bbb_sunflower_1080p_30fps_normal.mp4.zip"
NAME = "bbb_sunflower_1080p_30fps_normal.mp4"
SIZE = 276_134_947
SHA256 = "ae51005850b0ff757fe60c3dd7a12d754d3cd2397d87d939b55235e457f97658"
HERE = os.path.dirname(os.path.abspath(__file__))


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    """Download, extract and verify; a video already present and matching is left alone. 0 on success, 1 otherwise."""
    target = os.path.join(HERE, NAME)
    if os.path.isfile(target) and os.path.getsize(target) == SIZE and sha256_of(target) == SHA256:
        print(f"{NAME}: already present and verified")
        return 0
    archive = target + ".zip.part"
    print(f"downloading {URL} ...")
    with urllib.request.urlopen(URL, timeout=60) as r, open(archive, "wb") as out:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / total:6.1%}", end="", flush=True)
    print()
    tmp = target + ".part"
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.infolist() if m.filename.endswith(".mp4")]
        if len(members) != 1:
            print("unexpected archive contents:", [m.filename for m in z.infolist()], file=sys.stderr)
            return 1
        with z.open(members[0]) as src, open(tmp, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
    os.remove(archive)
    if os.path.getsize(tmp) != SIZE or sha256_of(tmp) != SHA256:
        os.remove(tmp)
        print("the downloaded video does not match the expected size / sha256; nothing kept", file=sys.stderr)
        return 1
    os.replace(tmp, target)
    print(f"{NAME}: {SIZE} bytes, sha256 verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
