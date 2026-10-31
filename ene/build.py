"""Profile `ene-1` (SPEC §6.1): media -> the model's fp32 vectors (the "master") -> one 1.0 packet under a storage policy.

One version per build, `REFERENCE` + `TIMELINE`, frames at the source's frame rate; the axis is that rate. Segments are frames
of the version (§2.3): a span given in seconds is rounded outwards (`floor(t0 · fps)`, `ceil(t1 · fps)`), a scene cut at
time `t` is the frame `[round(t · fps), +1)`. Every stage is timed through `Telemetry`: wall / CPU time, peak VRAM, RSS,
items/s, real-time factor. Media decoding (audio, frames at 1 fps, scene cuts) runs in parallel (three ffmpeg processes).

    built = build_title(path, Embedder(), title="Big Buck Bunny", year=2008, ids={"imdb": "tt1254207"}, synopsis="...", tags=[...])
    write_ene(built, "title.ene")                    # one packet under policy `compact` (§4.4)
    save_master(built, "title.master.npz")           # the fp32 rows, segments, texts, fingerprints and telemetry (for benchmarks)
    built = load_master("title.master.npz")          # back, without the model
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from functools import partial

import numpy as np

from . import __version__, media, probe
from .api import SectionSpec, VersionSpec, create, policy_of
from .container import ZERO16
from .telemetry import Telemetry

PROFILE = "ene-1"
WINDOW_S, STRIDE_S = 30, 15        # `audio.30s`, `video.30s` (role `window`)
SPEECH_S = 5                       # `audio.5s` (role `speech`)
CHAPTER_S = 150                    # `chapter` (fixed grid)
FRAME_W = 448                      # every picture is resized by ffmpeg (bicubic) to this width before the model
TOKENS = {"frame": 280, "title": 280, "window": 70, "chapter": 140}   # soft-token budgets per picture (§6.1, §7)
WINDOW_FRAMES, CHAPTER_FRAMES = 30, 29                                 # frames at 1 fps handed to the model per window / chapter
TITLE_FRAMES, TITLE_AUDIO = 6, 4                                       # key frames and 30 s audio segments in the title row
BATCH = {"images": 16, "audio": 8, "video": 2, "text": 32}
AUDIO_ONLY_FPS = (25, 1)           # the nominal rate a version without pictures declares (§2.3)
SCENE_THRESHOLD = 0.3


@dataclass
class SectionData:
    """One section as the pipeline produces it: unit fp32 rows, their segments in frames of the version (None in the title
    part), texts, provenance, the space and the input fingerprint `prep` (§2.7)."""
    id: str
    role: str
    modality: str
    vectors: np.ndarray
    times: np.ndarray | None = None
    texts: list[str] | None = None
    lang: str = ""
    ref: str = ""
    prep: dict | None = None
    space: str = field(kw_only=True)                 # the space the rows come from (§2.7): named by the pipeline, never implied

    count = property(lambda self: int(self.vectors.shape[0]))
    timeline = property(lambda self: self.times is not None)

    def spec(self, version_id: bytes) -> SectionSpec:
        return SectionSpec(self.id, version_id, modality=self.modality, role=self.role, vectors=self.vectors, times=self.times, texts=self.texts,
                           lang=self.lang, space=self.space, prep=self.prep, ref=self.ref)


@dataclass
class Built:
    """What `build_title` returns and `save_master` keeps: the title's identity, its one version, the description (§3.7), the
    sections, the timed stages."""
    title_id: bytes
    axis_fps: tuple[int, int]
    version: dict                                    # id, source_hash (16 bytes each), fps, duration (frames)
    description: dict
    sections: list[SectionData] = field(default_factory=list)
    timings: dict = field(default_factory=dict)      # {stage: wall seconds}
    telemetry: dict = field(default_factory=dict)    # Telemetry.to_dict() (+ "write")

    name = property(lambda self: (self.description.get("title") or {}).get("name"))
    duration_s = property(lambda self: self.version["duration"] * self.axis_fps[1] / self.axis_fps[0])
    n_vectors = property(lambda self: int(sum(s.count for s in self.sections)))

    def get(self, sid: str) -> SectionData:
        return next(s for s in self.sections if s.id == sid)

    def seconds(self, frames) -> np.ndarray:
        """Frames of the version (the reference: axis frames) as seconds."""
        return np.asarray(frames, dtype=np.float64) * (self.axis_fps[1] / self.axis_fps[0])

    def version_spec(self) -> VersionSpec:
        return VersionSpec(self.version["id"], source_hash=self.version["source_hash"], fps=self.version["fps"], duration=self.version["duration"], reference=True)

    @property
    def source(self) -> dict:
        return (self.description.get("versions") or {}).get(self.version["id"].hex(), {}).get("source", {})


# --- time -> frames (§2.3) ----------------------------------------------------------------------------------------------

def span_frames(t0: float, t1: float, fps: Fraction, duration: int | None = None) -> tuple[int, int]:
    """A span in seconds as frames `[f0, f1)`: floor / ceil, so it never shrinks; `f1` clipped to the version's length."""
    f0 = max(math.floor(t0 * fps), 0)
    f1 = math.ceil(t1 * fps)
    if duration is not None:
        f1 = min(f1, max(duration, f0 + 1))
    return f0, max(f1, f0 + 1)


def frame_at(t: float, fps: Fraction) -> int:
    return int(round(t * fps))


def parse_srt(path: str) -> tuple[list[str], np.ndarray]:
    """SRT -> (texts, spans in seconds, float64[n, 2]). HTML tags removed; multi-line cues joined with a space."""
    txt = open(path, encoding="utf-8-sig", errors="replace").read()
    blocks = re.split(r"\n\s*\n", txt.strip())
    ts = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
    texts, times = [], []
    for b in blocks:
        lines = b.strip().splitlines()
        hit = next(((i, ts.search(line)) for i, line in enumerate(lines) if ts.search(line)), None)
        if not hit:
            continue
        i, m = hit
        g = [int(x) for x in m.groups()]
        start = (g[0] * 60 + g[1]) * 60 + g[2] + g[3] / 1000
        end = (g[4] * 60 + g[5]) * 60 + g[6] + g[7] / 1000
        body = re.sub(r"<[^>]+>", "", " ".join(lines[i + 1:])).strip()
        if body:
            texts.append(body)
            times.append((start, end))
    return texts, np.array(times, dtype=np.float64).reshape(-1, 2)


def _centroid(x: np.ndarray) -> np.ndarray:
    c = x.mean(axis=0, keepdims=True)
    n = np.linalg.norm(c)
    return c / n if n else c


def _unit(x: np.ndarray) -> np.ndarray:
    """The model normalizes in bf16 (norms in [0.998, 1.004], §7); the writer renormalizes in fp32 before quantization."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


# --- the pipeline --------------------------------------------------------------------------------------------------------

def build_title(path: str, model, title: str, year: int | None = None, ids: dict | None = None, synopsis: str = "", synopsis_lang: str = "eng",
                tags: list[str] | None = None, people: list[str] | None = None, srt: dict[str, str] | None = None, kind: str = "movie",
                full_hash: bool = True, log=print, tel: Telemetry | None = None, contributor: str | None = None) -> Built:
    """Profile `ene-1` on one source file. `model`: an `ene.embed.Embedder` or anything with its interface; every section is
    written into `model.space_id`. `srt`: {lang: path}; `people`: one string per cast member or character
    ("Michael Knight, played by David Hasselhoff")."""
    tel = tel or Telemetry(title)
    ids, tags, people, srt = dict(ids or {}), list(tags or []), list(people or []), dict(srt or {})
    with tel.stage("probe") as rec:
        src = probe.source_dict(path, full_hash=full_hash)
        rec["source_mib"] = round(src["size_bytes"] / 2**20, 1)
    dur_s = src["duration_ms"] / 1000
    vid = src.get("video") or {}
    W, H = int(vid.get("width") or 0), int(vid.get("height") or 0)
    fps = Fraction(vid["fps"]) if W and vid.get("fps") and Fraction(vid["fps"]) > 0 else Fraction(*AUDIO_ONLY_FPS)
    axis_fps = (fps.numerator, fps.denominator)
    duration = int(vid["frames"]) if W and vid.get("frames") else math.ceil(dur_s * fps)
    log(f"[probe] {src['filename']}: {dur_s / 60:.1f} min, {W}x{H} @ {fps} ({duration} frames), {len(src['audio'])} audio, {len(src['subtitles'])} sub ({tel.stages[-1]['wall_s']:.1f}s)")

    # identities (§2.1, §2.2)
    sha = src["hash"].get("sha256")
    seed = bytes.fromhex(sha) if sha else hashlib.sha256(("fast:" + src["hash"]["fast"]).encode("utf-8")).digest()
    title_id = hashlib.sha256(("imdb:" + ids["imdb"]).encode("utf-8")).digest()[:16] if ids.get("imdb") else seed[:16]
    generator = f"ene {__version__} {PROFILE}"
    version_id = hashlib.sha256(seed + generator.encode("utf-8")).digest()[:16]
    source_hash = seed[:16] if sha else ZERO16

    # the input fingerprints of this profile (§2.7, §6.1): what moves a vector by more than the storage error
    fw, fh = media._scaled_size(W, H, FRAME_W) if W else (0, 0)
    p_text = {"profile": PROFILE, "prompt": "title: {title} | text: "}
    p_image = {"profile": PROFILE, "tokens": TOKENS["frame"], "resize": ["ffmpeg", "bicubic", fw, fh], "source": [W, H, vid.get("codec")], "lossy": None}
    p_video = dict(p_image, tokens=TOKENS["window"], frames=WINDOW_FRAMES, fps=1)

    def p_audio(seconds: float) -> dict:
        return {"profile": PROFILE, "sample_rate": media.SR, "max_length": int(seconds * media.SR)}

    p_chapter = dict(p_image, tokens=TOKENS["chapter"], frames=CHAPTER_FRAMES, fps=1) | {k: v for k, v in p_audio(CHAPTER_S).items() if k != "profile"}

    description = {
        "title": {"name": title, "year": year, "kind": kind, "ids": ids,
                  "languages": {"audio": [a["lang"] for a in src["audio"]], "subs": sorted(set(list(srt) + [s["lang"] for s in src["subtitles"]]))}},
        "versions": {version_id.hex(): {
            "source": src,
            "generator": {"tool": "ene", "version": __version__, "profile": PROFILE, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          **({"contributor": contributor} if contributor else {}), "compute": model.compute() | {"batch_size": BATCH}},
            "alignment": {"method": "reference"}}},
        "spaces": {model.space_id: model.space()},
        "sections": {},
    }
    built = Built(title_id, axis_fps, {"id": version_id, "source_hash": source_hash, "fps": axis_fps, "duration": duration}, description)
    tel.meta |= {"title": title, "file": src["filename"], "duration_s": round(dur_s, 1), "video": f"{W}x{H}@{fps}" if W else None,
                 "video_codec": vid.get("codec"), "audio_streams": len(src["audio"]), "source_mib": round(src["size_bytes"] / 2**20, 1)}
    sec_desc = description["sections"]
    Sec = partial(SectionData, space=model.space_id)

    def add(s: SectionData, **more) -> SectionData:
        s.vectors = _unit(s.vectors)
        built.sections.append(s)
        if more:
            sec_desc[f"{version_id.hex()}/{s.id}"] = more
        return s

    # --- media (three ffmpeg processes in parallel) ---------------------------------------------------------------------
    def dec_audio():
        with tel.stage("decode_audio", media_s=round(dur_s, 1)) as rec:
            a = media.audio_f32(path) if src["audio"] else np.zeros(0, np.float32)
            rec["samples"] = int(len(a)); rec["mib"] = round(a.nbytes / 2**20, 1)
            return a

    def dec_frames():
        with tel.stage("decode_frames", media_s=round(dur_s, 1)) as rec:
            f = media.frames_at_fps(path, W, H, fps=1.0, target_w=FRAME_W) if W else []
            rec["items"] = len(f); rec["frame_px"] = list(f[0].image.size) if f else None
            rec["mib"] = round(sum(im.image.size[0] * im.image.size[1] * 3 for im in f) / 2**20, 1)
            return f

    def dec_scenes():
        with tel.stage("scene_detect", media_s=round(dur_s, 1), threshold=SCENE_THRESHOLD) as rec:
            s = media.scene_frames(path, W, H, threshold=SCENE_THRESHOLD, target_w=FRAME_W) if W else []
            rec["items"] = len(s)
            return s

    with tel.stage("decode_media", media_s=round(dur_s, 1)) as rec:
        with ThreadPoolExecutor(3, thread_name_prefix="ffmpeg") as ex:
            fa, ff, fs = ex.submit(dec_audio), ex.submit(dec_frames), ex.submit(dec_scenes)
            audio, frames1, scenes = fa.result(), ff.result(), fs.result()
        if W and not scenes and frames1:
            scenes = [frames1[0]]
        rec["audio_s"] = round(len(audio) / media.SR, 1); rec["frames_1fps"] = len(frames1); rec["scene_cuts"] = len(scenes)
    log(f"[media] audio {len(audio) / media.SR:.0f}s, {len(frames1)} frames @1fps, {len(scenes)} scene cuts ({tel.stages[-1]['wall_s']:.1f}s, in parallel)")

    def frames_in(s: float, e: float) -> list:
        return [f.image for f in frames1 if s <= f.t < e]

    def spans(windows_s) -> np.ndarray:
        return np.array([span_frames(a, b, fps, duration) for a, b in windows_s], np.int32).reshape(-1, 2)

    # --- timeline part ---------------------------------------------------------------------------------------------------
    if scenes:
        with tel.stage("frame", items=len(scenes), batch=BATCH["images"], tokens_per_item=TOKENS["frame"]) as rec:
            v = model.images([f.image for f in scenes], max_soft_tokens=TOKENS["frame"], batch_size=BATCH["images"])
        n = np.array([min(frame_at(f.t, fps), max(duration - 1, 0)) for f in scenes], np.int32)
        keep = np.r_[True, n[1:] > n[:-1]]                                  # one row per frame
        add(Sec("frame", "frame", "image", v[keep], np.stack([n[keep], n[keep] + 1], 1), prep=p_image))
        log(f"[frame] {int(keep.sum())} frames ({rec['wall_s']:.1f}s, {rec['ms_per_item']:.0f} ms/frame)")
    if len(audio):
        for sid, role, win, stride in (("audio.5s", "speech", SPEECH_S, SPEECH_S), ("audio.30s", "window", WINDOW_S, STRIDE_S)):
            ws = media.windows(len(audio), win, stride)
            with tel.stage(sid, items=len(ws), media_s=round(sum(e - s for s, e in ws) / media.SR, 1), batch=BATCH["audio"], tokens_per_item=win * 25) as rec:
                v = model.audio([audio[s:e] for s, e in ws], batch_size=BATCH["audio"])
            add(Sec(sid, role, "audio", v, spans([(s / media.SR, e / media.SR) for s, e in ws]), prep=p_audio(win)))
            log(f"[{sid}] {len(ws)} windows ({rec['wall_s']:.1f}s, {rec['realtime_factor']}x real time)")
    for lang, srt_path in srt.items():
        texts, times = parse_srt(srt_path)
        if not texts:
            continue
        with tel.stage(f"cue.{lang}", items=len(texts), batch=BATCH["text"]) as rec:
            v = model.documents(texts, title)
        add(Sec(f"cue.{lang}", "cue", "text", v, spans(times.tolist()), texts, lang=lang, prep=p_text), origin="external")
        log(f"[cue.{lang}] {len(texts)} cues ({rec['wall_s']:.1f}s)")
    if frames1:
        ws = [(s, min(s + WINDOW_S, dur_s)) for s in np.arange(0, max(dur_s - STRIDE_S, 1e-9), STRIDE_S).tolist()]
        clips = [frames_in(s, e) for s, e in ws]
        keep = [i for i, c in enumerate(clips) if c]
        ws, clips = [ws[i] for i in keep], [clips[i] for i in keep]
        media_s = round(sum(e - s for s, e in ws), 1)
        with tel.stage("video.30s", items=len(ws), media_s=media_s, frames=sum(len(c) for c in clips), batch=BATCH["video"], tokens_per_item=WINDOW_FRAMES * TOKENS["window"]) as rec:
            v = model.video(clips, max_soft_tokens=TOKENS["window"], max_frames=WINDOW_FRAMES, batch_size=BATCH["video"])
        add(Sec("video.30s", "window", "video", v, spans(ws), prep=p_video))
        log(f"[video.30s] {len(ws)} windows ({rec['wall_s']:.1f}s, {rec['realtime_factor']}x real time)")
    if frames1 or len(audio):
        chap = [(s, min(s + CHAPTER_S, dur_s)) for s in np.arange(0, dur_s, CHAPTER_S).tolist()]
        with tel.stage("chapter", items=len(chap), media_s=round(dur_s, 1), batch=1, tokens_per_item=CHAPTER_FRAMES * TOKENS["chapter"] + CHAPTER_S * 25) as rec:
            vv = []
            for s, e in chap:
                fr = media.pick_uniform(frames_in(s, e), CHAPTER_FRAMES) if frames1 else []
                a = [audio[int(s * media.SR):int(e * media.SR)]] if len(audio) else None
                text = ("<|video|> " if fr else "") + ("<|audio|>" if a else "")
                vv.append(model.interleaved(text.strip(), audio=a, video=fr or None, video_tokens=TOKENS["chapter"], max_frames=CHAPTER_FRAMES))
        add(Sec("chapter", "chapter", "mixed", np.concatenate(vv), spans(chap), prep=p_chapter))
        log(f"[chapter] {len(chap)} chapters ({rec['wall_s']:.1f}s)")

    # --- title part ------------------------------------------------------------------------------------------------------
    with tel.stage("title", items=1 + bool(synopsis) + len(tags) + len(people)) as rec:
        kf = media.pick_uniform(scenes, TITLE_FRAMES) if scenes else []
        seg = []
        if len(audio):
            for s, _ in media.pick_uniform(media.windows(len(audio), WINDOW_S, WINDOW_S), TITLE_AUDIO):
                seg.append(audio[s:s + WINDOW_S * media.SR])
        desc = f"title: {title} | text: {synopsis}".strip()
        if tags:
            desc += " Tags: " + ", ".join(tags) + "."
        text = desc + " " + " ".join(["<|image|>"] * len(kf)) + " " + " ".join(["<|audio|>"] * len(seg))
        v0 = model.interleaved(text.strip(), images=[f.image for f in kf] or None, audio=seg or None, image_tokens=TOKENS["title"])
        p_title = dict(p_text)
        if kf:
            p_title |= {k: v for k, v in p_image.items() if k != "profile"} | {"images": len(kf), "tokens": TOKENS["title"]}
        if seg:
            p_title |= {k: v for k, v in p_audio(WINDOW_S).items() if k != "profile"} | {"audio_segments": len(seg)}
        add(Sec("title", "title", "mixed", v0, texts=[title], prep=p_title))
        if synopsis:
            add(Sec("synopsis", "synopsis", "text", model.documents([synopsis], title), texts=[synopsis], lang=synopsis_lang, prep=p_text))
        if tags:
            add(Sec("tag", "tag", "text", model.documents(tags, title), texts=list(tags), prep=p_text))
        if people:
            add(Sec("person", "person", "text", model.documents(people, title), texts=list(people), prep=p_text))
        for s in [s for s in built.sections if s.timeline]:                      # k = 1 summaries (§2.6, §8.3)
            add(Sec(f"summary.{s.id}", "summary", s.modality, _centroid(s.vectors), lang=s.lang, ref=s.id, prep=dict(s.prep or {}, summary="centroid")))
        rec["images"] = len(kf); rec["audio_segments"] = len(seg); rec["tags"] = len(tags); rec["people"] = len(people)
    log(f"[title] title row ({len(kf)} frames, {len(seg)} audio segments), {len(tags)} tags, {len(people)} people, summaries ({rec['wall_s']:.1f}s)")

    tel.meta["n_vectors"] = built.n_vectors
    tel.meta["model_load_s"] = round(getattr(model, "load_s", 0.0), 2)
    built.telemetry = tel.to_dict()
    built.timings = tel.timings()
    return built


# --- writing and the master ---------------------------------------------------------------------------------------------

def write_ene(built: Built, out_path: str, policy=None) -> int:
    """One packet for the title under the storage policy (§4.4): its version record, every section, the description."""
    pol = policy_of(policy)
    t0 = time.perf_counter()
    vid = built.version["id"]
    n = create(out_path, built.title_id, built.axis_fps, versions=[built.version_spec()], sections=[s.spec(vid) for s in built.sections],
               description=built.description, policy=pol)
    built.telemetry.setdefault("write", {})[out_path] = {"bytes": n, "wall_s": round(time.perf_counter() - t0, 3), "policy": pol.name}
    return n


MASTER_FORMAT = "ene-master-1"


def save_master(built: Built, path: str) -> None:
    """The fp32 rows, segments, texts, provenance, description and telemetry (npz), for benchmarks without re-embedding."""
    arrays, metas = {}, []
    for s in built.sections:
        arrays[f"v:{s.id}"] = s.vectors
        if s.times is not None:
            arrays[f"t:{s.id}"] = s.times
        if s.texts is not None:
            arrays[f"x:{s.id}"] = np.array(s.texts, dtype=object)
        metas.append({"id": s.id, "role": s.role, "modality": s.modality, "lang": s.lang, "ref": s.ref, "prep": s.prep, "space": s.space})
    meta = {"format": MASTER_FORMAT, "title_id": built.title_id.hex(), "axis_fps": list(built.axis_fps),
            "version": {"id": built.version["id"].hex(), "source_hash": built.version["source_hash"].hex(), "fps": list(built.version["fps"]), "duration": built.version["duration"]},
            "description": built.description, "sections": metas, "timings": built.timings, "telemetry": built.telemetry}
    arrays["meta"] = np.array(json.dumps(meta, ensure_ascii=False))
    np.savez_compressed(path, **arrays, allow_pickle=True)


def load_master(path: str) -> Built:
    z = np.load(path, allow_pickle=True)
    m = json.loads(str(z["meta"]))
    if m.get("format") != MASTER_FORMAT:
        raise ValueError(f"{path}: not an {MASTER_FORMAT} master")
    v = m["version"]
    b = Built(bytes.fromhex(m["title_id"]), tuple(m["axis_fps"]), {"id": bytes.fromhex(v["id"]), "source_hash": bytes.fromhex(v["source_hash"]), "fps": tuple(v["fps"]), "duration": int(v["duration"])},
              m["description"], timings=m.get("timings", {}), telemetry=m.get("telemetry", {}))
    for meta in m["sections"]:
        sid = meta["id"]
        b.sections.append(SectionData(sid, meta["role"], meta["modality"], z[f"v:{sid}"], z[f"t:{sid}"] if f"t:{sid}" in z else None,
                                      list(z[f"x:{sid}"]) if f"x:{sid}" in z else None, meta.get("lang", ""), meta.get("ref", ""), meta.get("prep"), space=meta["space"]))
    return b
