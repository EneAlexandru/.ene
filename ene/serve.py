"""Local web GUI for .ene: `python -m ene serve [--root DIR ...] [--port 8765]`.

FastAPI server + static page (ene/webui). API:
  GET  /api/status                      model, roots, number of titles
  GET  /api/library                     the titles (.ene) found under the roots
  POST /api/rescan                      re-scans the roots
  GET  /api/title/{id}                  summary, description, packets, sections (+ segments in ms, texts)
  GET  /api/title/{id}/section/{sid}/vectors?dim=   the fp32 rows of one section (Matryoshka at read time)
  POST /api/search                      form: q, image, audio, scope=title|library, title, dim, k
  GET  /api/title/{id}/frame?t=&w=      one JPEG frame from the video source at second t
  GET  /api/title/{id}/video            the video source (FileResponse with Range -> <video> seek)
  GET  /api/title/{id}/bench            the benchmark (markdown), if it exists
  GET  /api/media                       media files under the roots (for building)
  POST /api/build  /  GET /api/jobs[/{id}]   background build + log
"""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import tempfile
import threading
import time
import uuid
import webbrowser
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__, api
from . import container as C
from .api import DEFAULT_POLICY_NAME, POLICIES
from .quant import mrl

VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".mpg", ".mpeg", ".ts", ".wmv"}
SKIP_DIRS = {".venv", ".git", "__pycache__", "node_modules", ".cache"}
MIME = {".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska", ".mov": "video/quicktime", ".avi": "video/x-msvideo"}
STATIC = Path(__file__).parent / "webui"
MAX_QUERY_AUDIO_S = 300


def _ms(frames, fps: tuple[int, int]) -> np.ndarray | None:
    """Axis frames -> milliseconds (what the player and the timeline work in)."""
    if frames is None:
        return None
    return np.rint(np.asarray(frames, dtype=np.float64) * (1000 * fps[1] / fps[0])).astype(np.int64)


def _section_bytes(s: api.Section) -> int:
    """Bytes a section occupies; a shared `vectors` / `scale` block (§3.6) is attributed by row share."""
    r, rec, total = s.raw, s.raw.rec, 0
    for name, f_off, f_len, f_raw in C._BLOCK_FIELDS:
        if name != "vectors" and not r.encoded(name):
            continue
        enc, raw = int(rec[f_len]), int(rec[f_raw])
        if r.row_offset is not None and name in ("vectors", "scale") and raw:
            enc = int(round(enc * r.count * (r.row_bytes if name == "vectors" else 4) / raw))
        total += enc
    return total


# --------------------------------------------------------------------------------------- library
@dataclass
class Title:
    id: str
    path: str
    mtime: float
    size: int
    info: dict                                  # the summary built at scan time (no vectors)
    sections: list[dict]                        # static rows: id, part, role, modality, ... (no vectors)
    packets: list[dict]
    description: dict
    video: str | None
    master: str | None
    bench: str | None
    f: api.EneFile | None = None               # open and materialized by load()
    lock: threading.Lock = field(default_factory=threading.Lock)

    def summary(self) -> dict:
        return self.info | {"id": self.id, "path": self.path, "file": os.path.basename(self.path), "bytes": self.size,
                            "has_video": self.video is not None, "has_master": self.master is not None, "has_bench": self.bench is not None}

    def load(self) -> api.EneFile:
        with self.lock:
            if self.f is None:
                f = api.EneFile(self.path, verify=False)
                for s in f.title():
                    s.load()
                self.f = f
            return self.f

    def close(self) -> None:
        with self.lock:
            if self.f is not None:
                self.f.close()
                self.f = None

    def ordered(self) -> list[api.Section]:
        t = self.load().title()
        return sorted(t, key=lambda s: (s.raw.timeline, s.id, s.version_id))


def _tid(path: str) -> str:
    return hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:12]


def _source(desc: dict, ref_hex: str | None) -> dict:
    vs = desc.get("versions") or {}
    v = vs.get(ref_hex) if ref_hex else None
    if v is None and vs:
        v = next(iter(vs.values()))
    return (v or {}).get("source") or {}


def _find_video(path: str, src: dict) -> str | None:
    d = os.path.dirname(path)
    name = src.get("filename")
    if name and os.path.isfile(os.path.join(d, name)):
        return os.path.join(d, name)
    stem = os.path.splitext(os.path.basename(path))[0]
    for f in sorted(os.listdir(d)):
        if os.path.splitext(f)[1].lower() in VIDEO_EXT and (os.path.splitext(f)[0] == stem or name is None):
            return os.path.join(d, f)
    return None


def _find_bench(path: str, name: str | None, roots: list[str]) -> str | None:
    stem = os.path.splitext(path)[0]
    if os.path.isfile(stem + ".bench.md"):
        return stem + ".bench.md"
    for root in roots:
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in SKIP_DIRS]
            for f in fns:
                if f.endswith(".bench.md"):
                    p = os.path.join(dp, f)
                    try:
                        first = open(p, encoding="utf-8").readline()
                    except OSError:
                        continue
                    if name and name in first:
                        return p
    return None


class Library:
    def __init__(self, roots: list[str]):
        self.roots = [os.path.abspath(r) for r in roots]
        self.titles: dict[str, Title] = {}
        self.lock = threading.Lock()
        self.scanned_at = 0.0

    def scan(self) -> None:
        found: dict[str, Title] = {}
        for root in self.roots:
            for dp, dns, fns in os.walk(root):
                dns[:] = [d for d in dns if d not in SKIP_DIRS]
                for f in fns:
                    if f.lower().endswith(".ene"):
                        p = os.path.join(dp, f)
                        tid = _tid(p)
                        st = os.stat(p)
                        old = self.titles.get(tid)
                        if old and old.mtime == st.st_mtime and old.size == st.st_size:
                            found[tid] = old
                            continue
                        try:
                            found[tid] = self._open(p, tid, st)
                        except Exception as e:  # noqa: BLE001
                            print(f"[library] {p}: {e}")
        with self.lock:
            for tid, t in self.titles.items():
                if found.get(tid) is not t:
                    t.close()
            self.titles = found
            self.scanned_at = time.time()

    def _open(self, p: str, tid: str, st: os.stat_result) -> Title:
        with api.EneFile(p, verify=True) as f:
            t = f.title()
            desc = t.description
            ref = t.reference or next((v for v in t.versions.values() if v.timeline), None)
            fps = ref.fps if ref is not None and ref.timeline else t.axis_fps
            src = _source(desc, ref.id.hex() if ref is not None else None)
            vid = src.get("video") or {}
            dur_ms = int(round(ref.duration * 1000 * ref.fps[1] / ref.fps[0])) if ref is not None and ref.timeline else 0
            created = max((((v.get("generator") or {}).get("created") or "") for v in (desc.get("versions") or {}).values()), default="")
            secs, n_vec, tags = [], 0, None
            for s in sorted(t, key=lambda s: (s.raw.timeline, s.id, s.version_id)):
                r = s.raw
                n_vec += r.count
                secs.append({"id": r.id, "version": r.version_id.hex(), "part": s.part, "role": r.role, "modality": r.modality_name, "lang": r.lang, "count": r.count,
                             "space": r.space, "dim": r.dim, "dim_real": r.dim_real, "quant": r.quant, "codec": r.codec, "rev": r.rev, "ref": r.ref, "bytes": _section_bytes(s),
                             "has_text": bool(r.flags & C.S_HAS_TEXT), "has_time": r.timeline, "prep": s.prep_object, "shared": r.row_offset is not None})
                if r.role == "tag" and not r.timeline and tags is None:
                    tags = s.texts
            packets = [{"index": pk.index, "offset": pk.offset, "length": pk.length, "versions": len(pk.versions), "sections": len(pk.sections), "hash_ok": pk.hash_ok,
                        "one_block": pk.one_block, "description": pk.desc_len > 0} for pk in f.packets]
            title = desc.get("title") or {}
            info = {"name": t.name, "year": title.get("year"), "kind": title.get("kind"), "ids": title.get("ids") or {}, "title_id": t.title_id.hex(),
                    "axis_fps": list(t.axis_fps), "fps": list(fps), "duration_ms": dur_ms, "video": f"{vid['width']}x{vid['height']}" if vid.get("width") else None,
                    "source_bytes": src.get("size_bytes"), "source_file": src.get("filename"), "packets": len(f.packets), "versions": len(t.versions),
                    "n_sections": len(t), "n_vectors": n_vec, "created": created, "tags": tags}
        stem = os.path.splitext(p)[0]
        master = stem + ".master.npz" if os.path.isfile(stem + ".master.npz") else None
        return Title(tid, p, st.st_mtime, st.st_size, info, secs, packets, desc, _find_video(p, src), master, _find_bench(p, info["name"], self.roots))

    def get(self, tid: str) -> Title:
        with self.lock:
            t = self.titles.get(tid)
        if t is None:
            raise HTTPException(404, "unknown title")
        return t

    def widths(self) -> list[int]:
        """The read-time widths the files offer (§6.2): every space's Matryoshka ladder from the descriptions, else the native width."""
        with self.lock:
            ts = list(self.titles.values())
        ws: set[int] = set()
        for t in ts:
            spaces = (t.description or {}).get("spaces") or {}
            for s in t.sections:
                ws.update(int(w) for w in (spaces.get(s["space"]) or {}).get("mrl") or [s["dim_real"] or s["dim"]])
        return sorted(ws, reverse=True)

    def media_files(self) -> list[dict]:
        enes = {os.path.splitext(t.path)[0]: t.id for t in self.titles.values()}
        videos = {t.video: t.id for t in self.titles.values() if t.video}
        out = []
        for root in self.roots:
            for dp, dns, fns in os.walk(root):
                dns[:] = [d for d in dns if d not in SKIP_DIRS]
                for f in fns:
                    if os.path.splitext(f)[1].lower() in VIDEO_EXT:
                        p = os.path.join(dp, f)
                        out.append({"path": p, "file": f, "dir": os.path.relpath(dp, root), "bytes": os.path.getsize(p),
                                    "ene": videos.get(p) or enes.get(os.path.splitext(p)[0])})
        return sorted(out, key=lambda m: m["path"])


# --------------------------------------------------------------------------------------- model
class Model:
    def __init__(self):
        self.embedder = None
        self.space: dict = {}             # the loaded model's `spaces` entry (§3.7)
        self.lock = threading.Lock()      # GPU: one user at a time
        self.state = "not loaded"
        self.error = None

    def start_preload(self) -> None:
        threading.Thread(target=self._safe_get, daemon=True, name="model-preload").start()

    def _safe_get(self):
        try:
            self.get()
        except Exception:  # noqa: BLE001
            pass

    def get(self):
        with self.lock:
            if self.embedder is None:
                self.state, self.error = "loading", None
                try:
                    from .embed import Embedder
                    self.embedder = Embedder()
                    self.space = self.embedder.space()
                    self.state = "loaded"
                except Exception as e:  # noqa: BLE001
                    self.error = f"{type(e).__name__}: {e}"
                    self.state = "error"
                    raise
            return self.embedder

    def info(self) -> dict:
        d = {"state": self.state, "error": self.error}
        e = self.embedder
        if e is not None:
            d |= {"name": e.name, "model_id": e.model_id, "space": e.space_id, "dim": e.dim, "widths": [int(w) for w in self.space.get("mrl") or [e.dim]],
                  "device": e.device, "dtype": str(e.dtype).replace("torch.", ""), "load_s": round(e.load_s, 1), "vram_gib": round(e.vram(), 2), "revision": e.revision}
        return d

    def embed_query(self, text: str | None, image: bytes | None, audio: bytes | None) -> tuple[np.ndarray, dict]:
        """Text -> the query prompt; image / audio -> direct; combinations -> interleaved (SPEC §7)."""
        m = self.get()
        from PIL import Image
        img = Image.open(io.BytesIO(image)).convert("RGB") if image else None
        clip = None
        if audio:
            from . import media
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
                f.write(audio)
                tmp = f.name
            try:
                clip = media.audio_f32(tmp)[: MAX_QUERY_AUDIO_S * media.SR]
            finally:
                os.unlink(tmp)
        kind = "+".join(k for k, v in (("text", text), ("image", img), ("audio", clip is not None)) if v)
        if not kind:
            raise HTTPException(400, "empty query: give a text, an image or an audio clip")
        t0 = time.perf_counter()
        with self.lock:
            if kind == "text":
                q = m.queries([text])
            elif kind == "image":
                q = m.images([img])
            elif kind == "audio":
                q = m.audio([clip])
            else:
                t = (text or "") + (" <|image|>" if img else "") + (" <|audio|>" if clip is not None else "")
                q = m.interleaved(t.strip(), images=[img] if img else None, audio=[clip] if clip is not None else None)
        return q[0], {"kind": kind, "embed_ms": round((time.perf_counter() - t0) * 1000, 1), "audio_s": None if clip is None else round(len(clip) / 16000, 1)}


# --------------------------------------------------------------------------------------- frames
class Frames:
    def __init__(self, cap: int = 400):
        self.cache: OrderedDict[tuple, bytes] = OrderedDict()
        self.cap = cap
        self.lock = threading.Lock()

    def get(self, video: str, t: float, w: int) -> bytes:
        key = (video, round(t * 4) / 4, w)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{key[1]:.3f}", "-i", video, "-frames:v", "1", "-vf", f"scale={w}:-2", "-q:v", "5",
               "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"]
        out = subprocess.run(cmd, capture_output=True).stdout
        if not out:
            raise HTTPException(404, "frame unavailable")
        with self.lock:
            self.cache[key] = out
            while len(self.cache) > self.cap:
                self.cache.popitem(last=False)
        return out


# --------------------------------------------------------------------------------------- build
@dataclass
class Job:
    id: str
    params: dict
    status: str = "pending"
    log: list[str] = field(default_factory=list)
    started: float = 0.0
    ended: float = 0.0
    out: str | None = None
    title_id: str | None = None
    error: str | None = None

    def dict(self) -> dict:
        return {"id": self.id, "status": self.status, "log": self.log, "started": self.started, "ended": self.ended,
                "elapsed_s": round((self.ended or time.time()) - self.started, 1) if self.started else 0,
                "out": self.out, "title_id": self.title_id, "error": self.error, "params": self.params}


class Jobs:
    def __init__(self, lib: Library, model: Model):
        self.lib, self.model = lib, model
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def start(self, params: dict) -> Job:
        with self.lock:
            if any(j.status == "running" for j in self.jobs.values()):
                raise HTTPException(409, "a build is already running")
            job = Job(uuid.uuid4().hex[:8], params)
            self.jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True, name=f"build-{job.id}").start()
        return job

    def _run(self, job: Job) -> None:
        from .build import build_title, save_master, write_ene
        p = job.params
        job.status, job.started = "running", time.time()

        def log(m: str) -> None:
            job.log.append(f"{time.time() - job.started:6.1f}s {m}")

        try:
            log("[model] loading...")
            m = self.model.get()
            log(f"[model] {m.name} on {m.device} {m.dtype} ready in {m.load_s:.1f}s")
            srt = {}
            for line in (p.get("srt") or "").splitlines():
                if "=" in line:
                    lang, path = line.split("=", 1)
                    srt[lang.strip()] = path.strip()
            with self.model.lock:
                built = build_title(p["file"], m, title=p["title"], year=p.get("year"), ids={"imdb": p["imdb"]} if p.get("imdb") else {},
                                    synopsis=p.get("synopsis") or "", tags=p.get("tags") or [], srt=srt, full_hash=not p.get("no_sha256"), log=log)
            out = p.get("out") or os.path.splitext(p["file"])[0] + ".ene"
            n = write_ene(built, out, p.get("policy") or DEFAULT_POLICY_NAME)
            master = os.path.splitext(out)[0] + ".master.npz"
            save_master(built, master)
            log(f"[written] {out}: {n:,} B | master fp32: {master} | peak VRAM {m.vram():.2f} GiB")
            job.out, job.status = out, "done"
            self.lib.scan()
            job.title_id = _tid(out)
        except Exception as e:  # noqa: BLE001
            job.error, job.status = f"{type(e).__name__}: {e}", "error"
            log(f"[error] {job.error}")
        finally:
            job.ended = time.time()


# --------------------------------------------------------------------------------------- app
def _j(o):
    """JSON-safe: numpy -> lists / scalars, floats rounded to 4 decimals."""
    if isinstance(o, np.ndarray):
        return [_j(v) for v in o.tolist()] if o.dtype.kind == "f" else o.tolist()
    if isinstance(o, (np.floating, float)):
        return round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, dict):
        return {k: _j(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_j(v) for v in o]
    return o


def create_app(roots: list[str], preload: bool = True) -> FastAPI:
    lib, model, frames = Library(roots), Model(), Frames()
    jobs = Jobs(lib, model)
    lib.scan()
    if preload:
        model.start_preload()
    t_start = time.time()
    app = FastAPI(title=".ene", version=__version__)
    app.state.lib, app.state.model, app.state.jobs = lib, model, jobs

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-cache"})

    @app.get("/api/status")
    def status():
        return {"version": __version__, "roots": lib.roots, "titles": len(lib.titles), "scanned_at": lib.scanned_at, "uptime_s": round(time.time() - t_start),
                "model": model.info(), "widths": lib.widths(), "policies": list(POLICIES), "jobs": [j.dict() | {"log": j.log[-1:]} for j in jobs.jobs.values()]}

    @app.get("/api/library")
    def library():
        with lib.lock:
            ts = list(lib.titles.values())
        return [t.summary() for t in sorted(ts, key=lambda t: (t.info.get("name") or "").lower())]

    @app.post("/api/rescan")
    def rescan():
        lib.scan()
        return {"titles": len(lib.titles)}

    @app.get("/api/title/{tid}")
    def title(tid: str):
        t = lib.get(tid)
        f = t.load()
        fps = tuple(t.info["axis_fps"])
        by_key = {(s.version_id.hex(), s.id): s for s in f.title()}
        secs = []
        for row in t.sections:
            s = by_key.get((row["version"], row["id"]))
            if s is None:
                continue
            ms = _ms(s.axis_times, fps)
            secs.append(row | {"times": None if ms is None else ms.tolist(), "frames": None if s.times is None else s.times.tolist(), "texts": s.texts})
        return {"summary": t.summary(), "description": t.description, "packets": t.packets, "sections": secs, "video": t.video, "master": t.master, "bench": t.bench}

    @app.get("/api/title/{tid}/section/{sid}/vectors")
    def section_vectors(tid: str, sid: str, dim: int | None = None):
        t = lib.get(tid)
        f = t.load()
        try:
            s = f.section(sid)
        except KeyError:
            raise HTTPException(404, "unknown section")
        m = (mrl(s.vectors, dim) if dim else s.vectors).astype("<f4")
        return Response(m.tobytes(), media_type="application/octet-stream", headers={"X-Shape": f"{m.shape[0]}x{m.shape[1]}"})

    @app.post("/api/search")
    async def search(q: str | None = Form(None), scope: str = Form("title"), title: str | None = Form(None), dim: int | None = Form(None), k: int = Form(8),
                     image: UploadFile | None = File(None), audio: UploadFile | None = File(None)):
        if dim is not None and dim not in lib.widths():
            raise HTTPException(400, f"dim must be a width the files declare, {lib.widths()} (default: the native width of every section)")
        img = await image.read() if image else None
        aud = await audio.read() if audio else None
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(_search, q, scope, title, dim, max(1, min(k, 50)), img, aud)

    def _hit(s: api.Section, i: int, score: float, fps) -> dict:
        ms = _ms(s.axis_times, fps)
        tx = s.texts
        return {"i": int(i), "score": float(score), "t0": None if ms is None else int(ms[i][0]), "t1": None if ms is None else int(ms[i][1]),
                "text": tx[i] if tx else None, "section": s.id}

    def _search(q, scope, tid, dim, k, img, aud):
        qv, qinfo = model.embed_query((q or "").strip() or None, img or None, aud or None)
        qd = mrl(qv[None, :], dim)[0] if dim else qv
        qinfo |= {"dim": dim or int(qv.shape[0]), "k": k, "text": q}
        if scope == "library":
            with lib.lock:
                ts = list(lib.titles.values())
            out = []
            for t in ts:
                t.load()
                fps = tuple(t.info["axis_fps"])
                parts, roles, best = {}, {}, None
                for s in t.ordered():
                    sc = s.scores(qd, dim)[0]
                    i = int(np.argmax(sc))
                    parts[s.part] = max(parts.get(s.part, -1.0), float(sc[i]))
                    roles[s.role] = max(roles.get(s.role, -1.0), float(sc[i]))
                    if best is None or sc[i] > best["score"]:
                        best = _hit(s, i, sc[i], fps)
                out.append({"id": t.id, "name": t.info["name"], "year": t.info["year"], "duration_ms": t.info["duration_ms"], "has_video": t.video is not None,
                            "score": max(parts.values()) if parts else 0.0, "parts": parts, "roles": roles, "best": best})
            out.sort(key=lambda r: -r["score"])
            return _j({"query": qinfo, "titles": out})
        t = lib.get(tid)
        t.load()
        fps = tuple(t.info["axis_fps"])
        secs = []
        for s in t.ordered():
            t0 = time.perf_counter()
            sc = s.scores(qd, dim)[0]
            ms = (time.perf_counter() - t0) * 1000
            order = np.argsort(-sc)[:k]
            secs.append({"id": s.id, "version": s.version_id.hex(), "part": s.part, "role": s.role, "modality": s.modality_name, "count": int(sc.shape[0]), "ms": round(ms, 2),
                         "min": float(sc.min()), "max": float(sc.max()), "mean": float(sc.mean()), "scores": sc, "top": [_hit(s, i, sc[i], fps) for i in order]})
        return _j({"query": qinfo, "title": t.summary(), "sections": secs})

    @app.get("/api/title/{tid}/frame")
    def frame(tid: str, t: float = 0.0, w: int = 320):
        tt = lib.get(tid)
        if not tt.video:
            raise HTTPException(404, "no video source")
        w = max(64, min(int(w), 1280))
        return Response(frames.get(tt.video, max(0.0, t), w), media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    @app.get("/api/title/{tid}/video")
    def video(tid: str):
        t = lib.get(tid)
        if not t.video:
            raise HTTPException(404, "no video source")
        return FileResponse(t.video, media_type=MIME.get(os.path.splitext(t.video)[1].lower(), "application/octet-stream"))

    @app.get("/api/title/{tid}/bench")
    def bench(tid: str):
        t = lib.get(tid)
        if not t.bench:
            raise HTTPException(404, "no benchmark")
        return Response(open(t.bench, encoding="utf-8").read(), media_type="text/markdown; charset=utf-8")

    @app.get("/api/media")
    def media_list():
        return lib.media_files()

    @app.post("/api/build")
    def build(file: str = Form(...), title: str = Form(...), year: int | None = Form(None), imdb: str | None = Form(None), synopsis: str | None = Form(None),
              tags: str | None = Form(None), srt: str | None = Form(None), policy: str = Form(DEFAULT_POLICY_NAME), no_sha256: bool = Form(False), out: str | None = Form(None)):
        if not os.path.isfile(file):
            raise HTTPException(400, f"file not found: {file}")
        if policy not in POLICIES:
            raise HTTPException(400, f"unknown policy {policy!r}")
        job = jobs.start({"file": file, "title": title.strip(), "year": year, "imdb": (imdb or "").strip() or None, "synopsis": (synopsis or "").strip(),
                          "tags": [x.strip() for x in (tags or "").split(",") if x.strip()], "srt": srt or "", "policy": policy, "no_sha256": no_sha256,
                          "out": (out or "").strip() or None})
        return job.dict()

    @app.get("/api/jobs")
    def jobs_list():
        return [j.dict() for j in jobs.jobs.values()]

    @app.get("/api/jobs/{jid}")
    def job_get(jid: str):
        j = jobs.jobs.get(jid)
        if j is None:
            raise HTTPException(404, "unknown job")
        return j.dict()

    @app.post("/api/model/load")
    def model_load():
        model.start_preload()
        return model.info()

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def serve(roots: list[str], host: str = "127.0.0.1", port: int = 8765, preload: bool = True, open_browser: bool = True) -> None:
    import uvicorn
    app = create_app(roots, preload=preload)
    url = f"http://{host}:{port}/"
    print(f"[ene] web GUI at {url}  (roots: {', '.join(app.state.lib.roots)}; {len(app.state.lib.titles)} titles)")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
