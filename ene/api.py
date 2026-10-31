"""Public Python API on the 1.0 container (SPEC-ene.md §9): open, select, search, write, edit, overlay, compact, rewrite,
the in-memory local space (§5) and a small library over several files.

    import ene

    f = ene.open("film.ene")                             # EneFile: packets grouped by title, overlays resolved (§3.9)
    t = f.title()                                        # Title (a file normally holds one); f.titles for all of them
    t.name, t.description, t.axis_fps, t.versions, t.sections
    secs = t.select(part="timeline", modality="image")   # by part / role / modality / lang / version / space / axis frames / span
    s = t.section("frame")                               # by id (and by version when the id exists in several versions)
    s.vectors, s.times, s.axis_times, s.texts, s.scale, s.prep, s.prep_object
    s.scores(q, dim=256)                                 # float32 [nq, count]; the int8 scale folded in; Matryoshka at read time
    hits = f.search(q, k=10, modality="image")           # list[Hit] across the selected sections, best first
    hits = f.search(q, k=10, frames=(a0, a1), axis=True) # rows overlapping an axis span; segments returned on the axis

    # edits are staged on the object and written by save() as ONE packet appended to the file (§0.8, §3.9)
    f.set(title={"name": "Big Buck Bunny", "year": 2008})                      # description keys, merged key by key
    f.add_version(version_id, fps=(25, 1), duration=n, map_a=1.2, map_b=0.0)   # a new contribution
    f.set_alignment(version_id, a, b)                    # a corrected map = the version record again, rev + 1
    f.add_section("cue.ron", version_id, space=space_id, modality="text", role="cue", lang="ron", vectors=x, times=t, texts=cues, prep={...})
    f.remove_section("tag", version_id)                  # a tombstone
    f.save()                                             # append one packet; f.save("copy.ene") = a compacted copy with the changes
    f.compact()                                          # rewrite in place as one packet per title (lossless, §3.10)

    ene.create(path, title_id, axis_fps, versions=[VersionSpec(...)], sections=[SectionSpec(...)], description={...}, policy="compact")
    ene.overlay("film.ene", "contribution.ene")          # append another file's packets (cat, §3.1)
    ene.compact("film.ene", "film.c.ene")
    ene.rewrite("film.ene", "film.fast.ene", policy="fast")

    sp = ene.Space(); sp.ingest("a.ene"); sp.ingest("b.ene")   # the local space (§5): one matrix per key, duplicates by `prep`
    sp.search(q, k=10, part="timeline", frames=(a0, a1))
    ene.Library(["a.ene", "b.ene"]).titles(q)            # which title? then .search(q): which moment?

Everything here works without the embedding model and without torch: queries are vectors the caller already has.

Loading path (§8.2): a title written by policy `fast` (one packet, ONE_BLOCK) is scored by one GEMV on the file bytes (fp32 rows)
or one cast and one GEMV (int8 rows, in parallel from PARALLEL_ROWS on); other sections are cast chunk by chunk for a one-off
query and materialize a float32 matrix only from the third query on. The int8 scale is folded into the scores.
"""

from __future__ import annotations

import builtins
import itertools
import os
import queue
import shutil
import tempfile
import threading
from bisect import bisect_right

import numpy as np

from . import container as C
from .container import MODALITIES, V_REFERENCE, V_TIMELINE, ZERO16, Version, prep_hash
from .quant import mrl

__all__ = ["open", "create", "overlay", "compact", "rewrite", "EneFile", "Title", "Version", "Section", "Hit", "Writer", "VersionSpec",
           "SectionSpec", "Policy", "POLICIES", "DEFAULT_POLICY", "DEFAULT_POLICY_NAME", "Space", "SpaceHit", "SectionRef", "Library"]

# --- storage policies (§4.4) -------------------------------------------------------------------------------------------

COMPACT_SMALL_ROWS = 32            # `compact`: fp16 + zstd below, int8 + ene-xdelta from here on (§4.4; the threshold is not swept, §10)
FAST_INT8_ROWS = 1024              # `fast`: fp32 raw below this many rows in the packet, int8 raw from here on (bench/loading.md §1b)


class Policy:
    """How a writer stores a section (§4.4): quant and codec from the section's row count and the packet's; `shared` policies
    write every section as a row range of one raw block (`fast`), with the form chosen by the packet's row count unless forced."""
    __slots__ = ("name", "small", "large", "threshold", "shared", "form")

    def __init__(self, name: str, small=None, large=None, threshold: int = 0, shared: bool = False, form: str | None = None):
        self.name, self.small, self.large, self.threshold, self.shared, self.form = name, small, large, threshold, shared, form

    def storage(self, rows: int, packet_rows: int) -> tuple[str, str]:
        if self.shared:
            return (self.form or ("int8" if packet_rows >= FAST_INT8_ROWS else "fp32"), "none")
        return self.small if rows < self.threshold else self.large

    def __repr__(self) -> str:
        return f"Policy({self.name})"


POLICIES = {
    "compact": Policy("compact", ("fp16", "zstd"), ("int8", "ene-xdelta"), COMPACT_SMALL_ROWS),
    "fast": Policy("fast", shared=True),
    "fast-fp32": Policy("fast-fp32", shared=True, form="fp32"),
    "fast-int8": Policy("fast-int8", shared=True, form="int8"),
    "lossless": Policy("lossless", ("fp32", "zstd"), ("fp32", "zstd")),
}
DEFAULT_POLICY_NAME = "compact"
DEFAULT_POLICY = POLICIES[DEFAULT_POLICY_NAME]


def policy_of(policy) -> Policy:
    if policy is None:
        return DEFAULT_POLICY
    if isinstance(policy, Policy):
        return policy
    if policy not in POLICIES:
        raise KeyError(f"unknown policy {policy!r}; one of {list(POLICIES)}")
    return POLICIES[policy]


# --- scoring kernels (bench/loading.md; unchanged from the measured reader) ---------------------------------------------

CHUNK_ROWS = 256                   # 256 x 768 float32 = 786 KB: the cast scratch stays in L2, no fresh pages are touched per query
PARALLEL_ROWS = 1024               # raw blocks with at least this many rows are cast by PARALLEL_THREADS threads
PARALLEL_THREADS = min(6, os.cpu_count() or 1)   # the calling thread plus PARALLEL_THREADS - 1 workers
SUPER_ROWS = 8192                  # rows per parallel pass (bounds the shared float32 scratch to 24 MiB at 768 dims)
PULL_ROWS = 128                    # rows per chunk pulled by a thread of the parallel cast (~35 us of work)
_NP = {"int8": np.dtype(np.int8), "fp16": np.dtype("<f2"), "fp32": np.dtype("<f4")}
_scratch = threading.local()


class _Job:
    """One parallel cast: chunks of `chunk` rows pulled from a shared counter (atomic under the GIL); the thread that finishes
    the last chunk releases the waiter."""
    __slots__ = ("fn", "n", "chunk", "next", "done", "total", "ev")

    def __init__(self, fn, n: int, chunk: int):
        self.fn, self.n, self.chunk = fn, n, chunk
        self.next = itertools.count()
        self.done = itertools.count()
        self.total = -(-n // chunk)
        self.ev = threading.Event()

    def work(self) -> None:
        fn, n, ch, nxt, dn, total = self.fn, self.n, self.chunk, self.next, self.done, self.total
        while True:
            a = next(nxt) * ch
            if a >= n:
                return
            fn(a, min(a + ch, n))
            if next(dn) + 1 == total:
                self.ev.set()

    def wait(self) -> None:
        if self.total:
            self.ev.wait()


class _CastPool:
    """Dedicated cast workers, one SimpleQueue each (a dispatch costs ~1 us per worker; workers block in C between jobs and
    never hold the GIL while waiting). Work is pulled in chunks by the calling thread and by every worker as it arrives, so a
    worker whose core was parked (tens to hundreds of microseconds of wake-up after an idle period) simply takes fewer chunks
    and the calling thread never waits for a worker that has not started."""

    def __init__(self, n: int):
        self.n = n
        self.qs = [queue.SimpleQueue() for _ in range(n)]
        self.err: list = []
        for i in range(n):
            threading.Thread(target=self._run, args=(i,), daemon=True, name=f"ene-cast-{i}").start()

    def _run(self, i: int) -> None:
        qi = self.qs[i]
        while True:
            job = qi.get()
            try:
                job.work()
            except BaseException as e:  # noqa: BLE001 - re-raised on the calling thread
                self.err.append(e)
                job.ev.set()

    def run(self, fn, n: int, chunk: int) -> None:
        """fn(a0, a1) over [0, n) in chunks of `chunk` rows, on the calling thread and the workers together."""
        job = _Job(fn, n, chunk)
        for qi in self.qs:
            qi.put(job)
        job.work()
        job.wait()
        if self.err:
            e = self.err[:]
            self.err.clear()
            raise e[0]


_CAST_POOL: _CastPool | None = None
_FULL: np.ndarray | None = None
_PAR_LOCK = threading.Lock()


def _cast_pool() -> _CastPool:
    global _CAST_POOL
    if _CAST_POOL is None:
        _CAST_POOL = _CastPool(PARALLEL_THREADS - 1)
    return _CAST_POOL


def _full_scratch(rows: int, dim: int) -> np.ndarray:
    """The process-wide float32 scratch the parallel cast writes into (persistent: no page faults after the first use)."""
    global _FULL
    if _FULL is None or _FULL.shape[0] < rows or _FULL.shape[1] < dim:
        _FULL = np.empty((max(rows, min(SUPER_ROWS, 2048)), max(dim, 768)), np.float32)
    return _FULL


def _scratch_buf(rows: int, dim: int) -> np.ndarray:
    b = getattr(_scratch, "buf", None)
    if b is None or b.shape[0] < rows or b.shape[1] < dim:
        b = np.empty((max(rows, CHUNK_ROWS), max(dim, 768)), np.float32)
        _scratch.buf = b
    return b


def _score_rows(src: np.ndarray, d: int, full: int, qd: np.ndarray, sc: np.ndarray, norms: np.ndarray | None) -> None:
    """sc[i] = qd . float32(src[i, :d]) for every row, and norms[i] = |float32(src[i, :d])| when asked. Small blocks: one cast
    into the L2 scratch and one GEMV; medium: 256-row chunks; from PARALLEL_ROWS on: all rows cast in parallel into the shared
    float32 scratch (PULL_ROWS-row chunks pulled by the calling thread and PARALLEL_THREADS - 1 workers), then ONE GEMV over it."""
    total = src.shape[0]
    qv = qd[0]
    if total >= PARALLEL_ROWS and PARALLEL_THREADS > 1:
        with _PAR_LOCK:
            pool = _cast_pool()
            for a in range(0, total, SUPER_ROWS):
                b = min(a + SUPER_ROWS, total)
                n = b - a
                buf = _full_scratch(n, d)[:n, :d]
                blk = src[a:b, :d] if d < full else src[a:b]

                def cast(a0: int, a1: int) -> None:
                    np.copyto(buf[a0:a1], blk[a0:a1], casting="unsafe")
                pool.run(cast, n, PULL_ROWS)
                np.dot(buf, qv, out=sc[a:b])
                if norms is not None:
                    np.sqrt(np.einsum("ij,ij->i", buf, buf), out=norms[a:b])
        return
    if total <= 2 * CHUNK_ROWS and norms is None:                            # small block: one cast, one GEMV, no loop
        c = _scratch_buf(total, d)[:total, :d]
        np.copyto(c, src[:, :d] if d < full else src, casting="unsafe")
        np.dot(c, qv, out=sc)
        return
    buf = _scratch_buf(CHUNK_ROWS, d)
    for a in range(0, total, CHUNK_ROWS):
        b = min(a + CHUNK_ROWS, total)
        c = buf[: b - a, :d]
        np.copyto(c, src[a:b, :d], casting="unsafe")
        np.dot(c, qv, out=sc[a:b])
        if norms is not None:
            np.sqrt(np.einsum("ij,ij->i", c, c), out=norms[a:b])


def _score_block(src: np.ndarray, scale: np.ndarray | None, q2: np.ndarray, d: int) -> np.ndarray:
    """Cosine scores of ONE unit query (`q2`: float32 [1, >= d]) against every row of a block in its stored element type
    (`src`: [n, full]). fp32 rows at full width: one GEMV in place, nothing copied. Otherwise the rows are cast chunk by chunk;
    at full width the int8 scale is folded into the result (|m8_i| = 1 / s_i), truncated the norms are computed on the way."""
    n, full = src.shape
    d = min(d, q2.shape[1])                                                   # a narrower query scores the rows' prefix (§4.3)
    qd = mrl(q2, d)                                                          # truncated and renormalized when wider than d
    sc = np.empty(n, np.float32)
    if src.dtype == np.float32:
        if d == full:
            np.dot(src, qd[0], out=sc)
            if scale is not None:
                sc *= scale
            return sc
        sub = src[:, :d]
        np.dot(sub, qd[0], out=sc)
        norms = np.sqrt(np.einsum("ij,ij->i", sub, sub))
        norms[norms == 0] = 1.0
        sc /= norms
        return sc
    norms = np.empty(n, np.float32) if d < full else None
    _score_rows(src, d, full, qd, sc, norms)
    if norms is not None:
        norms[norms == 0] = 1.0
        sc /= norms
    elif scale is not None:
        sc *= scale
    return sc


def _topk(sc: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k best scores, best first (`-inf` rows never returned)."""
    n = len(sc)
    if k >= n:
        idx = np.argsort(sc)[::-1]
    elif n <= 1024:                                                           # small block: one full argsort beats partition + sort
        idx = np.argsort(sc)[:-k - 1:-1]
    else:
        part = np.argpartition(sc, n - k)[n - k:]
        idx = part[np.argsort(sc[part])[::-1]]
    if len(idx) and not np.isfinite(sc[idx[-1]]):
        idx = idx[np.isfinite(sc[idx])]
    return idx


# --- small helpers -------------------------------------------------------------------------------------------------------

def _unit(x) -> np.ndarray:
    """float32 rows, L2-normalized (a 1-D input becomes one row)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None]
    if x.ndim != 2:
        raise ValueError("vectors must be a matrix")
    n = np.sqrt(np.einsum("ij,ij->i", x, x))[:, None]
    n[n == 0] = 1.0
    return x / n


def _unit_query(q) -> np.ndarray:
    """One query as a unit float32 [1, d] row; a unit contiguous float32 vector passes through untouched."""
    if isinstance(q, np.ndarray) and q.ndim == 1 and q.dtype == np.float32 and q.flags.c_contiguous:
        n2 = float(np.dot(q, q))
        return q[None] if abs(n2 - 1.0) < 1e-6 or n2 == 0.0 else q[None] / np.float32(n2 ** 0.5)
    q2 = _unit(q)
    if q2.shape[0] != 1:
        raise ValueError("search takes one query vector; use scores() for several")
    return q2


def _merge(a: dict, b: dict) -> dict:
    """Description overlay (§3.9): key by key, at every level, the later winning on every key it carries."""
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _version_id(v) -> bytes:
    if isinstance(v, (Version, VersionSpec)):
        return v.id
    if isinstance(v, str):
        v = bytes.fromhex(v)
    b = bytes(v)
    if len(b) != 16:
        raise ValueError("a version id is 16 bytes")
    return b


def _to_axis(map_a: float, map_b: float, segments: np.ndarray) -> np.ndarray:
    """Version frames to axis frames (§2.8): floor(a·f0 + b), ceil(a·f1 + b); int64[..., 2]."""
    s = np.asarray(segments, dtype=np.float64)
    out = np.empty(s.shape, dtype=np.int64)
    out[..., 0] = np.floor(map_a * s[..., 0] + map_b)
    out[..., 1] = np.ceil(map_a * s[..., 1] + map_b)
    return out


def _block_view(r: C.Section) -> np.ndarray:
    """The whole raw `vectors` block a section belongs to (codec `none`), as a zero-copy array [rows, dim]."""
    rec = r.rec
    mv = r.packet.block(int(rec["vec_off"]), int(rec["vec_len"]), int(rec["vec_raw"]), 0)
    return np.frombuffer(mv, dtype=_NP[r.quant]).reshape(-1, r.dim)


def _scale_block_view(r: C.Section) -> np.ndarray | None:
    if not r.flags & C.S_HAS_SCALE:
        return None
    rec = r.rec
    mv = r.packet.block(int(rec["scale_off"]), int(rec["scale_len"]), int(rec["scale_raw"]), int(rec["aux_codec"]))
    return np.frombuffer(mv, dtype="<f4")


# --- what a writer takes (§2.2, §2.4, §2.5) -------------------------------------------------------------------------------

class VersionSpec:
    """A version record to write (§2.2). `reference` = this version defines the axis (identity map at the axis rate);
    `timeline` defaults to "has a frame rate". `rev` replaces an earlier record of the same id when higher."""
    __slots__ = ("id", "source_hash", "fps", "duration", "map_a", "map_b", "rev", "flags")

    def __init__(self, id, *, source_hash: bytes = ZERO16, fps: tuple[int, int] = (0, 0), duration: int = 0, map_a: float = 1.0,  # noqa: A002
                 map_b: float = 0.0, reference: bool = False, timeline: bool | None = None, rev: int = 0, flags: int | None = None):
        self.id = _version_id(id)
        self.source_hash = bytes(source_hash)
        self.fps = (int(fps[0]), int(fps[1]))
        self.duration, self.map_a, self.map_b, self.rev = int(duration), float(map_a), float(map_b), int(rev)
        if flags is None:
            if timeline is None:
                timeline = self.fps != (0, 0) or reference
            flags = (V_REFERENCE if reference else 0) | (V_TIMELINE if timeline else 0)
        self.flags = int(flags)

    @classmethod
    def of(cls, v: Version, **changes) -> VersionSpec:
        """A copy of a stored record, with changes (`map_a`, `map_b`, `rev`, ...)."""
        kw = dict(source_hash=v.source_hash, fps=v.fps, duration=v.duration, map_a=v.map_a, map_b=v.map_b, rev=v.rev, flags=v.flags)
        kw.update(changes)
        return cls(v.id, **kw)

    reference = property(lambda self: bool(self.flags & V_REFERENCE))
    timeline = property(lambda self: bool(self.flags & V_TIMELINE))

    def __repr__(self) -> str:
        return f"VersionSpec({self.id.hex()[:8]}…, fps {self.fps[0]}/{self.fps[1]}, map {self.map_a}·f+{self.map_b}, rev {self.rev})"


class SectionSpec:
    """A section to write (§2.5): float32 rows (normalized here), the version they belong to, the space they come from (§2.7,
    always named by the caller), part by the presence of `times`
    (frames of the version, `i32[count, 2]`), optional texts, the input fingerprint `prep` (an object, §2.7, also written into the
    description) and, optionally, a quant/codec that overrides the policy."""
    __slots__ = ("id", "version", "modality", "role", "vectors", "times", "texts", "lang", "space", "prep", "ref", "rev", "quant", "codec", "dim_real")

    def __init__(self, id, version, *, modality, role: str, vectors, times=None, texts: list[str] | None = None, lang: str = "",  # noqa: A002
                 space: str, prep=None, ref: str = "", rev: int = 0, quant: str | None = None, codec: str | None = None,
                 dim_real: int | None = None):
        self.id, self.version, self.modality, self.role, self.lang, self.space = str(id), _version_id(version), modality, str(role), str(lang), str(space)
        self.vectors = _unit(vectors)
        self.times = None if times is None else np.ascontiguousarray(times, dtype=np.int32).reshape(-1, 2)
        self.texts = None if texts is None else [str(t) for t in texts]
        n = self.vectors.shape[0]
        if self.times is not None and self.times.shape[0] != n:
            raise ValueError("times must have one [f0, f1] row per vector")
        if self.texts is not None and len(self.texts) != n:
            raise ValueError("texts must have one entry per vector")
        self.prep, self.ref, self.rev, self.quant, self.codec = prep, str(ref), int(rev), quant, codec
        self.dim_real = None if dim_real is None else int(dim_real)

    count = property(lambda self: int(self.vectors.shape[0]))
    dim = property(lambda self: int(self.vectors.shape[1]))
    timeline = property(lambda self: self.times is not None)


class Writer:
    """One packet of one title, written under a storage policy: versions, sections (normalized, quantized and encoded as the policy
    says; `prep` objects copied into the description), tombstones, description. `fast` policies put every section without an
    explicit quant/codec into one shared raw block (§3.6)."""

    def __init__(self, title_id: bytes, axis_fps: tuple[int, int], policy=None, description: dict | None = None):
        self.pw = C.PacketWriter(title_id, axis_fps)
        self.policy = policy_of(policy)
        self.versions: dict[bytes, int] = {}
        self.specs: list[SectionSpec] = []
        self.tombstones: list[tuple[str, bytes, int]] = []
        self.description: dict = dict(description) if description else {}
        self._bytes: bytes | None = None

    title_id = property(lambda self: self.pw.title_id)
    axis_fps = property(lambda self: self.pw.axis_fps)

    def add_version(self, version, **kw) -> bytes:
        spec = version if isinstance(version, VersionSpec) else VersionSpec(version, **kw)
        if spec.id in self.versions:
            raise ValueError(f"version {spec.id.hex()} is already in this packet")
        self.versions[spec.id] = self.pw.add_version(spec.id, source_hash=spec.source_hash, fps=spec.fps, map_a=spec.map_a, map_b=spec.map_b,
                                                     duration=spec.duration, rev=spec.rev, flags=spec.flags)
        self._bytes = None
        return spec.id

    def add_section(self, section, version=None, **kw) -> SectionSpec:
        spec = section if isinstance(section, SectionSpec) else SectionSpec(section, version, **kw)
        if spec.version not in self.versions:
            raise ValueError(f"section {spec.id!r} names version {spec.version.hex()}, which is not in this packet; add_version first")
        self.specs.append(spec)
        self._bytes = None
        return spec

    def add_tombstone(self, id: str, version, rev: int = 0) -> None:  # noqa: A002
        vid = _version_id(version)
        if vid not in self.versions:
            raise ValueError("a tombstone names a version of this packet; add_version first")
        self.tombstones.append((str(id), vid, int(rev)))
        self._bytes = None

    def set_description(self, desc: dict) -> None:
        self.description = dict(desc)
        self._bytes = None

    def update_description(self, desc: dict) -> None:
        self.description = _merge(self.description, desc)
        self._bytes = None

    def to_bytes(self) -> bytes:
        if self._bytes is not None:
            return self._bytes
        total = sum(s.count for s in self.specs)
        desc = self.description
        shared: dict[tuple[str, int], list[dict]] = {}
        for s in self.specs:
            quant, codec = self.policy.storage(s.count, total)
            forced = s.quant is not None or s.codec is not None
            quant, codec = s.quant or quant, s.codec or codec
            if isinstance(s.prep, dict):
                desc = _merge(desc, {"sections": {f"{s.version.hex()}/{s.id}": {"prep": s.prep}}})
            kw = dict(id=s.id, version=self.versions[s.version], space=s.space, modality=s.modality, role=s.role, vectors=s.vectors, lang=s.lang,
                      dim_real=s.dim_real, times=s.times, texts=s.texts, rev=s.rev, ref=s.ref, prep=s.prep)
            if self.policy.shared and not forced and codec == "none":
                shared.setdefault((quant, s.dim), []).append(kw)
            else:
                self.pw.add_section(quant=quant, codec=codec, **kw)
        for (quant, _dim), items in shared.items():
            self.pw.add_shared(items, quant=quant, codec="none")
        for sid, vid, rev in self.tombstones:
            self.pw.add_tombstone(sid, self.versions[vid], rev)
        if desc:
            self.pw.set_description(desc)
        self._bytes = self.pw.to_bytes()
        return self._bytes

    def write(self, path: str, append: bool = False) -> int:
        data = self.to_bytes()
        with builtins.open(path, "ab" if append else "wb") as fh:
            fh.write(data)
        return len(data)


# --- reading -------------------------------------------------------------------------------------------------------------

class Hit:
    """One row found by a search: its section, row index, cosine score, segment (`f0`, `f1`: version frames, or axis frames when
    the search asked `axis=True`; None in the title part) and text."""
    __slots__ = ("section", "row", "score", "f0", "f1", "text")

    def __init__(self, section: Section, row: int, score: float, f0: int | None = None, f1: int | None = None, text: str | None = None):
        self.section, self.row, self.score, self.f0, self.f1, self.text = section, row, score, f0, f1, text

    id = property(lambda self: self.section.id)
    version_id = property(lambda self: self.section.version_id)
    role = property(lambda self: self.section.role)
    modality = property(lambda self: self.section.modality_name)

    @property
    def axis(self) -> tuple[int, int] | None:
        """The row's segment on the title's axis, whatever the search returned."""
        t = self.section.axis_times
        return None if t is None else (int(t[self.row, 0]), int(t[self.row, 1]))

    def __eq__(self, o) -> bool:
        return isinstance(o, Hit) and (self.section.version_id, self.section.id, self.row, self.score, self.f0, self.f1, self.text) == \
            (o.section.version_id, o.section.id, o.row, o.score, o.f0, o.f1, o.text)

    def __repr__(self) -> str:
        t = f" [{self.f0}, {self.f1})" if self.f0 is not None else ""
        x = f" {self.text[:50]!r}" if self.text else ""
        return f"Hit({self.section.id}#{self.row} {self.score:.3f}{t}{x})"


class Section:
    """A resolved section of an open title: the container's section (record fields, columns on demand) plus what search needs
    cached: the scoring matrix, the int8 scale, times, axis times, texts, row norms per width."""
    __slots__ = ("title", "raw", "_m", "_scale", "_vectors", "_norms", "_times", "_axis", "_texts", "_queries", "_src")

    def __init__(self, title: Title, raw: C.Section):
        self.title, self.raw = title, raw
        self._m = self._scale = self._vectors = self._src = None
        self._norms: dict[int, np.ndarray] = {}
        self._times = self._axis = self._texts = False
        self._queries = 0

    def __getattr__(self, name):                          # id, version_id, space, role, lang, modality, modality_name, quant, codec, dim,
        return getattr(self.raw, name)                    # dim_real, count, rev, flags, ref, prep, row_offset, timeline, key, version

    part = property(lambda self: "timeline" if self.raw.timeline else "title")
    loaded = property(lambda self: self._m is not None)

    @property
    def scale(self) -> np.ndarray | None:
        """The per-row int8 scale (a copy), None for other quantizations."""
        if self._scale is None and self.raw.quant == "int8":
            self._scale = np.array(self.raw.scales(), dtype=np.float32)
        return self._scale

    def _rows(self) -> np.ndarray:
        """The stored rows in their element type: a zero-copy view for codec `none`, a decoded array (kept) otherwise."""
        if self.raw.codec == "none":
            return self.raw.vectors(raw=True)
        if self._src is None:
            self._src = self.raw.vectors(raw=True)
        return self._src

    def load(self) -> Section:
        """Materialize the float32 scoring matrix (idempotent): the int8 cast (unscaled) or the dequantized rows."""
        if self._m is None:
            r = self.raw
            if r.quant == "int8":
                self._m = self._rows().astype(np.float32)
                _ = self.scale
            elif r.quant == "bin":
                self._m = np.array(r.vectors(), dtype=np.float32)
            else:
                self._m = np.array(self._rows(), dtype=np.float32)
            self._src = None
        return self

    @property
    def matrix(self) -> np.ndarray:
        """float32 [count, dim] used for scoring: the int8 cast (unscaled) or the dequantized vectors."""
        return self.load()._m

    @property
    def vectors(self) -> np.ndarray:
        """The true float32 vectors (scale applied for int8)."""
        self.load()
        if self._scale is None:
            return self._m
        if self._vectors is None:
            self._vectors = self._m * self._scale[:, None]
        return self._vectors

    def at(self, dim: int) -> np.ndarray:
        """The Matryoshka prefix of the vectors, renormalized (a copy; §4.3)."""
        return mrl(self.vectors, dim)

    @property
    def times(self) -> np.ndarray | None:
        """`i32[count, 2]` segments in the version's frames (a copy); None in the title part."""
        if self._times is False:
            t = self.raw.times()
            self._times = None if t is None else np.array(t, dtype=np.int32)
        return self._times

    @property
    def axis_times(self) -> np.ndarray | None:
        """The segments on the title's axis through the version's current map (§2.8); None in the title part."""
        if self._axis is False:
            t = self.times
            self._axis = None if t is None else self.raw.version.to_axis(t)
        return self._axis

    @property
    def texts(self) -> list[str] | None:
        if self._texts is False:
            self._texts = self.raw.texts()
        return self._texts

    @property
    def prep_object(self):
        """The input fingerprint object from the description (§3.7), None when the description does not carry it."""
        secs = self.title.description.get("sections") or {}
        return (secs.get(f"{self.raw.version_id.hex()}/{self.raw.id}") or {}).get("prep")

    def norms(self, dim: int) -> np.ndarray:
        """Row norms of the scoring matrix truncated to `dim` (cached). At full width int8 rows are unit up to rounding, so
        |m8_i| = 1 / s_i without a pass over the matrix."""
        n = self._norms.get(dim)
        if n is None:
            if dim >= self.raw.dim and self.scale is not None:
                n = 1.0 / self.scale
            else:
                m = self.matrix[:, :dim] if dim < self.raw.dim else self.matrix
                n = np.sqrt(np.einsum("ij,ij->i", m, m))
                n[n == 0] = 1.0
            self._norms[dim] = n
        return n

    def scores(self, q, dim: int | None = None) -> np.ndarray:
        """Cosine scores of one or more queries (normalized here): float32 [nq, count]; `dim` truncates both sides (§4.3)."""
        return self._scores_unit(_unit(q), dim)

    def _scores_chunked(self, q2: np.ndarray, d: int) -> np.ndarray | None:
        """Scores straight from the stored rows: CHUNK_ROWS rows at a time cast into a reusable scratch, one GEMV per chunk.
        No float32 matrix is built for a one-off query."""
        r = self.raw
        if r.quant == "bin":
            return None
        src = self._rows()
        scale = self.scale
        qd = mrl(q2, d)
        if r.count <= 64 and d == r.dim:                                      # tiny section: one cast, one GEMV, no scratch
            out = qd @ src.astype(np.float32).T
            if scale is not None:
                out *= scale[None, :]
            return out
        out = np.empty((qd.shape[0], r.count), np.float32)
        buf = _scratch_buf(CHUNK_ROWS, d)
        need_norms = d < r.dim
        norms = np.empty(r.count, np.float32) if need_norms else None
        for a in range(0, r.count, CHUNK_ROWS):
            b = min(a + CHUNK_ROWS, r.count)
            c = buf[: b - a, :d]
            np.copyto(c, src[a:b, :d], casting="unsafe")
            out[:, a:b] = qd @ c.T
            if need_norms:
                np.sqrt(np.einsum("ij,ij->i", c, c), out=norms[a:b])
        if need_norms:
            norms[norms == 0] = 1.0
            out /= norms[None, :]
        elif scale is not None:
            out *= scale[None, :]
        return out

    def _scores_unit(self, q2: np.ndarray, dim: int | None = None) -> np.ndarray:
        """`q2`: float32 [nq, >= dim], rows already unit. The first two queries are answered from the stored rows; the third
        materializes the float32 matrix (one GEMV plus a per-row scale afterwards)."""
        d = min(self.raw.dim if dim is None or dim >= self.raw.dim else dim, q2.shape[1])
        if self._m is None:
            self._queries += 1
            if self._queries <= 2:
                out = self._scores_chunked(q2, d)
                if out is not None:
                    return out
        m = self.matrix
        if d < self.raw.dim:
            s = mrl(q2, d) @ m[:, :d].T
            s *= (1.0 / self.norms(d))[None, :]
        else:
            s = mrl(q2, d) @ m.T
            if self._scale is not None:
                s *= self._scale[None, :]
        return s

    def __repr__(self) -> str:
        return repr(self.raw)


class Title:
    """Everything a file says about one title, overlays resolved: versions, sections (keyed by `(version_id, id)`), the
    description; selection and search over its sections."""

    def __init__(self, file: EneFile, ct: C.Title):
        self.file, self._t = file, ct
        self.sections: dict[tuple[bytes, str], Section] = {k: Section(self, s) for k, s in ct.sections.items()}
        self._order = list(self.sections.values())
        self._fast = None
        self._fast_queries = 0

    title_id = property(lambda self: self._t.title_id)
    axis_fps = property(lambda self: self._t.axis_fps)
    versions = property(lambda self: self._t.versions)
    packets = property(lambda self: self._t.packets)
    description = property(lambda self: self._t.description)

    @property
    def name(self) -> str | None:
        """The title's name: `description.title.name`, else the text of a `title` row."""
        n = (self.description.get("title") or {}).get("name")
        if n:
            return n
        for s in self._order:
            if s.raw.role == "title" and not s.raw.timeline:
                tx = s.texts
                if tx:
                    return tx[0]
        return None

    @property
    def reference(self) -> Version | None:
        for v in self._t.versions.values():
            if v.reference:
                return v
        return None

    def version(self, v) -> Version:
        return self._t.versions[_version_id(v)]

    def section(self, id: str, version=None) -> Section:  # noqa: A002
        """A section by id, and by version when the id exists in several versions."""
        if version is not None:
            return self.sections[(_version_id(version), id)]
        hits = [s for (vid, sid), s in self.sections.items() if sid == id]
        if len(hits) != 1:
            raise KeyError(f"{len(hits)} sections named {id!r}; give the version")
        return hits[0]

    def __iter__(self):
        return iter(self._order)

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, id: str) -> bool:  # noqa: A002
        return any(sid == id for _, sid in self.sections)

    def select(self, part: str | None = None, role: str | None = None, modality=None, lang: str | None = None, version=None,
               space: str | None = None, frames: tuple[int, int] | None = None, span: tuple[float, float] | None = None,
               ids=None, min_count: int = 1) -> list[Section]:
        """Sections by part (`title` / `timeline`), role, modality, language, version, space, axis frame range (sections with a
        row overlapping `[a0, a1)`), span (sections whose median row length on the axis lies in `[lo, hi]`), ids, row count."""
        if part not in (None, "title", "timeline"):
            raise ValueError("part is 'title' or 'timeline'")
        vid = None if version is None else _version_id(version)
        mod = None if modality is None else C.modality_code(modality)
        out = []
        for s in self._order:
            r = s.raw
            if part is not None and (part == "timeline") != r.timeline:
                continue
            if role is not None and r.role != role:
                continue
            if mod is not None and r.modality != mod:
                continue
            if lang is not None and r.lang != lang:
                continue
            if vid is not None and r.version_id != vid:
                continue
            if space is not None and r.space != space:
                continue
            if ids is not None and r.id not in ids:
                continue
            if r.count < min_count:
                continue
            if frames is not None or span is not None:
                if not r.timeline:
                    continue
                ax = s.axis_times
                if frames is not None and not ((ax[:, 1] > frames[0]) & (ax[:, 0] < frames[1])).any():
                    continue
                if span is not None:
                    med = float(np.median(ax[:, 1] - ax[:, 0]))
                    if not span[0] <= med <= span[1]:
                        continue
            out.append(s)
        return out

    def scores(self, section, q, dim: int | None = None) -> np.ndarray:
        s = self.section(section) if isinstance(section, str) else section
        return s.scores(q, dim)

    def _fast_state(self):
        """Policy `fast`, one packet with ONE_BLOCK: every section a row range of one raw block, the table in row order."""
        if self._fast is None:
            self._fast = False
            pks = self._t.packets
            if len(pks) == 1 and pks[0].one_block and self._order and len(self._order) == len(pks[0].sections):
                secs = sorted(self._order, key=lambda s: s.raw.row_offset)
                r0 = secs[0].raw
                if r0.quant in _NP and len({s.raw.space for s in secs}) == 1:
                    self._fast = (r0.quant, r0.dim, [s.raw.row_offset for s in secs], secs)
        return self._fast or None

    def _search_fast(self, q2: np.ndarray, dim: int | None, k: int, axis: bool) -> list[Hit]:
        quant, full, offs, secs = self._fast_state()
        r0 = secs[0].raw
        sc = _score_block(_block_view(r0), _scale_block_view(r0), q2, full if dim is None or dim >= full else dim)
        out = []
        for i in _topk(sc, k).tolist():
            j = bisect_right(offs, i) - 1
            out.append(_hit(secs[j], i - offs[j], float(sc[i]), axis))
        return out

    def search(self, q, k: int = 10, dim: int | None = None, *, axis: bool = False, part: str | None = None, role: str | None = None,
               modality=None, lang: str | None = None, version=None, space: str | None = None, frames: tuple[int, int] | None = None,
               span=None, ids=None, min_count: int = 1) -> list[Hit]:
        """Top-k rows across the selected sections (the filters of `select`; `frames` also keeps only the rows overlapping the
        axis span), best first. Segments in version frames, or on the axis with `axis=True`."""
        q2 = _unit_query(q)
        filtered = any(x is not None for x in (part, role, modality, lang, version, space, frames, span, ids)) or min_count > 1
        if not filtered and self._fast_queries < 2 and self._fast_state() is not None:
            self._fast_queries += 1
            return self._search_fast(q2, dim, k, axis)
        chosen = self.select(part, role, modality, lang, version, space, frames, span, ids, min_count)
        if space is None and len({s.raw.space for s in chosen}) > 1:
            raise ValueError("the selection spans several spaces, whose scores are not comparable; name one with space=")
        cand: list[tuple[float, Section, int]] = []
        for s in chosen:
            sc = s._scores_unit(q2, dim)[0]
            if frames is not None and s.raw.timeline:
                ax = s.axis_times
                sc = np.where((ax[:, 1] > frames[0]) & (ax[:, 0] < frames[1]), sc, -np.inf)
            cand.extend((float(sc[i]), s, int(i)) for i in _topk(sc, k).tolist())
        cand.sort(key=lambda c: -c[0])
        return [_hit(s, i, score, axis) for score, s, i in cand[:k]]

    def describe(self) -> str:
        return self._t.describe()

    def __repr__(self) -> str:
        return f"Title({self.title_id.hex()[:8]}…, {self.name!r}, {len(self._t.versions)} version(s), {len(self._order)} section(s))"


def _hit(s: Section, row: int, score: float, axis: bool) -> Hit:
    t = s.axis_times if axis else s.times
    tx = s.texts
    f0, f1 = (int(t[row, 0]), int(t[row, 1])) if t is not None else (None, None)
    return Hit(s, row, score, f0, f1, tx[row] if tx else None)


class _Pending:
    """Staged changes of one title: description keys, version records (new or corrected), sections (new or replacements), tombstones."""
    __slots__ = ("desc", "versions", "sections", "removed")

    def __init__(self):
        self.desc: dict = {}
        self.versions: dict[bytes, VersionSpec] = {}
        self.sections: list[SectionSpec] = []
        self.removed: list[tuple[str, bytes]] = []

    def __bool__(self) -> bool:
        return bool(self.desc or self.versions or self.sections or self.removed)


class EneFile:
    """An open file: its packets, its titles (resolved on first access) and, for the usual single-title file, the title's
    sections and search directly on the file. Edits are staged and written by `save()` as one packet per title."""

    def __init__(self, path: str, verify: bool = True, policy=None):
        self.path, self._verify = path, verify
        self.policy = policy_of(policy)
        self._pending: dict[bytes, _Pending] = {}
        self._open()

    def _open(self) -> None:
        self._c = C.EneFile(self.path, verify=self._verify)
        self._titles: dict[bytes, Title] | None = None

    @property
    def titles(self) -> dict[bytes, Title]:
        if self._titles is None:
            self._titles = {tid: Title(self, ct) for tid, ct in self._c.titles.items()}
        return self._titles

    def title(self, title_id: bytes | None = None) -> Title:
        if title_id is not None:
            return self.titles[title_id]
        if len(self.titles) != 1:
            raise KeyError(f"{len(self.titles)} titles in {self.path}; give the title_id")
        return next(iter(self.titles.values()))

    packets = property(lambda self: self._c.packets)
    mapped = property(lambda self: self._c.mapped)

    def verify(self) -> bool:
        return self._c.verify()

    def close(self) -> None:
        if self._titles:
            for t in self._titles.values():
                for s in t._order:
                    s._m = s._vectors = s._scale = s._src = None
                    s._times = s._axis = s._texts = False
                    s._norms.clear()
        self._c.close()

    def reload(self) -> None:
        self.close()
        self._open()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # the single title's view
    description = property(lambda self: self.title().description)
    name = property(lambda self: self.title().name)
    axis_fps = property(lambda self: self.title().axis_fps)
    versions = property(lambda self: self.title().versions)
    sections = property(lambda self: self.title().sections)

    def section(self, id: str, version=None) -> Section:  # noqa: A002
        return self.title().section(id, version)

    def __getitem__(self, id: str) -> Section:  # noqa: A002
        return self.title().section(id)

    def __contains__(self, id: str) -> bool:  # noqa: A002
        return id in self.title()

    def __iter__(self):
        return iter(self.title())

    def select(self, *a, **kw) -> list[Section]:
        return self.title().select(*a, **kw)

    def scores(self, section, q, dim: int | None = None) -> np.ndarray:
        return self.title().scores(section, q, dim)

    def search(self, q, k: int = 10, dim: int | None = None, **kw) -> list[Hit]:
        return self.title().search(q, k, dim, **kw)

    def describe(self) -> str:
        return "\n".join(t.describe() for t in self.titles.values())

    # --- staged edits (written as one packet by save) -------------------------------------------------------------------
    def _pend(self, title_id: bytes | None) -> tuple[Title, _Pending]:
        t = self.title(title_id)
        return t, self._pending.setdefault(t.title_id, _Pending())

    def set(self, _desc: dict | None = None, *, title_id: bytes | None = None, **keys) -> EneFile:
        """Stage description keys (§3.7), merged key by key into what the file holds: `f.set(title={"name": ...})`."""
        _, p = self._pend(title_id)
        p.desc = _merge(_merge(p.desc, _desc or {}), keys)
        return self

    def add_version(self, version, *, title_id: bytes | None = None, **kw) -> bytes:
        """Stage a version record (§2.2): a new contribution, or a replacement of a held record with a higher `rev`."""
        t, p = self._pend(title_id)
        spec = version if isinstance(version, VersionSpec) else VersionSpec(version, **kw)
        held = t.versions.get(spec.id)
        if held is not None and spec.rev <= held.rev:
            raise ValueError(f"version {spec.id.hex()} is held at rev {held.rev}; a replacement needs a higher rev (or set_alignment)")
        p.versions[spec.id] = spec
        return spec.id

    def set_alignment(self, version, map_a: float, map_b: float, *, title_id: bytes | None = None) -> EneFile:
        """Stage a corrected map for a version: its record again with `rev + 1` (§2.8); rows are never rewritten."""
        t, p = self._pend(title_id)
        vid = _version_id(version)
        if vid in p.versions:
            s = p.versions[vid]
            p.versions[vid] = VersionSpec(vid, source_hash=s.source_hash, fps=s.fps, duration=s.duration, map_a=map_a, map_b=map_b, rev=s.rev, flags=s.flags)
        else:
            held = t.versions[vid]
            p.versions[vid] = VersionSpec.of(held, map_a=map_a, map_b=map_b, rev=held.rev + 1)
        return self

    def add_section(self, section, version=None, *, title_id: bytes | None = None, **kw) -> SectionSpec:
        """Stage a section (the arguments of `SectionSpec`); an existing `(version, id)` is replaced with `rev + 1` on save."""
        t, p = self._pend(title_id)
        spec = section if isinstance(section, SectionSpec) else SectionSpec(section, version, **kw)
        if spec.version not in t.versions and spec.version not in p.versions:
            raise ValueError(f"unknown version {spec.version.hex()}; add_version first")
        p.sections = [s for s in p.sections if (s.version, s.id) != (spec.version, spec.id)]
        p.sections.append(spec)
        return spec

    def remove_section(self, id: str, version=None, *, title_id: bytes | None = None) -> EneFile:  # noqa: A002
        """Stage a tombstone for `(version, id)`; the version may be omitted when the id exists in one version only."""
        t, p = self._pend(title_id)
        if version is None:
            vids = [vid for (vid, sid) in t.sections if sid == id] + [s.version for s in p.sections if s.id == id]
            if len(set(vids)) != 1:
                raise KeyError(f"{len(set(vids))} sections named {id!r}; give the version")
            version = vids[0]
        vid = _version_id(version)
        staged = [s for s in p.sections if (s.version, s.id) == (vid, id)]
        p.sections = [s for s in p.sections if (s.version, s.id) != (vid, id)]
        if (vid, id) in t.sections:
            p.removed.append((id, vid))
        elif not staged:
            raise KeyError(id)
        return self

    @property
    def dirty(self) -> bool:
        return any(bool(p) for p in self._pending.values())

    def _staged_bytes(self, policy=None) -> bytes:
        out = []
        for tid, p in self._pending.items():
            if not p:
                continue
            t = self.titles[tid]
            w = Writer(tid, t.axis_fps, policy or self.policy, description=p.desc or None)
            for spec in p.versions.values():
                w.add_version(spec)
            needed = {s.version for s in p.sections} | {vid for _, vid in p.removed}
            for vid in needed:
                if vid not in w.versions:
                    w.add_version(VersionSpec.of(t.versions[vid]))
            for s in p.sections:
                held = t.sections.get((s.version, s.id))
                if held is not None and s.rev <= held.rev:
                    s.rev = held.rev + 1
                w.add_section(s)
            for sid, vid in p.removed:
                held = t.sections.get((vid, sid))
                w.add_tombstone(sid, vid, held.rev if held is not None else 0)
            out.append(w.to_bytes())
        return b"".join(out)

    def save(self, path: str | None = None, policy=None) -> str:
        """No path: append ONE packet per changed title with the staged changes (§0.8) and reload. With a path: write a compacted
        copy with the changes applied, leaving this file and the staged changes as they are."""
        if path is not None:
            shutil.copyfile(self.path, path)
            data = self._staged_bytes(policy)
            if data:
                with builtins.open(path, "ab") as fh:
                    fh.write(data)
            with C.EneFile(path) as f:
                packed = C.compact_bytes(f)
            with builtins.open(path, "wb") as fh:
                fh.write(packed)
            return path
        data = self._staged_bytes(policy)
        if not data:
            return self.path
        self.close()
        with builtins.open(self.path, "ab") as fh:
            fh.write(data)
        self._pending = {}
        self._open()
        return self.path

    def compact(self, out: str | None = None) -> str:
        """One packet per title, overlays resolved, payloads copied as they are (§3.10). In place when `out` is None (the staged
        changes stay staged)."""
        data = C.compact_bytes(self._c)
        if out is not None:
            with builtins.open(out, "wb") as fh:
                fh.write(data)
            return out
        fd, tmp = tempfile.mkstemp(suffix=".ene", dir=os.path.dirname(os.path.abspath(self.path)))
        os.close(fd)
        try:
            with builtins.open(tmp, "wb") as fh:
                fh.write(data)
            self.close()
            shutil.move(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        self._open()
        return self.path


# --- module-level operations ------------------------------------------------------------------------------------------------

def open(path: str, verify: bool = True, policy=None) -> EneFile:  # noqa: A001 - deliberate, like zipfile / h5py
    return EneFile(path, verify=verify, policy=policy)


def create(path: str, title_id: bytes, axis_fps: tuple[int, int], versions=(), sections=(), description: dict | None = None,
           policy=None, append: bool = False) -> int:
    """Write one packet for one title: its version records, its sections under the policy, its description. `append=True` adds
    the packet to an existing file. Returns the bytes written."""
    w = Writer(title_id, axis_fps, policy, description)
    for v in versions:
        w.add_version(v)
    for s in sections:
        w.add_section(s)
    return w.write(path, append)


def overlay(dst: str, src: str, check_title: bool = True) -> int:
    """Append the packets of `src` to `dst` (`cat a.ene b.ene`, §3.1); `src` is verified first. With `check_title`, `src` must
    carry at least one title that `dst` already holds. Returns the new size of `dst`."""
    with C.EneFile(src) as s:
        src_titles = set(s.titles)
    if check_title:
        with C.EneFile(dst, verify=False) as d:
            if src_titles.isdisjoint(set(d.titles)):
                raise ValueError(f"{src} carries none of the titles of {dst} (pass check_title=False to append anyway)")
    with builtins.open(src, "rb") as fh:
        data = fh.read()
    with builtins.open(dst, "ab") as fh:
        fh.write(data)
    return os.path.getsize(dst)


def compact(path: str, out: str | None = None) -> str:
    with EneFile(path) as f:
        return f.compact(out)


def rewrite(path: str, out: str, policy=DEFAULT_POLICY_NAME) -> int:
    """Re-emit a file under another storage policy: every row dequantized, renormalized and stored again (lossless from fp32 or
    fp16 rows; a second rounding from int8 rows). Versions, segments, texts, `prep` hashes, `ref`, `rev` and the description are
    carried over. Returns the bytes written."""
    total = 0
    with EneFile(path) as f:
        for i, (tid, t) in enumerate(f.titles.items()):
            w = Writer(tid, t.axis_fps, policy, description=t.description or None)
            for v in t.versions.values():
                w.add_version(VersionSpec.of(v))
            for s in t._order:
                r = s.raw
                w.add_section(SectionSpec(r.id, r.version_id, modality=r.modality, role=r.role, vectors=s.vectors, times=s.times, texts=s.texts,
                                          lang=r.lang, space=r.space, prep=r.prep, ref=r.ref, rev=r.rev, dim_real=r.dim_real))
            total += w.write(out, append=i > 0)
    return total


# --- the local space (§5) ----------------------------------------------------------------------------------------------------

_SPACE_COLS = (("title", "<u4"), ("version", "<u4"), ("part", "u1"), ("f0", "<i4"), ("f1", "<i4"), ("vf0", "<i4"), ("vf1", "<i4"),
               ("modality", "u1"), ("role", "<u2"), ("lang", "<u2"), ("section", "<u4"))


class SectionRef:
    """A section as the local space holds it: its provenance, the key and chunk of its rows, its texts."""
    __slots__ = ("index", "title_id", "version_id", "id", "ref", "prep", "rev", "key", "timeline", "modality", "role", "lang", "chunk", "count", "texts", "alive")

    def __init__(self, index, title_id, version_id, id, ref, prep, rev, key, timeline, modality, role, lang, chunk, count, texts):  # noqa: A002
        self.index, self.title_id, self.version_id, self.id, self.ref, self.prep, self.rev = index, title_id, version_id, id, ref, prep, rev
        self.key, self.timeline, self.modality, self.role, self.lang, self.chunk, self.count, self.texts = key, timeline, modality, role, lang, chunk, count, texts
        self.alive = True

    part = property(lambda self: "timeline" if self.timeline else "title")
    modality_name = property(lambda self: MODALITIES[self.modality] if self.modality < len(MODALITIES) else str(self.modality))

    def __repr__(self) -> str:
        return f"SectionRef({self.id!r}, v {self.version_id.hex()[:8]}…, {self.modality_name}/{self.role}, {self.count} rows, rev {self.rev})"


class _TitleRef:
    __slots__ = ("index", "title_id", "axis_fps", "versions", "sections")

    def __init__(self, index: int, title_id: bytes, axis_fps: tuple[int, int]):
        self.index, self.title_id, self.axis_fps = index, title_id, axis_fps
        self.versions: dict[bytes, _VersionRef] = {}
        self.sections: dict[tuple[bytes, str], SectionRef] = {}


class _VersionRef:
    __slots__ = ("index", "id", "fps", "map_a", "map_b", "duration", "rev", "flags")

    def __init__(self, index: int, v: Version):
        self.index, self.id = index, v.id
        self.update(v)

    def update(self, v: Version) -> None:
        self.fps, self.map_a, self.map_b, self.duration, self.rev, self.flags = v.fps, v.map_a, v.map_b, v.duration, v.rev, v.flags


class _KeyRows:
    """One key of the local space: the rows in the key's element type (plus the int8 scale) and the provenance columns (§5.1),
    appended chunk by chunk (one chunk per ingested section) and stacked on demand."""
    __slots__ = ("key", "dtype", "chunks", "scales", "alive", "cols", "starts", "srefs", "n", "_cache")

    def __init__(self, key: tuple[str, str, int]):
        self.key = key
        self.dtype = _NP[key[1]]
        self.chunks: list[np.ndarray] = []
        self.scales: list[np.ndarray | None] = []
        self.alive: list[np.ndarray] = []
        self.cols: dict[str, list[np.ndarray]] = {name: [] for name, _ in _SPACE_COLS}
        self.starts: list[int] = []
        self.srefs: list[SectionRef] = []
        self.n = 0
        self._cache: dict | None = None

    def append(self, rows: np.ndarray, scale: np.ndarray | None, cols: dict, alive: np.ndarray, sref: SectionRef) -> int:
        self.chunks.append(rows)
        self.scales.append(scale)
        self.alive.append(alive)
        for name, dt in _SPACE_COLS:
            self.cols[name].append(np.asarray(cols[name], dtype=dt))
        self.starts.append(self.n)
        self.srefs.append(sref)
        self.n += len(rows)
        self._cache = None
        return len(self.chunks) - 1

    def invalidate(self) -> None:
        self._cache = None

    def stacked(self) -> dict:
        """The key's matrix, scale, alive mask and columns as single arrays (cached until the next append, kill or re-map)."""
        if self._cache is None:
            one = len(self.chunks) == 1
            c = {"matrix": self.chunks[0] if one else np.concatenate(self.chunks),
                 "scale": None if self.scales[0] is None else (self.scales[0] if one else np.concatenate(self.scales)),
                 "alive": self.alive[0] if one else np.concatenate(self.alive)}
            for name in self.cols:
                c[name] = self.cols[name][0] if one else np.concatenate(self.cols[name])
            self._cache = c
        return self._cache

    def live(self) -> int:
        return int(sum(int(a.sum()) for a in self.alive))


class SpaceHit:
    """A row of the local space found by a search: its section (provenance), row within it, score, axis segment, text, key."""
    __slots__ = ("section", "row", "score", "f0", "f1", "text", "key")

    def __init__(self, section: SectionRef, row: int, score: float, f0: int | None, f1: int | None, text: str | None, key):
        self.section, self.row, self.score, self.f0, self.f1, self.text, self.key = section, row, score, f0, f1, text, key

    title_id = property(lambda self: self.section.title_id)
    version_id = property(lambda self: self.section.version_id)
    id = property(lambda self: self.section.id)

    def __repr__(self) -> str:
        t = f" [{self.f0}, {self.f1})" if self.f0 is not None else ""
        x = f" {self.text[:50]!r}" if self.text else ""
        return f"SpaceHit({self.section.title_id.hex()[:8]}…/{self.section.id}#{self.row} {self.score:.3f}{t}{x})"


class Space:
    """The local multidimensional search space (§5.1): per key `space / quant / dim` one matrix of rows from any number of
    titles, with the provenance columns that say where each row comes from and, for timeline rows, when it happens on the
    title's axis. `ingest` applies §5.2: replaced versions re-map their rows, replaced or deleted sections drop theirs, and rows
    that are the same information twice (same key, version, part, modality, role, lang and non-zero `prep`, same segment or
    text) are kept once, from the higher `rev`, the later file on a tie. `bin` sections (a sieve, not a space) are skipped."""

    def __init__(self):
        self.title_refs: dict[bytes, _TitleRef] = {}                       # by title_id; `titles(q)` is the search
        self.title_list: list[bytes] = []
        self.version_list: list[tuple[bytes, bytes]] = []
        self.roles: list[str] = []
        self.langs: list[str] = []
        self._role_idx: dict[str, int] = {}
        self._lang_idx: dict[str, int] = {}
        self.keys: dict[tuple[str, str, int], _KeyRows] = {}
        self.sections: list[SectionRef] = []
        self._dups: dict[tuple, list[SectionRef]] = {}

    def __len__(self) -> int:
        """Live rows over every key."""
        return sum(kr.live() for kr in self.keys.values())

    @property
    def spaces(self) -> list[str]:
        return sorted({k[0] for k in self.keys})

    def _idx(self, index: dict, names: list, name: str) -> int:
        i = index.get(name)
        if i is None:
            i = index[name] = len(names)
            names.append(name)
        return i

    def ingest(self, src, verify: bool = True) -> int:
        """Ingest a file (path, `EneFile` or container file) or one `Title`. Returns the live rows appended."""
        if isinstance(src, Title):
            return self._ingest_title(src._t)
        own = isinstance(src, str)
        f = C.EneFile(src, verify) if own else (src._c if isinstance(src, EneFile) else src)
        try:
            return sum(self._ingest_title(t) for t in f.titles.values())
        finally:
            if own:
                f.close()

    def _ingest_title(self, ct: C.Title) -> int:
        T = self.title_refs.get(ct.title_id)
        if T is None:
            T = self.title_refs[ct.title_id] = _TitleRef(len(self.title_list), ct.title_id, ct.axis_fps)
            self.title_list.append(ct.title_id)
        elif ct.axis_fps != T.axis_fps:
            raise ValueError(f"title {ct.title_id.hex()} arrives with axis {ct.axis_fps}, held at {T.axis_fps}")
        # §3.9 rule 2 against what the space holds: tombstones of the incoming packets
        for pk in ct.packets:
            recs = pk.sections
            for i in range(len(recs)):
                rec = recs[i]
                if int(rec["flags"]) & C.S_DELETED:
                    held = T.sections.get((pk.version_id(int(rec["version"])), bytes(rec["id"]).decode("utf-8")))
                    if held is not None and int(rec["rev"]) >= held.rev:
                        self._drop(T, held)
        # §3.9 rule 1: version records; a replaced map re-maps the rows already held (§5.2 step 2)
        for vid, v in ct.versions.items():
            held = T.versions.get(vid)
            if held is None:
                T.versions[vid] = _VersionRef(len(self.version_list), v)
                self.version_list.append((ct.title_id, vid))
            elif v.rev >= held.rev:
                changed = (v.map_a, v.map_b) != (held.map_a, held.map_b)
                held.update(v)
                if changed:
                    self._remap(T, held)
        # sections: a held `(version, id)` is replaced by a higher rev, or by an equal rev from this later file
        rows = 0
        for (vid, sid), s in ct.sections.items():
            held = T.sections.get((vid, sid))
            if held is not None:
                if s.rev < held.rev:
                    continue
                self._drop(T, held)
            rows += self._append(T, s)
        return rows

    def _drop(self, T: _TitleRef, held: SectionRef) -> None:
        kr = self.keys[held.key]
        kr.alive[held.chunk][:] = False
        kr.invalidate()
        held.alive = False
        T.sections.pop((held.version_id, held.id), None)

    def _remap(self, T: _TitleRef, vref: _VersionRef) -> None:
        for sref in T.sections.values():
            if sref.version_id == vref.id and sref.timeline:
                kr = self.keys[sref.key]
                vt = np.stack([kr.cols["vf0"][sref.chunk], kr.cols["vf1"][sref.chunk]], 1)
                ax = _to_axis(vref.map_a, vref.map_b, vt)
                kr.cols["f0"][sref.chunk] = ax[:, 0].astype("<i4")
                kr.cols["f1"][sref.chunk] = ax[:, 1].astype("<i4")
                kr.invalidate()

    def _append(self, T: _TitleRef, s: C.Section) -> int:
        if s.quant == "bin":
            return 0
        key = s.key
        kr = self.keys.get(key)
        if kr is None:
            kr = self.keys[key] = _KeyRows(key)
        vref = T.versions[s.version_id]
        count = s.count
        vt = s.times()
        ax = None if vt is None else _to_axis(vref.map_a, vref.map_b, vt)
        texts = s.texts()
        alive = np.ones(count, bool)
        sref = SectionRef(len(self.sections), T.title_id, s.version_id, s.id, s.ref, s.prep, s.rev, key, s.timeline, s.modality, s.role, s.lang,
                          -1, count, texts)
        if s.prep != 0:                                                       # §5.2 step 3: duplicates against sections of the same provenance
            dk = (key, T.title_id, s.version_id, s.timeline, s.modality, s.role, s.lang, s.prep)
            for held in self._dups.get(dk, ()):
                if not held.alive:
                    continue
                hk = self.keys[held.key]
                if s.timeline:
                    pairs = zip(hk.cols["vf0"][held.chunk].tolist(), hk.cols["vf1"][held.chunk].tolist())
                    index = {}
                    for j, p in enumerate(pairs):
                        index.setdefault(p, j)
                    mine = [tuple(p) for p in vt.tolist()]
                else:
                    if texts is None or held.texts is None:
                        continue                                              # rows without text are never duplicates
                    index = {}
                    for j, t in enumerate(held.texts):
                        index.setdefault(t, j)
                    mine = texts
                newer = s.rev >= held.rev                                     # equal rev: this file is the later one
                for i, m in enumerate(mine):
                    j = index.get(m)
                    if j is None:
                        continue
                    if newer:
                        hk.alive[held.chunk][j] = False
                        hk.invalidate()
                    else:
                        alive[i] = False
            self._dups.setdefault(dk, []).append(sref)
        rows = np.array(s.vectors(raw=True))                                  # a copy in the stored element type (the file may close)
        scale = None if s.quant != "int8" else np.array(s.scales(), dtype="<f4")
        cols = {"title": np.full(count, T.index), "version": np.full(count, vref.index), "part": np.full(count, 1 if s.timeline else 0),
                "f0": np.zeros(count) if ax is None else ax[:, 0], "f1": np.zeros(count) if ax is None else ax[:, 1],
                "vf0": np.zeros(count) if vt is None else vt[:, 0], "vf1": np.zeros(count) if vt is None else vt[:, 1],
                "modality": np.full(count, s.modality), "role": np.full(count, self._idx(self._role_idx, self.roles, s.role)),
                "lang": np.full(count, self._idx(self._lang_idx, self.langs, s.lang)), "section": np.full(count, sref.index)}
        sref.chunk = kr.append(rows, scale, cols, alive, sref)
        self.sections.append(sref)
        T.sections[(s.version_id, s.id)] = sref
        return int(alive.sum())

    def rows(self, key) -> dict:
        """The stacked arrays of one key: `matrix`, `scale`, `alive` and the columns of §5.1 (plus `vf0`, `vf1` in version frames)."""
        return self.keys[key].stacked()

    def _mask(self, kr: _KeyRows, c: dict, part, role, modality, lang, title, version, frames) -> np.ndarray:
        mask = c["alive"].copy()
        if part is not None:
            mask &= c["part"] == (1 if part == "timeline" else 0)
        if role is not None:
            i = self._role_idx.get(role)
            mask &= (c["role"] == i) if i is not None else False
        if lang is not None:
            i = self._lang_idx.get(lang)
            mask &= (c["lang"] == i) if i is not None else False
        if modality is not None:
            mask &= c["modality"] == C.modality_code(modality)
        if title is not None:
            T = self.title_refs.get(bytes(title))
            mask &= (c["title"] == T.index) if T is not None else False
        if version is not None:
            vid = _version_id(version)
            idx = [v.index for T in self.title_refs.values() for v in T.versions.values() if v.id == vid]
            mask &= np.isin(c["version"], idx) if idx else False
        if frames is not None:
            mask &= (c["part"] == 1) & (c["f1"] > frames[0]) & (c["f0"] < frames[1])
        return mask

    def _scored(self, q2: np.ndarray, dim, space, key, part, role, modality, lang, title, version, frames):
        """(scores, key rows, stacked) per key searched; rows that fail a filter are -inf."""
        keys = [key] if key is not None else [k for k in self.keys if space is None or k[0] == space]
        if len({k[0] for k in keys}) > 1:
            raise ValueError(f"the local space holds several spaces {self.spaces}; name one with space=")
        out = []
        for k in keys:
            kr = self.keys[k]
            if kr.n == 0:
                continue
            c = kr.stacked()
            full = k[2]
            d = full if dim is None or dim >= full else dim
            sc = _score_block(c["matrix"], c["scale"], q2, d)
            sc[~self._mask(kr, c, part, role, modality, lang, title, version, frames)] = -np.inf
            out.append((sc, kr, c))
        return out

    def _hit(self, kr: _KeyRows, c: dict, i: int, score: float) -> SpaceHit:
        j = bisect_right(kr.starts, i) - 1
        sref = kr.srefs[j]
        row = i - kr.starts[j]
        f0, f1 = (int(c["f0"][i]), int(c["f1"][i])) if sref.timeline else (None, None)
        return SpaceHit(sref, row, score, f0, f1, sref.texts[row] if sref.texts else None, kr.key)

    def search(self, q, k: int = 10, dim: int | None = None, *, space: str | None = None, key=None, part: str | None = None,
               role: str | None = None, modality=None, lang: str | None = None, title: bytes | None = None, version=None,
               frames: tuple[int, int] | None = None) -> list[SpaceHit]:
        """Top-k rows over the keys of one space (every width stacks at `dim`, or at its own width), best first; `frames` keeps
        the timeline rows overlapping an axis span."""
        q2 = _unit_query(q)
        cand = []
        for sc, kr, c in self._scored(q2, dim, space, key, part, role, modality, lang, title, version, frames):
            cand.extend((float(sc[i]), kr, c, int(i)) for i in _topk(sc, k).tolist())
        cand.sort(key=lambda x: -x[0])
        return [self._hit(kr, c, i, score) for score, kr, c, i in cand[:k]]

    def titles(self, q, k: int = 10, dim: int | None = None, **filters) -> list[SpaceHit]:
        """Which title? The best row of each title (the filters of `search`; `part="title"` for the title part alone), ranked."""
        q2 = _unit_query(q)
        best: dict[int, tuple[float, _KeyRows, dict, int]] = {}
        for sc, kr, c in self._scored(q2, dim, filters.pop("space", None), filters.pop("key", None), filters.get("part"), filters.get("role"),
                                      filters.get("modality"), filters.get("lang"), filters.get("title"), filters.get("version"), filters.get("frames")):
            order = np.argsort(-sc)
            seen = set()
            for i in order.tolist():
                ti = int(c["title"][i])
                if ti in seen:
                    continue
                if not np.isfinite(sc[i]):
                    break
                seen.add(ti)
                if ti not in best or best[ti][0] < float(sc[i]):
                    best[ti] = (float(sc[i]), kr, c, i)
        ranked = sorted(best.values(), key=lambda x: -x[0])[:k]
        return [self._hit(kr, c, i, score) for score, kr, c, i in ranked]


class Library:
    """Several files searched together: `titles(q)` = which title, `search(q)` = which moment; `.space` = all of them ingested
    into one local space (§5). For thousands of files use `ene.index` (§5.4)."""

    def __init__(self, paths: list[str], verify: bool = False):
        self.files = [EneFile(p, verify=verify) for p in paths]
        self._space: Space | None = None

    @property
    def space(self) -> Space:
        if self._space is None:
            sp = Space()
            for f in self.files:
                sp.ingest(f)
            self._space = sp
        return self._space

    def titles(self, q, k: int | None = None, dim: int | None = None, part: str | None = "title", **kw) -> list[tuple[EneFile, Hit]]:
        """The best row of each file's title part (or any part with `part=None`), ranked; files without a hit are left out."""
        out = []
        for f in self.files:
            hits = f.search(q, 1, dim, part=part, **kw)
            if hits:
                out.append((f, hits[0]))
        out.sort(key=lambda fh: -fh[1].score)
        return out if k is None else out[:k]

    def search(self, q, k: int = 10, per_title: int = 1, dim: int | None = None, **kw) -> list[tuple[EneFile, Hit]]:
        """The best `per_title` rows of each file under the filters of `Title.search`, merged and ranked."""
        out = []
        for f in self.files:
            out += [(f, h) for h in f.search(q, per_title, dim, **kw)]
        out.sort(key=lambda fh: -fh[1].score)
        return out[:k]

    def close(self) -> None:
        for f in self.files:
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
