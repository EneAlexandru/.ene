"""The persisted local space (SPEC §5.4): the title-part rows of many `.ene` files stacked into one memory-mapped packet.

    import ene
    n = ene.index.build(["a.ene", "b.ene", ...], "library.ene")   # fp32, codec none, 64-byte aligned: one matrix per (space, width)
    idx = ene.index.open("library.ene")                            # one mmap, two tables, a description of a few hundred bytes
    idx.titles(q, k=10)        # "which title?": the best row of each file -> [IndexHit(file, score, section, row)]
    idx.hits(q, k=10)          # the best rows overall
    idx.files                  # [{"path", "title_id", "name", "year", "kind", "duration_s", "sections": [...]}], parsed on first use
    idx.close()

Layout: a packet of the §3 layout. Its sections are the stacked matrices (one per key `space / fp32 / dim`, in the title
part, role `index`, no `prep`), so any reader of the format sees them as ordinary rows. Two more blocks per packet are
addressed through the description only: per key an `origin` column `i64[count, 2]` = (file index, section index << 32 | row),
and the file table (zstd JSON). Rows are grouped by file in build order, which makes "best row per file" one
`np.maximum.reduceat` instead of a loop; the file table is parsed on first use, so opening an index of 1 000 files costs one
mmap and a description of a few hundred bytes, not a 500 KB JSON parse.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from . import api
from . import container as C
from .quant import mrl

INDEX_VERSION = hashlib.sha256(b"ene.index").digest()[:16]
ROLE = "index"


class IndexHit:
    __slots__ = ("file", "score", "section", "row", "key")

    def __init__(self, file: dict, score: float, section: str, row: int, key: tuple):
        self.file, self.score, self.section, self.row, self.key = file, score, section, row, key

    def __repr__(self) -> str:
        return f"IndexHit({self.file.get('name')!r} {self.score:.3f} {self.section}#{self.row})"


def build(paths: list[str], out: str, part: str | None = "title", roles=None, verify: bool = False) -> int:
    """Stack the rows of the given part (and roles, when given) of every file; the vectors are dequantized and renormalized.
    Returns the number of stacked rows."""
    files, mats, origins = [], {}, {}
    for fi, path in enumerate(paths):
        with api.EneFile(path, verify=verify) as f:
            t = f.title()
            secs, listed = [], []
            for s in t.select(part=part):
                if roles is not None and s.role not in roles or not s.count:
                    continue
                key = (s.space, s.dim)
                si = len(listed)
                listed.append({"id": s.id, "version": s.version_id.hex(), "role": s.role, "modality": s.modality_name, "lang": s.lang, "count": s.count})
                x = np.ascontiguousarray(s.vectors, dtype=np.float32)
                n = np.linalg.norm(x, axis=1, keepdims=True)
                n[n == 0] = 1.0
                mats.setdefault(key, []).append(x / n)
                origins.setdefault(key, []).append(np.stack([np.full(s.count, fi, np.int64), (np.int64(si) << 32) | np.arange(s.count, dtype=np.int64)], 1))
            ref = t.reference or next((v for v in t.versions.values() if v.timeline), None)
            files.append({"path": os.path.abspath(path), "title_id": t.title_id.hex(), "name": t.name, "year": (t.description.get("title") or {}).get("year"),
                          "kind": (t.description.get("title") or {}).get("kind"),
                          "duration_s": round(ref.duration * ref.fps[1] / ref.fps[0], 3) if ref is not None and ref.timeline else None, "sections": listed})
    title_id = hashlib.sha256(b"ene.index:" + "\n".join(e["path"] for e in files).encode("utf-8")).digest()[:16]
    w = C.PacketWriter(title_id, (1, 1))
    v = w.add_version(INDEX_VERSION)
    keys, n = {}, 0
    for (space, dim), parts in mats.items():
        m = np.concatenate(parts)
        o = np.concatenate(origins[(space, dim)])
        sid = f"{space}.{dim}"
        w.add_section(sid, v, space=space, modality="mixed", role=ROLE, vectors=m, quant="fp32", codec="none", dim_real=dim)
        keys[sid] = {"section": sid, "space": space, "dim": dim, "count": int(m.shape[0]), "origin": w.add_block(o.astype("<i8").tobytes(), "none")}
        n += int(m.shape[0])
    files_raw = json.dumps(files, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    fb = w.add_block(files_raw, "zstd")
    offs = w.block_offsets()
    for k in keys.values():
        bi = k.pop("origin")
        k["origin"] = [offs[bi], len(w._blocks[bi].data), w._blocks[bi].raw_len]
    w.set_description({"index": {"n_files": len(files), "part": part, "roles": None if roles is None else sorted(roles), "keys": keys,
                                 "files": [offs[fb], len(w._blocks[fb].data), w._blocks[fb].raw_len]}})
    w.write(out)
    return n


class Index:
    def __init__(self, path: str):
        self.path = path
        self._f = C.EneFile(path, verify=False)
        pk = self._f.packets[0]
        desc = pk.description() if len(self._f.packets) == 1 else None
        if not desc or "index" not in desc:
            self._f.close()
            raise ValueError(f"{path}: not a library index")
        self.meta = desc["index"]
        self.n_files: int = int(self.meta.get("n_files", 0))
        self._files: list[dict] | None = None
        self._mats: dict[tuple, np.ndarray] = {}
        self._origin: dict[tuple, np.ndarray] = {}
        self._starts: dict[tuple, np.ndarray] = {}
        recs = {bytes(r["id"]).decode("utf-8"): r for r in pk.sections}
        for sid, k in self.meta["keys"].items():
            r = recs[sid]
            key = (k["space"], "fp32", int(k["dim"]))
            mv = pk.block(int(r["vec_off"]), int(r["vec_len"]), int(r["vec_raw"]), int(r["codec"]))      # zero copy on the mmap
            self._mats[key] = np.frombuffer(mv, dtype="<f4").reshape(int(r["count"]), int(r["dim"]))
            off, ln, raw = k["origin"]
            o = np.frombuffer(pk.block(int(off), int(ln), int(raw), 0), dtype="<i8").reshape(-1, 2)
            self._origin[key] = o
            fidx = o[:, 0]
            self._starts[key] = np.flatnonzero(np.r_[True, fidx[1:] != fidx[:-1]])
        self.keys = list(self._mats)

    @property
    def files(self) -> list[dict]:
        if self._files is None:
            off, ln, raw = self.meta["files"]
            self._files = json.loads(bytes(self._f.packets[0].block(int(off), int(ln), int(raw), C.CODECS.index("zstd"))).decode("utf-8"))
        return self._files

    @property
    def spaces(self) -> list[str]:
        return sorted({k[0] for k in self.keys})

    def close(self) -> None:
        self._mats.clear(); self._origin.clear(); self._starts.clear()
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __len__(self) -> int:
        return self.n_files or len(self.files)

    @property
    def n_rows(self) -> int:
        return int(sum(m.shape[0] for m in self._mats.values()))

    def _keys(self, space: str | None, key=None) -> list[tuple]:
        keys = [key] if key is not None else [k for k in self.keys if space is None or k[0] == space]
        if len({k[0] for k in keys}) > 1:
            raise ValueError(f"the index holds several spaces {self.spaces}; name one with space=")
        return keys

    def scores(self, q, key=None, dim: int | None = None) -> np.ndarray:
        """float32 [nq, count] over one key (the only one when the index has one)."""
        key = key or (self.keys[0] if len(self.keys) == 1 else self._keys(None)[0])
        m = self._mats[key]
        q2 = api._unit(q)
        d = min(dim or m.shape[1], m.shape[1], q2.shape[1])
        if d == m.shape[1] and q2.shape[1] == d:
            return q2 @ m.T                                                   # no copy, no decode: the mmap pages are the operand
        return mrl(q2, d) @ mrl(np.asarray(m, dtype=np.float32), d).T

    def _hit(self, key: tuple, i: int, score: float) -> IndexHit:
        fi, packed = (int(x) for x in self._origin[key][i])
        f = self.files[fi]
        return IndexHit(f, float(score), f["sections"][packed >> 32]["id"], packed & 0xFFFFFFFF, key)

    def hits(self, q, k: int = 10, dim: int | None = None, space: str | None = None, key=None) -> list[IndexHit]:
        """The best rows overall."""
        out = []
        for kk in self._keys(space, key):
            sc = self.scores(q, kk, dim)[0]
            n = min(k, len(sc))
            idx = np.argpartition(-sc, n - 1)[:n]
            out += [self._hit(kk, int(i), sc[i]) for i in idx]
        out.sort(key=lambda h: -h.score)
        return out[:k]

    def best_per_file(self, q, key=None, dim: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized "best row of each file": (file indices, best scores, row indices) for the files present under the key."""
        key = key or self._keys(None)[0]
        sc = self.scores(q, key, dim)[0]
        starts = self._starts[key]
        best = np.maximum.reduceat(sc, starts)                                 # one pass, rows grouped by file
        ends = np.r_[starts[1:], len(sc)]
        rep = np.repeat(best, ends - starts)
        cand = np.flatnonzero(sc == rep)
        grp = np.searchsorted(starts, cand, side="right") - 1
        _, first = np.unique(grp, return_index=True)
        rows = cand[first]
        return self._origin[key][starts, 0], best, rows

    def titles(self, q, k: int = 10, dim: int | None = None, space: str | None = None, key=None) -> list[IndexHit]:
        """"Which title?": the best row of each file, ranked; one hit per file."""
        agg: dict[int, tuple[float, int, tuple]] = {}
        for kk in self._keys(space, key):
            fidx, best, rows = self.best_per_file(q, kk, dim)
            for fi, b, r in zip(fidx.tolist(), best.tolist(), rows.tolist()):
                if fi not in agg or b > agg[fi][0]:
                    agg[fi] = (b, r, kk)
        top = sorted(agg.items(), key=lambda kv: -kv[1][0])[:k]
        return [self._hit(kk, r, b) for _, (b, r, kk) in top]


def open(path: str) -> Index:  # noqa: A001
    return Index(path)
