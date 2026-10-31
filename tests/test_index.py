"""The persisted local space (ene.index, SPEC §5.4): build, open (mmap, lazy file table), titles / hits - synthetic data, no model.

Run: uv run pytest tests/test_index.py
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ene  # noqa: E402
from ene.api import SectionSpec, VersionSpec  # noqa: E402

SPACE = "eg2-768"                                    # the example space of SPEC §2.7; every section names it

rng = np.random.default_rng(5)


def _unit(n, d=768):
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _title(path, i):
    tid = hashlib.sha256(f"imdb:tt{i:07d}".encode()).digest()[:16]
    vid = hashlib.sha256(f"source-{i}".encode()).digest()[:16]
    v0, vt, v1, v2 = _unit(1), _unit(3), _unit(4), _unit(20)
    ene.create(path, tid, (25, 1), versions=[VersionSpec(vid, fps=(25, 1), duration=15_000, reference=True)],
               description={"title": {"name": f"Title {i}", "year": 2000 + i, "kind": "movie"}}, sections=[
                   SectionSpec("title", vid, space=SPACE, modality="mixed", role="title", vectors=v0, texts=[f"Title {i}"]),
                   SectionSpec("tag", vid, space=SPACE, modality="text", role="tag", vectors=vt, texts=["a", "b", "c"]),
                   SectionSpec("summary.frame", vid, space=SPACE, modality="image", role="summary", vectors=v2.mean(0, keepdims=True), ref="frame"),
                   SectionSpec("chapter", vid, space=SPACE, modality="mixed", role="chapter", vectors=v1, times=np.array([[j * 3750, (j + 1) * 3750] for j in range(4)], np.int32)),
                   SectionSpec("frame", vid, space=SPACE, modality="image", role="frame", vectors=v2, times=np.array([[j * 700, j * 700 + 1] for j in range(20)], np.int32))])
    return v0, vt, v1, v2


def test_index():
    with tempfile.TemporaryDirectory() as d:
        paths, rows = [], []
        for i in range(6):
            p = os.path.join(d, f"t{i}.ene")
            rows.append(_title(p, i)); paths.append(p)
        out = os.path.join(d, "library.ene")
        n = ene.index.build(paths, out)                                                   # the title part: title, tag, summary
        assert n == 6 * (1 + 3 + 1)
        with ene.index.open(out) as idx:
            assert len(idx) == 6 and idx.n_rows == n and idx.keys == [("eg2-768", "fp32", 768)] and idx.spaces == ["eg2-768"]
            assert idx.files[2]["name"] == "Title 2" and idx.files[2]["year"] == 2002 and idx.files[2]["duration_s"] == 600.0
            m = idx._mats[idx.keys[0]]
            assert m.dtype == np.float32 and m.flags.c_contiguous and not m.flags.owndata                   # a view on the file, not a copy
            q = rows[4][0][0] + 0.2 * rng.standard_normal(768).astype(np.float32) / np.sqrt(768)
            t = idx.titles(q, k=3)
            assert t[0].file["name"] == "Title 4" and t[0].section == "title" and t[0].row == 0 and len(t) == 3
            assert len({h.file["title_id"] for h in t}) == 3                                                # one hit per file
            qt = rows[3][1][1]                                                                            # tag "b" of title 3
            assert idx.titles(qt, k=1)[0].section == "tag" and idx.titles(qt, k=1)[0].row == 1 and idx.titles(qt, k=1)[0].file["name"] == "Title 3"
            assert idx.hits(qt, k=1)[0].section == "tag" and idx.hits(qt, k=1)[0].score > 0.99
            s = idx.scores(np.stack([q, qt]), dim=256)
            assert s.shape == (2, n)
        with ene.index.open(out) as idx:
            assert idx._files is None and len(idx) == 6                                                     # the file table is not parsed at open
            fidx, best, rr = idx.best_per_file(q)
            sc = idx.scores(q)[0]; o = idx._origin[idx.keys[0]].copy()                                    # a copy: views on the mmap must not outlive the index
            for fi, b, row in zip(fidx, best, rr):
                m = o[:, 0] == fi
                assert abs(b - sc[m].max()) < 1e-6 and o[row, 0] == fi and sc[row] == sc[m].max()          # equals the per-file loop
            assert idx._files is None
            _ = idx.titles(q, k=1)[0].file["name"]
            assert idx._files is not None
        # every part: timeline rows are stacked too, and "which moment?" works across files
        out2 = os.path.join(d, "all.ene")
        assert ene.index.build(paths, out2, part=None) == n + 6 * 24
        with ene.index.open(out2) as idx:
            q1 = rows[1][2][2]                                                                            # chapter 2 of title 1 (fp16: exact)
            h = idx.hits(q1, k=2)
            assert h[0].file["name"] == "Title 1" and h[0].section == "chapter" and h[0].row == 2 and h[0].score > 0.999
            assert idx.titles(q1, k=1)[0].file["name"] == "Title 1"
        # only some roles
        out3 = os.path.join(d, "tags.ene")
        assert ene.index.build(paths, out3, roles={"tag"}) == 18
        with ene.index.open(out3) as idx:
            assert all(h.section == "tag" for h in idx.hits(qt, k=5))
        try:
            ene.index.open(paths[0]); raise AssertionError("a title file is not an index")
        except ValueError:
            pass
    print("index: OK")


if __name__ == "__main__":
    test_index()
    print("ALL TESTS PASSED")
