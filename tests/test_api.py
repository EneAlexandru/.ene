"""Tests for the public API on the 1.0 container (ene.api, SPEC §9): create / open / select / search, the `fast` policy, staged
edits written as one packet, overlay, compaction, rewrite, the local space (§5.2: rows of several versions on one axis,
duplicates by `prep`, several spaces and widths; tests/traceability.md rows 4, 5 and 14) and the library. Synthetic data, no model.

Run: uv run pytest tests/test_api.py   (or uv run python tests/test_api.py)
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ene  # noqa: E402
from ene import container as C  # noqa: E402
from ene.api import Hit, SectionSpec, Space, VersionSpec  # noqa: E402
from ene.quant import mrl  # noqa: E402

rng = np.random.default_rng(11)
TITLE = hashlib.sha256(b"imdb:tt1254207").digest()[:16]
OTHER = hashlib.sha256(b"imdb:tt0000001").digest()[:16]
V_A, V_B, V_C = (hashlib.sha256(s).digest()[:16] for s in (b"source-a", b"source-b", b"contrib-c"))
SPACE = "eg2-768"                                    # the example space of SPEC §2.7; every section names it
KEY = (SPACE, "int8", 768)
KEY16 = (SPACE, "fp16", 768)                     # what `compact` writes below 32 rows
PREP_IMG = {"profile": "ene-1", "tokens": 280, "resize": ["ffmpeg", "bicubic", 448, 252], "source": [1920, 1080, "h264"], "lossy": None}
PREP_IMG140 = dict(PREP_IMG, tokens=140)
PREP_TXT = {"profile": "ene-1", "prompt": "title: {title} | text: "}


def unit(n, d=768):
    """n unit rows with temporal coherence (a random walk: neighbours at cos ~0.8)."""
    x = rng.standard_normal((n, d)).astype(np.float32)
    for i in range(1, n):
        x[i] = 0.8 * x[i - 1] + 0.2 * x[i]
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def segs(n, step, length, start=0):
    a = start + np.arange(n) * step
    return np.stack([a, a + length], 1).astype(np.int32)


def noisy(v, amount=0.3):
    q = v + amount * rng.standard_normal(v.shape[-1]).astype(np.float32) / np.sqrt(v.shape[-1])
    return (q / np.linalg.norm(q)).astype(np.float32)


def make_title(path, policy="compact", n_frames=40, n_cues=50):
    """One title, one reference version at 30 fps. Title part: name, synopsis (eng), 12 tags, a summary of `frame`. Timeline:
    `frame` rows (one frame every 30) and `cue.ron` rows (20 frames every 24, with text). Returns the fp32 rows."""
    x = {"title": unit(1), "synopsis": unit(1), "tag": unit(12), "frame": unit(n_frames), "cue.ron": unit(n_cues)}
    c = x["frame"].mean(0, keepdims=True)
    x["summary.frame"] = c / np.linalg.norm(c)
    secs = [SectionSpec("title", V_A, space=SPACE, modality="mixed", role="title", vectors=x["title"], texts=["Big Buck Bunny"], prep=PREP_TXT),
            SectionSpec("synopsis", V_A, space=SPACE, modality="text", role="synopsis", lang="eng", vectors=x["synopsis"], texts=["A big rabbit wakes up."], prep=PREP_TXT),
            SectionSpec("tag", V_A, space=SPACE, modality="text", role="tag", vectors=x["tag"], texts=[f"tag {i}" for i in range(12)], prep=PREP_TXT),
            SectionSpec("summary.frame", V_A, space=SPACE, modality="image", role="summary", vectors=x["summary.frame"], ref="frame", prep=PREP_IMG),
            SectionSpec("frame", V_A, space=SPACE, modality="image", role="frame", vectors=x["frame"], times=segs(n_frames, 30, 1), prep=PREP_IMG),
            SectionSpec("cue.ron", V_A, space=SPACE, modality="text", role="cue", lang="ron", vectors=x["cue.ron"], times=segs(n_cues, 24, 20),
                        texts=[f"replica {i}" for i in range(n_cues)], prep=PREP_TXT)]
    ene.create(path, TITLE, (30, 1), versions=[VersionSpec(V_A, source_hash=hashlib.sha256(b"file").digest()[:16], fps=(30, 1), duration=1500, reference=True)],
               sections=secs, description={"title": {"name": "Big Buck Bunny", "year": 2008}}, policy=policy)
    return x


def brute(f, q, k, dim=None):
    """The per-section reference ranking (for the one-pass path): [(section id, row)] and the best score."""
    ref = sorted([(float(s.scores(q, dim)[0][i]), s.id, i) for s in f for i in range(s.count)], reverse=True)
    return [(sid, i) for _, sid, i in ref[:k]], ref[0][0]


def test_create_open_select_search():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.ene")
        x = make_title(p)
        with ene.open(p) as f:
            assert f.verify() and len(f.packets) == 1 and len(f.titles) == 1 and not f.mapped
            t = f.title()
            assert t.name == "Big Buck Bunny" and f.description["title"]["year"] == 2008 and t.axis_fps == (30, 1)
            assert list(t.versions) == [V_A] and t.reference.id == V_A and t.version(V_A).duration == 1500
            assert len(t) == 6 and {sid for _, sid in t.sections} == {"title", "synopsis", "tag", "summary.frame", "frame", "cue.ron"}
            fr, tg, cue = f["frame"], f["tag"], f.section("cue.ron", V_A)
            # policy `compact` (§4.4): fp16 + zstd below 32 rows, int8 + ene-xdelta from 32 on
            assert (tg.quant, tg.codec) == ("fp16", "zstd") and (fr.quant, fr.codec) == ("int8", "ene-xdelta") and cue.quant == "int8"
            assert fr.part == "timeline" and tg.part == "title" and fr.modality_name == "image" and cue.lang == "ron" and fr.version.id == V_A
            assert fr.key == KEY and not fr.loaded
            assert np.abs(x["frame"] @ x["frame"].T - fr.vectors @ fr.vectors.T).max() < 0.02                # int8 fidelity
            assert fr.loaded and np.abs(tg.vectors - x["tag"]).max() < 2e-3                                  # fp16
            assert fr.at(256).shape == (40, 256) and np.allclose(np.linalg.norm(fr.at(256), axis=1), 1, atol=1e-3)
            assert cue.texts[2] == "replica 2" and tg.texts == [f"tag {i}" for i in range(12)] and tg.times is None and tg.axis_times is None
            assert fr.times[7].tolist() == [210, 211] and fr.axis_times[7].tolist() == [210, 211]            # the reference: identity map
            assert fr.prep == C.prep_hash(PREP_IMG) and fr.prep_object == PREP_IMG and cue.prep_object == PREP_TXT
            assert f["summary.frame"].ref == "frame" and f["summary.frame"].part == "title"
            # selection (§9)
            ids = lambda secs: sorted(s.id for s in secs)  # noqa: E731
            assert ids(t.select(part="title")) == ["summary.frame", "synopsis", "tag", "title"]
            assert ids(t.select(part="timeline", modality="image")) == ["frame"] and ids(t.select(lang="ron")) == ["cue.ron"]
            assert ids(t.select(role="cue")) == ["cue.ron"] and ids(t.select(version=V_A, part="timeline")) == ["cue.ron", "frame"]
            assert ids(t.select(frames=(300, 330))) == ["cue.ron", "frame"] and ids(t.select(frames=(1300, 1400))) == []
            assert ids(t.select(span=(0, 2))) == ["frame"] and ids(t.select(span=(10, 30))) == ["cue.ron"]
            assert ids(t.select(space="eg2-768", min_count=12)) == ["cue.ron", "frame", "tag"] and ids(t.select(space="other")) == []
            # search
            q = noisy(x["frame"][7])
            hits = f.search(q, k=3)
            assert isinstance(hits[0], Hit) and hits[0].id == "frame" and hits[0].row == 7 and (hits[0].f0, hits[0].f1) == (210, 211)
            assert hits[0].axis == (210, 211) and hits[0].version_id == V_A and hits[0].modality == "image" and hits[0].role == "frame"
            assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
            assert f.search(q, k=3, axis=True)[0] == hits[0]
            tp = f.search(q, k=3, part="title")
            assert len(tp) == 3 and all(h.section.part == "title" and h.f0 is None for h in tp)
            win = f.search(q, k=20, frames=(300, 330))
            assert len(win) == 3 and all(h.axis[1] > 300 and h.axis[0] < 330 for h in win) and {h.id for h in win} == {"cue.ron", "frame"}
            qc = noisy(x["cue.ron"][5])
            hc = f.search(qc, k=2, modality="text", lang="ron")
            assert hc[0].id == "cue.ron" and hc[0].row == 5 and hc[0].text == "replica 5" and all(h.id == "cue.ron" for h in hc)
            assert f.scores("cue.ron", np.stack([qc, qc])).shape == (2, 50)
            assert np.abs(fr.scores(q, dim=256) - mrl(q[None], 256) @ fr.at(256).T).max() < 1e-5         # Matryoshka at read time
            assert f.search(q, k=1, dim=256)[0].row == 7 and f.search(mrl(q[None], 256)[0], k=1, dim=256)[0].row == 7
            assert "frame" in f and "nope" not in f and len(list(f)) == 6 and f.describe().startswith("title ")
    print("create + open + select + search: OK")


def test_fast_policy():
    """`fast` (§4.4): one packet, ONE_BLOCK, every section a row range of one raw block; fp32 rows below FAST_INT8_ROWS rows in the
    packet, int8 from there on (`fast-int8` / `fast-fp32` force the form). The one-pass search on the file bytes equals the
    per-section path; a forced quant on one section takes that section out of the shared block."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "f.ene")
        x = make_title(p, policy="fast")
        with ene.open(p, verify=False) as f:
            t = f.title()
            assert f.packets[0].one_block and all(s.quant == "fp32" and s.codec == "none" and s.row_offset is not None for s in t)
            assert min(s.row_offset for s in t) == 0 and t._fast_state() is not None and t._fast_state()[0] == "fp32"
            q = noisy(x["frame"][5])
            direct = f.search(q, k=4)                                                               # one GEMV on the file bytes
            assert t._fast_queries == 1 and not any(s.loaded for s in t)
            ref, top = brute(f, q, 4)
            assert [(h.id, h.row) for h in direct] == ref and abs(direct[0].score - top) < 1e-6
            assert direct[0].id == "frame" and direct[0].row == 5 and direct[0].f0 == 150
            d256 = f.search(q, k=3, dim=256)
            ref256, top256 = brute(f, q, 3, 256)
            assert [(h.id, h.row) for h in d256] == ref256 and abs(d256[0].score - top256) < 1e-5 and t._fast_queries == 2
            third = f.search(q, k=4)                                                                # from the third query on: per section
            assert [(h.id, h.row) for h in third] == [(h.id, h.row) for h in direct]
            assert np.abs(f["frame"].vectors - x["frame"]).max() < 1e-6 and f["tag"].texts[4] == "tag 4"   # fp32: exact
        # the int8 form from FAST_INT8_ROWS rows in the packet; the folded scale equals explicit dequantization
        pbig = os.path.join(d, "big.ene")
        xb = make_title(pbig, policy="fast", n_frames=ene.api.FAST_INT8_ROWS)
        with ene.open(pbig, verify=False) as f:
            s = f["frame"]
            assert f.packets[0].one_block and s.quant == "int8" and s.codec == "none" and f.title()._fast_state()[0] == "int8"
            q = unit(2)
            assert np.abs(s.scores(q) - q @ s.vectors.T).max() < 1e-5 and np.abs(np.linalg.norm(s.vectors, axis=1) - 1).max() < 0.01
            assert np.abs(s.scores(q, dim=256) - mrl(q, 256) @ s.at(256).T).max() < 1e-5
            qq = noisy(xb["frame"][700])
            h = f.search(qq, k=3)                                                                   # parallel cast + one GEMV
            refr, top = brute(f, qq, 3)
            assert [(hh.id, hh.row) for hh in h] == refr and abs(h[0].score - top) < 1e-5 and h[0].row == 700
        for name, form in (("fast-int8", "int8"), ("fast-fp32", "fp32")):
            pf = os.path.join(d, name + ".ene")
            make_title(pf, policy=name)
            with ene.open(pf, verify=False) as f:
                assert f.packets[0].one_block and {s.quant for s in f} == {form} and f.verify()
        pm = os.path.join(d, "mixed.ene")
        ene.create(pm, TITLE, (30, 1), versions=[VersionSpec(V_A, fps=(30, 1), duration=100, reference=True)], policy="fast",
                   sections=[SectionSpec("a", V_A, space=SPACE, modality="text", role="tag", vectors=unit(3), texts=list("abc")),
                             SectionSpec("b", V_A, space=SPACE, modality="text", role="tag", vectors=unit(2), texts=list("de"), quant="fp16", codec="zstd")])
        with ene.open(pm) as f:
            assert not f.packets[0].one_block and f["a"].codec == "none" and f["b"].codec == "zstd" and len(f.search(unit(1)[0], k=5)) == 5
    print("fast policy: OK")


def test_edits_save_overlay_compact():
    """Staged edits written as ONE packet (§0.8, §3.9): description keys merged, a new version with its sections, a replacement
    (rev + 1), a tombstone, a corrected alignment as a version record alone (§2.8); save(path) = a compacted copy; compact() in
    place; overlay = cat (§3.1)."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.ene")
        x = make_title(p)
        n1 = os.path.getsize(p)
        xb, xs, xt = unit(10), unit(1), unit(3)
        with ene.open(p) as f:
            assert not f.dirty
            f.set(title={"year": 2009, "kind": "movie"})                                            # merged: the name survives
            f.add_version(V_C)                                                                      # a contribution without a timeline
            f.add_section("synopsis", V_C, space=SPACE, modality="text", role="synopsis", lang="ron", vectors=xs, texts=["Un iepure mare se trezește."], prep=PREP_TXT)
            f.add_version(V_B, fps=(25, 1), duration=1250, map_a=1.2, map_b=0.0)                  # a 25 fps transfer in real time
            f.add_section("frame", V_B, space=SPACE, modality="image", role="frame", vectors=xb, times=segs(10, 25, 1), prep=PREP_IMG)
            f.add_section("tag", V_A, space=SPACE, modality="text", role="tag", vectors=xt, texts=list("abc"), prep=PREP_TXT)   # replacement
            f.remove_section("summary.frame")
            assert f.dirty
            try:
                f.add_section("x", V_C.hex()[:4], space=SPACE, modality="text", role="tag", vectors=unit(1)); raise AssertionError("a bad version id must raise")
            except ValueError:
                pass
            try:
                f.add_version(V_A, fps=(30, 1), duration=1500, reference=True); raise AssertionError("a held version needs a higher rev")
            except ValueError:
                pass
            f.save()
            assert not f.dirty and len(f.packets) == 2 and os.path.getsize(p) > n1 and f.verify()
        with ene.open(p) as f:
            t = f.title()
            assert t.name == "Big Buck Bunny" and t.description["title"] == {"name": "Big Buck Bunny", "year": 2009, "kind": "movie"}
            assert set(t.versions) == {V_A, V_B, V_C} and not t.version(V_C).timeline and t.version(V_B).map_a == 1.2 and t.version(V_B).timeline
            assert "summary.frame" not in f and len(t) == 7
            assert f.section("synopsis", V_C).texts == ["Un iepure mare se trezește."] and f.section("synopsis", V_C).lang == "ron"
            try:
                f["synopsis"]; raise AssertionError("an id in two versions must ask for the version")
            except KeyError:
                pass
            tg = f["tag"]
            assert tg.rev == 1 and tg.count == 3 and tg.texts == ["a", "b", "c"] and tg.quant == "fp16"
            fb = f.section("frame", V_B)
            assert fb.prep_object == PREP_IMG and fb.times[4].tolist() == [100, 101] and fb.axis_times[4].tolist() == [120, 122]   # floor / ceil
            fa = f.section("frame", V_A)
            assert np.abs(x["frame"] @ x["frame"].T - fa.vectors @ fa.vectors.T).max() < 0.02                # untouched
            assert f.search(noisy(xb[4]), k=1, version=V_B, axis=True)[0].f0 == 120
            # a corrected alignment: one packet with the version record alone (rev + 1); rows are not rewritten
            f.set_alignment(V_B, 1.2, -6.0)
            f.save()
            assert len(f.packets) == 3 and len(f.packets[2].sections) == 0 and len(f.packets[2].versions) == 1
            assert f.title().version(V_B).rev == 1 and f.section("frame", V_B).axis_times[4].tolist() == [114, 116]
            # save(path): a compacted copy with the staged changes; the file and the staged changes stay
            f.set(title={"year": 2010}).remove_section("tag")
            pc = f.save(os.path.join(d, "copy.ene"))
            assert f.dirty and len(f.packets) == 3
        with ene.open(pc) as c:
            assert len(c.packets) == 1 and c.description["title"]["year"] == 2010 and "tag" not in c and len(c.title()) == 6
            assert c.section("frame", V_B).axis_times[4].tolist() == [114, 116] and c.verify()
        with ene.open(p) as f:
            assert f.description["title"]["year"] == 2009 and "tag" in f
            f.compact()
            assert len(f.packets) == 1 and "tag" in f and f.title().version(V_B).rev == 1 and f.verify()
            assert f.section("frame", V_B).axis_times[4].tolist() == [114, 116] and len(f.title()) == 7
        # overlay: another contributor's packet for the same title
        po = os.path.join(d, "o.ene")
        ene.create(po, TITLE, (30, 1), versions=[VersionSpec(V_A, fps=(30, 1), duration=1500, reference=True)],
                   sections=[SectionSpec("cue.eng", V_A, space=SPACE, modality="text", role="cue", lang="eng", vectors=unit(4), times=segs(4, 100, 30), texts=list("wxyz"), prep=PREP_TXT)])
        ene.overlay(p, po)
        with ene.open(p) as f:
            assert len(f.packets) == 2 and f.section("cue.eng").texts == list("wxyz") and f.verify() and len(f.title()) == 8
        pz = os.path.join(d, "z.ene")
        ene.create(pz, OTHER, (24, 1), versions=[VersionSpec(V_C)], sections=[SectionSpec("tag", V_C, space=SPACE, modality="text", role="tag", vectors=unit(1), texts=["z"])])
        try:
            ene.overlay(p, pz); raise AssertionError("a foreign title must be refused")
        except ValueError:
            pass
        ene.overlay(p, pz, check_title=False)
        with ene.open(p) as f:
            assert set(f.titles) == {TITLE, OTHER} and f.titles[OTHER].section("tag").texts == ["z"]
            try:
                f.title(); raise AssertionError("two titles: the id is needed")
            except KeyError:
                pass
    print("edits + save + overlay + compact: OK")


def test_rewrite():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.ene")
        x = make_title(p)
        out = os.path.join(d, "l.ene")
        ene.rewrite(p, out, policy="lossless")
        with ene.open(out) as r, ene.open(p) as a:
            assert {(s.quant, s.codec) for s in r} == {("fp32", "zstd")} and r.name == "Big Buck Bunny" and r.description == a.description
            for s in a:
                rs = r.section(s.id, s.version_id)
                assert (rs.prep, rs.rev, rs.ref, rs.lang, rs.modality, rs.role) == (s.prep, s.rev, s.ref, s.lang, s.modality, s.role)
                assert rs.texts == s.texts and (rs.times is None and s.times is None or np.array_equal(rs.times, s.times))
                assert np.abs(rs.vectors - s.vectors / np.linalg.norm(s.vectors, axis=1, keepdims=True)).max() < 1e-6   # renormalized, then exact
            assert np.abs(x["frame"] @ x["frame"].T - r["frame"].vectors @ r["frame"].vectors.T).max() < 0.02
            assert r.search(noisy(x["frame"][3]), k=1)[0].row == 3
        out2 = os.path.join(d, "f.ene")
        ene.rewrite(out, out2, policy="fast")
        with ene.open(out2, verify=False) as f, ene.open(out) as r:
            assert f.packets[0].one_block and {s.quant for s in f} == {"fp32"} and f.search(noisy(x["frame"][3]), k=1)[0].row == 3
            assert np.abs(f["frame"].vectors - r["frame"].vectors).max() < 1e-6 and f["cue.ron"].prep_object == PREP_TXT
    print("rewrite: OK")


def test_space_versions_on_one_axis():
    """tests/traceability.md row 4 (§2.8, §5.2): rows of three versions at different frame rates land on one axis; a corrected
    version record re-maps rows already held; a replacement and a tombstone from later files drop rows; a stale rev is ignored."""
    with tempfile.TemporaryDirectory() as d:
        n, k = 20, 7
        xa = unit(n)
        xb, xc = np.stack([noisy(v, 0.05) for v in xa]), np.stack([noisy(v, 0.05) for v in xa])
        pa, pb, pc = (os.path.join(d, s) for s in ("a.ene", "b.ene", "c.ene"))
        # A: the reference at 25 fps, one frame per second; B: a 30 fps real-time re-render (a = 25/30), the same instants at frames
        # 30i; C: 29.97 fps with a 3:2 pulldown and a 100-frame leader (a = 1.25, b = -125), the same instants at frames 100 + 20i
        ene.create(pa, TITLE, (25, 1), versions=[VersionSpec(V_A, fps=(25, 1), duration=500, reference=True)],
                   sections=[SectionSpec("frame", V_A, space=SPACE, modality="image", role="frame", vectors=xa, times=segs(n, 25, 1), prep=PREP_IMG)])
        ene.create(pb, TITLE, (25, 1), versions=[VersionSpec(V_B, fps=(30, 1), duration=600, map_a=25 / 30, map_b=0.0)],
                   sections=[SectionSpec("frame", V_B, space=SPACE, modality="image", role="frame", vectors=xb, times=segs(n, 30, 1), prep=PREP_IMG140)])
        ene.create(pc, TITLE, (25, 1), versions=[VersionSpec(V_C, fps=(30000, 1001), duration=700, map_a=1.25, map_b=-125.0)],
                   sections=[SectionSpec("frame", V_C, space=SPACE, modality="image", role="frame", vectors=xc, times=segs(n, 20, 1, start=100), prep=PREP_IMG140)])
        sp = Space()
        assert sp.ingest(pa) == n and sp.ingest(pb) == n and sp.ingest(pc) == n and len(sp) == 3 * n
        assert list(sp.keys) == [KEY16] and len(sp.title_list) == 1 and len(sp.version_list) == 3 and sp.spaces == ["eg2-768"]
        c = sp.rows(KEY16)
        a_rows, b_rows, c_rows = (c["version"] == i for i in range(3))
        assert c["f0"][a_rows].tolist() == [25 * i for i in range(n)] and c["f1"][a_rows].tolist() == [25 * i + 1 for i in range(n)]
        assert c["f0"][b_rows].tolist() == c["f0"][a_rows].tolist() and c["f1"][b_rows].tolist() == c["f1"][a_rows].tolist()     # coincide
        assert c["f0"][c_rows].tolist() == c["f0"][a_rows].tolist() and c["f1"][c_rows].tolist() == [25 * i + 2 for i in range(n)]  # 1.25 frames, rounded outwards: overlap
        assert c["vf0"][c_rows].tolist() == [100 + 20 * i for i in range(n)] and (c["part"] == 1).all()
        hits = sp.search(xa[k], k=3, part="timeline")
        assert {h.version_id for h in hits} == {V_A, V_B, V_C} and all(h.row == k for h in hits) and hits[0].version_id == V_A
        assert (hits[0].f0, hits[0].f1) == (25 * k, 25 * k + 1) and hits[0].title_id == TITLE and hits[0].key == KEY16
        at = sp.search(xa[k], k=10, frames=(25 * k, 25 * k + 1))                                    # everything at that instant
        assert sorted(h.version_id for h in at) == sorted([V_A, V_B, V_C]) and all(h.row == k for h in at)
        assert {h.version_id for h in sp.search(xa[k], k=5, version=V_C)} == {V_C} and sp.search(xa[k], k=1, title=TITLE)[0].row == k
        assert sp.search(xa[k], k=5, part="title") == [] and sp.search(xa[k], k=5, role="cue") == []
        # a corrected alignment for B arrives as a version record alone (rev 1): the rows already held move by +5 axis frames
        pd_ = os.path.join(d, "d.ene")
        ene.create(pd_, TITLE, (25, 1), versions=[VersionSpec(V_B, fps=(30, 1), duration=600, map_a=25 / 30, map_b=5.0, rev=1)])
        assert sp.ingest(pd_) == 0 and len(sp) == 3 * n
        c = sp.rows(KEY16)
        assert c["f0"][b_rows].tolist() == [25 * i + 5 for i in range(n)] and c["f1"][b_rows].tolist() == [25 * i + 6 for i in range(n)]
        assert sp.search(xb[k], k=1, version=V_B)[0].f0 == 25 * k + 5
        # a replacement of B's section from a later file (rev 1, 10 rows) drops the 20 held rows; a tombstone for C's drops C's
        pe = os.path.join(d, "e.ene")
        ene.create(pe, TITLE, (25, 1), versions=[VersionSpec(V_B, fps=(30, 1), duration=600, map_a=25 / 30, map_b=5.0, rev=1)],
                   sections=[SectionSpec("frame", V_B, space=SPACE, modality="image", role="frame", vectors=xb[:10], times=segs(10, 30, 1), prep=PREP_IMG140, rev=1)])
        assert sp.ingest(pe) == 10 and len(sp) == 2 * n + 10
        w = ene.Writer(TITLE, (25, 1))
        w.add_version(VersionSpec(V_C, fps=(30000, 1001), duration=700, map_a=1.25, map_b=-125.0))
        w.add_tombstone("frame", V_C)
        pf = os.path.join(d, "f.ene")
        w.write(pf)
        assert sp.ingest(pf) == 0 and len(sp) == n + 10
        assert {h.version_id for h in sp.search(xa[k], k=10)} == {V_A, V_B}
        assert sp.ingest(pb) == 0 and len(sp) == n + 10                                              # a stale revision, ignored
    print("local space: versions on one axis: OK")


def test_space_duplicates_by_prep():
    """tests/traceability.md row 5 (§2.7, §5.2): rows of the same key, version, part, modality, role, lang and `prep` on the same
    segment (or with the same text) are the same information twice: kept once, from the higher rev, the later file on a tie. A
    different `prep`, `prep = 0` or a row without text is a variant, kept."""
    with tempfile.TemporaryDirectory() as d:
        xs = unit(30)

        def contrib(name, sid, rows, start=None, texts=None, prep=PREP_IMG, rev=0, role="frame", modality="image"):
            p = os.path.join(d, name)
            times = None if start is None else segs(len(rows), 30, 1, start=30 * start)
            ene.create(p, TITLE, (30, 1), versions=[VersionSpec(V_A, fps=(30, 1), duration=1000, reference=True)],
                       sections=[SectionSpec(sid, V_A, space=SPACE, modality=modality, role=role, vectors=rows, times=times, texts=texts, prep=prep, rev=rev)])
            return p

        sp = Space()
        assert sp.ingest(contrib("1.ene", "frame", xs[:10], 0)) == 10                               # contributor 1: frames S0..S9
        assert sp.ingest(contrib("1t.ene", "tag", xs[:5], texts=[f"t{i}" for i in range(5)], prep=PREP_TXT, role="tag", modality="text")) == 5
        # contributor 2, same version and prep, another id: frames S5..S14 -> S5..S9 twice; the later file keeps them
        assert sp.ingest(contrib("2.ene", "frame.b", xs[5:15], 5)) == 10 and len(sp) == 20
        one, two = sp.sections[0], sp.sections[2]
        c = sp.rows(KEY16)
        assert c["alive"][c["section"] == one.index].tolist() == [True] * 5 + [False] * 5 and c["alive"][c["section"] == two.index].all()
        h = sp.search(xs[7], k=1)[0]
        assert h.id == "frame.b" and h.row == 2 and (h.f0, h.f1) == (210, 211)
        # the title part: duplicates by text; rows without text never duplicate
        assert sp.ingest(contrib("2t.ene", "tag.b", xs[3:6], texts=["t3", "t4", "t5"], prep=PREP_TXT, role="tag", modality="text")) == 3 and len(sp) == 21
        assert sp.search(xs[3], k=1, part="title")[0].id == "tag.b" and sp.search(xs[1], k=1, part="title")[0].id == "tag"
        assert sp.ingest(contrib("3t.ene", "tag.notext", xs[:5], prep=PREP_TXT, role="tag", modality="text")) == 5 and len(sp) == 26
        # another prep (140 tokens) and prep = 0: variants, kept beside the originals
        assert sp.ingest(contrib("3.ene", "frame.c", xs[:5], 0, prep=PREP_IMG140)) == 5 and len(sp) == 31
        assert sp.ingest(contrib("4.ene", "frame.zero", xs[:5], 0, prep=None)) == 5 and len(sp) == 36
        assert {h.id for h in sp.search(xs[2], k=3, part="timeline")} == {"frame", "frame.c", "frame.zero"}
        # the higher rev wins whatever the order of arrival
        assert sp.ingest(contrib("5.ene", "frame.d", xs[:3], 0, rev=2)) == 3 and len(sp) == 36      # contributor 1's S0..S2 die
        c = sp.rows(KEY16)
        assert c["alive"][c["section"] == one.index].tolist() == [False] * 3 + [True] * 2 + [False] * 5
        assert sp.ingest(contrib("6.ene", "frame.e", xs[:3], 0, rev=1)) == 0 and len(sp) == 36      # rev 1 < held rev 2: dead on arrival
        e = sp.sections[-1]
        c = sp.rows(KEY16)
        assert not c["alive"][c["section"] == e.index].any() and c["alive"][c["section"] == sp.sections[-2].index].all()
        assert sp.search(xs[0], k=10, role="frame") and all(h.id != "frame.e" for h in sp.search(xs[0], k=20))
    print("local space: duplicates by prep: OK")


def test_space_two_spaces_two_widths():
    """tests/traceability.md row 14 (§0.4, §2.7, §4.3): two spaces on the same segments never share a matrix; two widths of one
    space stack after truncation and renormalization; a search names the space when several are held."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.ene")
        x, face = unit(40), unit(40, 512)
        ene.create(p, TITLE, (30, 1), versions=[VersionSpec(V_A, fps=(30, 1), duration=1500, reference=True)], sections=[
            SectionSpec("frame", V_A, space=SPACE, modality="image", role="frame", vectors=x, times=segs(40, 30, 1), prep=PREP_IMG),
            SectionSpec("frame.256", V_A, space=SPACE, modality="image", role="frame", vectors=mrl(x, 256), dim_real=768, times=segs(40, 30, 1), prep=PREP_IMG),
            SectionSpec("face", V_A, modality="image", role="face", space="face-512", vectors=face, times=segs(40, 30, 1), prep={"profile": "face-1"})])
        with ene.open(p) as f:
            assert f["frame.256"].dim == 256 and f["frame.256"].dim_real == 768 and f["frame.256"].key == ("eg2-768", "int8", 256)
            assert f["face"].key == ("face-512", "int8", 512) and f["frame"].key == KEY
            try:
                f.search(x[3], k=1); raise AssertionError("two spaces in one selection must raise")
            except ValueError:
                pass
            top = f.search(x[3], k=1, space="eg2-768")[0]                                     # a tie: the same row at 768 and at 256
            assert top.id in ("frame", "frame.256") and top.row == 3 and f.search(face[3], k=1, space="face-512")[0].row == 3
            two = f.search(x[3], k=2, space="eg2-768", dim=256)
            assert {h.id for h in two} == {"frame", "frame.256"} and all(h.row == 3 for h in two) and abs(two[0].score - two[1].score) < 5e-3
        sp = Space()
        assert sp.ingest(p) == 120
        assert sorted(sp.keys) == [("eg2-768", "int8", 256), KEY, ("face-512", "int8", 512)] and sp.spaces == ["eg2-768", "face-512"]
        try:
            sp.search(x[3], k=1); raise AssertionError("several spaces: name one")
        except ValueError:
            pass
        assert [h.id for h in sp.search(face[3], k=2, space="face-512")] == ["face", "face"] and sp.search(face[3], k=1, space="face-512")[0].row == 3
        two = sp.search(x[3], k=2, space="eg2-768", dim=256)
        assert {h.id for h in two} == {"frame", "frame.256"} and all(h.row == 3 for h in two) and abs(two[0].score - two[1].score) < 5e-3
        full = sp.search(x[3], k=2, space="eg2-768")                                               # no dim: each key at its own width
        assert {h.id for h in full} == {"frame", "frame.256"} and all(h.row == 3 for h in full)
        assert sp.search(x[3], k=1, key=("eg2-768", "int8", 256))[0].id == "frame.256"
        assert sp.search(mrl(x[3:4], 256)[0], k=1, space="eg2-768", dim=256)[0].row == 3        # a 256-wide query
    print("local space: two spaces, two widths: OK")


def test_library():
    with tempfile.TemporaryDirectory() as d:
        pa, pb = os.path.join(d, "a.ene"), os.path.join(d, "b.ene")
        xa = make_title(pa)
        xb = {"title": unit(1), "tag": unit(5), "frame": unit(30)}
        ene.create(pb, OTHER, (24, 1), versions=[VersionSpec(V_B, fps=(24, 1), duration=800, reference=True)], description={"title": {"name": "Other"}},
                   sections=[SectionSpec("title", V_B, space=SPACE, modality="mixed", role="title", vectors=xb["title"], texts=["Other"], prep=PREP_TXT),
                             SectionSpec("tag", V_B, space=SPACE, modality="text", role="tag", vectors=xb["tag"], texts=list("abcde"), prep=PREP_TXT),
                             SectionSpec("frame", V_B, space=SPACE, modality="image", role="frame", vectors=xb["frame"], times=segs(30, 24, 1), prep=PREP_IMG)])
        with ene.Library([pa, pb]) as lib:
            ranked = lib.titles(noisy(xb["tag"][2]))
            assert ranked[0][0].name == "Other" and ranked[0][1].id == "tag" and ranked[0][1].row == 2 and ranked[1][0].name == "Big Buck Bunny"
            res = lib.search(noisy(xa["frame"][9]), k=3, part="timeline")
            assert res[0][0].name == "Big Buck Bunny" and res[0][1].id == "frame" and res[0][1].row == 9 and len(res) == 2
            sp = lib.space
            assert len(sp.title_list) == 2 and len(sp) == 105 + 36
            best = sp.titles(noisy(xa["frame"][9]), part="timeline")
            assert [h.title_id for h in best] == [TITLE, OTHER] and best[0].row == 9 and best[0].id == "frame"
    print("library: OK")


if __name__ == "__main__":
    test_create_open_select_search()
    test_fast_policy()
    test_edits_save_overlay_compact()
    test_rewrite()
    test_space_versions_on_one_axis()
    test_space_duplicates_by_prep()
    test_space_two_spaces_two_widths()
    test_library()
    print("ALL TESTS PASSED")
