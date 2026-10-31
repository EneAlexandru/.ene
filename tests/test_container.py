"""Container 1.0 (SPEC-ene.md §3): record layout, canonical JSON and the `prep` hash, a round trip with both parts,
overlay across packets, the `fast` shared block, compaction, concatenation and the error paths. Synthetic data."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ene import container as C  # noqa: E402

rng = np.random.default_rng(7)
TITLE = hashlib.sha256(b"imdb:tt1254207").digest()[:16]
V_REF = hashlib.sha256(b"source-a").digest()[:16]
V_SUB = hashlib.sha256(b"contrib-b").digest()[:16]


def unit(n, d):
    x = rng.standard_normal((n, d)).astype(np.float32)
    for i in range(1, n):
        x[i] = 0.8 * x[i - 1] + 0.2 * x[i]
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_record_layout():
    assert C.PREFIX_DT.itemsize == 64 and C.VERSION_DT.itemsize == 128 and C.SECTION_DT.itemsize == 256
    off = lambda dt, f: dt.fields[f][1]
    assert off(C.PREFIX_DT, "title_id") == 16 and off(C.PREFIX_DT, "desc_len") == 48 and off(C.PREFIX_DT, "trailer_pad") == 57
    assert off(C.VERSION_DT, "map_a") == 40 and off(C.VERSION_DT, "duration") == 56 and off(C.VERSION_DT, "flags") == 68
    assert off(C.SECTION_DT, "space") == 40 and off(C.SECTION_DT, "flags") == 77 and off(C.SECTION_DT, "dim") == 80
    assert off(C.SECTION_DT, "vec_off") == 104 and off(C.SECTION_DT, "text_raw") == 192
    assert off(C.SECTION_DT, "ref") == 200 and off(C.SECTION_DT, "prep") == 232 and off(C.SECTION_DT, "reserved2") == 240
    assert C.CODECS.index("ene-xdelta") == 3 and C.CODECS.index("lz4") == 4 and C.QUANTS.index("int8") == 2


def test_canonical_json_and_prep_hash():
    obj = {"b": 1, "a": [1.0, 0.5, 1e-7, 0.000001, 1e21, 123.456, 1.5e22, -0.0, True, None, "ș\n\"x\""], "é": {"z": 30, "y": 30.0}}
    assert C.canonical_json(obj) == '{"a":[1,0.5,1e-7,0.000001,1e+21,123.456,1.5e+22,0,true,null,"ș\\n\\"x\\""],"b":1,"é":{"y":30,"z":30}}'.encode("utf-8")
    h = C.prep_hash({"profile": "ene-1", "tokens": 280})
    assert h == int.from_bytes(hashlib.sha256(b'{"profile":"ene-1","tokens":280}').digest()[:8], "little") and h != 0
    assert C.prep_hash({"tokens": 280, "profile": "ene-1"}) == h            # key order does not matter
    assert C.prep_hash({"tokens": 280.0, "profile": "ene-1"}) == h          # 280.0 and 280 are one number
    assert C.prep_hash(None) == 0 and C.prep_hash(h) == h and C.prep_hash(b'{"profile":"ene-1","tokens":280}') == h
    assert C.prep_hash({"a": 1}) != C.prep_hash({"a": 2})


PREP_IMG = {"profile": "ene-1", "tokens": 280, "resize": ["ffmpeg", "bicubic", 448, 252], "source": [1920, 1080, "h264"], "lossy": None}
PREP_TXT = {"profile": "ene-1", "prompt": "title: {title} | text: "}


def write_title(path):
    """A title with both parts: 1 title row, 1 synopsis, 12 tags, 1 summary; 40 frames (int8 + ene-xdelta, unsorted input),
    50 cues (fp16 + zstd, texts). Returns what was written for the checks."""
    w = C.PacketWriter(TITLE, axis_fps=(30, 1))
    v = w.add_version(V_REF, source_hash=hashlib.sha256(b"file").digest()[:16], fps=(30, 1), duration=1000, flags=C.V_REFERENCE | C.V_TIMELINE)
    x_title, x_syn, x_tag, x_sum = unit(1, 768), unit(1, 768), unit(12, 768), unit(1, 768)
    x_frame, x_cue = unit(40, 768), unit(50, 768)
    cuts = np.sort(rng.choice(np.arange(0, 999), 40, replace=False)).astype(np.int32)
    t_frame = np.stack([cuts, cuts + 1], 1)
    perm = rng.permutation(40)                                              # written unsorted on purpose
    t_cue = np.stack([np.arange(50) * 20, np.arange(50) * 20 + 15], 1).astype(np.int32)
    cues = [f"replica {i} ăîșț" for i in range(50)]
    w.add_section("title", v, space="eg2-768", modality="mixed", role="title", vectors=x_title, quant="fp16", codec="zstd", texts=["Big Buck Bunny"], prep=PREP_TXT)
    w.add_section("synopsis", v, space="eg2-768", modality="text", role="synopsis", lang="eng", vectors=x_syn, quant="fp16", texts=["A big rabbit..."], prep=PREP_TXT)
    w.add_section("tag", v, space="eg2-768", modality="text", role="tag", lang="eng", vectors=x_tag, quant="fp16", texts=[f"tag{i}" for i in range(12)], prep=PREP_TXT)
    w.add_section("summary.frame", v, space="eg2-768", modality="image", role="summary", vectors=x_sum, quant="fp16", ref="frame", prep=PREP_IMG)
    w.add_section("frame", v, space="eg2-768", modality="image", role="frame", vectors=x_frame[perm], quant="int8", codec="ene-xdelta", times=t_frame[perm], prep=PREP_IMG)
    w.add_section("cue.ron", v, space="eg2-768", modality="text", role="cue", lang="ron", vectors=x_cue, quant="fp16", codec="zstd", times=t_cue, texts=cues, prep=PREP_TXT)
    desc = {"title": {"name": "Big Buck Bunny", "year": 2008}, "spaces": {"eg2-768": {"model": "google/embeddinggemma-2", "dim": 768}},
            "sections": {V_REF.hex() + "/frame": {"prep": PREP_IMG}}}
    w.set_description(desc)
    n = w.write(path)
    return dict(n=n, x_title=x_title, x_syn=x_syn, x_tag=x_tag, x_frame=x_frame, t_frame=t_frame, x_cue=x_cue, t_cue=t_cue, cues=cues, desc=desc)


def test_roundtrip_two_parts():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.ene")
        g = write_title(p)
        assert g["n"] % 64 == 0 and os.path.getsize(p) == g["n"]
        with C.EneFile(p) as f:
            assert len(f.packets) == 1 and f.packets[0].hash_ok is True and f.packets[0].length == g["n"]
            pk = f.packets[0]
            assert pk.title_id == TITLE and pk.axis_fps == (30, 1) and not pk.one_block
            # table order: title part first, ascending row count, ties by id (§2.5)
            ids = [bytes(r["id"]).decode() for r in pk.sections]
            assert ids == ["summary.frame", "synopsis", "title", "tag", "frame", "cue.ron"]
            for r in pk.sections:                                            # every block 64-byte aligned
                for name, f_off, f_len, _ in C._BLOCK_FIELDS:
                    if r[f_len]:
                        assert int(r[f_off]) % 64 == 0
            t = f.title()
            assert set(t.versions) == {V_REF} and t.versions[V_REF].reference and t.versions[V_REF].duration == 1000
            s_frame, s_cue, s_tag = t.section("frame"), t.section("cue.ron"), t.section("tag")
            assert s_frame.key == ("eg2-768", "int8", 768) and s_frame.timeline and s_frame.prep == C.prep_hash(PREP_IMG)
            assert s_frame.version is t.versions[V_REF] and s_frame.ref == "" and t.section("summary.frame").ref == "frame"
            assert s_tag.lang == "eng" and s_tag.role == "tag" and s_tag.modality_name == "text" and not s_tag.timeline and s_tag.times() is None
            # timeline rows came back sorted, with vectors permuted along
            assert np.array_equal(s_frame.times(), g["t_frame"])
            y = s_frame.vectors()
            assert np.abs((g["x_frame"] @ g["x_frame"].T) - (y @ y.T)).max() < 0.02
            assert np.allclose(s_cue.vectors(), g["x_cue"], atol=1e-3) and s_cue.texts() == g["cues"]
            assert np.allclose(t.section("title").vectors(), g["x_title"], atol=1e-3) and t.section("title").texts() == ["Big Buck Bunny"]
            assert np.array_equal(s_cue.axis_times(), g["t_cue"].astype(np.int64))   # identity map
            assert s_frame.scales().shape == (40,) and s_frame.vectors(raw=True).dtype == np.int8
            assert t.description == g["desc"]
            assert t.description["sections"][V_REF.hex() + "/frame"]["prep"] == PREP_IMG
            print(t.describe())


def test_overlay():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.ene")
        g = write_title(p)
        # packet 2: corrected alignment (rev 1), cue.ron replaced (rev 1, 60 rows), tag deleted, a stale tombstone for frame (rev 0 vs 0 → removes),
        # a new version without timeline carrying a Romanian synopsis, a partial description
        w = C.PacketWriter(TITLE, axis_fps=(30, 1))
        v = w.add_version(V_REF, fps=(30, 1), map_a=1.0, map_b=-100.0, duration=1000, rev=1, flags=C.V_TIMELINE)
        v2 = w.add_version(V_SUB, flags=0)
        x_cue2 = unit(60, 768); t_cue2 = np.stack([np.arange(60) * 15, np.arange(60) * 15 + 10], 1).astype(np.int32)
        w.add_section("cue.ron", v, space="eg2-768", modality="text", role="cue", lang="ron", vectors=x_cue2, quant="fp16", times=t_cue2, rev=1, prep=PREP_TXT)
        w.add_tombstone("tag", v, rev=0)
        x_syn_ro = unit(1, 768)
        w.add_section("synopsis", v2, space="eg2-768", modality="text", role="synopsis", lang="ron", vectors=x_syn_ro, quant="fp16", texts=["Un iepure mare..."], prep=PREP_TXT)
        w.set_description({"title": {"kind": "movie"}, "versions": {V_SUB.hex(): {"generator": {"contributor": "b"}}}})
        n2 = w.write(p, append=True)
        assert os.path.getsize(p) == g["n"] + n2
        with C.EneFile(p) as f:
            assert len(f.packets) == 2 and len(f.titles) == 1
            t = f.title()
            assert t.versions[V_REF].map_b == -100.0 and t.versions[V_REF].rev == 1 and not t.versions[V_REF].reference
            assert set(t.versions) == {V_REF, V_SUB} and not t.versions[V_SUB].timeline
            keys = {(vid.hex()[:4], sid) for vid, sid in t.sections}
            assert (V_REF.hex()[:4], "tag") not in keys and (V_SUB.hex()[:4], "synopsis") in keys and (V_REF.hex()[:4], "synopsis") in keys
            s = t.section("cue.ron")
            assert s.rev == 1 and s.count == 60 and np.allclose(s.vectors(), x_cue2, atol=1e-3)
            assert np.array_equal(s.axis_times()[:, 0], t_cue2[:, 0] - 100)       # the corrected map applies to the rows
            assert np.array_equal(t.section("frame").axis_times(), g["t_frame"].astype(np.int64) - 100)
            try:
                t.section("synopsis"); assert False
            except KeyError:
                pass
            assert t.section("synopsis", V_SUB).texts() == ["Un iepure mare..."]
            assert t.description["title"] == {"name": "Big Buck Bunny", "year": 2008, "kind": "movie"}   # merged key by key
            assert t.description["versions"][V_SUB.hex()]["generator"]["contributor"] == "b"
        # a tombstone older than the section it names is ignored
        w = C.PacketWriter(TITLE, axis_fps=(30, 1))
        v = w.add_version(V_REF, fps=(30, 1), map_b=-100.0, duration=1000, rev=1, flags=C.V_TIMELINE)
        w.add_tombstone("cue.ron", v, rev=0)
        w.write(p, append=True)
        with C.EneFile(p) as f:
            assert f.title().section("cue.ron").count == 60


def test_compact_lossless():
    with tempfile.TemporaryDirectory() as d:
        p, q = os.path.join(d, "t.ene"), os.path.join(d, "c.ene")
        write_title(p)
        w = C.PacketWriter(TITLE, axis_fps=(30, 1))
        v = w.add_version(V_REF, fps=(30, 1), map_b=-100.0, duration=1000, rev=1, flags=C.V_TIMELINE)
        w.add_tombstone("tag", v)
        w.set_description({"title": {"kind": "movie"}})
        w.write(p, append=True)
        n = C.compact(p, q)
        with C.EneFile(p) as a, C.EneFile(q) as b:
            assert len(b.packets) == 1 and b.packets[0].length == n and os.path.getsize(q) == n
            ta, tb = a.title(), b.title()
            assert set(ta.sections) == set(tb.sections) and len(tb.sections) == 5
            for key, sa in ta.sections.items():
                sb = tb.sections[key]
                assert bytes(sa.vector_bytes()) == bytes(sb.vector_bytes()) and sa.fields() == sb.fields()   # payloads copied as they are
                assert sa.codec == sb.codec and sa.texts() == sb.texts()
                ta_t, tb_t = sa.times(), sb.times()
                assert (ta_t is None and tb_t is None) or np.array_equal(ta_t, tb_t)
            assert tb.versions[V_REF].map_b == -100.0 and tb.description == ta.description
            ids = [bytes(r["id"]).decode() for r in b.packets[0].sections]
            assert ids == ["summary.frame", "synopsis", "title", "frame", "cue.ron"]


def test_fast_one_block():
    with tempfile.TemporaryDirectory() as d:
        p, q = os.path.join(d, "f.ene"), os.path.join(d, "fc.ene")
        w = C.PacketWriter(TITLE, axis_fps=(25, 1))
        v = w.add_version(V_REF, fps=(25, 1), duration=500, flags=C.V_REFERENCE | C.V_TIMELINE)
        xs = {"title": unit(1, 256), "tag": unit(5, 256), "frame": unit(30, 256), "window": unit(12, 256)}
        tf = np.stack([np.arange(30) * 10, np.arange(30) * 10 + 1], 1).astype(np.int32)
        tw = np.stack([np.arange(12) * 40, np.arange(12) * 40 + 50], 1).astype(np.int32)
        w.add_shared([dict(id="frame", version=v, space="eg2-768", modality="image", role="frame", vectors=xs["frame"], times=tf, dim_real=768),
                      dict(id="title", version=v, space="eg2-768", modality="mixed", role="title", vectors=xs["title"], dim_real=768),
                      dict(id="window", version=v, space="eg2-768", modality="video", role="window", vectors=xs["window"], times=tw, dim_real=768),
                      dict(id="tag", version=v, space="eg2-768", modality="text", role="tag", vectors=xs["tag"], dim_real=768)], quant="int8", codec="none")
        w.write(p)
        with C.EneFile(p) as f:
            pk = f.packets[0]
            assert pk.one_block
            ids = [bytes(r["id"]).decode() for r in pk.sections]
            assert ids == ["title", "tag", "window", "frame"]                        # title part first, ascending count
            offs = [int(r["row_offset"]) for r in pk.sections]
            assert offs == [0, 1, 6, 18] and len({int(r["vec_off"]) for r in pk.sections}) == 1
            t = f.title()
            for sid, x in xs.items():
                s = t.section(sid)
                assert s.row_offset is not None and s.dim_real == 768 and s.dim == 256
                y = s.vectors()
                assert np.abs((x @ x.T) - (y @ y.T)).max() < 0.02
                raw = s.vectors(raw=True)
                assert raw.dtype == np.int8 and raw.shape == (len(x), 256)
            assert np.array_equal(t.section("window").times(), tw)
            # the whole packet in one pass: the shared block as one matrix, sections by bisection over row offsets
            r0 = pk.sections[0]
            m = np.frombuffer(pk.block(int(r0["vec_off"]), int(r0["vec_len"]), int(r0["vec_raw"]), int(r0["codec"]), 256), np.int8).reshape(-1, 256)
            assert m.shape == (48, 256) and np.array_equal(m[6:18], t.section("window").vectors(raw=True))
        C.compact(p, q)
        with C.EneFile(p) as f, C.EneFile(q) as g:
            assert g.packets[0].one_block and [bytes(r["id"]).decode() for r in g.packets[0].sections] == ["title", "tag", "window", "frame"]
            for sid in xs:                                                           # compaction keeps the shared layout and the bytes
                assert np.array_equal(g.title().section(sid).vectors(raw=True), f.title().section(sid).vectors(raw=True))
                assert np.array_equal(g.title().section(sid).scales(), f.title().section(sid).scales())


def test_concat_and_errors():
    with tempfile.TemporaryDirectory() as d:
        a, b, ab = (os.path.join(d, n) for n in ("a.ene", "b.ene", "ab.ene"))
        write_title(a)
        other = hashlib.sha256(b"imdb:tt0000001").digest()[:16]
        w = C.PacketWriter(other, axis_fps=(24000, 1001))
        v = w.add_version(V_SUB, fps=(30000, 1001), map_a=0.8, map_b=12.0, duration=100, flags=C.V_TIMELINE)
        w.add_section("frame", v, space="eg2-768", modality="image", role="frame", vectors=unit(3, 128), quant="fp32", codec="none",
                      times=np.array([[0, 1], [10, 11], [20, 21]], np.int32))
        w.write(b)
        with open(ab, "wb") as o:
            o.write(open(a, "rb").read() + open(b, "rb").read())              # cat a.ene b.ene is a valid file (§3.1)
        with C.EneFile(ab) as f:
            assert len(f.packets) == 2 and set(f.titles) == {TITLE, other}
            s = f.titles[other].section("frame")
            assert np.array_equal(s.axis_times(), np.array([[12, 13], [20, 21], [28, 29]]))   # floor(0.8·f0+12), ceil(0.8·f1+12)
            try:
                f.title(); assert False
            except KeyError:
                pass
        # errors: bad magic, truncation, corrupted payload, timeline on a version without TIMELINE, overlong id
        data = bytearray(open(a, "rb").read())
        bad = os.path.join(d, "bad.ene")
        open(bad, "wb").write(b"XXXX" + bytes(data[4:]))
        for broken, msg in ((b"XXXX" + bytes(data[4:]), "magic"), (bytes(data[:-64]), "truncated")):
            open(bad, "wb").write(broken)
            try:
                C.EneFile(bad); assert False
            except ValueError as e:
                assert msg in str(e)
        data[1000] ^= 0xFF
        open(bad, "wb").write(bytes(data))
        try:
            C.EneFile(bad); assert False
        except ValueError as e:
            assert "sha256" in str(e)
        assert C.EneFile(bad, verify=False).packets[0].verify() is False
        w = C.PacketWriter(TITLE, axis_fps=(30, 1))
        v = w.add_version(V_SUB)
        for kw in (dict(times=np.array([[0, 1]], np.int32)),):
            try:
                w.add_section("x", v, space="s", modality="text", role="r", vectors=unit(1, 8), **kw); assert False
            except ValueError:
                pass
        try:
            w.add_section("x" * 33, v, space="s", modality="text", role="r", vectors=unit(1, 8)); assert False
        except ValueError:
            pass
        try:
            w.add_version(V_REF, fps=(25, 1), flags=C.V_REFERENCE | C.V_TIMELINE); assert False    # not the axis rate
        except ValueError:
            pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print(name, "OK")
    print("ALL TESTS PASSED")
