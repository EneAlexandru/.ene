"""The build pipeline (ene.build, profile `ene-1`, SPEC §6.1) without the model: time -> frames, SRT parsing, the master round
trip, the packet it writes, and a smoke run of the whole pipeline on a synthetic video (ffmpeg) with a stub in place of
the model (tests/traceability.md rows 1, 6 and 10).

Run: uv run pytest tests/test_build.py
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction

import numpy as np
import pytest

SPACE = "eg2-768"                                    # the example space of SPEC §2.7; every section names it

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ene  # noqa: E402
from ene import build as B  # noqa: E402
from ene.container import prep_hash  # noqa: E402

rng = np.random.default_rng(13)
FPS = Fraction(30, 1)


def _unit(n, d=768):
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_frames_and_srt():
    assert B.span_frames(0.5, 1.2, FPS) == (15, 36) and B.span_frames(0.0, 0.0, FPS) == (0, 1)             # never shrinks; at least one frame
    assert B.span_frames(10.0, 11.0, FPS, duration=310) == (300, 310) and B.span_frames(10.5, 10.6, FPS, duration=300) == (315, 316)
    assert B.span_frames(1.0, 2.0, Fraction(24000, 1001)) == (23, 48) and B.frame_at(16.3, FPS) == 489
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.srt")
        open(p, "w", encoding="utf-8").write("1\n00:00:01,000 --> 00:00:02,500\n<i>Hello</i>\nthere\n\n2\n00:01:00,200 --> 00:01:01,000\n\n\n3\n01:00:00.000 --> 01:00:00.040\nEnd\n")
        texts, times = B.parse_srt(p)
        assert texts == ["Hello there", "End"] and times.tolist() == [[1.0, 2.5], [3600.0, 3600.04]]
    print("frames + srt: OK")


def _built():
    tid, vid = hashlib.sha256(b"imdb:tt0000003").digest()[:16], hashlib.sha256(b"src").digest()[:16]
    desc = {"title": {"name": "Synthetic", "year": 2026, "kind": "movie", "ids": {"imdb": "tt0000003"}}, "versions": {vid.hex(): {"source": {"filename": "s.mp4"}}},
            "spaces": {"eg2-768": {"dim": 768}}, "sections": {f"{vid.hex()}/cue.ron": {"origin": "external"}}}
    b = B.Built(tid, (30, 1), {"id": vid, "source_hash": vid, "fps": (30, 1), "duration": 9000}, desc, timings={"total": 1.0}, telemetry={"stages": []})
    p_img = {"profile": "ene-1", "tokens": 280, "resize": ["ffmpeg", "bicubic", 448, 252], "source": [1920, 1080, "h264"], "lossy": None}
    fr = _unit(40)
    b.sections += [B.SectionData("frame", "frame", "image", fr, np.stack([np.arange(40) * 200, np.arange(40) * 200 + 1], 1).astype(np.int32), space=SPACE, prep=p_img),
                   B.SectionData("cue.ron", "cue", "text", _unit(5), np.array([[i * 1000, i * 1000 + 60] for i in range(5)], np.int32), [f"replica {i}" for i in range(5)], lang="ron",
                                 space=SPACE, prep={"profile": "ene-1", "prompt": "title: {title} | text: "}),
                   B.SectionData("title", "title", "mixed", _unit(1), texts=["Synthetic"], space=SPACE, prep={"profile": "ene-1", "prompt": "title: {title} | text: "}),
                   B.SectionData("summary.frame", "summary", "image", B._centroid(fr), ref="frame", space=SPACE, prep=dict(p_img, summary="centroid"))]
    return b


def test_master_roundtrip_and_packet():
    with tempfile.TemporaryDirectory() as d:
        b = _built()
        m = os.path.join(d, "s.master.npz")
        B.save_master(b, m)
        b2 = B.load_master(m)
        assert b2.title_id == b.title_id and b2.axis_fps == (30, 1) and b2.version == b.version and b2.description == b.description and b2.timings == b.timings
        assert [s.id for s in b2.sections] == [s.id for s in b.sections]
        for s, s2 in zip(b.sections, b2.sections):
            assert (s.role, s.modality, s.lang, s.ref, s.prep, s.texts, s.space) == (s2.role, s2.modality, s2.lang, s2.ref, s2.prep, s2.texts, s2.space)
            assert np.array_equal(s.vectors, s2.vectors) and (s.times is None and s2.times is None or np.array_equal(s.times, s2.times))
        p = os.path.join(d, "s.ene")
        n = B.write_ene(b2, p)
        assert os.path.getsize(p) == n and b2.telemetry["write"][p]["policy"] == "compact"
        with ene.open(p) as f:
            t = f.title()
            assert t.name == "Synthetic" and t.axis_fps == (30, 1) and t.reference.id == b.version["id"] and t.reference.duration == 9000 and len(t) == 4
            fr, cue, sm = f["frame"], f["cue.ron"], f["summary.frame"]
            assert (fr.quant, fr.codec) == ("int8", "ene-xdelta") and fr.role == "frame" and fr.part == "timeline" and fr.times[3].tolist() == [600, 601]
            assert fr.prep == prep_hash(b.get("frame").prep) and fr.prep_object == b.get("frame").prep
            assert cue.lang == "ron" and cue.texts[2] == "replica 2" and cue.quant == "fp16" and f.description["sections"][f"{b.version['id'].hex()}/cue.ron"] == {"origin": "external", "prep": cue.prep_object}
            assert sm.ref == "frame" and sm.part == "title" and sm.prep_object["summary"] == "centroid" and f["title"].texts == ["Synthetic"]
            assert f.search(fr.vectors[7], k=1)[0].row == 7
        pf = os.path.join(d, "f.ene")
        B.write_ene(b2, pf, "fast")
        with ene.open(pf, verify=False) as f:
            assert f.packets[0].one_block and f.search(b.get("frame").vectors[7], k=1)[0].row == 7
        try:
            B.load_master(os.path.join(d, "nope.npz")); raise AssertionError
        except FileNotFoundError:
            pass
    print("master round trip + packet: OK")


class StubModel:
    """Deterministic unit vectors in place of the model: the pipeline's segmentation, bookkeeping and writing are what is tested."""
    name, model_id, space_id = "stub", "stub", "stub-768"
    device, dtype, dim, load_s, revision = "cpu", "float32", 768, 0.0, "stub"

    def _rows(self, n: int, seed: str) -> np.ndarray:
        r = np.random.default_rng(int.from_bytes(hashlib.sha256(seed.encode()).digest()[:4], "little"))
        x = r.standard_normal((n, self.dim)).astype(np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def images(self, imgs, max_soft_tokens=280, batch_size=16):
        return self._rows(len(imgs), f"img{max_soft_tokens}")

    def audio(self, clips, sr=16000, batch_size=8):
        return self._rows(len(clips), "audio")

    def video(self, clips, max_soft_tokens=70, max_frames=32, batch_size=2):
        assert all(len(c) <= max_frames for c in clips)
        return self._rows(len(clips), "video")

    def interleaved(self, text, images=None, audio=None, video=None, image_tokens=280, video_tokens=70, max_frames=32, sr=16000):
        assert text.count("<|image|>") == len(images or []) and text.count("<|audio|>") == len(audio or []) and ("<|video|>" in text) == bool(video)
        return self._rows(1, text[:40])

    def documents(self, texts, title):
        return self._rows(len(texts), "doc" + title)

    def queries(self, texts):
        return self._rows(len(texts), "q")

    def space(self):
        return {"model": "stub", "dim": 768, "mrl": [768, 512, 256, 128], "normalized": True, "pooling": "mean", "modalities": ["text", "image", "video", "audio"], "libs": {}}

    def compute(self):
        return {"device": "cpu", "dtype": "float32"}

    def vram(self):
        return 0.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg is not on PATH")
def test_pipeline_smoke():
    with tempfile.TemporaryDirectory() as d:
        video = os.path.join(d, "clip.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=25:duration=40", "-f", "lavfi", "-i", "sine=frequency=440:duration=40",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", video], check=True)
        srt = os.path.join(d, "clip.eng.srt")
        open(srt, "w", encoding="utf-8").write("1\n00:00:01,000 --> 00:00:02,500\nHello\n\n2\n00:00:30,000 --> 00:00:31,000\nBye\n")
        logs = []
        b = B.build_title(video, StubModel(), "Clip", year=2026, ids={"imdb": "tt0000004"}, synopsis="A test pattern.", tags=["test", "pattern"], people=["Nobody, played by no one"],
                          srt={"eng": srt}, log=logs.append)
        ids = {s.id for s in b.sections}
        assert {"frame", "audio.5s", "audio.30s", "video.30s", "chapter", "cue.eng", "title", "synopsis", "tag", "person"} <= ids
        assert {f"summary.{s.id}" for s in b.sections if s.timeline} <= ids and b.axis_fps == (25, 1) and b.version["duration"] == 1000 and b.duration_s == 40.0
        assert b.title_id == hashlib.sha256(b"imdb:tt0000004").digest()[:16] and b.version["source_hash"] == bytes.fromhex(b.source["hash"]["sha256"])[:16]
        a5, a30, v30, ch, fr, cue = (b.get(s) for s in ("audio.5s", "audio.30s", "video.30s", "chapter", "frame", "cue.eng"))
        assert a5.count in (8, 9) and a5.times[1].tolist() == [125, 250] and a5.role == "speech" and a5.prep == {"profile": "ene-1", "sample_rate": 16000, "max_length": 80000}   # AAC may pad the end
        assert a30.count == 2 and a30.times.tolist() == [[0, 750], [375, 1000]] and v30.count == 2 and v30.times.tolist() == [[0, 750], [375, 1000]]
        assert v30.prep["frames"] == 30 and v30.prep["tokens"] == 70 and v30.prep["resize"] == ["ffmpeg", "bicubic", 448, 252] and v30.prep["source"] == [320, 180, "h264"]
        assert ch.count == 1 and ch.times.tolist() == [[0, 1000]] and ch.modality == "mixed" and ch.prep["max_length"] == 150 * 16000
        assert fr.count >= 1 and (fr.times[:, 1] == fr.times[:, 0] + 1).all() and (fr.times[:, 0] < 1000).all() and fr.prep["lossy"] is None
        assert cue.texts == ["Hello", "Bye"] and cue.times.tolist() == [[25, 63], [750, 775]] and cue.lang == "eng"
        assert b.get("tag").texts == ["test", "pattern"] and b.get("person").texts == ["Nobody, played by no one"] and b.get("synopsis").lang == "eng"
        assert b.get("summary.frame").ref == "frame" and b.get("summary.frame").prep["summary"] == "centroid" and b.get("title").prep["images"] >= 1
        assert all(abs(np.linalg.norm(s.vectors, axis=1) - 1).max() < 1e-5 for s in b.sections)
        desc = b.description
        v = desc["versions"][b.version["id"].hex()]
        assert desc["title"] == {"name": "Clip", "year": 2026, "kind": "movie", "ids": {"imdb": "tt0000004"}, "languages": {"audio": ["und"], "subs": ["eng"]}}
        assert v["source"]["filename"] == "clip.mp4" and v["generator"]["profile"] == "ene-1" and v["generator"]["compute"]["device"] == "cpu" and v["alignment"] == {"method": "reference"}
        assert desc["sections"][f"{b.version['id'].hex()}/cue.eng"] == {"origin": "external"} and desc["spaces"] == {"stub-768": StubModel().space()} and {s.space for s in b.sections} == {"stub-768"}
        assert {s["name"] for s in b.telemetry["stages"]} >= {"probe", "decode_media", "frame", "audio.5s", "video.30s", "chapter", "title", "cue.eng"} and b.telemetry["meta"]["n_vectors"] == b.n_vectors
        p = os.path.join(d, "clip.ene")
        B.write_ene(b, p)
        with ene.open(p) as f:
            t = f.title()
            assert f.verify() and t.name == "Clip" and t.reference.fps == (25, 1) and len(t) == len(b.sections)
            assert f["cue.eng"].axis_times.tolist() == [[25, 63], [750, 775]] and f["audio.5s"].prep_object == a5.prep
            assert f.description["sections"][f"{b.version['id'].hex()}/video.30s"]["prep"] == v30.prep
        m = os.path.join(d, "clip.master.npz")
        B.save_master(b, m)
        assert B.load_master(m).get("cue.eng").texts == ["Hello", "Bye"]
    print("pipeline smoke: OK")


if __name__ == "__main__":
    test_frames_and_srt()
    test_master_roundtrip_and_packet()
    test_pipeline_smoke()
    print("ALL TESTS PASSED")
