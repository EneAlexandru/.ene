"""The web GUI (ene/serve.py) on a synthetic 1.0 file, with the model replaced by a stub.

Run: uv run pytest tests/test_serve.py
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile

import numpy as np
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ene  # noqa: E402
from ene.api import SectionSpec, VersionSpec  # noqa: E402
from ene.serve import create_app  # noqa: E402

rng = np.random.default_rng(7)
FPS = 25
DUR_S = 120
TITLE = hashlib.sha256(b"imdb:tt0000001").digest()[:16]
VID = hashlib.sha256(b"synthetic").digest()[:16]
SPACE = "eg2-768"                                    # the example space of SPEC §2.7; every section names it


def _unit(n, d=768):
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def make_ene(path: str, name: str = "Synthetic") -> dict:
    x2, x3 = _unit(7), _unit(5)
    t2 = np.array([[i * 15 * FPS, min(i * 15 * FPS + 30 * FPS, DUR_S * FPS)] for i in range(7)], np.int32)        # 30 s windows, 15 s stride
    t3 = np.array([[i * 20 * FPS, i * 20 * FPS + 3 * FPS] for i in range(5)], np.int32)                           # cues of 3 s every 20 s
    desc = {"title": {"name": name, "year": 2026, "kind": "movie", "ids": {"imdb": "tt0000001"}},
            "versions": {VID.hex(): {"source": {"filename": "does-not-exist.mp4", "size_bytes": 12345, "duration_ms": DUR_S * 1000,
                                                "video": {"width": 640, "height": 360, "fps": "25/1"}, "audio": [], "subtitles": []},
                                     "generator": {"tool": "test", "version": "0", "created": "2026-01-01T00:00:00+00:00"}}},
            "spaces": {SPACE: {"dim": 768, "mrl": [768, 512, 256, 128]}}}
    ene.create(path, TITLE, (FPS, 1), versions=[VersionSpec(VID, fps=(FPS, 1), duration=DUR_S * FPS, reference=True)], description=desc, sections=[
        SectionSpec("title", VID, space=SPACE, modality="mixed", role="title", vectors=_unit(1), texts=[name]),
        SectionSpec("video.30s", VID, space=SPACE, modality="video", role="window", vectors=x2, times=t2),
        SectionSpec("cue.ron", VID, space=SPACE, modality="text", role="cue", lang="ron", vectors=x3, times=t3, texts=[f"line {i}" for i in range(5)]),
        SectionSpec("tag", VID, space=SPACE, modality="text", role="tag", vectors=_unit(3), texts=["a", "b", "c"])])
    return {"video.30s": x2, "cue.ron": x3}


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "Synthetic.ene")
        x = make_ene(path)
        app = create_app([d], preload=False)
        calls = []

        def stub_embed(text, image, audio):   # instead of the model: the query = the 4th window of video.30s + noise
            if not (text or image or audio):
                raise HTTPException(400, "empty query")
            calls.append((text, bool(image), bool(audio)))
            q = x["video.30s"][3] + 0.005 * rng.standard_normal(768).astype(np.float32)   # ||noise|| ~ 0.14 -> cos ~ 0.99
            return q / np.linalg.norm(q), {"kind": "stub", "embed_ms": 0.0, "audio_s": None}

        app.state.model.embed_query = stub_embed
        c = TestClient(app)

        r = c.get("/"); assert r.status_code == 200 and "<title>.ene</title>" in r.text
        r = c.get("/static/app.js"); assert r.status_code == 200 and "queryBox" in r.text
        st = c.get("/api/status").json(); assert st["titles"] == 1 and st["model"]["state"] == "not loaded" and "compact" in st["policies"], st
        lib = c.get("/api/library").json()
        assert len(lib) == 1 and lib[0]["name"] == "Synthetic" and lib[0]["n_vectors"] == 16 and not lib[0]["has_video"] and lib[0]["duration_ms"] == DUR_S * 1000, lib
        assert lib[0]["packets"] == 1 and lib[0]["tags"] == ["a", "b", "c"] and lib[0]["title_id"] == TITLE.hex() and lib[0]["video"] == "640x360"
        tid = lib[0]["id"]

        t = c.get(f"/api/title/{tid}").json()
        ids = [s["id"] for s in t["sections"]]
        assert ids == ["tag", "title", "cue.ron", "video.30s"], ids                       # the title part first, then by id
        s3 = next(s for s in t["sections"] if s["id"] == "cue.ron")
        assert s3["texts"] == [f"line {i}" for i in range(5)] and s3["times"][1] == [20000, 23000] and s3["frames"][1] == [500, 575] and s3["has_text"]
        assert s3["part"] == "timeline" and s3["role"] == "cue" and s3["lang"] == "ron" and s3["quant"] == "fp16" and s3["bytes"] > 0
        assert t["packets"][0]["hash_ok"] and t["video"] is None and t["description"]["title"]["name"] == "Synthetic"
        assert next(s for s in t["sections"] if s["id"] == "tag")["times"] is None
        assert c.get("/api/title/nope").status_code == 404

        r = c.get(f"/api/title/{tid}/section/video.30s/vectors?dim=256")
        assert r.status_code == 200 and r.headers["x-shape"] == "7x256" and len(r.content) == 7 * 256 * 4
        m = np.frombuffer(r.content, "<f4").reshape(7, 256)
        assert np.allclose(np.linalg.norm(m, axis=1), 1, atol=1e-3)                    # Matryoshka renormalized at read time
        assert c.get(f"/api/title/{tid}/section/nope/vectors").status_code == 404

        # search in the title: window 3 of video.30s must come first; its segment in ms on the axis
        r = c.post("/api/search", data={"q": "something", "scope": "title", "title": tid, "dim": 768, "k": 3})
        assert r.status_code == 200, r.text
        res = r.json()
        s2 = next(s for s in res["sections"] if s["id"] == "video.30s")
        assert len(s2["scores"]) == 7 and s2["top"][0]["i"] == 3 and s2["top"][0]["t0"] == 45000 and s2["top"][0]["t1"] == 75000 and s2["top"][0]["score"] > 0.9, s2["top"]
        assert [h["score"] for h in s2["top"]] == sorted((h["score"] for h in s2["top"]), reverse=True) and len(s2["top"]) == 3
        assert s2["part"] == "timeline" and s2["role"] == "window" and s2["modality"] == "video"
        s3 = next(s for s in res["sections"] if s["id"] == "cue.ron"); assert s3["top"][0]["text"].startswith("line ")
        assert res["query"]["dim"] == 768 and res["query"]["k"] == 3 and calls[-1] == ("something", False, False)

        res = c.post("/api/search", data={"q": "something", "scope": "title", "title": tid, "dim": 128, "k": 2}).json()
        assert next(s for s in res["sections"] if s["id"] == "video.30s")["top"][0]["i"] == 3

        res = c.post("/api/search", data={"q": "something", "scope": "library", "dim": 512}).json()
        top = res["titles"][0]
        assert top["id"] == tid and top["best"]["section"] == "video.30s" and top["parts"]["timeline"] > 0.9 and top["roles"]["window"] > 0.9, res["titles"]
        assert "title" in top["parts"] and top["best"]["t0"] == 45000

        res = c.post("/api/search", data={"scope": "title", "title": tid}, files={"image": ("x.png", b"\x89PNG fake", "image/png")})
        assert res.status_code == 200 and calls[-1] == (None, True, False)

        assert c.post("/api/search", data={"q": "x", "scope": "title", "title": tid, "dim": 99}).status_code == 400
        assert c.post("/api/search", data={"q": "   ", "scope": "title", "title": tid}).status_code == 400
        assert c.get(f"/api/title/{tid}/frame?t=1").status_code == 404
        assert c.get(f"/api/title/{tid}/video").status_code == 404
        assert c.get(f"/api/title/{tid}/bench").status_code == 404
        assert c.post("/api/build", data={"file": os.path.join(d, "missing.mkv"), "title": "x"}).status_code == 400
        assert c.get("/api/jobs").json() == [] and c.get("/api/media").json() == []

        make_ene(os.path.join(d, "Other.ene"), "Other")
        assert c.post("/api/rescan").json()["titles"] == 2 and len(c.get("/api/library").json()) == 2
        for t in app.state.lib.titles.values():
            t.close()
    print("test_serve: OK")


def test_serve():
    main()


if __name__ == "__main__":
    main()
