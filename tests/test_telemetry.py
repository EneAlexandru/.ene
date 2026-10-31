"""Telemetry + analyses on synthetic data (no model, no media): stages, geometry, alignment between sections, queries,
the byte profile and the cold read of a written file, the policy rows.

Run: uv run pytest tests/test_telemetry.py
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ene import analysis  # noqa: E402
from ene.api import POLICIES  # noqa: E402
from ene.build import Built, SectionData, write_ene  # noqa: E402
from ene.telemetry import Telemetry, host_info  # noqa: E402

rng = np.random.default_rng(3)
FPS = 25
TITLE = hashlib.sha256(b"imdb:tt0000002").digest()[:16]
VID = hashlib.sha256(b"synthetic-2").digest()[:16]
SPACE = "eg2-768"                                    # the example space of SPEC §2.7; every section names it


def _walk(n, d=768, rho=0.8):
    x = rng.standard_normal((n, d)).astype(np.float32)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + (1 - rho) * x[i]
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _built(n2=40) -> Built:
    dur_s = 15 * (n2 - 1) + 30
    a = _walk(n2)
    v = a + 0.5 * rng.standard_normal(a.shape).astype(np.float32) / np.sqrt(a.shape[1])   # "video" related to "audio" on the same window
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    t2 = np.array([[i * 15 * FPS, i * 15 * FPS + 30 * FPS] for i in range(n2)], np.int32)
    chap = np.array([[s * FPS, min(s + 150, dur_s) * FPS] for s in range(0, 15 * (n2 - 1), 150)], np.int32)   # chapters that contain windows
    c = np.stack([v[(t2[:, 0] >= s) & (t2[:, 0] < e)].mean(0) for s, e in chap]); c /= np.linalg.norm(c, axis=1, keepdims=True)
    m = a + v; m /= np.linalg.norm(m, axis=1, keepdims=True)
    desc = {"title": {"name": "Synthetic", "year": 2026, "kind": "movie"},
            "versions": {VID.hex(): {"source": {"filename": "s.mp4", "size_bytes": 10_000_000, "duration_ms": dur_s * 1000}}}, "spaces": {"eg2-768": {"dim": 768}}}
    b = Built(TITLE, (FPS, 1), {"id": VID, "source_hash": bytes(16), "fps": (FPS, 1), "duration": dur_s * FPS}, desc)
    p_audio, p_video = {"profile": "ene-1", "sample_rate": 16000, "max_length": 480000}, {"profile": "ene-1", "tokens": 70, "frames": 30, "fps": 1}
    b.sections += [
        SectionData("audio.30s", "window", "audio", a, t2, space=SPACE, prep=p_audio),
        SectionData("video.30s", "window", "video", v, t2, space=SPACE, prep=p_video),
        SectionData("chapter", "chapter", "mixed", c, chap, space=SPACE, prep={"profile": "ene-1", "tokens": 140}),
        SectionData("title", "title", "mixed", m.mean(0, keepdims=True) / np.linalg.norm(m.mean(0)), texts=["Synthetic"], space=SPACE, prep={"profile": "ene-1", "prompt": "title: {title} | text: "}),
        SectionData("summary.audio.30s", "summary", "audio", a.mean(0, keepdims=True) / np.linalg.norm(a.mean(0)), ref="audio.30s", space=SPACE, prep=dict(p_audio, summary="centroid")),
        SectionData("summary.frame", "summary", "image", v.mean(0, keepdims=True) / np.linalg.norm(v.mean(0)), ref="frame", space=SPACE, prep={"profile": "ene-1", "tokens": 280, "summary": "centroid"}),
        SectionData("tag", "tag", "text", _walk(3), texts=["a", "b", "c"], space=SPACE, prep={"profile": "ene-1", "prompt": "title: {title} | text: "}),
    ]
    return b


def test_telemetry_stages():
    tel = Telemetry("t")
    with tel.stage("a", items=10, media_s=5.0) as rec:
        s = sum(range(100000)); rec["sum"] = s
    with tel.stage("outer"):
        with tel.stage("inner"):
            pass
    d = tel.to_dict()
    names = [s["name"] for s in d["stages"]]
    assert names == ["a", "inner", "outer"], names                      # stages are recorded in the order they close
    a = d["stages"][0]
    assert a["items_per_s"] > 0 and a["realtime_factor"] > 0 and a["wall_s"] >= 0 and "cpu_s" in a and a["sum"] == 4999950000
    assert d["stages"][1].get("parallel") is True and "parallel" not in d["stages"][2]
    assert tel.timings()["a"] == a["wall_s"] and "total" in tel.timings()
    h = host_info()
    assert h["cpu_threads"] >= 1 and h["ram_gib"] > 0 and "python" in h["libs"] and "numpy" in h["libs"]
    print("telemetry: OK")


def test_structure_alignment_queries():
    b = _built()
    assert b.name == "Synthetic" and b.duration_s == 615.0 and b.n_vectors == 40 * 2 + 4 + 1 + 2 + 3
    st = analysis.section_structure(b.get("audio.30s").vectors)
    assert st["count"] == 40 and 0.7 < st["adjacent_cos"]["mean"] < 1.0                 # a random walk: temporal neighbours are close
    assert st["nn_is_temporal_neighbour"] > 0.9 and st["effective_rank"] > 1 and st["prefix_energy"]["512"] > 0.5
    assert all(k in st["mrl_cos_err"] for k in ("512", "256", "128"))
    al = analysis.cross_alignment(b)
    assert al["windows_common"] == 40
    p = al["pairs"]["audio.30s→video.30s"]
    assert p["recall@1"] > 0.8 and p["diag_cos"] > p["offdiag_cos"] and al["cos_audio_video_same_window"] > 0.8
    assert al["chapter_vs_video.30s_centroid"]["recall@1"] == 1.0 and "chapter_vs_audio.30s_centroid" in al
    assert set(al["title_cos"]) == {"summary.audio.30s", "summary.frame", "tag", "chapter_mean"} and "modality_gap_cos" in al
    # queries: audio window 7 + noise -> moment 7 (105 s) in every section; the language pair = the same vector
    q = b.get("audio.30s").vectors[7] + 0.3 * rng.standard_normal(768).astype(np.float32) / np.sqrt(768)
    qv = np.stack([q, q, b.get("video.30s").vectors[15]]); qv /= np.linalg.norm(qv, axis=1, keepdims=True)
    qa = analysis.query_analysis(b, ["ro", "en", "other"], qv, [(0, 1)])
    s = qa["sections"]["audio.30s"]
    assert s["top1_t_s"][0] == 7 * 15 and s["top1_t_s"][2] == 15 * 15 and s["margin_mean"] >= 0 and s["role"] == "window"
    assert qa["language_pairs"]["audio.30s"]["within_30s"] == 1.0 and qa["language_pairs"]["query_cos_pairs"]["min"] > 0.999
    assert qa["agreement"]["audio.30s|video.30s"]["within_60s"] >= 0.6 and len(qa["best"]) == 3 and "title" not in qa["sections"]
    print("structure / alignment / queries: OK")


def test_read_queries_and_file_profile():
    with tempfile.TemporaryDirectory() as d:
        qp = os.path.join(d, "q.txt")
        open(qp, "w", encoding="utf-8").write("# comment\nsingle\nro unu || en one\n\nro doi||en two\n")   # `a || b` lines are language pairs
        texts, pairs = analysis.read_queries(qp)
        assert texts == ["single", "ro unu", "en one", "ro doi", "en two"] and pairs == [(1, 2), (3, 4)]
        b = _built()
        p = os.path.join(d, "s.ene")
        n = write_ene(b, p)
        assert os.path.getsize(p) == n and list(b.telemetry["write"].values())[0] == {"bytes": n, "wall_s": list(b.telemetry["write"].values())[0]["wall_s"], "policy": "compact"}
        prof = analysis.file_profile(p)
        assert prof["bytes"] == n and prof["n_vectors"] == b.n_vectors and prof["bytes_per_min"] > 0 and prof["ratio_vs_fp32"] > 1 and prof["duration_min"] == 10.2
        assert set(prof["by_part"]) == {"title", "timeline"} and prof["sections"]["audio.30s"]["quant"] == "int8" and prof["sections"]["chapter"]["quant"] == "fp16"
        assert prof["padding_bytes"] >= 0 and prof["tables_bytes"] == 64 + 128 + 256 * 7 and prof["description_bytes"] > 0 and prof["packets"][0]["hash_ok"]
        assert prof["tables_bytes"] + prof["description_bytes"] + sum(prof["by_block_kind"].values()) <= n and prof["ratio_vs_source"] > 1
        cr = analysis.cold_read(p, reps=2)
        assert cr["cold"]["n_vectors"] == prof["n_vectors"] and cr["cold"]["total_ms"] > 0 and cr["warm"]["total_ms"] > 0 and len(cr["cold"]["sections"]) == 7
        assert cr["title_part_only_ms"] > 0 and cr["open_noverify_ms"] > 0
        grid = {"audio.30s": [{"dim": 768, "quant": "fp16", "codec": "zstd", "B/vec": 1180.0}, {"dim": 768, "quant": "int8", "codec": "ene-xdelta", "B/vec": 430.0, "ratio": 7.1, "recall@10": 0.97, "recall±1": 0.99, "cos_err": 0.001}]}
        pr = analysis.policy_rows(grid, "compact", {"audio.30s": 40, "title": 1})
        assert pr["audio.30s"]["B/vec"] == 430.0 and pr["audio.30s"]["quant"] == "int8"           # 40 rows: int8 + ene-xdelta under `compact`
        assert analysis.policy_rows(grid, POLICIES["compact"], {"audio.30s": 10})["audio.30s"]["B/vec"] == 1180.0   # 10 rows: fp16 + zstd
    print("queries / file profile / cold read: OK")


if __name__ == "__main__":
    test_telemetry_stages()
    test_structure_alignment_queries()
    test_read_queries_and_file_profile()
    print("ALL TESTS PASSED")
