"""Quantization and codecs (ene.quant, SPEC §4): round trip of every quant x codec, the quantization error, int8 and bin scores.

Run: uv run pytest tests/test_quant.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ene import quant as Q  # noqa: E402

rng = np.random.default_rng(1)


def unit(n, d):
    x = rng.standard_normal((n, d)).astype(np.float32)
    for i in range(1, n):                                      # similar neighbours (overlapping windows): a random walk
        x[i] = 0.8 * x[i - 1] + 0.2 * x[i]
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_codecs_roundtrip():
    x = unit(200, 768)
    for quant in Q.QUANTS:
        qz = Q.quantize(x, quant)
        rb = len(qz.payload) // qz.count
        for codec in Q.CODECS:
            enc = Q.encode(qz.payload, codec, row_bytes=rb)
            assert Q.decode(enc, codec, len(qz.payload), rb) == qz.payload, (quant, codec)
    for quant, tol in (("fp16", 1e-3), ("int8", 0.02)):
        q = Q.quantize(x, quant)
        y = Q.dequantize(q.payload, q.quant, q.count, q.dim, q.scale)
        err = np.abs((x @ x.T) - (y @ y.T)).max()
        assert err < tol, (quant, err)
    raw = Q.raw_view(Q.quantize(x, "fp32").payload, "fp32", 200, 768)
    assert raw.dtype == np.dtype("<f4") and np.array_equal(raw, x)
    assert Q.mrl(x, 256).shape == (200, 256) and np.allclose(np.linalg.norm(Q.mrl(x, 256), axis=1), 1, atol=1e-5)
    print("codecs + quantization: OK")


def test_scores():
    x = unit(100, 256)
    q = x[::20] + 0.6 * rng.standard_normal((5, 256)).astype(np.float32) / 16
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    base = q @ x.T
    qz = Q.quantize(x, "int8"); qq = Q.quantize(q, "int8")
    s8 = Q.scores_int8(np.frombuffer(qq.payload, np.int8).reshape(5, 256), np.frombuffer(qq.scale, "<f4"),
                       np.frombuffer(qz.payload, np.int8).reshape(100, 256), np.frombuffer(qz.scale, "<f4"))
    assert np.abs(s8 - base).max() < 0.02
    sb = Q.scores_bin(np.frombuffer(Q.quantize(q, "bin").payload, np.uint8).reshape(5, 32), np.frombuffer(Q.quantize(x, "bin").payload, np.uint8).reshape(100, 32), 256)
    assert np.corrcoef(sb.ravel(), base.ravel())[0, 1] > 0.9
    print("int8 / bin scores: OK")


if __name__ == "__main__":
    test_codecs_roundtrip()
    test_scores()
    print("ALL TESTS PASSED")
