"""MRL, quantization and codecs for the vector blocks (SPEC §5, §7).

All functions work on `count x dim` numpy matrices, C-order, little-endian.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import zstandard as zstd

QUANTS = ("fp32", "fp16", "int8", "bin")
CODECS = ("none", "zstd", "ene-delta", "ene-xdelta", "lz4", "brotli", "zlib", "lzma", "rle")   # the numbering of SPEC §4.2


# --- MRL -------------------------------------------------------------------------

def mrl(x: np.ndarray, dim: int) -> np.ndarray:
    """Matryoshka truncation: the first `dim` components + L2 renormalization (SPEC §6.2)."""
    if dim >= x.shape[1]:
        return np.ascontiguousarray(x, dtype=np.float32)
    t = np.ascontiguousarray(x[:, :dim], dtype=np.float32)
    n = np.linalg.norm(t, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return t / n


# --- quantization ----------------------------------------------------------------

@dataclass
class Quantized:
    quant: str
    payload: bytes          # the `vectors` block
    scale: bytes | None     # the `scale` block (float32[count]) for int8
    count: int
    dim: int


def quantize(x: np.ndarray, quant: str) -> Quantized:
    x = np.ascontiguousarray(x, dtype=np.float32)
    count, dim = x.shape
    if quant == "fp32":
        return Quantized(quant, x.tobytes(), None, count, dim)
    if quant == "fp16":
        return Quantized(quant, x.astype("<f2").tobytes(), None, count, dim)
    if quant == "int8":
        s = np.abs(x).max(axis=1) / 127.0
        s[s == 0] = 1.0
        q = np.rint(x / s[:, None]).clip(-127, 127).astype(np.int8)
        return Quantized(quant, q.tobytes(), s.astype("<f4").tobytes(), count, dim)
    if quant == "bin":
        if dim % 8:
            raise ValueError("bin requires dim to be a multiple of 8")
        bits = np.packbits((x > 0).astype(np.uint8), axis=1)
        return Quantized(quant, bits.tobytes(), None, count, dim)
    raise ValueError(f"unknown quant: {quant}")


def dequantize(payload: bytes | memoryview, quant: str, count: int, dim: int, scale: bytes | None = None) -> np.ndarray:
    """Back to float32 (for `bin`: normalized +-1, as an approximation)."""
    if quant == "fp32":
        return np.frombuffer(payload, dtype="<f4").reshape(count, dim)
    if quant == "fp16":
        return np.frombuffer(payload, dtype="<f2").reshape(count, dim).astype(np.float32)
    if quant == "int8":
        q = np.frombuffer(payload, dtype=np.int8).reshape(count, dim).astype(np.float32)
        s = np.frombuffer(scale, dtype="<f4")
        return q * s[:, None]
    if quant == "bin":
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8).reshape(count, dim // 8), axis=1)
        v = bits.astype(np.float32) * 2 - 1
        return v / np.sqrt(dim)
    raise ValueError(f"unknown quant: {quant}")


def raw_view(payload: bytes | memoryview, quant: str, count: int, dim: int) -> np.ndarray:
    """Zero-copy view of the block, in its native dtype (SPEC §5)."""
    dt = {"fp32": "<f4", "fp16": "<f2", "int8": "i1", "bin": "u1"}[quant]
    w = dim // 8 if quant == "bin" else dim
    return np.frombuffer(payload, dtype=dt).reshape(count, w)


# --- scores ------------------------------------------------------------------------

def scores(q: np.ndarray, m: np.ndarray) -> np.ndarray:
    """cosine = dot product (normalized space): Q @ M.T"""
    return q @ m.T


def scores_int8(q8: np.ndarray, qs: np.ndarray, m8: np.ndarray, ms: np.ndarray) -> np.ndarray:
    """cos(a,b) ~= (q_a . q_b) * s_a * s_b, with int32 accumulation."""
    dots = q8.astype(np.int32) @ m8.astype(np.int32).T
    return dots * qs[:, None] * ms[None, :]


def scores_bin(qb: np.ndarray, mb: np.ndarray, dim: int) -> np.ndarray:
    """sim = dim - 2*popcount(a XOR b), normalized to [-1, 1]."""
    x = np.bitwise_xor(qb[:, None, :], mb[None, :, :])
    pop = np.unpackbits(x, axis=2).sum(axis=2, dtype=np.int32)  # signed int: otherwise dim - 2*pop wraps around (uint)
    return (dim - 2 * pop) / dim


# --- codecs -----------------------------------------------------------------------

def _delta_encode(b: bytes, row_bytes: int) -> bytes:
    a = np.frombuffer(b, dtype=np.uint8).reshape(-1, row_bytes)
    d = a.copy()
    d[1:] = (a[1:].astype(np.int16) - a[:-1].astype(np.int16)).astype(np.uint8)  # wrap mod 256, lossless
    return d.tobytes()


def _delta_decode(b: bytes, row_bytes: int) -> bytes:
    d = np.frombuffer(b, dtype=np.uint8).reshape(-1, row_bytes)
    a = np.cumsum(d, axis=0, dtype=np.uint8)  # native mod 256 accumulation = exact inverse of the delta
    return a.tobytes()


def _xdelta_encode(b: bytes, row_bytes: int) -> bytes:
    """XOR of each row with the previous one. Compresses ~6 % worse than the subtraction delta, but the inverse is a
    cumulative XOR, which has no carries and therefore runs on 64-bit words: ~7x faster to decode (SPEC 7, 15)."""
    a = np.frombuffer(b, dtype=np.uint8).reshape(-1, row_bytes)
    d = a.copy()
    d[1:] = a[1:] ^ a[:-1]
    return d.tobytes()


def _xdelta_decode(b: bytes, row_bytes: int) -> memoryview:
    """Returns a memoryview over the decoded array (no extra copy); every consumer uses np.frombuffer or bytes()."""
    d = np.frombuffer(b, dtype=np.uint8).reshape(-1, row_bytes)
    if row_bytes % 8 == 0:
        out = np.bitwise_xor.accumulate(d.view(np.uint64), axis=0).view(np.uint8)
    else:
        out = np.bitwise_xor.accumulate(d, axis=0)
    return memoryview(out.reshape(-1))


def _rle_encode(b: bytes) -> bytes:
    """Byte-level RLE: (value, run length <= 255). Included only as a baseline (SPEC §7: expected to be useless)."""
    a = np.frombuffer(b, dtype=np.uint8)
    if a.size == 0:
        return b""
    change = np.flatnonzero(np.diff(a)) + 1
    starts = np.concatenate(([0], change))
    lens = np.diff(np.concatenate((starts, [a.size])))
    out_vals, out_lens = [], []
    for v, n in zip(a[starts], lens):
        while n > 255:
            out_vals.append(v); out_lens.append(255); n -= 255
        out_vals.append(v); out_lens.append(n)
    return np.stack([np.array(out_vals, dtype=np.uint8), np.array(out_lens, dtype=np.uint8)], axis=1).tobytes()


def _rle_decode(b: bytes) -> bytes:
    pairs = np.frombuffer(b, dtype=np.uint8).reshape(-1, 2)
    return np.repeat(pairs[:, 0], pairs[:, 1]).tobytes()


def encode(b: bytes, codec: str, level: int | None = None, row_bytes: int | None = None) -> bytes:
    if codec == "none":
        return b
    if codec == "zstd":
        return zstd.ZstdCompressor(level=level or 6).compress(b)
    if codec == "lz4":
        import lz4.frame
        return lz4.frame.compress(b, compression_level=level or 9)
    if codec == "brotli":
        import brotli
        return brotli.compress(b, quality=level if level is not None else 9)
    if codec == "zlib":
        import zlib
        return zlib.compress(b, level or 6)
    if codec == "lzma":
        import lzma
        return lzma.compress(b, preset=level or 6)
    if codec == "ene-delta":
        if not row_bytes:
            raise ValueError("ene-delta requires row_bytes")
        return zstd.ZstdCompressor(level=level or 6).compress(_delta_encode(b, row_bytes))
    if codec == "ene-xdelta":
        if not row_bytes:
            raise ValueError("ene-xdelta requires row_bytes")
        return zstd.ZstdCompressor(level=level or 6).compress(_xdelta_encode(b, row_bytes))
    if codec == "rle":
        return _rle_encode(b)
    raise ValueError(f"unknown codec: {codec}")


def decode(b: bytes | memoryview, codec: str, raw_len: int, row_bytes: int | None = None) -> bytes:
    if codec == "none":
        return bytes(b)
    if codec == "zstd":
        return zstd.ZstdDecompressor().decompress(b, max_output_size=raw_len)
    if codec == "lz4":
        import lz4.frame
        return lz4.frame.decompress(b)
    if codec == "brotli":
        import brotli
        return brotli.decompress(b)
    if codec == "zlib":
        import zlib
        return zlib.decompress(b)
    if codec == "lzma":
        import lzma
        return lzma.decompress(b)
    if codec == "ene-delta":
        return _delta_decode(zstd.ZstdDecompressor().decompress(b, max_output_size=raw_len), row_bytes)
    if codec == "ene-xdelta":
        return _xdelta_decode(zstd.ZstdDecompressor().decompress(b, max_output_size=raw_len), row_bytes)
    if codec == "rle":
        return _rle_decode(b)
    raise ValueError(f"unknown codec: {codec}")


def json_zip_size(x: np.ndarray) -> int:
    import json, zlib
    """The "all text" baseline: JSON with floats, zlib-compressed (SPEC §7, first row)."""
    s = json.dumps(np.round(x, 6).tolist(), separators=(",", ":")).encode()
    return len(zlib.compress(s, 6))


def timed(fn, *a, **kw):
    import time
    t0 = time.perf_counter()
    r = fn(*a, **kw)
    return r, time.perf_counter() - t0
