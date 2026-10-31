"""Size / information / speed benchmark (SPEC §8.1).

For every section and every combination (Matryoshka width x quantization x codec):
  bytes, B/row, ratio to fp32 at the native width, encode / decode time, recall@10 and the cosine error against fp32.
Queries: text (the model's query prompt) + rows sampled from the timeline sections (cross-modal), all in the same space.
"""

from __future__ import annotations

import json
import time

import numpy as np

from . import quant as Q
from .build import Built

QUANTS = ("fp32", "fp16", "int8", "bin")
CODECS = ("none", "zstd", "lz4", "brotli", "zlib", "lzma", "ene-delta", "ene-xdelta", "rle")
K = 10


def _topk(s: np.ndarray, k: int) -> np.ndarray:
    k = min(k, s.shape[1])
    return np.argpartition(-s, k - 1, axis=1)[:, :k]


def recall_at_k(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean([len(set(t) & set(p)) / len(t) for t, p in zip(truth, pred)]))


def recall_tolerant(truth: np.ndarray, pred: np.ndarray, tol: int = 1) -> float:
    """A true neighbour counts as found if the prediction contains a row within `tol` positions of it.
    Sections are in temporal order, so ±1 = the adjacent row (an overlapping window, the next frame)."""
    hits, total = 0, 0
    for t, p in zip(truth, pred):
        p = np.asarray(p)
        total += len(t)
        hits += sum(int(np.abs(p - j).min() <= tol) for j in t)
    return hits / total


def _scores_for(quant: str, qz: Q.Quantized, queries: np.ndarray, dim: int) -> np.ndarray:
    """Scores using the stored representation: int8 -> int32 dot product with scales; bin -> popcount; the rest -> float."""
    qd = Q.mrl(queries, dim)
    if quant == "int8":
        m8 = np.frombuffer(qz.payload, dtype=np.int8).reshape(qz.count, dim)
        ms = np.frombuffer(qz.scale, dtype="<f4")
        qq = Q.quantize(qd, "int8")
        q8 = np.frombuffer(qq.payload, dtype=np.int8).reshape(-1, dim)
        qs = np.frombuffer(qq.scale, dtype="<f4")
        return Q.scores_int8(q8, qs, m8, ms)
    if quant == "bin":
        mb = np.frombuffer(qz.payload, dtype=np.uint8).reshape(qz.count, dim // 8)
        qb = np.frombuffer(Q.quantize(qd, "bin").payload, dtype=np.uint8).reshape(-1, dim // 8)
        return Q.scores_bin(qb, mb, dim)
    m = Q.dequantize(qz.payload, quant, qz.count, dim)
    return qd @ m.T


def bench_section(x: np.ndarray, queries: np.ndarray, dims=None, quants=QUANTS, codecs=CODECS, k: int = K) -> list[dict]:
    """x: normalized fp32 [count, dim]; queries: normalized fp32 [nq, dim]; `dims`: the widths to try (default: the native width only)."""
    dims = tuple(dims or (x.shape[1],))
    count = x.shape[0]
    base = queries @ x.T
    truth = _topk(base, k)
    base_bytes = count * x.shape[1] * 4
    rows = [{"dim": x.shape[1], "quant": "json+zlib", "codec": "-", "bytes": Q.json_zip_size(x), "recall@10": 1.0, "recall±1": 1.0, "cos_err": 0.0, "enc_ms": None, "dec_ms": None}]
    for dim in dims:
        xd = Q.mrl(x, dim)
        for quant in quants:
            if quant == "bin" and dim % 8:
                continue
            qz = Q.quantize(xd, quant)
            s = _scores_for(quant, qz, queries, dim)
            pred = _topk(s, k)
            rec = recall_at_k(truth, pred)
            rec_tol = recall_tolerant(truth, pred, 1)
            cos_err = float(np.abs(s - base).mean()) if quant != "bin" else float("nan")
            row_bytes = len(qz.payload) // count
            for codec in codecs:
                try:
                    enc, t_enc = Q.timed(Q.encode, qz.payload, codec, None, row_bytes)
                    dec, t_dec = Q.timed(Q.decode, enc, codec, len(qz.payload), row_bytes)
                    assert dec == qz.payload, "round-trip mismatch"
                except Exception as e:  # noqa: BLE001
                    rows.append({"dim": dim, "quant": quant, "codec": codec, "error": str(e)[:60]}); continue
                scale_bytes = len(Q.encode(qz.scale, "zstd")) if qz.scale else 0
                total = len(enc) + scale_bytes
                rows.append({"dim": dim, "quant": quant, "codec": codec, "bytes": total, "B/vec": total / count, "ratio": base_bytes / total,
                             "enc_ms": t_enc * 1000, "dec_ms": t_dec * 1000, "dec_MB/s": (len(qz.payload) / 2**20) / max(t_dec, 1e-9),
                             "recall@10": rec, "recall±1": rec_tol, "cos_err": cos_err})
    return rows


def make_queries(built: Built, model, text_queries: list[str], n_cross: int = 30, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Queries: text (the query prompt) + rows sampled from the timeline sections (cross-modal)."""
    rng = np.random.default_rng(seed)
    parts, info = [], {}
    if text_queries:
        parts.append(model.queries(text_queries)); info["text"] = len(text_queries)
    for s in built.sections:
        if s.timeline and s.count >= 5:
            idx = rng.choice(s.count, min(n_cross, s.count), replace=False)
            parts.append(s.vectors[idx]); info[s.id] = int(len(idx))
    return np.concatenate(parts), info


def widths(built: Built, s) -> tuple[int, ...]:
    """The widths a section is benchmarked at: its space's Matryoshka ladder (description §3.7) down from the native width, else
    the native width alone."""
    native = int(s.vectors.shape[1])
    ladder = ((built.description.get("spaces") or {}).get(s.space) or {}).get("mrl") or []
    return tuple(int(w) for w in ladder if int(w) <= native) or (native,)


def run(built: Built, queries: np.ndarray, qinfo: dict, out_md: str, out_json: str | None = None, min_count: int = 5) -> str:
    results = {}
    lines = [f"# Benchmark .ene — {built.name}", "",
             f"Queries: {qinfo} (total {queries.shape[0]}). Ground truth: top-{K} on fp32 at the native width. ",
             "int8 scores use an int32 dot product × scale, bin uses popcount; the rest use float32 after dequantization.", ""]
    for s in sorted(built.sections, key=lambda s: (s.timeline, s.id)):
        if s.count < min_count:
            continue
        t0 = time.perf_counter()
        rows = bench_section(s.vectors, queries, dims=widths(built, s))
        results[s.id] = rows
        lines += [f"## {s.id} — {s.count} rows, {'timeline' if s.timeline else 'title'} part, {s.role} / {s.modality} (benchmark {time.perf_counter() - t0:.1f}s)", "",
                  "| dim | quant | codec | bytes | B/vec | ratio | enc ms | dec MB/s | recall@10 | recall±1 | cos err |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if "error" in r:
                lines.append(f"| {r['dim']} | {r['quant']} | {r['codec']} | error: {r['error']} | | | | | | | |"); continue
            lines.append(f"| {r['dim']} | {r['quant']} | {r['codec']} | {r['bytes']:,} | {r.get('B/vec', r['bytes'] / s.count):.0f} | "
                         f"{r.get('ratio', 1.0):.1f}× | {'' if r['enc_ms'] is None else f'{r['enc_ms']:.1f}'} | "
                         f"{'' if r.get('dec_MB/s') is None else f'{r['dec_MB/s']:.0f}'} | {r['recall@10']:.3f} | {r['recall±1']:.3f} | "
                         f"{'' if r['cos_err'] != r['cos_err'] else f'{r['cos_err']:.4f}'} |")
        lines.append("")
    for metric, label in (("recall@10", "recall@10 strict"), ("recall±1", "recall@10 tolerant to ±1 row")):
        lines += [f"## Summary: smallest representation with {label} ≥ 0.95 (codec zstd / ene-delta / ene-xdelta)", "",
                  "| section | dim | quant | codec | B/vec | ratio | recall@10 | recall±1 |", "|---|---|---|---|---|---|---|---|"]
        for sid, rows in results.items():
            ok = [r for r in rows if "error" not in r and r["codec"] in ("zstd", "ene-delta", "ene-xdelta") and r[metric] >= 0.95]
            if ok:
                b = min(ok, key=lambda r: r["bytes"])
                lines.append(f"| {sid} | {b['dim']} | {b['quant']} | {b['codec']} | {b['B/vec']:.0f} | {b['ratio']:.1f}× | {b['recall@10']:.3f} | {b['recall±1']:.3f} |")
        lines.append("")
    md = "\n".join(lines)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"queries": qinfo, "results": results}, f, ensure_ascii=False, indent=1)
    return md
