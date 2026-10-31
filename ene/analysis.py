"""Analyses on the fp32 master vectors and on the written `.ene` file (SPEC §8): the metadata conclusions are drawn from.

  section_structure   geometry of one section: temporal coherence, neighbour separability, energy in the Matryoshka prefixes, effective rank
  cross_alignment     the two 30 s windows (audio / video) on the same grid, a chapter against the windows it contains, the title row
                      against the summaries, the gap between modalities
  query_analysis      what real queries do: top-1 score, margin, z-score, agreement between sections and between the two languages of a pair
  cold_read           open -> first scores on every section, cold and warm, through the public API; the title-part-only path
  file_profile        where the bytes go: per section, per part, per column, B/min, ratio against the source and against fp32
  policy_rows         the grid row a storage policy prescribes for each section
"""

from __future__ import annotations

import os
import time

import numpy as np

from . import api
from . import container as C
from .build import Built
from .quant import mrl

MRL_PREFIXES = (512, 256, 128)


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def _pct(a, ps=(10, 50, 90)):
    return {f"p{p}": _r(np.percentile(a, p)) for p in ps}


# --- geometry ------------------------------------------------------------------------------

def section_structure(x: np.ndarray) -> dict:
    """x: fp32 [n, dim], normalized, in temporal order."""
    n = x.shape[0]
    out: dict = {"count": int(n)}
    if n < 2:
        return out
    S = x @ x.T
    np.fill_diagonal(S, -np.inf)
    nn_idx = S.argmax(axis=1)
    top2 = -np.partition(-S, 1, axis=1)[:, :2]
    np.fill_diagonal(S, 1.0)
    adj = np.array([S[i, i + 1] for i in range(n - 1)])
    sum_all = float(S.sum()) - n
    out |= {
        "adjacent_cos": {"mean": _r(adj.mean())} | _pct(adj),                      # temporal coherence (the neighbouring row)
        "nn_cos_mean": _r(top2[:, 0].mean()),                                        # the nearest stored neighbour
        "nn_gap": {"mean": _r((top2[:, 0] - top2[:, 1]).mean())} | _pct(top2[:, 0] - top2[:, 1]),  # top1 - top2 between rows: how easily they swap
        "nn_is_temporal_neighbour": _r(np.mean(np.abs(nn_idx - np.arange(n)) == 1)),
        "pairwise_cos": {"mean": _r(sum_all / (n * (n - 1))), "std": _r(S[~np.eye(n, dtype=bool)].std())},
        "centroid_norm": _r(np.linalg.norm(x.mean(axis=0))),                       # 1 = all identical; anisotropy of the space
        "prefix_energy": {str(k): _r((x[:, :k] ** 2).sum(axis=1).mean()) for k in MRL_PREFIXES},  # how much of the norm sits in the prefix
    }
    xc = x - x.mean(axis=0)
    sv = np.linalg.svd(xc, compute_uv=False)
    e = sv ** 2
    out["effective_rank"] = _r((e.sum() ** 2) / (e ** 2).sum(), 1)
    cum = np.cumsum(e) / e.sum()
    out["dims_for_90pct_var"] = int(np.searchsorted(cum, 0.9) + 1)
    out["dims_for_99pct_var"] = int(np.searchsorted(cum, 0.99) + 1)
    full = S[np.triu_indices(n, 1)]
    out["mrl_cos_err"] = {}
    for k in MRL_PREFIXES:
        xk = mrl(x, k)
        Sk = (xk @ xk.T)[np.triu_indices(n, 1)]
        out["mrl_cos_err"][str(k)] = {"mean": _r(np.abs(Sk - full).mean()), "max": _r(np.abs(Sk - full).max())}
    return out


# --- alignment between sections ------------------------------------------------------------------

def _by_f0(sec) -> dict[int, int]:
    return {int(t[0]): i for i, t in enumerate(sec.times)} if sec.times is not None else {}


def _retrieval(a: np.ndarray, b: np.ndarray, tol: int = 1) -> dict:
    """Queries b with every row of a: the rank of the pair with the same index."""
    S = a @ b.T
    n = S.shape[0]
    ranks = np.array([(S[i] > S[i, i]).sum() for i in range(n)])
    top1 = S.argmax(axis=1)
    return {"recall@1": _r(np.mean(ranks == 0)), "recall@3": _r(np.mean(ranks < 3)), "recall@1±1": _r(np.mean(np.abs(top1 - np.arange(n)) <= tol)),
            "mean_rank": _r(ranks.mean(), 1), "diag_cos": _r(np.mean(np.diag(S))), "offdiag_cos": _r((S.sum() - np.trace(S)) / (n * (n - 1))) if n > 1 else None}


def _centroids_within(outer, inner) -> np.ndarray:
    cents = []
    for f0, f1 in outer.times:
        m = (inner.times[:, 0] >= f0) & (inner.times[:, 0] < f1)
        v = inner.vectors[m].mean(0) if m.any() else np.zeros(inner.vectors.shape[1], np.float32)
        cents.append(v / (np.linalg.norm(v) or 1))
    return np.stack(cents)


def cross_alignment(built: Built) -> dict:
    secs = {s.id: s for s in built.sections}
    out: dict = {}
    # the two 30 s windows on the same grid (aligned by f0)
    if "audio.30s" in secs and "video.30s" in secs:
        a, v = secs["audio.30s"], secs["video.30s"]
        ia, iv = _by_f0(a), _by_f0(v)
        common = sorted(set(ia) & set(iv))
        if len(common) > 2:
            A, V = np.stack([a.vectors[ia[t]] for t in common]), np.stack([v.vectors[iv[t]] for t in common])
            out["windows_common"] = len(common)
            out["pairs"] = {"audio.30s→video.30s": _retrieval(A, V), "video.30s→audio.30s": _retrieval(V, A)}
            out["cos_audio_video_same_window"] = _r((A * V).sum(1).mean())
            ca, cv = A.mean(0), V.mean(0)
            out["centroid_cos"] = {"audio.30s|video.30s": _r(ca @ cv / (np.linalg.norm(ca) * np.linalg.norm(cv)))}
    # a chapter against the centroid of the windows it contains
    if "chapter" in secs and secs["chapter"].count > 1:
        for sid in ("video.30s", "audio.30s"):
            if sid in secs:
                out[f"chapter_vs_{sid}_centroid"] = _retrieval(secs["chapter"].vectors, _centroids_within(secs["chapter"], secs[sid]), tol=0)
    # the title row against the summaries, the tags, the synopsis and the mean of the chapters
    if "title" in secs:
        t = secs["title"].vectors[0]
        d = {sid: _r((s.vectors @ t).mean()) for sid, s in secs.items() if s.role in ("summary", "tag", "synopsis", "person")}
        if "chapter" in secs:
            d["chapter_mean"] = _r((secs["chapter"].vectors @ t).mean())
        out["title_cos"] = d
    # the gap between modalities: the image summary against the audio summary
    if "summary.frame" in secs and "summary.audio.30s" in secs:
        out["modality_gap_cos"] = _r(secs["summary.frame"].vectors[0] @ secs["summary.audio.30s"].vectors[0])
    return out


# --- queries ----------------------------------------------------------------------------------

def read_queries(path: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Lines of the form `text` or `text A || text B` (the same query in two languages). Returns the texts and the pair indices."""
    texts, pairs = [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "||" in line:
            a, b = (x.strip() for x in line.split("||", 1))
            pairs.append((len(texts), len(texts) + 1)); texts += [a, b]
        else:
            texts.append(line)
    return texts, pairs


def query_analysis(built: Built, texts: list[str], qv: np.ndarray, pairs: list[tuple[int, int]], agree_s: float = 30.0, min_count: int = 5) -> dict:
    secs = [s for s in sorted(built.sections, key=lambda s: (s.timeline, s.id)) if s.count >= min_count and s.timeline]
    per_sec, top_time = {}, {}
    for s in secs:
        S = qv @ s.vectors.T
        order = np.argsort(-S, axis=1)
        t1, t2 = S[np.arange(len(qv)), order[:, 0]], S[np.arange(len(qv)), order[:, 1]]
        mu, sd = S.mean(axis=1), S.std(axis=1)
        z = (t1 - mu) / np.where(sd > 0, sd, 1)
        top_time[s.id] = built.seconds(s.times[order[:, 0], 0])
        per_sec[s.id] = {"count": int(s.count), "role": s.role, "modality": s.modality, "top1_mean": _r(t1.mean()), "top1_min": _r(t1.min()), "top1_max": _r(t1.max()),
                         "margin_mean": _r((t1 - t2).mean()), "z_mean": _r(z.mean(), 2), "score_mean": _r(mu.mean()), "score_std_mean": _r(sd.mean()),
                         "top1_t_s": [int(t) for t in top_time[s.id]], "top1_score": [_r(v, 3) for v in t1]}
    sids = list(per_sec)
    agree = {}
    for i, a in enumerate(sids):
        for b in sids[i + 1:]:
            d = np.abs(top_time[a] - top_time[b])
            agree[f"{a}|{b}"] = {"within_30s": _r(np.mean(d <= agree_s)), "within_60s": _r(np.mean(d <= 2 * agree_s)), "median_gap_s": _r(np.median(d), 1)}
    pair_agree = {}
    if pairs:
        for sid in sids:
            d = np.array([abs(top_time[sid][i] - top_time[sid][j]) for i, j in pairs])
            pair_agree[sid] = {"within_30s": _r(np.mean(d <= agree_s)), "within_60s": _r(np.mean(d <= 2 * agree_s)), "median_gap_s": _r(np.median(d), 1)}
        cs = np.array([float(qv[i] @ qv[j]) for i, j in pairs])
        pair_agree["query_cos_pairs"] = {"mean": _r(cs.mean()), "min": _r(cs.min())}
    best = []
    for qi, q in enumerate(texts):
        b = max(sids, key=lambda sid: per_sec[sid]["top1_score"][qi]) if sids else None
        best.append({"q": q, "section": b, "t_s": per_sec[b]["top1_t_s"][qi] if b else None, "score": per_sec[b]["top1_score"][qi] if b else None})
    return {"n_queries": len(texts), "n_pairs": len(pairs), "sections": per_sec, "agreement": agree, "language_pairs": pair_agree, "best": best}


# --- reading -----------------------------------------------------------------------------------

def cold_read(path: str, dim: int | None = None, reps: int = 3, seed: int = 0) -> dict:
    """Open -> first scores through the public API (`ene.open`, `Section.load`, `Section.scores`): the path a reader takes.
    First pass = cold within the process (tables, bytes, every section decoded lazily); later passes = warm. Also the
    title-part-only path ("which title?") and opening without verification. `dim`: the query width (default: the widest section)."""
    rng = np.random.default_rng(seed)
    q = None
    passes = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f = api.EneFile(path, verify=True)
        t1 = time.perf_counter()
        if q is None:
            q = rng.standard_normal(dim or max(s.dim for s in f)).astype(np.float32)
            q /= np.linalg.norm(q)
        per, nvec = {}, 0
        for s in f:
            a = time.perf_counter()
            s.load()
            b = time.perf_counter()
            s.scores(q, dim)
            c = time.perf_counter()
            per[s.id] = {"decode_ms": _r((b - a) * 1000, 2), "gemm_ms": _r((c - b) * 1000, 2), "count": int(s.count)}
            nvec += s.count
        t2 = time.perf_counter()
        f.close()
        passes.append({"open_verify_ms": _r((t1 - t0) * 1000, 2), "decode_ms": _r(sum(p["decode_ms"] for p in per.values()), 2),
                       "gemm_ms": _r(sum(p["gemm_ms"] for p in per.values()), 2), "total_ms": _r((t2 - t0) * 1000, 2), "sections": per, "n_vectors": nvec})
    t0 = time.perf_counter()
    f = api.EneFile(path, verify=False)
    t1 = time.perf_counter()
    for s in f.select(part="title"):
        s.scores(q, dim)
    t2 = time.perf_counter()
    f.close()
    return {"file_bytes": os.path.getsize(path), "cold": passes[0], "warm": passes[-1], "open_noverify_ms": _r((t1 - t0) * 1000, 2),
            "title_part_only_ms": _r((t2 - t0) * 1000, 2), "warm_total_ms_all": [p["total_ms"] for p in passes]}


# --- bytes --------------------------------------------------------------------------------------

def file_profile(path: str) -> dict:
    """Where the bytes of a file go: tables, description, per section (a shared block attributed by row share), per part, per
    column; B/row and B/min; ratios against the source and against fp32."""
    with C.EneFile(path, verify=True) as f:
        size = os.path.getsize(path)
        t = f.title()
        ref = next((v for v in t.versions.values() if v.reference), None) or next((v for v in t.versions.values() if v.timeline), None)
        dur_min = ref.duration * ref.fps[1] / ref.fps[0] / 60 if ref is not None and ref.timeline else 0
        src = ((t.description.get("versions") or {}).get(ref.id.hex(), {}) if ref is not None else {}).get("source", {})
        tables = sum(64 + 128 * len(pk.versions) + 256 * len(pk.sections) for pk in f.packets)
        desc_bytes = sum(pk.desc_len for pk in f.packets)
        trailers = sum(C.TRAILER_LEN + pk.trailer_pad for pk in f.packets)
        by_sec, by_part, by_kind, blocks = {}, {"title": 0, "timeline": 0}, {"vectors": 0, "scale": 0, "time": 0, "text": 0}, {}
        n_vec, dims = 0, set()
        for (vid, sid), s in t.sections.items():
            rec = s.rec
            d, raw = {}, {}
            for name, f_off, f_len, f_raw in C._BLOCK_FIELDS:
                if name != "vectors" and not s.encoded(name):
                    continue
                enc, rw = int(rec[f_len]), int(rec[f_raw])
                blocks[(s.packet.index, int(rec[f_off]))] = enc
                if s.row_offset is not None and name in ("vectors", "scale") and rw:
                    rb = s.row_bytes if name == "vectors" else 4
                    enc = int(round(enc * s.count * rb / rw))
                    rw = s.count * rb
                d[name], raw[name] = enc, rw
            tot = sum(d.values())
            key = sid if (vid, sid) not in by_sec and sum(1 for v2, s2 in t.sections if s2 == sid) == 1 else f"{vid.hex()[:8]}/{sid}"
            by_sec[key] = {"part": "timeline" if s.timeline else "title", "role": s.role, "modality": s.modality_name, "lang": s.lang, "count": s.count, "dim": s.dim,
                           "quant": s.quant, "codec": s.codec, "bytes": tot, "b_per_vec": _r(tot / max(s.count, 1), 1), "blocks": d,
                           "vectors_ratio": _r(raw["vectors"] / max(d["vectors"], 1), 2), "fp32_bytes": s.count * s.dim_real * 4}
            by_part["timeline" if s.timeline else "title"] += tot
            for k, v in d.items():
                by_kind[k] = by_kind.get(k, 0) + v
            n_vec += s.count
            dims.add(s.dim_real)
        blocks_bytes = sum(blocks.values())
        fp32 = sum(v["fp32_bytes"] for v in by_sec.values())
        return {"bytes": size, "tables_bytes": tables, "description_bytes": desc_bytes, "trailer_bytes": trailers,
                "padding_bytes": size - tables - desc_bytes - trailers - blocks_bytes,
                "n_sections": len(t.sections), "n_vectors": n_vec, "bytes_per_vector": _r(size / max(n_vec, 1), 1), "bytes_per_min": _r(size / dur_min, 0) if dur_min else None,
                "ratio_vs_fp32": _r(fp32 / size, 2), "ratio_vs_source": _r(src.get("size_bytes", 0) / size, 0), "source_bytes": src.get("size_bytes"),
                "duration_min": _r(dur_min, 1), "by_part": by_part, "by_block_kind": by_kind, "sections": by_sec,
                "packets": [{"index": pk.index, "length": pk.length, "hash_ok": pk.hash_ok, "one_block": pk.one_block} for pk in f.packets]}


def policy_rows(grid: dict, policy, counts: dict[str, int]) -> dict:
    """From the benchmark grid results, the (dim, quant, codec) row that the policy prescribes for each section (by its row
    count and the title's), at the native width."""
    pol = api.policy_of(policy)
    total = sum(counts.values())
    out = {}
    for sid, rows in grid.items():
        if sid not in counts:
            continue
        quant, codec = pol.storage(counts[sid], total)
        dim = max((r["dim"] for r in rows if "error" not in r), default=None)
        if dim is None:
            continue
        for r in rows:
            if "error" not in r and r["dim"] == dim and r["quant"] == quant and r["codec"] == codec:
                out[sid] = {"dim": dim, "quant": quant, "codec": codec} | {k: r.get(k) for k in ("B/vec", "ratio", "recall@10", "recall±1", "cos_err", "dec_MB/s", "enc_ms")}
    return out
