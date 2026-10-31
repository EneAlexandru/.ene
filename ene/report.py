"""Telemetry over all the titles in a manifest (SPEC §8):

    python -m ene telemetry bench/titles.json [--rebuild] [--only slug] [--bench-dir bench] [--policy compact]

For each title: build (if no master exists or --rebuild) -> .ene -> benchmark grid (width x quant x codec) -> analyses
(geometry, alignment between sections, queries, cold read, byte profile) -> `<bench-dir>/<slug>.telemetry.json`.
At the end: `<bench-dir>/telemetry.json` (all titles) and `<bench-dir>/telemetry.md` (aggregated tables); the conclusions
drawn from them live in `bench/conclusions.md`.

Manifest: a list of objects {"slug", "file", "title", "year"?, "imdb"?, "synopsis"?|"synopsis_file"?, "synopsis_lang"?, "tags"?: [...],
          "people"?: [...], "queries"?: "path.txt", "out"?: "path/stem" (default: the video file without extension),
          "srt"?: {"lang": "path.srt"}, "no_sha256"?: bool}.
"""

from __future__ import annotations

import json
import os
import time
import traceback

import numpy as np

from . import __version__, analysis, bench
from .api import DEFAULT_POLICY_NAME, policy_of
from .build import build_title, load_master, save_master, write_ene
from .telemetry import Telemetry, host_info

STAGE_ORDER = ["probe", "decode_media", "decode_audio", "decode_frames", "scene_detect", "frame", "audio.5s", "audio.30s", "video.30s", "chapter", "title"]


def _log_to(path: str):
    f = open(path, "a", encoding="utf-8")

    def log(m: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {m}"
        print(line, flush=True)
        f.write(line + "\n"); f.flush()
    return log, f


def run_title(entry: dict, model, bench_dir: str, policy_name: str, rebuild: bool, host: dict, log=print) -> dict:
    slug, path = entry["slug"], entry["file"]
    stem = entry.get("out") or os.path.splitext(path)[0]   # `out` = stem of the output names (<out>.ene, <out>.master.npz)
    master, ene_path = stem + ".master.npz", stem + ".ene"
    policy = policy_of(policy_name)
    tel = Telemetry(slug)
    tel.meta["slug"] = slug
    if rebuild or not os.path.isfile(master):
        synopsis = open(entry["synopsis_file"], encoding="utf-8").read().strip() if entry.get("synopsis_file") else entry.get("synopsis", "")
        ids = {"imdb": entry["imdb"]} if entry.get("imdb") else {}
        log(f"[{slug}] build: {os.path.basename(path)}")
        built = build_title(path, model, title=entry["title"], year=entry.get("year"), ids=ids, synopsis=synopsis, synopsis_lang=entry.get("synopsis_lang", "eng"),
                            tags=entry.get("tags") or [], people=entry.get("people") or [], srt=entry.get("srt") or {}, full_hash=not entry.get("no_sha256", False), log=log, tel=tel)
        with tel.stage("write_ene") as rec:
            n = write_ene(built, ene_path, policy)
            rec["bytes"] = n
        with tel.stage("save_master") as rec:
            save_master(built, master)
            rec["bytes"] = os.path.getsize(master)
        built.telemetry = tel.to_dict() | {"write": built.telemetry.get("write", {})}
        save_master(built, master)  # with the complete telemetry (incl. the write stages)
        log(f"[{slug}] wrote {ene_path}: {n:,} B; master {os.path.getsize(master):,} B; build {built.telemetry['total_wall_s']:.1f}s, peak VRAM {built.telemetry['vram_peak_gib']:.2f} GiB")
        build_tel = built.telemetry
    else:
        built = load_master(master)
        build_tel = built.telemetry or {"stages": [], "note": "master without telemetry", "timings": built.timings}
        if not os.path.isfile(ene_path):
            write_ene(built, ene_path, policy)
        log(f"[{slug}] existing master ({len(built.sections)} sections); .ene {os.path.getsize(ene_path):,} B")

    texts, pairs = analysis.read_queries(entry["queries"]) if entry.get("queries") else ([], [])
    qv = np.zeros((0, model.dim), np.float32)
    if texts:
        with tel.stage("queries_embed", items=len(texts)) as rec:
            qv = model.queries(texts)
        log(f"[{slug}] {len(texts)} text queries ({len(pairs)} language pairs) in {rec['wall_s']:.2f}s")

    with tel.stage("bench_grid") as rec:
        q_all, qinfo = bench.make_queries(built, model, texts) if texts else bench.make_queries(built, None, [])
        md_path = os.path.join(bench_dir, f"{slug}.bench.md")
        bench.run(built, q_all, qinfo, md_path, os.path.join(bench_dir, f"{slug}.bench.json"))
        grid = json.load(open(os.path.join(bench_dir, f"{slug}.bench.json"), encoding="utf-8"))["results"]
        rec["n_queries"] = int(q_all.shape[0]); rec["n_sections"] = len(grid)
    log(f"[{slug}] benchmark grid: {len(grid)} sections x {sum(len(v) for v in grid.values())} rows ({rec['wall_s']:.1f}s)")

    with tel.stage("analysis_structure"):
        structure = {s.id: analysis.section_structure(s.vectors) | {"part": "timeline" if s.timeline else "title", "role": s.role, "modality": s.modality}
                     for s in built.sections if s.count >= 2}
    with tel.stage("analysis_alignment"):
        alignment = analysis.cross_alignment(built)
    with tel.stage("analysis_queries"):
        queries = analysis.query_analysis(built, texts, qv, pairs) if texts else {}
    with tel.stage("cold_read"):
        cold = analysis.cold_read(ene_path)
    with tel.stage("file_profile"):
        profile = analysis.file_profile(ene_path)
    log(f"[{slug}] analyses done; cold read {cold['cold']['total_ms']} ms, warm {cold['warm']['total_ms']} ms; {profile['bytes']:,} B = {profile['bytes_per_min']} B/min")

    tel.close()
    out = {"slug": slug, "title": entry["title"], "ene": ene_path, "master": master, "ene_version": __version__, "policy": policy.name,
           "host": host, "title_id": built.title_id.hex(), "axis_fps": list(built.axis_fps), "duration_s": round(built.duration_s, 3),
           "description": built.description, "build": build_tel, "run": tel.to_dict(), "file": profile, "structure": structure, "alignment": alignment,
           "queries": queries, "cold_read": cold, "policy_rows": analysis.policy_rows(grid, policy, {s.id: s.count for s in built.sections}),
           "grid_summary": {sid: [r for r in rows if "error" not in r and r["codec"] in ("zstd", "ene-delta", "ene-xdelta", "none") and r["quant"] in ("fp16", "int8", "bin")] for sid, rows in grid.items()}}
    with open(os.path.join(bench_dir, f"{slug}.telemetry.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


# --- aggregation --------------------------------------------------------------------------------

def _stage(t: dict, name: str) -> dict | None:
    return next((s for s in t.get("build", {}).get("stages", []) if s["name"] == name), None)


def _f(x, nd=1, suffix=""):
    if x is None:
        return ""
    if isinstance(x, (int, np.integer)) and nd == 0:
        return f"{x:,}{suffix}"
    return f"{x:,.{nd}f}{suffix}" if nd else f"{x:,.0f}{suffix}"


def _source(t: dict) -> dict:
    vs = t.get("description", {}).get("versions") or {}
    return next(iter(vs.values()), {}).get("source", {}) if vs else {}


def _dig(d: dict, path: str):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def aggregate_md(results: list[dict]) -> str:
    L = [f"# .ene telemetry - {len(results)} titles", ""]
    h = results[0]["host"]
    L += ["## Software", "", f"- ene {results[0]['ene_version']}, policy {results[0]['policy']}; " + ", ".join(f"{k} {v}" for k, v in h["libs"].items()), ""]

    L += ["## Titles", "", "| title | duration | source | video | rows | build (s) | x real time | peak VRAM | peak RSS | .ene | B/min | source/.ene | fp32/.ene |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in results:
        b, f, m = t["build"], t["file"], t["build"].get("meta", {})
        dur = t["duration_s"]
        wall = b.get("total_wall_s")
        L.append(f"| {t['title']} | {dur / 60:.1f} min | {m.get('source_mib', _source(t).get('size_bytes', 0) / 2**20):,.0f} MiB {m.get('video_codec', '')} | {m.get('video') or ''} | {f['n_vectors']:,} | "
                 f"{_f(wall)} | {_f(dur / wall if wall else None)} | {_f(b.get('vram_peak_gib'), 2)} GiB | {_f(b.get('rss_peak_gib'), 2)} GiB | {f['bytes'] / 1024:,.0f} KB | {_f(f['bytes_per_min'], 0)} | {_f(f['ratio_vs_source'], 0)}x | {_f(f['ratio_vs_fp32'], 1)}x |")
    L.append("")

    stage_names = STAGE_ORDER + sorted({s["name"] for t in results for s in t.get("build", {}).get("stages", []) if s["name"].startswith("cue.")})
    L += ["## Stages (compute seconds per minute of film; in parentheses: real-time factor)", "", "| stage | " + " | ".join(t["slug"] for t in results) + " | mean |", "|---|" + "---|" * (len(results) + 1)]
    for name in stage_names:
        cells, vals = [], []
        for t in results:
            s = _stage(t, name)
            if not s:
                cells.append(""); continue
            dur_min = t["duration_s"] / 60
            v = s["wall_s"] / dur_min
            vals.append(v)
            extra = f" ({s['realtime_factor']}x)" if s.get("realtime_factor") else (f" ({s['ms_per_item']:.0f} ms/item)" if s.get("ms_per_item") else "")
            cells.append(f"{v:.2f}{extra}")
        L.append(f"| {name} | " + " | ".join(cells) + f" | {_f(np.mean(vals), 2) if vals else ''} |")
    cells = []
    for t in results:
        w = t["build"].get("total_wall_s"); dm = t["duration_s"] / 60
        cells.append(f"**{w / dm:.2f}**" if w else "")
    L.append("| **total build** | " + " | ".join(cells) + " | |")
    L.append("")
    L += ["### Throughput per stage (mean over titles; CPU/wall = cores kept busy by the Python process, excluding the child ffmpeg processes, so underestimated for decode*; GPU % sampled at 1 Hz)", "",
          "| stage | items | ms / item | x real time | peak VRAM (GiB) | CPU/wall | GPU util % | GPU W |", "|---|---|---|---|---|---|---|---|"]
    for name in stage_names[1:]:
        ss = [s for t in results if (s := _stage(t, name))]
        if not ss:
            continue
        gu = [s["gpu_util_mean"] for s in ss if s.get("gpu_util_mean") is not None]
        gw = [s["gpu_power_w_mean"] for s in ss if s.get("gpu_power_w_mean") is not None]
        L.append(f"| {name} | {sum(s.get('items', 0) for s in ss):,} | {_f(np.mean([s['ms_per_item'] for s in ss if s.get('ms_per_item')]), 0) if any(s.get('ms_per_item') for s in ss) else ''} | "
                 f"{_f(np.mean([s['realtime_factor'] for s in ss if s.get('realtime_factor')]) if any(s.get('realtime_factor') for s in ss) else None)} | "
                 f"{_f(max(s.get('vram_peak_gib', 0) for s in ss), 2)} | {_f(np.mean([s['cpu_s'] / s['wall_s'] for s in ss if s['wall_s']]), 2)} | "
                 f"{_f(np.mean(gu), 0) if gu else ''} | {_f(np.mean(gw), 0) if gw else ''} |")
    L.append("")

    L += ["## Where the bytes go (.ene, storage policy)", "", "| title | total | tables | description | vectors | scale | time | text | title part | timeline part |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t in results:
        f = t["file"]; k = f["by_block_kind"]; p = f["by_part"]
        L.append(f"| {t['slug']} | {f['bytes']:,} | {f['tables_bytes']:,} | {f['description_bytes']:,} | {k.get('vectors', 0):,} | {k.get('scale', 0):,} | {k.get('time', 0):,} | {k.get('text', 0):,} | "
                 f"{p.get('title', 0):,} | {p.get('timeline', 0):,} |")
    L.append("")

    def order(sid):
        st = next((t["structure"][sid] for t in results if sid in t["structure"]), {})
        return (st.get("part") == "timeline", sid)
    sids = sorted({sid for t in results for sid in t["structure"]}, key=order)
    L += ["## Sections under the storage policy (mean over titles; recall@10 vs fp32 at the native width)", "",
          "| section | storage | rows | B/vec | ratio | recall@10 | recall±1 | cos error | dec MB/s |", "|---|---|---|---|---|---|---|---|---|"]
    for sid in sids:
        rows = [t["policy_rows"][sid] for t in results if sid in t.get("policy_rows", {})]
        n = sum(t["structure"][sid]["count"] for t in results if sid in t["structure"])
        if not rows:
            continue
        mean = lambda k: np.mean([r[k] for r in rows if r.get(k) is not None]) if any(r.get(k) is not None for r in rows) else None  # noqa: E731
        L.append(f"| {sid} | {rows[0]['dim']} {rows[0]['quant']} {rows[0]['codec']} | {n:,} | {_f(mean('B/vec'), 0)} | {_f(mean('ratio'))}x | {_f(mean('recall@10'), 3)} | {_f(mean('recall±1'), 3)} | {_f(mean('cos_err'), 4)} | {_f(mean('dec_MB/s'), 0)} |")
    L.append("")
    L += ["## Section geometry (mean over titles)", "",
          "| section | temporal neighbour cos | NN cos | NN gap (top1-top2) | NN = temporal neighbour | mean pairwise cos | centroid norm | prefix energy 512/256/128 | effective rank | dims for 90 % var | MRL 256 cos error |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for sid in sids:
        ss = [t["structure"][sid] for t in results if sid in t["structure"] and t["structure"][sid]["count"] >= 3]
        if not ss:
            continue
        m = lambda path: np.mean([_dig(s, path) for s in ss if _dig(s, path) is not None])  # noqa: E731
        L.append(f"| {sid} | {_f(m('adjacent_cos.mean'), 3)} | {_f(m('nn_cos_mean'), 3)} | {_f(m('nn_gap.mean'), 4)} | {_f(m('nn_is_temporal_neighbour'), 2)} | {_f(m('pairwise_cos.mean'), 3)} | "
                 f"{_f(m('centroid_norm'), 3)} | {_f(m('prefix_energy.512'), 3)} / {_f(m('prefix_energy.256'), 3)} / {_f(m('prefix_energy.128'), 3)} | {_f(m('effective_rank'), 1)} | {_f(m('dims_for_90pct_var'), 0)} | {_f(m('mrl_cos_err.256.mean'), 4)} |")
    L.append("")

    L += ["## Alignment between sections (the 30 s windows on the same grid; a chapter against the centroid of the windows it contains)", "",
          "| title | windows | audio->video r@1 / r@1±1 | video->audio r@1 / r@1±1 | diag cos | off-diag cos | cos(audio, video) same window | chapter->video windows r@1 | chapter->audio windows r@1 | title·summaries | modality gap |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in results:
        a = t["alignment"]; p = a.get("pairs", {})
        av = p.get("audio.30s→video.30s", {}); va = p.get("video.30s→audio.30s", {})
        tc = a.get("title_cos", {})
        L.append(f"| {t['slug']} | {a.get('windows_common', '')} | {_f(av.get('recall@1'), 2)} / {_f(av.get('recall@1±1'), 2)} | {_f(va.get('recall@1'), 2)} / {_f(va.get('recall@1±1'), 2)} | "
                 f"{_f(av.get('diag_cos'), 3)} | {_f(av.get('offdiag_cos'), 3)} | {_f(a.get('cos_audio_video_same_window'), 3)} | "
                 f"{_f(a.get('chapter_vs_video.30s_centroid', {}).get('recall@1'), 2)} | {_f(a.get('chapter_vs_audio.30s_centroid', {}).get('recall@1'), 2)} | "
                 + ", ".join(f"{k} {_f(v, 2)}" for k, v in tc.items() if k.startswith("summary.")) + f" | {_f(a.get('modality_gap_cos'), 3)} |")
    L.append("")

    L += ["## Text queries per section", "", "| title | section | mean top-1 | margin top1-top2 | top-1 z-score | mean score | score std |", "|---|---|---|---|---|---|---|"]
    for t in results:
        for sid, s in t.get("queries", {}).get("sections", {}).items():
            L.append(f"| {t['slug']} | {sid} | {_f(s['top1_mean'], 3)} | {_f(s['margin_mean'], 4)} | {_f(s['z_mean'], 2)} | {_f(s['score_mean'], 3)} | {_f(s['score_std_mean'], 4)} |")
    L.append("")
    L += ["### Agreement between sections (fraction of queries whose top-1 moment coincides within ±30 s / ±60 s)", "", "| title | pair | ±30 s | ±60 s | median gap |", "|---|---|---|---|---|"]
    for t in results:
        for k, v in t.get("queries", {}).get("agreement", {}).items():
            L.append(f"| {t['slug']} | {k} | {_f(v['within_30s'], 2)} | {_f(v['within_60s'], 2)} | {_f(v['median_gap_s'], 0)} s |")
    L.append("")
    L += ["### Language pairs (the same query in two languages: same moment?)", "", "| title | section | ±30 s | ±60 s | median gap | cos(q_a, q_b) |", "|---|---|---|---|---|---|"]
    for t in results:
        pa = t.get("queries", {}).get("language_pairs", {})
        qc = pa.get("query_cos_pairs", {})
        for sid, v in pa.items():
            if sid == "query_cos_pairs":
                continue
            L.append(f"| {t['slug']} | {sid} | {_f(v['within_30s'], 2)} | {_f(v['within_60s'], 2)} | {_f(v['median_gap_s'], 0)} s | {_f(qc.get('mean'), 3)} (min {_f(qc.get('min'), 3)}) |")
    L.append("")

    L += ["## Reading (.ene -> first scores on every section, through the public API)", "", "| title | file | rows | cold: open+sha256 | cold: decode | cold: GEMM | cold total | warm total | open without verification | title part only (\"which title?\") |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t in results:
        c = t["cold_read"]
        L.append(f"| {t['slug']} | {c['file_bytes']:,} B | {c['cold']['n_vectors']:,} | {c['cold']['open_verify_ms']} ms | {c['cold']['decode_ms']} ms | {c['cold']['gemm_ms']} ms | "
                 f"**{c['cold']['total_ms']} ms** | {c['warm']['total_ms']} ms | {c['open_noverify_ms']} ms | {c.get('title_part_only_ms', '')} ms |")
    L.append("")
    return "\n".join(L)


def run_manifest(manifest_path: str, rebuild: bool = False, only: list[str] | None = None, bench_dir: str = "bench", policy: str = DEFAULT_POLICY_NAME) -> list[dict]:
    entries = json.load(open(manifest_path, encoding="utf-8"))
    if only:
        entries = [e for e in entries if e["slug"] in only]
    os.makedirs(bench_dir, exist_ok=True)
    log, lf = _log_to(os.path.join(bench_dir, "telemetry.log"))
    log(f"=== ene telemetry {__version__}: {len(entries)} titles, policy {policy}, rebuild={rebuild}")
    host = host_info()
    from .embed import Embedder
    model = Embedder()
    host["model"] = model.space() | {"load_s": round(model.load_s, 2), "vram_after_load_gib": round(model.vram(), 3)}
    log(f"[model] {model.name} on {model.device} {model.dtype} loaded in {model.load_s:.1f}s, VRAM {model.vram():.2f} GiB")
    results = []
    for e in entries:
        try:
            results.append(run_title(e, model, bench_dir, policy, rebuild, host, log))
        except Exception as ex:  # noqa: BLE001
            log(f"[{e['slug']}] ERROR: {type(ex).__name__}: {ex}")
            traceback.print_exc()
    if results:
        aggregate_from_disk(bench_dir)
        log(f"=== wrote {bench_dir}/telemetry.md + telemetry.json")
    lf.close()
    return results


def aggregate_from_disk(bench_dir: str = "bench") -> str:
    """Regenerates telemetry.md and telemetry.json from the existing <slug>.telemetry.json files (without running anything)."""
    results = []
    for f in sorted(os.listdir(bench_dir)):
        if f.endswith(".telemetry.json"):
            results.append(json.load(open(os.path.join(bench_dir, f), encoding="utf-8")))
    md = aggregate_md(results)
    with open(os.path.join(bench_dir, "telemetry.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    with open(os.path.join(bench_dir, "telemetry.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)
    return md
