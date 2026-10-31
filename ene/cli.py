"""CLI: python -m ene probe|build|rewrite|edit|overlay|compact|inspect|search|index|bench|telemetry|serve ..."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def _policies() -> list[str]:
    from .api import POLICIES
    return list(POLICIES)


def cmd_probe(a):
    from .probe import source_dict
    print(json.dumps(source_dict(a.file, full_hash=not a.no_sha256), ensure_ascii=False, indent=2))


def cmd_build(a):
    from .build import build_title, save_master, write_ene
    from .embed import Embedder
    model = Embedder()
    print(f"[model] {model.name} on {model.device} {model.dtype} loaded in {model.load_s:.1f}s")
    tags = [t.strip() for t in (a.tags or "").split(",") if t.strip()]
    srt = dict(kv.split("=", 1) for kv in (a.srt or []))
    synopsis = open(a.synopsis_file, encoding="utf-8").read().strip() if a.synopsis_file else (a.synopsis or "")
    ids = {"imdb": a.imdb} if a.imdb else {}
    built = build_title(a.file, model, title=a.title, year=a.year, ids=ids, synopsis=synopsis, synopsis_lang=a.synopsis_lang, tags=tags, people=a.person or [],
                        srt=srt, kind=a.kind, full_hash=not a.no_sha256, contributor=a.contributor)
    out = a.out or os.path.splitext(a.file)[0] + ".ene"
    n = write_ene(built, out, a.policy)
    master = os.path.splitext(out)[0] + ".master.npz"
    save_master(built, master)
    print(f"[written] {out}: {n:,} B (policy {a.policy}) | master fp32: {master} | timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in built.timings.items()) + f" | peak VRAM {model.vram():.2f} GiB")


def cmd_rewrite(a):
    """Re-emits a .ene under another storage policy: from a .master.npz (fp32, no re-embedding) or from a .ene (dequantized first)."""
    src = a.source
    if src.endswith(".npz"):
        from .build import load_master, write_ene
        out = a.out or os.path.splitext(src)[0].replace(".master", "") + ".ene"
        n = write_ene(load_master(src), out, a.policy)
    else:
        from .api import rewrite
        out = a.out or os.path.splitext(src)[0] + f".{a.policy}.ene"
        n = rewrite(src, out, a.policy)
    print(f"[written] {out}: {n:,} B (policy {a.policy})")


def cmd_edit(a):
    """Appends one packet with description changes and/or tombstones (SPEC §3.9)."""
    from .api import open as ene_open
    with ene_open(a.file) as f:
        title = {k: getattr(a, k) for k in ("name", "year", "kind") if getattr(a, k) is not None}
        if a.imdb:
            title["ids"] = {"imdb": a.imdb}
        if title:
            f.set(title=title)
        for ref in a.remove_section or []:
            sid, _, vid = ref.partition("@")
            f.remove_section(sid, vid or None)
        if not f.dirty:
            print("[edit] nothing to change"); return
        f.save()
        print(f"[written] {a.file}: {len(f.packets)} packets, {len(f.title())} sections; name: {f.name!r}")


def cmd_overlay(a):
    from .api import overlay
    n = overlay(a.dst, a.src, check_title=not a.force)
    print(f"[written] {a.dst}: {n:,} B")


def cmd_compact(a):
    from .api import compact
    out = compact(a.file, a.out)
    print(f"[written] {out}: {os.path.getsize(out):,} B (one packet per title)")


def cmd_inspect(a):
    from .api import open as ene_open
    with ene_open(a.file, verify=not a.no_verify) as f:
        print(f"{os.path.basename(a.file)}: {os.path.getsize(a.file):,} B, {len(f.packets)} packet(s)" + ("" if a.no_verify else ", sha256 OK"))
        print(f.describe())
        if a.description:
            for t in f.titles.values():
                print(json.dumps(t.description, ensure_ascii=False, indent=2))


def cmd_search(a):
    from .api import open as ene_open
    from .embed import Embedder
    q = Embedder(text_only=True).queries([a.query])[0]
    with ene_open(a.file, verify=False) as f:
        t = f.title()
        num, den = t.axis_fps
        sel = dict(part=a.part, role=a.role, modality=a.modality, lang=a.lang, ids=[a.section] if a.section else None)
        for s in t.select(min_count=1, **sel):
            t0 = time.perf_counter()
            sc = s.scores(q, a.dim)[0]
            dt = time.perf_counter() - t0
            order = np.argsort(-sc)[: a.k]
            times, texts = s.axis_times, s.texts
            print(f"\n[{s.id}] {s.count} rows, {s.part} part, {s.role} / {s.modality_name}, {s.quant}+{s.codec}, dim {a.dim or s.dim} - {dt * 1000:.1f} ms")
            for i in order:
                tm = f"{times[i][0] * den / num:7.1f}s–{times[i][1] * den / num:7.1f}s" if times is not None else ""
                x = f"  {texts[i][:70]}" if texts is not None else ""
                print(f"  {sc[i]:.3f}  {tm}{x}")


def cmd_bench(a):
    from .analysis import read_queries
    from .bench import make_queries, run
    from .build import load_master
    from .embed import Embedder
    built = load_master(a.master)
    queries = read_queries(a.queries)[0] if a.queries else []
    model = Embedder(text_only=True) if queries else None
    q, info = make_queries(built, model, queries, n_cross=a.n_cross)
    out_md = a.out or os.path.splitext(a.master)[0] + ".bench.md"
    md = run(built, q, info, out_md, os.path.splitext(out_md)[0] + ".json")
    print(md.split("## Summary")[-1] if a.quiet else md)
    print(f"\n[written] {out_md}")


def cmd_telemetry(a):
    from .report import aggregate_from_disk, run_manifest
    if a.aggregate_only:
        md = aggregate_from_disk(a.bench_dir)
        print(md if not a.quiet else md.split("\n## Titles")[1].split("\n## Stages")[0])
        return
    results = run_manifest(a.manifest, rebuild=a.rebuild, only=a.only, bench_dir=a.bench_dir, policy=a.policy)
    print(f"\n[done] {len(results)} titles -> {a.bench_dir}/telemetry.md")


def cmd_index(a):
    """The persisted local space (SPEC §5.4): the title-part rows of many files stacked into one memory-mapped packet."""
    from . import index
    n = index.build(a.files, a.out, part=None if a.all_parts else "title", roles=set(a.role) if a.role else None, verify=a.verify)
    with index.open(a.out) as idx:
        print(f"[written] {a.out}: {os.path.getsize(a.out):,} B, {len(idx)} files, {n:,} rows, keys {idx.keys}")


def cmd_serve(a):
    from .serve import serve
    serve(a.root or [os.getcwd()], host=a.host, port=a.port, preload=not a.no_preload, open_browser=not a.no_open)


def main(argv=None):
    from .api import DEFAULT_POLICY_NAME
    p = argparse.ArgumentParser(prog="ene", description="ENhanced Embedded - the embedding vectors of a video title in one file")
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("probe", help="ffprobe + hashes -> the source object"); s.add_argument("file"); s.add_argument("--no-sha256", action="store_true"); s.set_defaults(fn=cmd_probe)
    s = sp.add_parser("build", help="build the .ene of one title (profile ene-1, SPEC §6.1)")
    s.add_argument("file"); s.add_argument("--title", required=True); s.add_argument("--year", type=int); s.add_argument("--imdb"); s.add_argument("--kind", default="movie", choices=["movie", "episode", "clip"])
    s.add_argument("--synopsis"); s.add_argument("--synopsis-file"); s.add_argument("--synopsis-lang", default="eng", help="ISO 639 code of the synopsis (default eng)")
    s.add_argument("--tags", help="comma-separated tags"); s.add_argument("--person", action="append", help='one cast member or character, e.g. "Frank, the flying squirrel" (repeatable)')
    s.add_argument("--srt", action="append", help="lang=path.srt (repeatable)"); s.add_argument("--policy", default=DEFAULT_POLICY_NAME, choices=_policies())
    s.add_argument("--contributor", help="named in the description"); s.add_argument("--out"); s.add_argument("--no-sha256", action="store_true"); s.set_defaults(fn=cmd_build)
    s = sp.add_parser("rewrite", help="re-emit a .ene under another storage policy (from a .master.npz or from a .ene)"); s.add_argument("source", help=".master.npz (fp32) or .ene")
    s.add_argument("--policy", default=DEFAULT_POLICY_NAME, choices=_policies()); s.add_argument("--out"); s.set_defaults(fn=cmd_rewrite)
    s = sp.add_parser("edit", help="append one packet: change the title's description, remove sections"); s.add_argument("file")
    s.add_argument("--name"); s.add_argument("--year", type=int); s.add_argument("--imdb"); s.add_argument("--kind", choices=["movie", "episode", "clip"])
    s.add_argument("--remove-section", action="append", metavar="ID[@VERSION]", help="tombstone a section (repeatable)"); s.set_defaults(fn=cmd_edit)
    s = sp.add_parser("overlay", help="append the packets of SRC to DST (the same title)"); s.add_argument("dst"); s.add_argument("src"); s.add_argument("--force", action="store_true", help="skip the title check"); s.set_defaults(fn=cmd_overlay)
    s = sp.add_parser("compact", help="rewrite as one packet per title, overlays resolved (lossless)"); s.add_argument("file"); s.add_argument("--out"); s.set_defaults(fn=cmd_compact)
    s = sp.add_parser("inspect", help="list the packets, versions and sections"); s.add_argument("file"); s.add_argument("--description", action="store_true", help="print the description JSON"); s.add_argument("--no-verify", action="store_true"); s.set_defaults(fn=cmd_inspect)
    s = sp.add_parser("search", help="search a text in the sections of a file"); s.add_argument("file"); s.add_argument("query"); s.add_argument("--section"); s.add_argument("--part", choices=["title", "timeline"])
    s.add_argument("--role"); s.add_argument("--modality"); s.add_argument("--lang"); s.add_argument("--dim", type=int); s.add_argument("-k", type=int, default=5); s.set_defaults(fn=cmd_search)
    s = sp.add_parser("bench", help="size / information benchmark on a .master.npz"); s.add_argument("master"); s.add_argument("--queries", help="file with text queries, one per line"); s.add_argument("--n-cross", type=int, default=30); s.add_argument("--out"); s.add_argument("--quiet", action="store_true"); s.set_defaults(fn=cmd_bench)
    s = sp.add_parser("telemetry", help="run build + benchmark + analyses on all the titles in the manifest and aggregate the telemetry")
    s.add_argument("manifest", nargs="?", default="bench/titles.json"); s.add_argument("--rebuild", action="store_true", help="rebuild even if a master exists")
    s.add_argument("--only", action="append", help="only the given slug (repeatable)"); s.add_argument("--bench-dir", default="bench"); s.add_argument("--policy", default=DEFAULT_POLICY_NAME, choices=_policies())
    s.add_argument("--aggregate-only", action="store_true", help="only regenerate telemetry.md from the existing <slug>.telemetry.json files"); s.add_argument("--quiet", action="store_true"); s.set_defaults(fn=cmd_telemetry)
    s = sp.add_parser("index", help="stack the title-part rows of many .ene files into one memory-mapped index (SPEC §5.4)"); s.add_argument("out"); s.add_argument("files", nargs="+")
    s.add_argument("--all-parts", action="store_true", help="timeline rows too"); s.add_argument("--role", action="append", help="only these roles (repeatable)"); s.add_argument("--verify", action="store_true"); s.set_defaults(fn=cmd_index)
    s = sp.add_parser("serve", help="local web GUI: library, search, timeline, build"); s.add_argument("--root", action="append", help="directory with .ene files (repeatable; default: the current directory)")
    s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8765); s.add_argument("--no-preload", action="store_true", help="do not load the model at startup")
    s.add_argument("--no-open", action="store_true", help="do not open the browser"); s.set_defaults(fn=cmd_serve)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
