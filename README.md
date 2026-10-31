# .ene — ENhanced Embedded

**A small file next to every video that knows what is seen, heard and said in it.**

`.ene` is a container for the embedding vectors of a video title: its name, synopses, tags and people, and
every segment of its timeline (a frame, a subtitle cue, a scene, a window of sound) as vectors in one shared
space, with the provenance of every row. A two-hour film becomes about two megabytes that can be copied,
shared, extended by other people and loaded into a search space as fast as bytes can be read, without
touching the video again.

This repository holds the format specification ([`SPEC-ene.md`](SPEC-ene.md)), the reference implementation
(a Python library, a CLI and a local web GUI), the benchmark and telemetry harness, and the measured results
(`bench/`). One title is included as data: *Big Buck Bunny* (CC BY 3.0, Blender Foundation), source video fetched
from Blender's download server by a script.

---

## Contents

1. [The problem](#1-the-problem)
2. [What `.ene` does about it](#2-what-ene-does-about-it)
3. [What it becomes useful for](#3-what-it-becomes-useful-for)
4. [What it is not](#4-what-it-is-not)
5. [The format in brief](#5-the-format-in-brief)
6. [Install and run](#6-install-and-run)
7. [Python API](#7-python-api)
8. [Measured](#8-measured)
9. [Status](#9-status)
10. [Repository layout](#10-repository-layout)
11. [Licences and attribution](#11-licences-and-attribution)

---

## 1. The problem

A video library is opaque. Its files can be listed, sorted and played, but not asked a question. *The
documentary where a conductor explains a passage at a rehearsal*, *the moment the flying squirrel glides over
the meadow*, *the concert where the audience sings along*: nothing in a filename, a folder or a metadata record
answers that. Subtitles, where they exist, cover the dialogue in one language and nothing of what is seen or
heard; most personal collections have many titles with no subtitles at all, and many in languages a search
box does not speak.

Multimodal embedding models change what is possible. One model places text, images, video and audio in the
same vector space, so a sentence in any language can be compared directly with a frame, a sequence or ten
seconds of sound. Every moment of a film becomes a handful of vectors, a question becomes one more vector,
and search is a matrix product.

Computing those vectors is the expensive part: a large model, a GPU, minutes per title. Storing them is
cheap: a whole feature film fits in a couple of megabytes. Yet the vectors have no home. They end up in
ad-hoc arrays, pickles and databases tied to one tool, one model run and one machine. They cannot travel with
the film. They cannot be merged when two people processed the same title differently. They cannot be
re-aligned when the film exists in another cut or at another frame rate. They do not say what produced
them, so nobody can tell whether two sets of rows are the same information twice or two different readings
of the same scene. A few months later they are recomputed.

`.ene` is the missing file: the embedding vectors of a title, with their provenance, in a container built
to be copied next to the video, extended by anyone, and loaded into a search space with nothing to parse on
the way.

## 2. What `.ene` does about it

| the problem | what the file does |
|---|---|
| vectors tied to one tool, one run, one machine | **self-describing and model-agnostic**: fixed binary tables carry, for every section, the vector space, quantization, width, modality, role, language, version and an **input fingerprint** of how the vectors were produced; an optional JSON description names the model, the source file and the pipeline |
| no notion of time, or a grid imposed by one pipeline | **two parts**: the *title part* (name, synopses, tags, people, summaries; no clock) and the *timeline part*, segments `[f0, f1)` in **frames**, of any granularity: one frame, one cue, one scene, one chapter, all at once |
| the same film in another cut or frame rate | **versions**: every contribution keeps its own frames and is placed on the title's axis by `f_axis = a·f + b`; a corrected alignment is a new 128-byte record, never rewritten rows |
| several people, several pipelines, several languages | **append-only packets**: `cat a.ene b.ene` is a valid file; a later packet overrides what it carries by revision, tombstones remove, nothing already written is touched; compaction is lossless |
| "is this the same information twice?" | rows with the same key, version, part, modality, role, language, fingerprint and segment are **duplicates** (one kept); anything else is a **variant** and every variant is kept: the cumulated set of vectors is the asset |
| loading cost | **nothing parsed on the hot path**: a 64-byte prefix, fixed records, 64-byte-aligned columns usable in place; under the `fast` policy a file is one matrix product on its own bytes |
| size | int8 rows with a temporal delta and zstd: **6–7× smaller than fp32** at recall@10 ≥ 0.95; a two-hour film is 1.6–2.4 MB, about 1 000× below its source |
| trust | sha256 over every packet, verified on demand |
| dependencies | reading, searching, editing and merging need **numpy and zstandard only**: no model, no torch, no server |

## 3. What it becomes useful for

- **A library that answers questions.** *Which title?* is a matrix product over the stacked title parts of
  thousands of films (about two milliseconds for a thousand); *which moment?* decodes the candidates only. The
  query is text in any language, an image, a sound, or several of them.
- **Help with translation.** Cues, frames and sound of a film sit in the same space as text in any
  language: a Romanian subtitle lines up with an English one by meaning, not by timestamps; speech is
  searchable directly through its audio rows, without transcription.
- **Semantic layers shared without the media.** A subtitle in another language, a cast list, a synopsis,
  a scene segmentation, a second opinion from a finer pipeline: each is a small `.ene` that overlays onto
  someone else's file for the same title, with its own version record and its own fingerprint.
- **Servers and thin clients.** The `fast` policy keeps every row as a raw aligned block: thousands of
  files stay open and a query is one pass over bytes already in the page cache.
- **Tomorrow's model next to today's.** A second vector space (another model, a face model) lives in the
  same file under another key, on the same segments; earlier vectors are not discarded.
- **Reproducibility.** The fingerprint names exactly what moves a vector by more than the storage error
  (prompt, token budget, resize path, source rendering, lossy intermediates) and leaves out what does not
  (device, compute dtype, batch size), so two builds of the same version agree and a different one is
  visibly a variant.

## 4. What it is not

`.ene` is not a library index (an index is built *from* `.ene` files and is derived data), not a
transcription format (text is optional and lives in a column next to its vectors), not a search engine (how
a query is embedded, decomposed, scored or fused is outside the file; the file's one obligation to search is
that every row carries its full provenance), and not a model.

## 5. The format in brief

```
packet = prefix (64 B: magic "ENE\x01", flags, length, title_id, axis frame rate, table sizes)
       + version table   (128 B per version: id, source hash, fps, map a·f+b, duration, rev, flags)
       + section table   (256 B per section: id, version, lang, space, role, modality, quant, codec,
                          dim, count, rev, block offsets, summary ref, input fingerprint)
       + 64-byte-aligned blocks: vectors | scale | time | text, each with its own codec
       + description block (optional JSON, always last)
       + trailer (sha256 + end magic)
file   = one or more packets end to end, grouped by title_id on read
```

| part | rows | time | example roles |
|---|---|---|---|
| **title** | what identifies the production | none | `title`, `synopsis` (per language), `tag`, `person` (cast and characters as text), `summary` (k representatives of a timeline section) |
| **timeline** | what happens on a segment of frames | `[f0, f1)` in the version's frames, any length, overlapping and nesting freely | `cue` (a subtitle line), `frame` (one picture), `chapter`, `window`, `speech`, `scene`, anything a pipeline defines |

One vector space per key (`space / quant / dim`): every vector of a key comes from one model, so a text query
compares directly with a frame, a sound or a sequence, and rows written by different people at different
times stack into one matrix. The reference profile `ene-1` embeds with EmbeddingGemma 2 today (`google/embeddinggemma-2`,
768-d, Matryoshka 768/512/256/128, text + image + video + audio in one context); nothing in the format, the API or the GUI
assumes it: any other model is another space id, another key.

Storage is the writer's choice per section, and every reader understands every combination:

| policy | rule | for |
|---|---|---|
| `compact` (default) | fp16 + zstd below 32 rows; int8 + `ene-xdelta` (XOR against the previous row, then zstd) from 32 rows on; native width stored, Matryoshka at read time | transport: the smallest file that keeps recall@10 ≥ 0.95 against fp32 |
| `fast` | no codec; every section a row range of one shared raw block; fp32 rows below 1 024 rows per packet, int8 above | servers: one matrix product on the file bytes, or one parallel cast and one product |
| `lossless` | fp32 + zstd | archival, benchmarks |

The reference pipeline (profile `ene-1`, SPEC §6.1) writes, per title: the title row, a synopsis per
language, the tags, the people, one summary per timeline section, and a timeline of 150-second chapters,
30-second audio and video windows (15-second stride), one frame per scene cut, 5-second audio rows and the
subtitle cues per language. Nothing of that is part of the format: another pipeline may write one vector
per frame, scene-level rows from a shot detector, or only a cast list.

Full definition, byte offsets, ingestion rules, the measured properties of the model and the open questions:
[`SPEC-ene.md`](SPEC-ene.md).

## 6. Install and run

Reading, searching and editing `.ene` files needs Python ≥ 3.12 with numpy and zstandard. Building files
from video needs, in addition, `ffmpeg` / `ffprobe` on `PATH`, PyTorch with CUDA, sentence-transformers ≥ 6.1
and the `google/embeddinggemma-2` model (1.4 GiB, downloaded once from Hugging Face, Apache 2.0). A GPU with
bf16 and about 3 GB of free VRAM builds a title several times faster than real time; a CPU works, far
slower. Everything is pinned in [`uv.lock`](uv.lock).

```bash
git clone https://github.com/EneAlexandru/.ene.git ene && cd ene
uv sync                                                   # .venv from uv.lock (torch with CUDA 13)
uv run python "samples/Big Buck Bunny/fetch_video.py"     # the Big Buck Bunny video (263 MiB) from Blender's server, sha256-checked
uv run huggingface-cli download google/embeddinggemma-2   # once
uv run pytest tests                                       # container, API, build pipeline, index, GUI, telemetry, quantization
```

Build, inspect, search, edit, serve:

```bash
uv run python -m ene build "samples/Big Buck Bunny/bbb_sunflower_1080p_30fps_normal.mp4" \
    --title "Big Buck Bunny" --year 2008 --imdb tt1254207 \
    --synopsis-file "samples/Big Buck Bunny/synopsis.txt" \
    --tags "animation,rabbit,flying squirrel,meadow" \
    --out "samples/Big Buck Bunny/Big Buck Bunny.ene"
uv run python -m ene inspect "samples/Big Buck Bunny/Big Buck Bunny.ene" --header
uv run python -m ene search  "samples/Big Buck Bunny/Big Buck Bunny.ene" "a flying squirrel glides through the air"
uv run python -m ene edit    "samples/Big Buck Bunny/Big Buck Bunny.ene" --title "Big Buck Bunny (2008)"   # appends one packet
uv run python -m ene overlay "film.ene" "contribution.ene"     # append another file's packets to the same title
uv run python -m ene compact "film.ene"                        # one packet per title, lossless
uv run python -m ene index   library.ene a.ene b.ene c.ene   # the title-part rows of many files in one memory-mapped packet
uv run python -m ene serve                                     # local GUI: http://127.0.0.1:8765/
```

The GUI offers the library of `.ene` files found under its roots, search across titles and inside a title
(text, image, audio or combinations), a timeline with colour-coded scores, a player that seeks to the found
moment, thumbnails, the Matryoshka width chosen per query, and a build form with a live log.

Benchmark and telemetry:

```bash
uv run python -m ene bench "samples/Big Buck Bunny/Big Buck Bunny.master.npz" --queries bench/bbb-queries.txt
uv run python -m ene telemetry bench/titles.json --rebuild      # build + grid + analyses + JSON per title
uv run python -m ene telemetry --aggregate-only                 # regenerate bench/telemetry.md
```

## 7. Python API

`ene.api`, re-exported from the `ene` package, reads, searches, writes, edits and merges `.ene` files with
numpy alone. Queries are vectors the caller already has (from `ene.embed.Embedder`, from the GUI, or copied out
of another file). SPEC §9 has the full description.

```python
import ene

f = ene.open("film.ene")                             # packets grouped by title, overlays resolved
t = f.title()                                        # a file normally holds one title; f.titles for all
t.name, t.description, t.axis_fps, t.versions, t.sections
secs = t.select(part="timeline", modality="image")   # by part / role / modality / lang / version / space / axis frames / span
s = t.section("frame")                               # by id (and by version when the id exists in several)
s.vectors, s.times, s.axis_times, s.texts, s.scale, s.prep, s.prep_object
s.scores(q, dim=256)                                 # float32 [nq, count]; int8 scale folded in; Matryoshka at read time
hits = f.search(q, k=10, modality="image")           # Hit(section, row, score, f0, f1, text), best first
hits = f.search(q, k=10, frames=(a0, a1), axis=True) # rows overlapping an axis span; segments returned on the axis

# edits are staged on the object and written by save() as ONE packet appended to the file
f.set(title={"name": "Big Buck Bunny", "year": 2008})                      # description keys, merged key by key
f.add_version(version_id, fps=(25, 1), duration=n, map_a=1.2, map_b=0.0)   # a new contribution
f.set_alignment(version_id, a, b)                    # a corrected map: the version record again, rev + 1
f.add_section("cue.ron", version_id, modality="text", role="cue", lang="ron",
              vectors=x, times=t, texts=cues, prep={...})
f.remove_section("tag", version_id)                  # a tombstone
f.save()                                             # append one packet; f.save("copy.ene") = compacted copy with the changes
f.compact()                                          # rewrite in place as one packet per title (lossless)

ene.create(path, title_id, axis_fps, versions=[ene.VersionSpec(...)], sections=[ene.SectionSpec(...)],
           description={...}, policy="compact")
ene.overlay("film.ene", "contribution.ene")          # append another file's packets (cat)
ene.compact("film.ene", "film.c.ene")
ene.rewrite("film.ene", "film.fast.ene", policy="fast")

sp = ene.Space(); sp.ingest("a.ene"); sp.ingest("b.ene")   # the local space: one matrix per key, duplicates by fingerprint
sp.search(q, k=10, part="timeline", frames=(a0, a1))
ene.Library(["a.ene", "b.ene"]).titles(q)            # which title? then .search(q): which moment?

ene.index.build(["a.ene", "b.ene"], "library.ene")   # the local space persisted (§5.4): title-part rows of many files, fp32, memory-mapped
ene.index.open("library.ene").titles(q, k=10)        # which title? one matrix product over all of them, about two milliseconds for a thousand
```

## 8. Measured

Every design choice above rests on a measurement. Four titles were used (a 1080p animation, two 720p
documentaries, a 360p documentary; 10.6–64.9 min); only Big Buck Bunny is in the repository, the others
appear as aggregated numbers. Full write-ups: [`bench/conclusions.md`](bench/conclusions.md) (size,
fidelity, geometry, what each modality brings), [`bench/loading.md`](bench/loading.md) (loading time
against a hardware bound and against standard containers), [`bench/frames.md`](bench/frames.md) (the same
frame embedded on every path), [`bench/telemetry.md`](bench/telemetry.md) (the aggregated tables), and
SPEC §8.

| indicator | measured |
|---|---|
| file size | 12.8–19.7 KB per minute of video under `compact`; a two-hour film 1.5–2.4 MB, 960–1 325× below the source; 96–98.5 % of the bytes are vectors |
| fidelity | int8 + `ene-xdelta`: 468–538 B per row (5.7–6.6× over fp32), recall@10 against fp32 0.957–0.970 (0.968–0.986 within ±1 row), cosine error 0.0008–0.0009; int8 + `ene-delta` on the same rows 429–516 B |
| what lost | JSON + zlib as large as raw fp32; RLE twice raw int8; brotli / lzma / lz4 no smaller than zstd and 2–30× slower to decode; Matryoshka truncation at write time saves 40 % of bytes but its error at 256-d (0.013–0.018) equals the gap between stored neighbours (0.010–0.020), so it permutes results and is applied at read time only |
| build | 10.5–11.2× faster than real time at 720p/1080p (5.4–5.7 s of compute per minute of film), 15× at 360p; video decoding and frame preprocessing on the CPU are the bound |
| loading | a 65-minute title (1 879 rows) is ready for its first scores on every row 2.9 ms after `open` under `compact` and 0.74 ms under `fast`, a 10-minute one (356 rows) in 1.0 ms and 0.42 ms, in a warm process; the title part alone in 0.4 ms |
| against a bound | with finish lines stated before measuring: within 3× of a hardware bound built from measured machine constants (`fast` 1.37–1.58×, `compact` 1.88–2.11×, the library index 1.10–1.21×) in three consecutive runs; measured on the layout that preceded 1.0 with the same reader kernels (SPEC §8.2) |
| against other containers | `fast` loads the same vectors faster than `.npy`, `.npz`, `safetensors`, `pickle` and `torch.save` in both their int8 and fp32 forms, and `compact` faster than every compressed one, on every title in three runs and in 40 of 40 fresh processes |
| a library | *which title?* over 1 000 films through the stacked title parts (17 750 rows, 55 MB): one memory map and one matrix product, 1.7 ms warm, 0.26 ms to open |
| the same frame, other pipelines | compute path (batch, bf16/fp32, GPU/CPU): 1 − cos 2·10⁻⁵–7·10⁻⁵, the order of the int8 storage error; preprocessing (token budget, resize path, source rendering, JPEG): 10⁻³–4·10⁻², up to the distance of a frame one second away; hence what goes into the fingerprint and what does not |
| what the modalities say | a mixed audio+video window is the video window (cos 0.93–0.996), so the timeline is written per modality; sound and picture of the same moment do not match each other (chance level), so the best answer is the best moment from each modality; a chapter row is the centroid of mixed windows (recall@1 0.94–1.0) and nearer to the centroid of its own video windows than to any other chapter's in 42–100 % of the chapters |
| text queries | the cosine is high even for unrelated pairs (0.6–0.75): ranks and z-scores matter, not thresholds; speech is found directly through audio rows (z-score 3.3–3.8 on documentaries); the same query in two languages finds the same moment in 17–70 % of cases, so query in both and fuse |

## 9. Status

Every module reads and writes the 1.0 layout. `ene/container.py` and `ene/api.py` implement the specification
(`tests/test_container.py`, `tests/test_api.py`); `ene/build.py` writes profile `ene-1` (`tests/test_build.py` runs the
whole pipeline on a synthetic video with a stub in place of the model); `ene/index.py` persists the local space of
SPEC §5.4 as a packet of the same layout (`tests/test_index.py`); the GUI, the CLI, the benchmark and the telemetry sit on
`ene/api.py`. The included `Big Buck Bunny.ene` and its fp32 master were written by this pipeline, and `bench/bbb.*` and
`bench/telemetry.md` by this telemetry run. The write-ups `bench/conclusions.md`, `bench/loading.md` and `bench/frames.md`
describe the measurement campaign on the four titles; the pipeline measured there wrote the same rows as `ene-1` plus a
mixed audio+video window, dropped since (SPEC §6.1). Open questions are listed in SPEC §10.

## 10. Repository layout

```
README.md                 this document
LICENSE, NOTICE           Apache License 2.0 and the copyright notice
SPEC-ene.md               format specification 1.0: data model, physical layout, storage, ingestion, profiles, measurements
pyproject.toml, uv.lock   the environment (uv)
ene/                      reference implementation
  container.py            packets, tables, blocks, trailer, overlay resolution, compaction, the fingerprint hash
  api.py                  public API: open / create / overlay / compact / rewrite, EneFile, Title, Section, Space, Library
  quant.py                Matryoshka, quantization (fp32/fp16/int8/bin), codecs (zstd, ene-delta, ene-xdelta, ...)
  probe.py, media.py      ffprobe + hashes; audio / frame / scene-cut decoding (ffmpeg)
  embed.py                the model of profile ene-1, the one module that names it: text, image, audio, video, interleaved
  build.py                profile ene-1 (SPEC §6.1): video -> fp32 master -> .ene
  index.py                the persisted local space (SPEC §5.4): the title-part rows of many files stacked, fp32, memory-mapped
  bench.py                dim x quantization x codec grid, recall against fp32
  telemetry.py, analysis.py, report.py   timed stages, analyses, aggregated report
  serve.py, webui/        local web GUI (FastAPI + static page)
  cli.py                  python -m ene probe|build|rewrite|edit|overlay|compact|inspect|search|index|bench|telemetry|serve
tests/                    uv run pytest tests: container, API, build pipeline, index, GUI, telemetry, quantization
bench/                    manifest (titles.json), queries, Big Buck Bunny results, conclusions.md, loading.md, frames.md, telemetry.md
samples/Big Buck Bunny/
  fetch_video.py                         downloads the source video (263 MiB) from Blender's server and checks its sha256
  bbb_sunflower_1080p_30fps_normal.mp4   the source video, not in git (written by fetch_video.py)
  Big Buck Bunny.ene                     written by `ene build` (profile ene-1, policy compact)
  Big Buck Bunny.master.npz              its fp32 rows, segments, texts, fingerprints and build telemetry (for benchmarks)
  synopsis.txt                           the synopsis used for the title part
```

## 11. Licences and attribution

- **Big Buck Bunny** — © 2008 Blender Foundation / www.bigbuckbunny.org, licensed
  [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/); the included file is the "sunflower"
  1080p/30 fps render (2013).
- **EmbeddingGemma 2** — Google, [Apache 2.0](https://huggingface.co/google/embeddinggemma-2); not
  included in the repository, downloaded from Hugging Face.
- **The `.ene` code and specification** — © 2026 Ene Alexandru, released under the
  [Apache License 2.0](LICENSE); the copyright notice is [`NOTICE`](NOTICE).
