# .ene — ENhanced Embedded (format specification 1.0)

A transport container for the embedding vectors of a video title, in two parts: **the title**, everything
that identifies the production without a clock (its name, synopses, tags, people, summaries), and **the
timeline**, segments of frames of any granularity (one frame, a cue, a scene) carrying the embeddings of
what is written, heard and seen on them. Vectors come from any number of **versions** (files, pipelines,
contributors), live in **one vector space** per key, and are placed on the title's frame axis by an affine
map. The file exists to move those vectors, as fast as bytes can be read, from the source title into a
**local multidimensional search space** that holds rows from any number of titles.

Nothing time-related is fixed by the format: no levels, no grids, no strides. Granularity is whatever the
writer chose, declared row by row, in frames.

This document specifies what the file contains and how it is laid out. How a query is embedded,
decomposed, scored or fused is the search engine's business and is outside it; the file's only obligation
towards search is that every row carries its full provenance.

Sections marked **(normative)** define the format; sections marked **(informative)** record measurements,
profiles and implementation notes that do not constrain a reader or a writer.

## 0. Foundation (normative)

1. **A title has two parts.** The **title part** describes the production without time. The **timeline
   part** describes its content frame by frame.
2. **The title part** holds the rows that identify the production: the embedding of its name, of its
   synopsis (in any number of variants and languages), of its tags, of its people (cast and characters,
   as text) and summaries of the timeline part. A title-part row has no time.
3. **The timeline part** holds segments `[f0, f1)` in **frames** of a version, of any length: one frame,
   one subtitle cue, one scene, one chapter. Segments overlap and nest freely. On a segment live
   embeddings, one per modality: text (what is said or written there), audio (what is heard), image (one
   frame), video (the sequence, the action).
4. **One vector space per key.** Every vector of a key comes from one model, so a text query compares
   directly with a frame, a sound or a sequence, and vectors produced by different people at different
   times are mutually comparable. Several spaces (another model, a face model) may coexist in one file,
   on the same segments, under different keys.
5. **Versions coexist.** The same title exists as several files (another cut, another frame rate, a
   leader at the start) and is processed by several pipelines and contributors (one per frame, one per
   scene, one with a Romanian subtitle, one with a hand-written synopsis). Every such contribution is a
   **version**. Timeline rows keep the frames of their own version and are placed on the title's frame
   axis by `f_axis = a · f + b`. Rows from different versions that land on the same span are variants of
   the same segment.
6. **Every variant is kept.** The file never chooses between contributors, languages, granularities or
   modalities; the cumulated set of vectors is the asset, and choosing is the search engine's job.
7. **The file is a transport.** Its destination is the local search space: one matrix per
   (space, quantization, width), holding rows from any number of titles, plus the columns that say where
   each row comes from and, for timeline rows, when it happens. Every layout decision serves the shortest
   path from bytes on disk to rows in that matrix: fixed binary records, aligned columns, nothing to parse
   on the hot path, and a human-readable description that is optional and never needed to search or ingest.
8. **Growth is append-only.** New versions, corrected alignments, extra modalities and replaced rows
   arrive as further packets concatenated onto the file. Nothing already written is rewritten or recomputed.

## 1. Terms

| term | meaning |
|---|---|
| **title** | the production (film, episode, clip); identified by `title_id` (16 bytes); owns one **frame axis** with a declared frame rate |
| **title part** | the rows of a title that have no time: name, synopsis, tag, person, summary |
| **timeline part** | the rows of a title that have a segment in frames |
| **version** | one contribution to a title: a source file processed by one pipeline run, or a contribution without a file; identified by `version_id` (16 bytes); a version with a timeline owns a frame rate and an affine map to the axis |
| **frame** | the unit of time of the timeline part: one picture of the version's video at its declared frame rate |
| **segment** | a span `[f0, f1)` of a version's frames |
| **row** | one vector with its provenance (version, modality, role, language), its text when it has one, and, in the timeline part, its segment |
| **section** | the rows of one version, in one part, that share space, modality, role, language, quantization, codec and width, stored as columns; the unit of storage and of replacement |
| **packet** | the physical unit of the file: prefix, binary tables, blocks, trailer; a file is one or more packets end to end |
| **block** | a 64-byte-aligned run of bytes inside a packet: a column of a section, or the description |
| **space** | a vector space = one model at one revision and one native width; identified by a short id |
| **key** | `space / quant / dim`: rows with the same key are byte-compatible and stack into one matrix |
| **local space** | the destination: per key, one matrix of rows from any number of titles plus their provenance columns (§5) |

## 2. Data model (normative)

### 2.1 Title

| field | where | meaning |
|---|---|---|
| `title_id` | prefix | 16 bytes. Recommended derivation: the first 16 bytes of `sha256("imdb:" + id)` when an IMDb id is known, otherwise the first 16 bytes of the sha256 of the first version's source file. A reader never needs to know how it was derived; packets with the same `title_id` describe the same title |
| `axis_fps_num`, `axis_fps_den` | prefix | the frame rate of the title's axis, `num / den` frames per second (`24000 / 1001`, `25 / 1`, `30 / 1`). Normally the reference version's rate |
| names, year, kind, ids, languages | description block (§3.7) | human metadata; optional |

### 2.2 Version

| field | type | meaning |
|---|---|---|
| `version_id` | 16 bytes | unique per contribution; recommended: the first 16 bytes of `sha256(source_sha256 ‖ generator string)`, or of `sha256(contributor ‖ created)` for a contribution without a file |
| `source_hash` | 16 bytes | the first 16 bytes of the sha256 of the source file; all zeros when there is none |
| `fps_num`, `fps_den` | u32, u32 | the version's frame rate, `num / den` frames per second; `0 / 0` for a version without a timeline |
| `map_a`, `map_b` | f64, f64 | the affine map to the axis, **in frames**: `f_axis = map_a · f + map_b`. `map_a` = axis frames per version frame; `map_b` = axis frame of the version's frame 0 |
| `duration` | i64 | the version's length in its own frames (0 without a timeline) |
| `rev` | u32 | revision of this record; a later packet with the same `version_id` and a higher `rev` replaces it (this is how `map_a` / `map_b` are corrected without rewriting rows) |
| `flags` | u32 | bit 0 `REFERENCE` (this version defines the axis: `fps = axis fps`, `map_a = 1`, `map_b = 0`); bit 1 `TIMELINE` (the version has a frame rate and may own timeline sections); other bits reserved, 0 |
| source file details, generator, contributor, alignment method | description block | optional |

**The map.** `map_a` is the number of axis frames one version frame covers:

| case | `map_a` |
|---|---|
| same content in real time at another frame rate (a 30 fps re-render of a 24 fps film, axis at 24 fps) | `axis_fps / version_fps` = 0.8 |
| frame-for-frame transfer shown faster or slower (a 25 fps PAL transfer of a 24 fps film, axis at 24 fps) | 1 |
| 3:2 pulldown (29.97 fps video of a 23.976 fps film, axis at 23.976) | `30000/1001 ÷ 24000/1001` = 1.25 |

`map_b` is the axis frame at which the version's frame 0 falls: negative when the version has a leader
(the film's frame 0 is at version frame 100 → `map_b = −100` for `map_a = 1`), positive when the version
starts inside the film. A version with the identity map on the axis rate is the natural reference; it need
not exist.

### 2.3 Segment (timeline part)

A segment is `[f0, f1)` in the version's frames, `0 ≤ f0 ≤ f1`, both `i32`. Conventions:

| what | encoding |
|---|---|
| one frame `n` | `[n, n + 1)` |
| a span of time given in seconds or milliseconds (a subtitle cue, an audio window) | `f0 = floor(t0 · fps)`, `f1 = ceil(t1 · fps)`: a span never shrinks |
| a version with a variable frame rate | the writer declares the nominal rate at which it decoded (decoders can force a constant rate); frame indices are exact, the map to the axis uses the nominal rate |
| a version without pictures (an audio-only file) that still has a timeline | declares a nominal rate (the axis rate is the natural choice) |

Finer than a frame (audio samples) cannot be expressed; no row that serves search is that fine.

### 2.4 Row

| field | type | meaning |
|---|---|---|
| part | section flag | `TITLE` or `TIMELINE` |
| segment | `i32[2]` | timeline part only, §2.3 |
| modality | u8 | `0 text`, `1 image`, `2 video`, `3 audio`, `4 mixed` (interleaved text + media in one context) |
| role | ≤ 16 bytes UTF-8 | what the embedding is *of*. Reserved in the title part: `title` (the name), `synopsis`, `tag`, `person` (one cast member or character, as text: "Michael Knight, played by David Hasselhoff"), `summary` (a representative vector of a timeline section, §2.6). Reserved in the timeline part: `cue` (one subtitle line), `frame` (one picture). Any other value is free and belongs to the pipeline that wrote it (`scene`, `window`, `chapter`, `speech`, …) |
| lang | ≤ 4 bytes | ISO 639-2/3 code for text rows and for speech, empty otherwise |
| version | index | the contribution the row belongs to |
| vector | `dim` values | the embedding, L2-normalized before quantization |
| text | UTF-8, optional | the text the row embeds (the name, the synopsis, the cue, the tag label, the person) |

Part, modality, role, lang and version are properties of the **section**, so they cost nothing per row.

### 2.5 Section

A section is a set of rows of one version, in one part, with the same space, modality, role, lang,
quantization, codec, stored width and input fingerprint `prep` (§2.7), written as separate columns (§3.6).
Rules:

- `id`: ≤ 32 bytes UTF-8, unique within the version. `(version_id, id)` is the replacement key (§3.9).
  Ids are free; a pipeline names its sections as it likes (`cue.ron`, `frame`, `window.30s`, `synopsis`).
- timeline sections carry a `time` column and are **sorted by `f0`, then `f1`**, ascending: time-range
  filtering is a bisection, and the delta codecs (§4) profit from temporal coherence. Title sections carry
  no `time` column.
- `rev`: a later packet with the same key and a higher `rev` replaces the section; a record flagged
  `DELETED` hides it (tombstone).
- `dim ≤ dim_real`: `dim_real` is the space's native width; `dim < dim_real` means the stored vectors are
  the Matryoshka prefix, truncated and renormalized at write time (§4.3).
- sections of a packet are written **title part first, then timeline part**, each in ascending order of
  row count, ties by id: a reader that only wants the title part stops after a few hundred bytes.

### 2.6 Summaries

A `summary` section (title part) stands for one timeline section of the same version, named in its `ref`
field: `k ≥ 1` representative vectors of that section's rows, so that the title part carries evidence from
every modality without the timeline. `k = 1` is the centroid (L2-normalized mean); `k > 1` are
representatives chosen by the writer (cluster centres). `k` is the writer's choice; the measured guidance is
in §8.3.

### 2.7 Space and key

A space is one model at one revision with one native width. Its id is a short string (≤ 16 bytes),
deterministic from model and width, so that two files produced separately agree without negotiation:
`eg2-768` = `google/embeddinggemma-2` at 768 d. A fine-tuned or different model gets another id. The
description block (§3.7) spells out model name, revision, pooling, normalization, Matryoshka widths and
library versions.

`key = space / quant / dim`. Rows with the same key are byte-compatible: they concatenate into one matrix
without conversion. Two widths of the same space stack after truncating the wider one (`M[:, :k]` plus
renormalization, cheap). Rows of different spaces never share a matrix; they share segments and provenance.

The format is model-agnostic: it stores vectors and names where they came from. What it knows about a
model is the space id, `quant` and `dim` in the binary tables, and two entries of the description (§3.7):
the space's identity (`spaces`: model, revision, width, pooling, normalization) and, per section, the
**input fingerprint** `prep`, an object the format does not interpret, hashed into the section record
(§3.5). `prep` names everything on the input side that the space's profile has measured to move a vector
by more than the storage error of the policy; what does not (for EmbeddingGemma 2: device, compute dtype,
batch size, §8.4) stays out of it and is informative. The content of `prep` is the profile's (§6.1); its
meaning is the format's: rows of the same key, version, part, modality, role, lang and `prep`, on the same
segment (timeline part) or with the same text (title part), are **duplicates**, the same information
twice, and the local space keeps one (§5.2); rows that differ in `prep` are **variants** and are all kept.

### 2.8 Alignment

Alignment is applied when timeline rows leave the file (at ingestion into the local space or on read),
never when they are stored: `f_axis = map_a · f + map_b` is a multiply-add over the time column, and a
corrected map arrives as a new version record (§2.2), not as rewritten rows.

Rounding to axis frames: `f0_axis = floor(map_a · f0 + map_b)`, `f1_axis = ceil(map_a · f1 + map_b)`, so a
span never shrinks. Two rows from different versions describe the same segment when their axis spans
coincide (or overlap, for tolerant comparisons).

## 3. Physical layout (normative)

All integers are little-endian. All offsets are relative to the start of the packet they belong to.
Everything is 64-byte aligned: the tables area, every block, the trailer, the packet.

### 3.1 File

A file is one or more packets placed end to end. `cat a.ene b.ene` is a valid file. Packets need not
belong to the same title; a reader groups them by `title_id` (§3.9). A packet is skipped with one read of
its prefix (`packet_len`).

### 3.2 Packet

```
+----------------------------------------------------------------------------+
| prefix                                                        64 B         |
+----------------------------------------------------------------------------+
| version table      n_versions × 128 B                                      |
| section table      n_sections × 256 B   (title part first, then timeline)  |
| zero padding to a multiple of 64 B                                         |
+----------------------------------------------------------------------------+
| block | pad | block | pad | ... | block | pad     each block 64 B-aligned  |
| description block (optional, always the last block) | pad                  |
+----------------------------------------------------------------------------+
| trailer: sha256 (32 B) + end magic "\x01ENE" (4 B) | zero padding to 64 B   |
+----------------------------------------------------------------------------+
```

### 3.3 Prefix (64 bytes)

| offset | size | field | meaning |
|---|---|---|---|
| 0 | 4 | `magic` | `"ENE\x01"` |
| 4 | 4 | `flags` u32 | bit 0 `ONE_BLOCK`: every section is a row range of one shared raw `vectors` block and the section table is ordered by `row_offset` (§3.6); other bits reserved, 0 |
| 8 | 8 | `packet_len` u64 | the whole packet, prefix to final padding, a multiple of 64 |
| 16 | 16 | `title_id` | §2.1 |
| 32 | 4 + 4 | `axis_fps_num`, `axis_fps_den` u32 | the frame rate of the title's axis |
| 40 | 4 + 4 | `n_versions`, `n_sections` u32 | table sizes |
| 48 | 4 | `desc_len` u32 | encoded length of the description block, 0 if absent |
| 52 | 4 | `desc_raw` u32 | its decoded length |
| 56 | 1 | `desc_codec` u8 | codec of the description block (§4.2 numbering; `zstd` recommended) |
| 57 | 1 | `trailer_pad` u8 | zero bytes after the end magic (0..63) |
| 58 | 6 | reserved | 0 |

Derived positions: the version table starts at 64; the section table at `64 + 128·n_versions`; the block
area at `64 + 128·n_versions + 256·n_sections`, rounded up to a multiple of 64; the trailer at
`packet_len − trailer_pad − 36`; the description block, when present, at `trailer_offset − desc_len −
pad64(desc_len)`.

### 3.4 Version record (128 bytes)

| offset | size | field |
|---|---|---|
| 0 | 16 | `version_id` |
| 16 | 16 | `source_hash` |
| 32 | 4 + 4 | `fps_num`, `fps_den` u32 (`0 / 0` without a timeline) |
| 40 | 8 | `map_a` f64 |
| 48 | 8 | `map_b` f64 |
| 56 | 8 | `duration` i64 (version frames) |
| 64 | 4 | `rev` u32 |
| 68 | 4 | `flags` u32 (bit 0 `REFERENCE`, bit 1 `TIMELINE`) |
| 72 | 56 | reserved, 0 |

### 3.5 Section record (256 bytes)

| offset | size | field |
|---|---|---|
| 0 | 32 | `id` (UTF-8, zero-padded) |
| 32 | 4 | `version` u32: index into this packet's version table |
| 36 | 4 | `lang` (ISO 639, zero-padded, empty if none) |
| 40 | 16 | `space` (zero-padded) |
| 56 | 16 | `role` (zero-padded) |
| 72 | 1 | `modality` u8 (§2.4) |
| 73 | 1 | `quant` u8: `0 fp32`, `1 fp16`, `2 int8`, `3 bin` |
| 74 | 1 | `codec` u8 of the `vectors` block (§4.2) |
| 75 | 1 | `aux_codec` u8 of the `scale` and `time` blocks |
| 76 | 1 | `text_codec` u8 of the `text` block |
| 77 | 1 | `flags` u8: `1 HAS_SCALE`, `2 HAS_TIME`, `4 HAS_TEXT`, `8 ROW_RANGE`, `16 DELETED`, `32 TIMELINE` (clear = title part) |
| 78 | 2 | reserved, 0 |
| 80 | 4 × 6 | `dim`, `dim_real`, `count`, `row_offset`, `rev` u32, reserved u32 |
| 104 | 8 × 3 | `vec_off`, `vec_len`, `vec_raw` u64: offset, encoded length, decoded length of the `vectors` block |
| 128 | 8 × 3 | `scale_off`, `scale_len`, `scale_raw` |
| 152 | 8 × 3 | `time_off`, `time_len`, `time_raw` |
| 176 | 8 × 3 | `text_off`, `text_len`, `text_raw` |
| 200 | 32 | `ref`: for a `summary` section, the `id` of the timeline section it summarizes (same version); zeros otherwise |
| 232 | 8 | `prep` u64: the first 8 bytes of the SHA-256 of the section's canonical `prep` object (§2.7, §3.7); 0 = not given (such a section is never a duplicate of anything) |
| 240 | 16 | reserved, 0 |

A `DELETED` record carries only `id`, `version` and `rev`. A timeline section has `TIMELINE` and `HAS_TIME`
set and belongs to a version with `TIMELINE`. Everything a reader needs to locate, decode, filter, align and
ingest every row of the packet is in these two tables; nothing else has to be read.

### 3.6 Blocks (the columns of a section)

| column | content | required |
|---|---|---|
| `vectors` | `count × dim` matrix, C-order, in the quantization's element type (§4.1), no per-row header | yes |
| `scale` | `f32[count]`, the per-row scale of `int8` rows | `int8` only |
| `time` | `i32[count, 2]` = `f0, f1` in version frames, sorted (§2.5) | timeline sections; absent in the title part |
| `text` | the rows' texts, UTF-8, joined by `\n` (newlines inside a text replaced by spaces) | no |

Every block is encoded with its own codec and padded with zeros to a multiple of 64. A `none` block is
therefore directly usable as a numpy array on the file bytes or on a memory map (`np.frombuffer(...).
reshape(count, dim)`), aligned for SIMD.

**Shared block** (`ROW_RANGE`): several sections of the same quantization and width may share one
`vectors` block (and one `scale` block): each is the rows `[row_offset, row_offset + count)` of it. When
every section of the packet is a row range of one raw block and the table is ordered by `row_offset`, the
packet sets `ONE_BLOCK` and a reader scores the whole packet with one pass, mapping hits back to sections
by bisection over the row offsets. Written by the `fast` policy (§4.4). A section without `ROW_RANGE`
owns its block.

### 3.7 Description block (optional)

A JSON object, UTF-8, encoded with `desc_codec`, always the last block of the packet. It is the
human-readable side of the packet and is **never required** to read, search, align or ingest rows. A
reader parses it on demand (when a title's name, a source file's details or a section's parameters are
asked for). Recommended content:

```json
{
  "title": {"name": "Big Buck Bunny", "year": 2008, "kind": "movie", "ids": {"imdb": "tt1254207"},
            "languages": {"audio": ["und"], "subs": []}},
  "versions": {
    "<version_id hex>": {
      "source": {"filename": "...", "size_bytes": 276134947, "sha256": "...", "container": "mov",
                 "video": {"codec": "h264", "width": 1920, "height": 1080, "fps": "30/1", "frames": 19036},
                 "audio": [{"index": 1, "lang": "und", "codec": "mp3", "channels": 2, "sample_rate": 48000}],
                 "subtitles": []},
      "generator": {"tool": "ene", "version": "...", "profile": "ene-1", "created": "2026-10-28T10:00:00Z",
                    "contributor": "...", "compute": {"device": "cuda", "dtype": "bfloat16", "batch_size": 16}},
      "alignment": {"method": "reference | manual | frame-match | audio-xcorr", "note": "..."}
    }
  },
  "spaces": {"eg2-768": {"model": "google/embeddinggemma-2", "revision": "f9972ac2…", "dim": 768, "mrl": [768, 512, 256, 128],
                         "normalized": true, "pooling": "mean", "modalities": ["text", "image", "video", "audio"],
                         "libs": {"sentence-transformers": "6.2.0", "transformers": "5.20.2", "torch": "2.14.1"}}},
  "sections": {"<version_id hex>/<id>": {"prep": {"profile": "ene-1", "prompt": "title: {title} | text: ", "tokens": 70, "frames": 30, "fps": 1,
                                                  "resize": ["ffmpeg", "bicubic", 448, 252], "source": [1920, 1080, "h264"], "lossy": null},
                                         "origin": "embedded | external | asr | mt | manual",
                                         "tags": [{"label": "rabbit", "source": "manual", "weight": 1.0}]}}
}
```

The texts the rows embed (the name, the synopsis, the cues, the tag labels, the people) live in the
`text` columns of their sections, not here; the description only adds what has no row.

`prep` is the section's input fingerprint (§2.7): an object written exactly as the space's profile defines
it (§6.1 for `ene-1`), canonicalized (RFC 8785: sorted keys, no whitespace) before it is hashed into the
section record's `prep` field (§3.5). The format does not interpret its keys; a reader without a profile
for the space compares fingerprints and nothing else. Compute details (`generator.compute`: device, dtype,
batch size) are informative and never part of `prep`: they are measured not to change the information
(§8.4).

### 3.8 Trailer and integrity

`sha256` of the packet from byte 0 up to the trailer (prefix, tables, padding, every block including the
description), followed by the end magic `"\x01ENE"` and `trailer_pad` zero bytes. Verification costs about
1 ms per MB and is optional at open time (`verify=False` for bulk ingestion, `verify()` when integrity
matters).

### 3.9 Reading several packets (overlay)

Packets are walked in file order and grouped by `title_id`. Within a title:

1. a version record with a new `version_id` is added; the same `version_id` with a higher `rev` replaces
   the record (map, duration, flags); an equal `rev` keeps the later packet's record;
2. a section with a new `(version_id, id)` is added; a higher `rev` replaces; an equal `rev` keeps the
   later packet's; a `DELETED` record removes the section (tombstone);
3. description blocks merge key by key, the later packet winning on every key it carries.

A packet may therefore be partial: a new version with its sections; a single version record that corrects
an alignment; a tombstone and a replacement for one section; a synopsis in another language; a
description change. This is how `ene edit`, `ene overlay` and any third-party contribution add to a file
without rewriting it.

### 3.10 Compaction

`ene compact` rewrites a file as one packet per title with the overlays resolved, sections reordered
(§2.5) and stored payloads copied as they are (no requantization): lossless.

## 4. Storage of a section (normative list, informative policy)

### 4.1 Quantization

| `quant` | element | notes |
|---|---|---|
| `fp32` | `f32` | reference |
| `fp16` | `f16` | lossless in practice for unit vectors (values in [−1, 1]) |
| `int8` | `i8` + `scale f32[count]` | `q = round(v / s)`, `s = max|v| / 127` per row; `cos(a, b) ≈ (q_a · q_b) · s_a · s_b`; for a unit query the scale cancels after renormalization, so a reader can score on the raw cast |
| `bin` | `u8[dim / 8]` | `sign(v)`; `sim = dim − 2·popcount(a ⊕ b)`; a pre-filter sieve only |

### 4.2 Codecs

Numbering used in the records: `0 none`, `1 zstd`, `2 ene-delta`, `3 ene-xdelta`, `4 lz4`, `5 brotli`,
`6 zlib`, `7 lzma`, `8 rle`.

- `none`: the block is the raw column, aligned, usable in place.
- `zstd`: general entropy coding; decompression at hundreds of MB/s.
- `ene-delta`: each row minus the previous row, byte-wise modulo 256, then zstd. Profits from temporal
  coherence between neighbouring rows (that is why timeline rows are sorted by frame).
- `ene-xdelta`: each row XOR the previous row, then zstd. 5–6 % larger than `ene-delta`; its inverse is a
  carry-free cumulative XOR computed on 64-bit words, ~7× faster to decode. **Default for large sections.**
- `lz4`, `brotli`, `zlib`, `lzma`, `rle`: readable, measured, not recommended (§8.1).

### 4.3 Matryoshka width

The space's native width is stored (`dim = dim_real`, 768 for `eg2-768`). Narrower widths are produced
at read time by truncation plus renormalization: `M[:, :k]`. Storing a truncated prefix (`dim < dim_real`)
is allowed and declared in the record, but measured to permute neighbours at fine granularity (§8.1), so
the default policy does not do it: correctness forbids saving bytes here.

### 4.4 Storage policies (informative, measured; §8)

A policy is the writer's choice per section; every reader understands every combination of §4.1–§4.2.

| policy | rule | for |
|---|---|---|
| `compact` (default) | sections with fewer than 32 rows: `fp16 + zstd`; from 32 rows on: `int8 + ene-xdelta`; `dim = dim_real` | transport: the smallest file that keeps recall@10 ≥ 0.95 against fp32 (`codec = ene-delta` is 5 % smaller and 1.7× slower to decode; available, not default) |
| `fast` | codec `none` for every section, all sections row ranges of one shared block (`ONE_BLOCK`); `fp32` rows below 1 024 rows in the packet, `int8` rows from 1 024 on; `fast-fp32` / `fast-int8` force one form | servers that keep thousands of files open: nothing to decode, one GEMV on the file bytes (fp32) or one parallel cast and one GEMV (int8) |
| `lossless` | `fp32 + zstd` everywhere | archival, benchmarks |

The 32-row threshold sits between the measured sections: those of 1–26 rows (title part, chapters)
cost nothing at `fp16`, and every section of 42 rows or more was measured at `int8 + delta` with recall@10
≥ 0.95. The threshold itself has not been swept.

## 5. From the file to the local space

### 5.1 The local space (normative as a contract, informative as a layout)

Per key, one matrix `N × dim` in the key's element type (plus `scale f32[N]` for `int8`), and the
provenance columns that every row keeps:

| column | type | meaning |
|---|---|---|
| `title` | u32 | index into the space's title table (`title_id`, axis rate) |
| `version` | u32 | index into the space's version table (`version_id`, map, duration, flags) |
| `part` | u8 | title or timeline |
| `f0`, `f1` | i32 | the segment **on the title's axis** (§2.8); undefined for title-part rows |
| `modality` | u8 | §2.4 |
| `role` | u16 | index into the space's role dictionary |
| `lang` | u16 | index into the space's language dictionary |
| `section` | u32 | index into the space's section table (`(title, version, id)`, `ref`, text column reference) |

Rows of different titles, versions, parts, roles and granularities sit in the same matrix; the columns
are what separate them when a query asks. Nothing in the local space knows about levels, windows or grids.

### 5.2 Ingestion (normative behaviour)

For each packet of a file, in order:

1. read the prefix; read the two tables (one read of `128·n_versions + 256·n_sections` bytes);
2. resolve versions and sections against what the local space already holds for this `title_id` (§3.9):
   a replaced or deleted section is dropped from the space, a replaced version record re-maps the axis
   frames of its timeline rows (one multiply-add over the affected rows);
3. for each section, per key: decode the `vectors` block (or take it in place when `codec = none`),
   append it to the key's matrix; append `scale` when present; for a timeline section map the `time`
   column with the version's `map_a`, `map_b` (§2.8) and append; append the constant columns (`title`,
   `version`, `part`, `modality`, `role`, `lang`, `section`) as filled arrays. **Duplicates** are not
   appended: when the space already holds a section of the same key, version, part, modality, role, lang
   and the same non-zero `prep` (§2.7; one lookup on the binary record), the rows that agree with it on
   the segment (timeline part) or on the text (title part; rows without text are never duplicates) are
   the same information twice, and the space keeps the one from the higher `rev`, the later file on a tie.
   Rows that differ in any of these, or come from a section with `prep = 0`, are variants and are appended;
4. the description block is not touched.

Cost: with `codec = none` ingestion is a memory copy of the columns; with the delta codecs it is bound by
zstd and the XOR pass (§8.2: a 65-minute title, 2 134 rows, is ready for its first scores 2.6 ms after
open under `compact` and 0.45 ms under `fast`).

### 5.3 What the stored columns make possible (informative)

- one `q @ M.T` per key over rows of any number of titles, parts and versions;
- masks over any provenance column: a title, a version, a part, a modality, a role, a language, an
  axis frame range (`f1 > a and f0 < b`), a minimum span length; a frame-range filter on one section is
  a bisection, its rows being sorted;
- aggregation per title (rows grouped by title: `np.maximum.reduceat`), per version, per segment on the
  axis (variants from several versions overlap there);
- a cheap title part (names, synopses, tags, people, summaries of every modality) and a full timeline part
  behind it, both with the same provenance;
- several spaces side by side on the same segments;
- Matryoshka truncation per query by truncating both sides.

What a search engine does with these (decomposing a composite query, ranking, fusing modalities,
thresholds) is outside this document.

### 5.4 Persisting the local space (informative)

A persisted local space is a packet of the same layout with one extra column block per section,
`origin i64[count, 2]` = `(title index, source row)`, and with the title and version tables of all the
titles it holds in its description. The reference implementation keeps `ene.index` (one `fp32`, codec
`none`, memory-mapped matrix of title-part rows) as the first instance of it.

## 6. Build profiles (informative)

A profile is a pipeline's choice of segments, modalities and roles. It is documented so that its rows are
interpretable and its measurements reproducible; **none of it is part of the format**, and two profiles of
the same title coexist as two versions.

### 6.1 Profile `ene-1` (the reference pipeline, `ene build`)

One version, `REFERENCE` + `TIMELINE`, frames at the source's frame rate; the axis is that rate.

**Title part**

| section id | role | modality | rows | what goes into the model (8 192-token budget) |
|---|---|---|---|---|
| `title` | `title` | mixed | 1 | `title: … \| text: name + synopsis + tags` interleaved with ~6 key frames (280 tok each) and ~4 audio segments of 30 s (750 tok each) |
| `synopsis` | `synopsis` | text | 1 per language given | `title: {title} \| text: {synopsis}`; the synopsis in the text column |
| `tag` | `tag` | text | 10–40 | `title: {title} \| text: {label}` per tag; labels in the text column |
| `person` | `person` | text | one per cast member or character given | `title: {title} \| text: {character}, played by {actor}`; text in the text column |
| `summary.<id>` | `summary` | as the section | `k` per timeline section (`k = 1` until §8.3 says otherwise) | the centroid (or cluster centres) of that section's rows; `ref = <id>` |

**Timeline part** (rows per 45 minutes)

| section id | role | modality | segment | rows | what goes into the model |
|---|---|---|---|---|---|
| `chapter` | `chapter` | mixed | 150 s, fixed grid or scene cuts | 18 | video 1 fps, ≤ 29 frames at 140 tok + the chapter's audio (explicit `max_length`) |
| `audio.30s`, `video.30s` | `window` | audio, video | 30 s, 15 s stride | 2 × 180 | 30 s of audio (750 tok); 30 frames at 1 fps at 70 tok |
| `frame` | `frame` | image | one frame at each scene cut, `[n, n+1)` | 100–500 | the picture, 280 tok |
| `audio.5s` | `speech` | audio | 5 s | 540 | 5 s of audio, 125 tok |
| `cue.<lang>` | `cue` | text | the cue's span, rounded outwards to frames | 500–2 000 | `title: {title} \| text: {cue}`; cue text in the text column; `origin` in the description (`embedded`, `external`, `asr`, `mt`) |

**Input fingerprint `prep`** (§2.7, §3.7) of every `ene-1` section: `profile: "ene-1"`, then, per modality,
what §8.4 measured to move a vector by more than the int8 storage error. Text rows: `prompt` (the
template). Image rows: `tokens` (the soft-token budget), `resize` (`["ffmpeg", "bicubic", w, h]`, 448 px
wide), `source` (`[w, h, codec]` of the version's video), `lossy: null` (frames go from the decoder to the
model, never through JPEG). Video rows: the image fields plus `frames` and `fps`. Audio rows: `sample_rate`
(16 000) and `max_length` (samples). Mixed rows: the fields of every modality present. Left out on purpose,
measured: device, compute dtype, batch size (they go to `generator.compute`). Two `ene-1` builds of the
same version therefore agree in `prep` and their rows are duplicates (§5.2); a build at 140 tokens, from a
480p rendering or resized with PIL is a variant.

Measured on four titles (10.6–64.9 min; `bench/telemetry.md`, `bench/conclusions.md`): 5.4–5.7 s of compute per
minute of video at 720p/1080p (10.5–11.2× real time), 3.9 s/min at 360p (15×); ffmpeg decoding 35–55 % of it,
the video windows 26–38 % (CPU preprocessing bound, GPU 68 %), the chapters 8 %, audio 7–9 %, frames 3–10 %; peak
VRAM 2.7 GiB; 12.8–19.7 KB of `.ene` per minute of video under `compact`, 960–1 325× below the source; 96–98.5 %
of the bytes are vectors.

What the data said about this profile (`bench/conclusions.md` §5): a mixed audio+video window is the
video window (cos 0.93–0.996; silence instead of sound changes nothing), and in any mixed context the
visual tokens dominate (§7), so the timeline part is written **per modality** and never mixed except for
the chapter rows; a chapter row was the centroid of the mixed windows that pipeline wrote (recall@1 0.94–1.0) and is nearer
to the centroid of its own video windows than to any other chapter's in 42–100 % of the chapters under `ene-1` (§8.3),
so it could be derived instead of encoded (open, §10).

### 6.2 Other profiles the format carries without change

- one vector per frame (`frame` role, `[n, n+1)` for every `n`; 2 h at 30 fps = 216 000 rows = 166 MB
  of `int8` rows, 1.7 MB of frames);
- scene-level `video` rows from a shot detector, of any length;
- a subtitle in another language from another contributor (`cue.ron`, a new version with the map that
  aligns that contributor's file to the axis; `origin: mt` when machine-translated);
- a hand-written synopsis in a second language, or a cast list (`synopsis`, `person`): a version without
  a timeline, title part only, no source file;
- a second space (a face model) on the same `frame` segments, under its own key.

## 7. The model of profile `ene-1`: EmbeddingGemma 2 (informative; what the documentation says, what was verified)

Sources read in full: the HF and Google model cards (v2), the multimodal and getting-started guides, the
Google launch, AI Edge and developer-guide posts, the sentence-transformers 6.0/6.1 release notes, the
sbert input-formats guide, the model's configuration files (`config.json`, `processor_config.json`,
`config_sentence_transformers.json`) and the processor source in transformers 5.20.

| aspect | documentation | verified (bf16) |
|---|---|---|
| model | 740M (text 270M + vision 170M + audio 300M), Gemma 4, Apache 2.0, not gated | 1.42 GiB downloaded; load 3.8 s; 1.40 GiB VRAM; `modalities = text, image, audio, video, message` |
| space | 768-d, mean pooling, L2-normalized; Matryoshka 768/512/256/128 with renormalization | mean cosine error vs 768: 0.004 at 512, 0.010 at 256, 0.087 at 128 (RO text) |
| context | 8 192 tokens, shared by all modalities | |
| image | 280 soft tokens by default; budget ∈ {70, 140, 280, 560, 1120}; aspect-preserving resize, patch 16, pooling 3 | 720p → 142 ms; a frame with a conductor → top "orchestră condusă de un dirijor" |
| video | 140 tok/frame by default, 1 fps, `max_frames` 32, `overflow_strategy` uniform/truncate, `add_timestamps` needs `VideoMetadata.fps` | 10 frames @ 1 fps → 278 ms |
| audio | mono 16 kHz, 25 tok/s, "~327 s" within the budget | **by default only the first 30 s**: `Gemma4AudioFeatureExtractor` has `max_length=480_000` samples and `truncation=True`; 60 s and 120 s gave the same 750 tokens as 30 s (cos 1.0). Raised per call with `processing_kwargs={"audio": {"max_length": N}}`: 120 s → 3 000 tokens, 217 ms, 1.77 GiB VRAM. The pipeline always sets an explicit `max_length` |
| audio → text | MSEB retrieval MRR@10 69.5 | 20 s of RO speech → "A person speaking in Romanian" 0.78 (top); speech is searchable by text without transcription (z-score 3.3–3.8 on documentaries) |
| interleaved | `{"text": "... <\|image\|> ... <\|audio\|>", "image": …, "audio": …}` → one vector | title + frame + audio → top: the title's own synopsis sentence about the conductor and his orchestra, 0.76 |
| interleaved: who dominates | undocumented | **the visual tokens dominate the mean**: 30 frames + 30 s audio → cos 0.997 with video alone, 0.68 with audio alone; audio counts only when alone in the context |
| prompts | text: `task: search result \| query: ` (query), `title: {t} \| text: ` (document; the real title helps); media without prefix | |
| precision | bf16 or fp32; **never fp16** (silent NaN). Storing the *resulting* vectors in fp16/int8 is unaffected | the model's own L2 normalization runs in bf16 and leaves norms in [0.998, 1.004]; the pipeline renormalizes in fp32 before quantization |
| determinism | undocumented | the output is a deterministic function of (input, batch shape): 32 batch sizes → 11 kernel classes, bit-identical within a class, 1 − cos ≤ 8.6·10⁻⁴ between classes (the order of the int8 storage error); fp32 and CPU land in the same regime. Preprocessing moves vectors 30–2 000× more (token budget, resize path, source rendering, JPEG; §8.4): these, and not device, dtype or batch, are the `prep` of an `ene-1` section (§6.1) |
| scores | the cosine is high even for unrelated pairs (~0.6–0.75) | "interviu" vs "peisaj" = 0.75 → ranks and relative scores, not thresholds |
| faces, names | not a face model | a person is found by text (`person` rows), not by the picture |
| languages | multilingual | the same query in RO and EN finds the same moment in 17–70 % of cases (cos between the two queries 0.79–0.88): text rows in every available language are the coverage |
| selective loading | `config_kwargs={"vision_config": None}` etc.; vectors stay compatible across configurations | |
| dependencies | sentence-transformers ≥ 6.1, transformers 5 | 6.2.0 / 5.20.2; audio and video decoded with ffmpeg, without torchcodec |

## 8. Measured (informative; `bench/`)

### 8.1 Size and fidelity (Big Buck Bunny grid, confirmed on four titles)

The metric: bytes per row and per minute of video for size; `recall@10` against `fp32` at 768 d (strict,
and tolerant to ±1 neighbouring row where rows overlap) and the mean cosine error for information; MB/s at
decode and time to the first `q @ M.T` for speed. fp32 vectors are noise to classical compressors; the gain
comes from quantization, then entropy coding of the temporal delta.

| stage | candidate | measured (768 d) | verdict |
|---|---|---|---|
| reference | JSON text + zlib | 2 990–3 520 B/row, as much as raw fp32 | eliminated |
| reference | raw fp32 | 3 072 B/row | baseline |
| quantization | fp16 + zstd | 1 182 B/row, 2.6×, recall 1.000 | small sections |
| quantization | int8 + scale | 770 B/row, 4×, cos error 0.0008–0.001, recall 0.95–0.98 | the base |
| coding | int8 + zstd | 518–569 B/row, 5.4–5.9×, decode 450–1 000 MB/s | good |
| coding | int8 + brotli / lzma | same bytes as zstd, decode 17–300 MB/s | eliminated |
| coding | int8 + lz4 | 740–778 B/row | eliminated |
| coding | RLE | 1 484–1 508 B/row, 2× worse than raw int8 | eliminated |
| ENE codec | int8 + `ene-delta` | 428–517 B/row, 6–7.2×; decode 150–230 MB/s | available |
| ENE codec | int8 + `ene-xdelta` | 445–540 B/row, +5–6 %; decode 7× faster (0.5 ms vs 3.3 ms per 2 134 rows) | **default (`compact`)** |
| Matryoshka at write time | 512 / 256 / 128 + int8 + delta | 290–350 / 150–187 / ~100 B/row; strict recall 0.86–0.96 / 0.79–0.90 / 0.58–0.81 | **at read time only**: the truncation error at 256 d (0.013–0.018) equals the gap between stored neighbours (0.010–0.020), so it permutes them |
| bin | 768 bits + zstd | 87–98 B/row, recall 0.54–0.80 | sieve only |

Confirmed on four titles (10.6–64.9 min) under `compact`: int8 + `ene-xdelta` at 468–538 B/row with recall@10
0.957–0.970 (0.968–0.986 at ±1 row) and cos error 0.0008–0.0009 on every large section (`bench/telemetry.md`); int8 +
`ene-delta` on the same rows 429–516 B/row. The delta gains from temporal
coherence: neighbouring 30 s windows have cos 0.92–0.95, 5 s audio rows 0.80–0.84, and the nearest stored
neighbour is the temporal neighbour in ~90 % of rows. Also measured: audio and video rows of the same
moment do not match each other (recall@1 of the pair 1–12 %, chance level), which is why every modality
is stored as its own rows; for absent concepts the model still returns a top-1 (0.62–0.65 vs 0.72–0.80
when the concept exists), hence the need for annotated ground truth (§10).

### 8.2 Loading time (`bench/loading.md`)

Measured with the reference reader on the files `ene-1` writes under the §3 layout (13 sections per title, 8 of them
in the title part): best of 7 in a warm process on one machine, through `ene.open` and `EneFile.search`, the files in
the page cache, `verify=False`.

| scenario | `compact` with `ene-delta` | `compact` | `fast` |
|---|---|---|---|
| one title, 1 879 rows: open → first scores on every row | 5.3 ms | 2.9 ms | 0.74 ms (int8 form) |
| one title, 1 044–1 130 rows: the same | 3.0–3.2 ms | 1.7 ms | 0.47–0.54 ms (int8 form) |
| one title, 356 rows: the same | 1.5 ms | 1.0 ms | 0.42 ms (fp32 form) |
| one title: open → title part only (8 sections, "which title?") | 0.4 ms | 0.4 ms | — |
| one title, first call of a fresh process (after `import ene`): open → first scores on every row | 2.1–6.2 ms | 1.7–3.8 ms | — |
| 1 000 titles: open all + title part of each, through the files (warm) | — | 0.42 s | — |
| 1 000 titles: persisted index (§5.4) over 17 750 title-part rows: build / size | — | 0.9 s / 55 MB | same |
| 1 000 titles: index open / warm scores over the title parts / first call of a process | — | 0.26 ms / 1.7 ms / 15 ms | same |

Two finish lines were stated before measuring and met in three consecutive runs on the layout that preceded §3, with
the same reader kernels (`bench/loading.md`): within 3× of a hardware bound built from machine constants (`fast`
1.37–1.58×, `compact` 1.88–2.11×, index 1.10–1.21×), and no standard container (`.npy`, `.npz`, `safetensors`, `pickle`,
`torch.save`, in both int8 and fp32 form) loading the same vectors faster: `fast` beat every uncompressed one and
`compact` every compressed one on every title (13–22 µs ahead on the smallest title, 42–218 µs on the others; ahead in
40 of 40 fresh processes). The §3 figures above are of the same order as those runs' (`compact`: 2.6 ms for 2 134 rows
then, 2.9 ms for 1 879 rows now; `fast`: 0.43–0.50 ms then, 0.42–0.74 ms now; the title part: 0.36 ms over four
sections then, 0.4 ms over eight now); the bound and the container comparison have not been rebuilt on them (§10).

What the reader does for it: `os.open` + two `lseek` + one `os.read` of exactly the file size for files
≤ 4 MiB (a memory map above); the binary tables parsed with one `tolist()` (~5 µs); nothing else parsed;
sections decoded lazily; a one-off query never materializes a float32 matrix (fp32 rows multiplied in
place; int8 rows cast 256 at a time into an L2-resident scratch below 1 024 rows, and above that cast in
parallel on `min(6, cores)` threads into one persistent scratch, in 128-row chunks pulled from a shared
counter, followed by one GEMV on the calling thread); the int8 scale folded into the scores; verification
on demand. Alternatives measured and rejected are itemized in `bench/loading.md` §1c.

### 8.3 Summaries

Measured so far: on the pipeline that preceded `ene-1`, whose 30 s windows were mixed audio+video like the
chapter rows, a chapter row was the centroid of the windows it contained (recall@1 0.94–1.0, cos 0.966 with the
encoded chapter), which is what `k = 1` relies on. Against the per-modality windows `ene-1` writes, the chapter row
is nearer to the centroid of its own video windows than to any other chapter's in 42–100 % of the chapters, and to
that of its audio windows in 20–78 % (`bench/telemetry.md`): a derived chapter would be a centroid per modality,
not of mixed rows (§10, item 7). Not yet measured: whether `k > 1`
representatives per timeline section raise the rank of the right title for queries about rare content
(a concept present in two scenes of a two-hour film) over the centroid alone. The fp32 masters of the
four titles (`*.master.npz`) allow this without re-embedding; until then profile `ene-1` writes `k = 1`.

### 8.4 The same frame on every path (`bench/frames.md`)

100 random frames of Big Buck Bunny, decoded frame-exactly, embedded on the `ene-1` path (ffmpeg bicubic
448 px, 280 tokens, bf16) and on every path another contributor could take, plus 7 600 repeated runs in
random order with random batch sizes 1–32. Yardsticks (1 − cos, median): the next frame 3·10⁻³, one second
away 2.7·10⁻², int8 storage 1.8·10⁻⁴; the query margin top-1 − top-2 over the 100 frames 7·10⁻³.

| what differs between two contributors | 1 − cos (median) | frame recovered as itself |
|---|---|---|
| batch size, bf16/fp32, GPU/CPU | 2·10⁻⁵ – 7·10⁻⁵ (worst 8.6·10⁻⁴) | 100 % |
| resize filter (PIL), 720p re-encode | 2·10⁻³ | 99–100 % |
| 480p / 360p source, JPEG q90–q50 | 9·10⁻³ – 3·10⁻² | 88–97 % |
| native frame handed to the processor | 2·10⁻² | 95 % |
| token budget (140, 560, 70 instead of 280) | 1.5·10⁻² – 4·10⁻² | 83–91 % |
| *the next frame, n+1* | *3·10⁻³* | *96 %* |
| *one second later, n+30* | *2.7·10⁻²* | *52 %* |

Within one machine the output is a deterministic function of the batch shape (11 classes over 32 sizes;
every class seen by run 143 of 7 600); 21 of 24 queries kept their top-1 frame in every run, the three that
flipped have margins ≤ 3·10⁻⁴. What follows for the format (§2.7): `prep` names the budget,
the resize path, the source rendering and lossy intermediates, not the device, dtype or batch; a single
image row locates a moment to about one second, so image matching between versions checks a map `a·f + b`
by consensus and does not compute it frame-exactly (§10, item 6); writers renormalize in fp32.

## 9. Reference implementation (informative)

The package `ene/` (Python ≥ 3.12; numpy + zstandard for reading, writing and searching; torch +
sentence-transformers only for `ene build`): `container.py` (packets, tables, blocks, trailer, overlay
resolution), `quant.py` (§4), `api.py` (`ene.open`, `EneFile`, `Version`, `Section`, staged edits written as
one packet, `create`, `overlay`, `compact`, `rewrite`, `Space` = the local space of §5 in memory, `Library`), `index.py` (the persisted local space,
§5.4, as a packet of this layout), `build.py` (profile `ene-1`, §6.1), `bench.py`, `telemetry.py`, `serve.py` + `webui/` (local GUI),
`cli.py` (`python -m ene probe|build|rewrite|edit|overlay|compact|inspect|search|index|bench|telemetry|serve`).

Selection in the API is by part, role, modality, language, version, axis frame range and span length;
frames are returned in version frames or on the axis, as asked; `add_version(...)` and
`set_alignment(version, a, b)` write version records; `add_section(...)`, `remove_section(...)` and
`set(...)` (description keys) are staged and written as one packet by `save()`.

Every module reads and writes this layout: `container.py` and `api.py` implement §2–§5 (`tests/test_container.py`,
`tests/test_api.py`); `build.py` writes profile `ene-1` (`tests/test_build.py` runs the whole pipeline on a synthetic video
with a stub in place of the model); `index.py` persists the local space of §5.4 (`tests/test_index.py`); the GUI, the CLI,
the benchmark and the telemetry sit on `api.py`. The sample and the benchmark files of `bench/` were produced by this
pipeline.

## 10. Open questions (informative)

1. `k` for summaries (§8.3): to be measured on the four fp32 masters before profile `ene-1` writes anything
   but `k = 1`.
2. An annotated semantic ground truth (Big Buck Bunny first, then a documentary), to measure search quality
   rather than fidelity to fp32 (§8.1).
3. The 32-row threshold of policy `compact` (§4.4), not yet swept.
4. The hardware bound and the comparison with the standard containers of §8.2, rebuilt on the §3 files (they were
   measured on the layout that preceded §3; the loading figures themselves are re-measured there).
5. Per-section PCA or product quantization for very large local spaces (per-frame profiles, §6.2).
6. Automatic alignment between versions: audio cross-correlation of the `speech` rows as the frame-exact
   witness; the `frame` rows of two versions matched by their image vectors as a consensus check of
   `a`, `b` (an image row locates a moment to about one second, not a frame, §8.4); manual override kept.
7. Profile `ene-1`: deriving the `chapter` rows from the windows they contain (centroid; recall@1 0.94–1.0,
   −6 % build time) instead of encoding them; one decoding pass with frames decoded per segment instead of
   all in memory; batches above 1 for the chapter rows.
8. Forward compatibility: what a reader does with an unknown `quant`, `codec` or `space`, a set reserved
   bit or a non-zero reserved field (skip the section, skip the packet, or fail) is not yet specified.

---

This specification and the reference implementation in this repository are published under the Apache License 2.0
(`LICENSE`); README §11 lists the licences of the included data and of the model.
