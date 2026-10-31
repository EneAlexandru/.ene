# Conclusions from telemetry — 4 titles

Source of the numbers: `bench/telemetry.md` (aggregated tables), `bench/<slug>.telemetry.json` (everything, per title),
`bench/<slug>.bench.md` (the dim × quantization × codec grid), all produced by `python -m ene telemetry`; EmbeddingGemma 2 in bf16.
Section names are those of profile `ene-1` (SPEC §6.1): `title` the title row, `tag`, `chapter` 150 s chapters, `audio.30s` /
`video.30s` 30 s windows, `frame` one frame per scene cut, `audio.5s` 5 s audio rows. The pipeline measured here also wrote `av.30s`,
a 30 s window with audio and video in one context; §5 shows it is redundant and `ene-1` does not write it.

**Re-run under `ene-1`** (`bench/telemetry.md`: the same four titles, the mixed window gone, `ene-xdelta` instead of `ene-delta`):
5.4–5.7 s of compute per minute at 720p/1080p (10.5–11.2× real time; §1 predicted 5.4) and 3.9 s/min at 360p; 12.8–19.7 KB/min,
960–1 325× below the source; int8 + `ene-xdelta` at 468–538 B/row with recall@10 0.957–0.970 (0.968–0.986 at ±1 row); a title is
ready for its first scores 1.0–2.9 ms after `open` under `compact` (SPEC §8.2). One number moved: against per-modality windows the
chapter row is the centroid of its own video windows in 42–100 % of the chapters (§5 measured 0.94–1.0 against mixed windows),
so a derived chapter would be a centroid per modality.

| title | duration | source | content | vectors | .ene |
|---|---|---|---|---|---|
| Big Buck Bunny (bbb) | 10.6 min | 1080p30 h264, 263 MiB | animation, music, no speech | 394 | 204 KB |
| Title A (title-a) | 43.5 min | 720p25 av1, 614 MiB | documentary, RO speech, classical music, archive footage | 1 214 | 576 KB |
| Title B (title-b) | 64.9 min | 720p25 av1, 933 MiB | concert, audience, live music, RO speech | 2 134 | 1 036 KB |
| Title C (title-c) | 44.3 min | 360p25 av1, 622 MiB | documentary, EN narrator, filmed locations | 1 303 | 643 KB |

All four runs passed (hash OK, 903 grid rows × 7 sections per title, 24–27 text queries per title).

## 1. What building costs

**7.2 s of compute per minute of film** (8.3× faster than real time) for 720p and 1080p, 5.7 s/min at 360p.
A 45 min film builds in ~5.4 min, a 2 h one in ~14 min. Constant across all titles: the speed does not depend
on the content, only on the resolution (through decoding).

Where the time goes (average over the titles, s per minute of film):

| stage | s/min | share | what limits it |
|---|---|---|---|
| media decoding (ffmpeg: audio + 1 fps frames + scene cuts) | 2.7 (1.4 at 360p) | 24–41 % | **CPU**: the three ffmpeg processes run in parallel took as long as a single one (27 s vs 30 s sequential on bbb) — AV1/H.264 decoding saturates the CPU cores; GPU 3 % |
| av.30s (audio+video interleaved, batch 1) | 1.83 | 25 % | **preprocessing CPU** (4.2 cores busy, GPU only 59 %) |
| video.30s (30 frames × 70 tok, batch 2) | 1.46 | 20 % | the same (3.5 cores, GPU 68 %) |
| chapter (29 frames × 140 tok + 150 s audio, batch 1) | 0.45 | 6 % | 1.06 s per chapter, GPU 74 % |
| frame (280 tok, batch 16) | 0.31 | 4 % | 45 ms/frame, GPU 83 % — the only GPU-bound stage |
| audio.5s + audio.30s | 0.35 | 5 % | audio is cheap: 290–800× real time, 17–39 ms/window |
| title part, probe, write | < 0.1 | — | |

- The GPU is used on average only **37–47 %** over the course of a build; peak VRAM 2.68 GiB (model 1.4 + activations).
  So several parallel builds, or larger batches, fit on one GPU.
- Peak RSS 2.7–4.3 GiB, proportional to the duration (all the 1 fps frames stay in memory: 3 894 frames × 338 KB = 1.3 GB
  for 65 min). A 3 h film would need ~12 GiB; for long titles the frames must be decoded per window, not all at once.
- Image preprocessing (resize + patches, in the HF processor, on the CPU) is the bottleneck of the video stages,
  not the model: CPU/wall 3.5–4.2 at GPU 57–71 %.

**What follows.** (1) Without `av.30s` (see §5: it is redundant) the build drops to **5.4 s/min (11.1× real time)**, 3.9 s/min
at 360p. (2) A single ffmpeg pass with two outputs (or NVDEC decoding on the GPU) would cut another ~1.2 s/min. (3) Batches > 1
for the chapter rows and preprocessing in separate threads would raise GPU utilization. Realistic target: ~3 s/min, i.e. 2 h in 6 min.

## 2. How much space the file takes

**13.6–19.7 KB per minute of film** under `compact` with `ene-delta` (768 d everywhere, int8 + ene-delta on the windows, frames and 5 s audio rows, fp16 on the title part and the chapters):
a 2 h film = 1.6–2.4 MB, **920–1 325× smaller than the source** and 6× smaller than the fp32 vectors.
Big Buck Bunny is the most "expensive" per minute because it has 11 scene cuts per minute (key frames) versus 3–8 for the documentaries.

- 97–98.5 % of the bytes are vectors; the tables and the description are 6–7 KB; the times, the int8 scales and the text are under 1 %.
  There is nothing left to gain from the header or the metadata: any reduction comes from the number of vectors or from the bits per vector.
- frames + 5 s audio rows = 54–63 % of the file; the 30 s windows = 26–38 %; title part + chapters = 3–13 %.
  `audio.5s` alone = 35–43 % (the most windows: 12/min).
- Other policies, on the same vectors (`ene rewrite`): Matryoshka at write time (windows at 512, frames and 5 s audio at 256) = 8.4–12.2 KB/min (−40 %, but with
  the recall refuted in SPEC §8.1); lossless fp16 = 33–45 KB/min (2.3×); without `av.30s` = 11.9–17.9 KB/min (−10 %).

## 3. Compression under int8 + `ene-delta` (fidelity to fp32 768)

| section | storage | B/vec | ratio | recall@10 strict | recall ±1 window | cos error | decode |
|---|---|---|---|---|---|---|---|
| audio.30s / video.30s / av.30s | int8 + ene-delta | 429–445 | 6.9–7.2× | 0.966–0.969 | 0.980–0.984 | 0.0008–0.0009 | 300–320 MB/s |
| frame | int8 + ene-delta | 516 | 6.0× | 0.969 | 0.973 | 0.0008 | 332 MB/s |
| audio.5s | int8 + ene-delta | 478 | 6.4× | 0.954 | 0.965 | 0.0009 | 313 MB/s |
| title part, chapter | fp16 + zstd | 1 185 | 2.6× | 1.000 | 1.000 | 0 | 500–700 MB/s |

- The int8 + delta storage is confirmed on all four titles: int8 loses under 5 % of the top-10 neighbors (under 3.5 % with a
  one-window tolerance), with a cosine error of 0.001 — 10–20× below the distance between the stored neighbors (§4).
- `ene-delta` gains from **temporal coherence**: on the 30 s windows (overlapping by 50 %) the cosine with the neighboring window is 0.92–0.95
  and the nearest neighbor of a row is the temporal neighbor in 89–92 % of the cases → 429–445 B/vec; on the frames and 5 s audio rows (no
  overlap) the neighbor cosine is 0.80–0.84, the NN is temporal in only 11–13 % of the cases → 478–516 B/vec.
  The worst-compressed section (`frame`) is also the least "temporal" one.

## 4. The geometry of a title's vectors (why the policy is what it is)

- **The space is very anisotropic**: within a section the mean cosine between any two rows is 0.78–0.89
  and the centroid norm 0.88–0.95. All the vectors of a title sit in a narrow cone; the absolute scores (0.56–0.75 for
  text queries) mean nothing on their own — what matters are the ranks and the z-scores (§6). Confirms SPEC §7.
- **MRL**: the 512 prefix holds 78 % of the norm, 256 holds 53 %, 128 holds 39 % — identical across all modalities and titles.
  The cosine error of truncation to 256 on the section's own pairs is 0.013–0.018, **as large as the distance between
  top-1 and top-2** (0.010–0.020). This is the mechanism that rules out Matryoshka truncation at write time: it permutes nearly
  equal neighbors. Int8 (error 0.0009) sits an order of magnitude below the threshold, which is why it does not permute.
- **Effective rank** 16–32 on the timeline sections; 90 % of the variance in 41–136 dimensions (audio.5s 136, video.30s 41). The variation
  *within a title* is low-dimensional → a per-section PCA projection (int8 on 64–136 coefficients) would give another
  3–4× over int8 on 768, but the basis (768 × k × 2 B ≈ 100–200 KB per section) cancels the gain below ~2 h of film.
  An `ene-pca` codec is worth it only for series / stacked libraries, where the basis is amortized. Not now.

## 5. The modalities: what each one brings and what it does not

- **`av.30s` is `video.30s`**. The cosine between the audio+video window and the video-only window is 0.93–0.996;
  the mixed window is closer to audio than to video in 0–2 % of the windows. On bbb, replacing the sound with
  silence does not change the vector (cos 0.999). It is not a matter of token count: with 1 frame (70 tokens) against 750 tokens
  of audio, the mix stays on the video side (0.925 vs 0.823). In the common EG2 space, the visual tokens dominate the mean
  (mean pooling) of the context; audio matters only on its own.
- **The sound and the image of the same moment do not match each other**: searching for the audio window with the video window
  from the same time, the correct pair comes first in 1–12 % of the cases (pure chance: 0.4–2.4 %); the cosine on the diagonal
  (0.65–0.74) is practically equal to the one off it (0.65–0.72). The space is common *semantically* (a text finds both sound
  and image), but the sound of a moment does not describe its image — not even in documentaries with a narrator.
- **A chapter is the centroid of the 30 s windows** it contains (recall@1 0.94–1.0, cos 0.966) and, like av.30s, it is dominated by
  video (cos 0.988 with video-only vs 0.711 with audio-only on bbb).
- **The title row** is dominated by text (cos 0.91 with the text alone), then by the frames (0.82); audio adds the least (0.73).
  For "which film?" the title row with all modalities gave slightly better scores than the text alone (0.76 vs 0.74 for the same
  RO query, 0.66 for an unrelated one).

**What follows**: the mixed window is dropped from the profile (−25 % build time, −10 % bytes, zero information
lost; SPEC §6.1 writes the timeline **per modality**) and the audio + video fusion is done **at query time** (max score or weighted
sum per window, both sections being on the same grid). `chapter` can be *derived* from the windows (centroid) instead
of encoded (−6 % time), with a loss of 0.03 cosine; whether the uniformity is worth it is open (SPEC §10).

## 6. What real queries do

- The top-1 scores are 0.64–0.75; the mean score per section 0.56–0.69; the standard deviation 0.02–0.04; the top-1 − top-2 margin only
  0.004–0.02. The top-1 z-score is 1.3–2.5 on bbb (no speech) and 2.5–3.8 on the documentaries — highest on
  `audio.5s` (3.3–3.8): **speech is searched directly through audio**, without transcription (SPEC §7).
- The sections **do not agree** with each other: the top-1 moment of the same query differs between modalities by minutes
  (median 1–20 min), except `av.30s` ~ `video.30s` (0.41–0.63 at ±30 s, one more proof for §5). This is not a defect:
  "mocking laughter" makes sense in audio, "butterflies on a meadow" in video. For the user, the good answer is the fusion
  across sections (the best moment from each modality), not a single list.
- **RO vs EN**: the same query in two languages has cos 0.79–0.88 and finds the same moment (±30 s) in 17–70 % of the cases
  (bbb audio.30s 0.70, title-c av.30s 0.58, title-b video.30s 0.00). Multilingualism is real, but not symmetric: for
  stable results it is worth querying in both languages and fusing.
- **Visual verification** (32 frames extracted at the top-1 moments and looked at): on Big Buck Bunny 5 out of 8
  visual queries are clearly correct (the flying squirrel, the butterfly, the credits, the rodents with acorns, the rabbit in its burrow);
  on Title A 5 out of 8 (the conductor, the orchestra, the black-and-white photographs, the interview, the name card); on Title B and Title C
  most are only thematically close (crowds / Sigiriya) — my queries assumed content that **does not exist in
  the film** (pyramids, candles). For such queries the model still returns a "top-1" with a score of 0.62–0.65, versus
  0.72–0.80 when the concept exists. So a relative threshold (z ≥ 3 or top-1 ≥ ~0.70) can signal "not found", but
  the calibration requires the annotated ground truth (SPEC §10).
- GUI detail: the frame extracted exactly at the time of the scene cut can be the last frame of the *previous* scene
  (rounding to ms); the thumbnails for `frame` must be requested at `t0 + 0.3 s`.

## 7. Reading the file

Opening with sha256 verification + decoding all the sections + the first `Q @ M.T`: **11–17 ms per title** cold
(sha256 7–10 ms, decoding 1.9–6.7 ms, GEMM < 1 ms), 2–7 ms warm, 0.2 ms for opening without verification.
A library of 1 000 titles loads in full in ~13 s with verification (~5 s without); the library index from the title part
(a few vectors per title) is practically free.

Loading (`bench/loading.md`, SPEC §8.2): with the XOR delta (policy `compact`) a 65-minute title is ready for its
first query in 2.62–2.63 ms, with the uncompressed `fast` policy in 0.43–0.50 ms; a binary section table ahead of the description means a
search never parses JSON, sections are decoded lazily, files are opened with `os.open` and one exact-size `os.read`, under
`fast` a file is one GEMV on the file bytes (fp32 rows, below 1 024 vectors) or one parallel cast + one GEMV (int8 rows above), and a
stacked index answers "which title?" over 1 000 films in 1.14–1.17 ms. Against a hardware-bound model with a ≤ 3× rule fixed before
measuring, `compact` is within 1.88–2.11×, `fast` within 1.37–1.58× and the index within 1.10–1.21× in three consecutive runs; against external
containers holding the same vectors in both int8 and fp32 form, `compact` beats every compressed one and `fast` beats every uncompressed one
on every title in all three runs and in 40 further fresh processes (smallest margin 13–22 µs, on the smallest title).


## 8. What the measurements suggest

1. **Drop `av.30s`**; per-modality fusion at query time (max score per window). SPEC §6.1.
2. **`chapter` derived from the windows** (centroid) — open (SPEC §10); encoded until the ground truth exists.
3. **Pipeline**: ffmpeg decoding in a single pass (or NVDEC), frames per window (not all in RAM), batches > 1 for
   the chapter rows, preprocessing in separate threads. Target: ~3 s/min.
4. **Semantic ground truth** (manually annotated moments on Big Buck Bunny, then on a documentary): the only way to measure
   the quality of the search, not just the fidelity to fp32; calibration of the "not found" threshold.
5. **Bilingual querying** (RO + EN) with fusion, as an option in the GUI.
6. `frame` thumbnails at `t0 + 0.3 s` in the GUI.
7. The text in the file (the `text` column, SPEC §3.6): it costs < 0.1 % of the bytes, so the "smaller file" argument
   does not exist; only the licensing / confidentiality question of a given subtitle remains.
8. Per-section `ene-pca`: to be tested only on stacked libraries (series), not on isolated titles.
