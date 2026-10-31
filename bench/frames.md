# Frame stability: the same frame embedded on every path

Question: when two contributors embed the *same frame* of the *same title*, how far apart are their vectors, and does the
difference change what a query finds? If it is below the gap between neighbouring stored vectors, the two rows carry the
same information (the local space may merge or deduplicate them); if it is above, they are genuinely different variants.

Setup: Big Buck Bunny (1080p, 30 fps, 19 036 frames), 100 frames drawn uniformly at random (seed 20261028), decoded
frame-exactly (sequential decode, `select=eq(n,…)`); EmbeddingGemma 2, image rows at 280 soft tokens. Baseline = the
`ene build` path: ffmpeg scale to 448×252, batch 16, bf16 on the GPU. Cosines are computed after renormalizing in
fp32 (the model's own normalization is done in bf16 and leaves norms in [0.9981, 1.0039]). Queries: the 24 RO/EN benchmark queries; ranks are over the 100 frames.

## Yardsticks (what a difference has to be compared with)

| quantity | min | median |
|---|---|---|
| cosine to the next frame (n+1) | 0.9310 | 0.9970 |
| cosine to the frame one second later (n+30) | 0.7081 | 0.9725 |
| cosine to the nearest of the other 99 frames | 0.8079 | 0.9622 |
| nearest-neighbour gap (top-1 − top-2 cosine among the 100) | 0.0000 | 0.0158 |
| cosine after int8 storage (`compact`) | 0.999779 | 0.999817 |
| cosine after fp16 storage | 1.000000 | 1.000000 |
| query margin top-1 − top-2 over the 100 frames | 0.0001 | 0.0068 |

## Variants against the baseline (100 frames each unless noted)

| variant | frames | bit-identical | cos min | cos median | 1 − cos (median) | 1 − cos (worst) | top-1 kept | top-5 kept | rank shift median | rank shift max | max score change |
|---|---|---|---|---|---|---|---|---|---|---|---|
| batch size 1 | 100 | 0 % | 0.99980 | 0.99998 | 1.8e-05 | 2.0e-04 | 96 % | 99 % | 0.0 | 7 | 0.0017260313034057617 |
| batch size 4 | 100 | 4 % | 0.99914 | 0.99994 | 5.9e-05 | 8.6e-04 | 88 % | 94 % | 0.0 | 8 | 0.005906403064727783 |
| batch size 32 | 100 | 100 % | 1.00000 | 1.00000 | 0.0e+00 | 1.8e-07 | 100 % | 100 % | 0.0 | 0 | 0.0 |
| fp32 on the GPU, batch 16 | 100 | 0 % | 0.99918 | 0.99993 | 7.0e-05 | 8.2e-04 | 92 % | 95 % | 0.0 | 9 | 0.0046506524085998535 |
| fp32 on the GPU, batch 1 | 100 | 0 % | 0.99918 | 0.99993 | 7.0e-05 | 8.2e-04 | 92 % | 95 % | 0.0 | 9 | 0.004648804664611816 |
| fp32 on the CPU (20 frames) | 20 | 0 % | 0.99957 | 0.99994 | 6.0e-05 | 4.3e-04 | — | — | — | — | — |
| native 1080p handed to the processor (no pre-resize) | 100 | 0 % | 0.86994 | 0.97792 | 2.2e-02 | 1.3e-01 | 46 % | 65 % | 6.0 | 92 | 0.0996469259262085 |
| native -> PIL bicubic 448 | 100 | 0 % | 0.99059 | 0.99813 | 1.9e-03 | 9.4e-03 | 83 % | 86 % | 2.0 | 43 | 0.02198582887649536 |
| native -> PIL bilinear 448 | 100 | 0 % | 0.97744 | 0.99416 | 5.8e-03 | 2.3e-02 | 79 % | 84 % | 4.0 | 57 | 0.04295319318771362 |
| native -> PIL Lanczos 448 | 100 | 0 % | 0.98562 | 0.99598 | 4.0e-03 | 1.4e-02 | 79 % | 82 % | 3.0 | 41 | 0.022669970989227295 |
| pipeline frame -> JPEG q90 | 100 | 0 % | 0.96594 | 0.98909 | 1.1e-02 | 3.4e-02 | 67 % | 70 % | 5.0 | 48 | 0.051291704177856445 |
| JPEG q75 | 100 | 0 % | 0.94058 | 0.97187 | 2.8e-02 | 5.9e-02 | 50 % | 72 % | 6.0 | 65 | 0.05206674337387085 |
| JPEG q50 | 100 | 0 % | 0.93333 | 0.96657 | 3.3e-02 | 6.7e-02 | 33 % | 64 % | 7.0 | 64 | 0.05903327465057373 |
| same frame from a 720p re-encode (x264 crf 23) | 100 | 0 % | 0.98897 | 0.99788 | 2.1e-03 | 1.1e-02 | 92 % | 88 % | 2.0 | 37 | 0.023888587951660156 |
| 480p re-encode (crf 28) | 100 | 0 % | 0.96202 | 0.99066 | 9.3e-03 | 3.8e-02 | 62 % | 75 % | 5.0 | 56 | 0.04766160249710083 |
| 360p re-encode (crf 30) | 100 | 0 % | 0.94290 | 0.98327 | 1.7e-02 | 5.7e-02 | 42 % | 67 % | 7.0 | 69 | 0.05959552526473999 |
| 140 soft tokens instead of 280 | 100 | 0 % | 0.88884 | 0.97815 | 2.2e-02 | 1.1e-01 | 29 % | 59 % | 7.0 | 67 | 0.06293344497680664 |
| 70 soft tokens | 100 | 0 % | 0.86194 | 0.96200 | 3.8e-02 | 1.4e-01 | 42 % | 56 % | 9.0 | 71 | 0.07114088535308838 |
| 560 soft tokens | 100 | 0 % | 0.95633 | 0.98534 | 1.5e-02 | 4.4e-02 | 62 % | 65 % | 5.0 | 57 | 0.04589402675628662 |

## What this says

Three regimes, measured against the yardsticks:

1. **Compute path** (batch shape, bf16 vs fp32, GPU vs CPU): `1 − cos` of 2·10⁻⁵ to 7·10⁻⁵ at the median, 9·10⁻⁴ at worst. That is the order of the int8 storage error (1.8·10⁻⁴) and 30× below the distance to the next frame (3·10⁻³). Rows of the same segment from pipelines that differ only here carry the **same information**: a local space may deduplicate them.
2. **Resize filter and a good re-encode** (PIL instead of ffmpeg's bicubic, a 720p x264 source): 2·10⁻³, the distance to the next frame. Still the same picture, no longer interchangeable at rank level (top-1 kept for 83–92 % of the queries).
3. **Source quality (480p, 360p), JPEG round trips, the processor's own resize of a native frame, and above all the token budget**: 10⁻² to 4·10⁻², the distance to a frame one second away (2.7·10⁻²). Rankings reshuffle for half of the queries; the cosine between two renderings of the same frame drops to 0.86. These are **different variants**, not copies, and the format keeps all of them.

Rankings among close frames are fragile by nature: the query margin top-1 − top-2 over 100 frames (median 0.007, min 0.0001) is smaller than most of the differences above, so even a batch-size change flips the top-1 frame for 4–12 % of the queries. "The" vector of a frame does not exist; its neighbourhood does. A search engine working on these rows has to be tolerant (overlapping spans, evidence per version), and the file has to keep every variant.

Consequences for the format and the profile:

- the key `space / quant / dim` guarantees **comparability**, not equivalence: two rows of the same segment under the same key may differ by up to 0.14 in cosine depending on preprocessing;
- the description of a section must carry the **preprocessing fingerprint**: source resolution and codec, resize width and filter, soft-token budget. Profile `ene-1` fixes them (ffmpeg bicubic to 448 px wide, 280 tokens); rows whose fingerprints agree are regime 1 and deduplicable, otherwise they are variants;
- the **token budget is the largest single factor** (70 vs 280 tokens: cos down to 0.86): it is part of the identity of a row's information and is stored per section;
- the model's own L2 normalization runs in bf16 and leaves norms in [0.998, 1.004]: writers renormalize in fp32 before quantization (`ene build` does) and readers must not take unit norm from the model for granted;
- for contributors who want their rows to be *copies* of a reference pipeline, the cheap requirement is: same source rendering or a ≥ 720p re-encode, the same resize path, the same budget; GPU, precision and batch size do not matter.

## 7600 repeated runs of the same 100 frames (random order, random batch size 1..32, bf16 GPU)

| quantity | value |
|---|---|
| runs × frames | 7600 × 100 = 760000 encodings |
| outputs bit-identical to the baseline | 55.97 % |
| distinct output vectors per frame over all runs | min 10, median 10, max 11 |
| cosine to the baseline | min 0.999138, median 1.000000 |
| 1 − cos, worst over all runs and frames | 8.6e-04 |
| top-1 frame kept per query, over runs | min 88 %, mean 98.3 % |
| top-5 set kept per query, over runs | min 94 %, mean 99.0 % |
| max score change vs baseline | 5.91e-03 |

Bit-identical fraction by batch size (batch 16 is the baseline's):

| batch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identical | 0 % | 0 % | 95 % | 4 % | 0 % | 69 % | 71 % | 92 % | 95 % | 0 % | 0 % | 0 % | 9 % | 94 % | 86 % | 92 % | 96 % | 86 % | 0 % | 0 % | 15 % | 0 % | 96 % | 92 % | 96 % | 75 % | 78 % | 96 % | 84 % | 86 % | 94 % | 92 % |

Bit-identical fraction by batch size reads as a partition, not as a probability (next section).

## Conclusions

The loop was stopped at 7600 of the planned 10 000 runs: every frame had shown all of its distinct outputs by run 143
(median 88), and runs 144–7600 added none. The result is exhaustive for the within-machine axis on one machine
(one GPU, one torch build, one sentence-transformers build); other GPUs, drivers or library versions were not measured,
the fp32 CPU run (6·10⁻⁵ median) suggests they land in the same regime.

**1. Within one machine the output is not noisy, it is a finite function of the batch shape.** For every batch size that
divides 100 (every batch full) each frame produced exactly one vector in 200+ runs; for the others at most two (the full
batch and the remainder batch). Position in the batch and the other images in the batch do not matter. The 32 batch
sizes fall into 11 classes by the vector they produce:

| class | batch sizes |
|---|---|
| the `ene build` class (batch 16) | 3, 8, 9, 14, 15, 16, 17, 18, 23–32 |
| one class each | 1 · 2 · 4 · 5 · 10 · 11 · 12 · 13 |
| shared | 6, 7 · 19, 20, 21, 22 |

Two contributors with the same preprocessing produce bit-identical rows when their batch sizes fall in the same class,
and rows at 1 − cos ≤ 8.6·10⁻⁴ otherwise (334 629 non-identical outputs: median 1.9·10⁻⁵, p90 8.3·10⁻⁵, p99 2.1·10⁻⁴;
a single frame of 100 exceeds 5·10⁻⁴).

**2. A frame is recoverable as itself from every within-pipeline variant, and from most preprocessing variants.** Of the
1025 distinct vectors seen in the repeated runs, 100 % have their own frame as the nearest baseline frame, margin
≥ 0.0010 (median 0.038). The spread inside a frame (max 1 − cos between two of its variants, median 7·10⁻⁵, max 9.2·10⁻⁴)
is at least 27× below the distance to the nearest other frame (median ratio 519). Across preprocessing, nearest baseline
frame = same frame for:

| variant | self-nearest | variant | self-nearest |
|---|---|---|---|
| batch 1 / batch 4 / fp32 | 100 % | native frame → processor | 95 % |
| 720p re-encode | 100 % | 360p re-encode | 93 % |
| PIL resize (bicubic, bilinear, Lanczos) | 99–100 % | JPEG q75 | 92 % |
| JPEG q90 | 97 % | 560 tokens | 91 % |
| 480p re-encode | 96 % | 140 tokens · JPEG q50 | 88 % |
| *the next frame (n+1)* | *96 %* | 70 tokens | 83 % |
| *one second later (n+30)* | *52 %* | | |

Changing the token budget moves a vector further than moving one frame in time. Conversely, a single image row locates a
moment to the order of a second, not a frame: embedding-based matching between versions can *check* a map a·f + b by
consensus over many rows, it cannot compute it frame-exactly.

**3. Rank stability is a property of the query margin, not of the pipeline.** Over 7600 runs, 21 of the 24 queries kept
the same top-1 frame in every run. The three that flipped (6–27 % of runs) have top-1 − top-2 margins of 0.0001–0.0003,
ties at the resolution of the model's bf16 output; every query with a margin ≥ 0.0017 never flipped. The per-run figure
"top-1 kept 98.3 %, min 88 %" is these three queries.

### What this means for integrating other contributors' rows

- **Same preprocessing fingerprint ⇒ same information.** Rows for the same version, segment, modality and role that
  share the fingerprint differ at most by the batch-shape class: 1 − cos ≤ 10⁻³, the order of the int8 storage error,
  and the frame is recoverable as itself from any of them. The local space may keep one of them (dedup is a local-space
  decision; the file transports every row it was given).
- **Different fingerprint ⇒ a variant; keep it.** The fingerprint has to name exactly the factors that moved vectors by
  more than the batch-class spread: the soft-token budget (the largest: 70 vs 280 → cos down to 0.86; 560 vs 280 →
  1.5·10⁻²), the resize path (library, filter, target size: PIL vs ffmpeg bicubic 2–6·10⁻³; the processor's own resize of
  a native frame 2·10⁻²), the source rendering (720p 2·10⁻³, 480p 9·10⁻³, 360p 1.7·10⁻²), any lossy intermediate (JPEG q90
  1.1·10⁻², q50 3.3·10⁻²). It need not name the device, the compute dtype or the batch size (≤ 9·10⁻⁴, measured).
- **Ties are ties.** Score differences below ~10⁻³ between candidates are below what the same pipeline reproduces on
  another day; nothing in the file can or should resolve them.

### What the specification takes from this

The format stays model-agnostic: what it took from this measurement is one generic rule, not these numbers.

1. §2.7, §3.5, §3.7: every section carries an **input fingerprint** `prep`, an object the format does not
   interpret (hashed into the section record as a u64); its content is the space profile's, chosen by one
   criterion: what moves a vector by more than the storage error of the policy.
2. §5.2: rows of the same key, version, part, modality, role, lang and `prep`, on the same segment (timeline
   part) or text (title part), are duplicates and the local space keeps one (higher `rev`, later file on a
   tie); anything else is a variant and is kept.
3. §6.1: profile `ene-1` puts into `prep` the prompt, the soft-token budget, the resize path (`ffmpeg`,
   `bicubic`, 448 px), the source rendering, the lossy intermediate (none), frames/fps for video, sample
   rate/max_length for audio; it leaves out device, compute dtype and batch size (`generator.compute`).
4. §7, §8.4: the determinism by batch shape and the fp32 renormalization are recorded as properties of
   EmbeddingGemma 2, not of the format; automatic alignment between versions (§10) uses image matching only as a
   consensus check.

The per-variant and per-run figures are in `frames.json` (the repeated runs under `conclusions`); the scripts and the raw
arrays (`axes.npz`, `repeat.npz`) are kept outside the repository.
