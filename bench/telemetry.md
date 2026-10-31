# .ene telemetry - 4 titles

## Software

- ene 1.0.0, policy compact; python 3.13.14, torch 2.14.1+cu130, transformers 5.20.2, sentence_transformers 6.2.0, numpy 2.5.4, zstandard 0.25.0, PIL 13.0.0, av 19.0.2, fastapi 0.159.1, ffmpeg 8.0-full_build-www.gyan.dev

## Titles

| title | duration | source | video | rows | build (s) | x real time | peak VRAM | peak RSS | .ene | B/min | source/.ene | fp32/.ene |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Big Buck Bunny | 10.6 min | 263 MiB h264 | 1920x1080@30 | 356 | 60.2 | 10.5 | 2.68 GiB | 2.70 GiB | 204 KB | 19,729 | 1,323x | 5.2x |
| Title A | 43.5 min | 614 MiB av1 | 1280x720@25 | 1,044 | 235.5 | 11.1 | 2.68 GiB | 3.50 GiB | 544 KB | 12,799 | 1,155x | 5.8x |
| Title B | 64.9 min | 933 MiB av1 | 1280x720@25 | 1,879 | 349.1 | 11.2 | 2.68 GiB | 4.25 GiB | 992 KB | 15,656 | 963x | 5.7x |
| Title C | 44.3 min | 622 MiB av1 | 466x360@25 | 1,130 | 172.2 | 15.4 | 2.68 GiB | 4.00 GiB | 605 KB | 13,973 | 1,053x | 5.6x |

## Stages (compute seconds per minute of film; in parentheses: real-time factor)

| stage | bbb | title-a | title-b | title-c | mean |
|---|---|---|---|---|---|
| probe | 0.02 | 0.01 | 0.01 | 0.01 | 0.02 |
| decode_media | 2.65 (22.6x) | 3.00 (20.0x) | 2.80 (21.4x) | 1.38 (43.5x) | 2.46 |
| decode_audio | 0.18 (341.4x) | 0.09 (684.5x) | 0.07 (862.4x) | 0.06 (991.5x) | 0.10 |
| decode_frames | 2.65 (22.7x) | 3.00 (20.0x) | 2.80 (21.4x) | 1.38 (43.5x) | 2.46 |
| scene_detect | 2.65 (22.6x) | 2.92 (20.6x) | 2.76 (21.7x) | 1.36 (44.2x) | 2.42 |
| frame | 0.57 (50 ms/item) | 0.14 (45 ms/item) | 0.35 (43 ms/item) | 0.21 (44 ms/item) | 0.32 |
| audio.5s | 0.22 (269.5x) | 0.21 (289.1x) | 0.20 (301.8x) | 0.20 (298.3x) | 0.21 |
| audio.30s | 0.17 (690.5x) | 0.15 (775.8x) | 0.15 (799.4x) | 0.15 (781.9x) | 0.16 |
| video.30s | 1.47 (80.2x) | 1.44 (83.1x) | 1.43 (83.7x) | 1.47 (81.2x) | 1.45 |
| chapter | 0.51 (117.0x) | 0.43 (138.5x) | 0.42 (142.6x) | 0.44 (136.1x) | 0.45 |
| title | 0.06 (36 ms/item) | 0.01 (48 ms/item) | 0.01 (49 ms/item) | 0.01 (54 ms/item) | 0.02 |
| **total build** | **5.69** | **5.41** | **5.38** | **3.88** | |

### Throughput per stage (mean over titles; CPU/wall = cores kept busy by the Python process, excluding the child ffmpeg processes, so underestimated for decode*; GPU % sampled at 1 Hz)

| stage | items | ms / item | x real time | peak VRAM (GiB) | CPU/wall | GPU util % | GPU W |
|---|---|---|---|---|---|---|---|
| decode_media | 0 |  | 26.9 | 1.41 | 0.11 | 2 | 34 |
| decode_audio | 0 |  | 720.0 | 1.41 | 0.52 | 1 | 41 |
| decode_frames | 9,800 | 41 | 26.9 | 1.41 | 0.11 | 2 | 34 |
| scene_detect | 1,006 | 444 | 27.3 | 1.41 | 0.11 | 2 | 34 |
| frame | 1,006 | 45 |  | 2.68 | 2.58 | 81 | 199 |
| audio.5s | 1,961 | 17 | 289.7 | 1.53 | 1.31 | 39 | 114 |
| audio.30s | 652 | 39 | 761.9 | 2.14 | 1.29 | 65 | 149 |
| video.30s | 652 | 364 | 82.0 | 2.53 | 3.47 | 68 | 192 |
| chapter | 67 | 1,067 | 133.6 | 2.58 | 2.40 | 64 | 194 |
| title | 51 | 47 |  | 2.00 | 2.84 | 82 | 223 |

## Where the bytes go (.ene, storage policy)

| title | total | tables | description | vectors | scale | time | text | title part | timeline part |
|---|---|---|---|---|---|---|---|---|---|
| bbb | 208,640 | 3,520 | 1,038 | 200,034 | 1,289 | 1,169 | 597 | 30,513 | 172,576 |
| title-a | 557,056 | 3,520 | 1,052 | 545,147 | 3,577 | 2,517 | 327 | 24,463 | 527,105 |
| title-b | 1,016,000 | 3,520 | 1,026 | 998,486 | 6,268 | 5,545 | 241 | 24,362 | 986,178 |
| title-c | 619,392 | 3,520 | 1,031 | 606,743 | 3,826 | 2,925 | 311 | 22,874 | 590,931 |

## Sections under the storage policy (mean over titles; recall@10 vs fp32 at the native width)

| section | storage | rows | B/vec | ratio | recall@10 | recall±1 | cos error | dec MB/s |
|---|---|---|---|---|---|---|---|---|
| tag | 768 fp16 zstd | 43 | 1,402 | 2.2x | 0.999 | 1.000 | 0.0001 | 449 |
| audio.30s | 768 int8 ene-xdelta | 652 | 483 | 6.4x | 0.969 | 0.981 | 0.0009 | 849 |
| audio.5s | 768 int8 ene-xdelta | 1,961 | 506 | 6.1x | 0.957 | 0.968 | 0.0009 | 861 |
| chapter | 768 fp16 zstd | 67 | 1,408 | 2.2x | 0.999 | 1.000 | 0.0001 | 760 |
| frame | 768 int8 ene-xdelta | 1,006 | 538 | 5.7x | 0.970 | 0.974 | 0.0008 | 894 |
| video.30s | 768 int8 ene-xdelta | 652 | 468 | 6.6x | 0.968 | 0.986 | 0.0008 | 838 |

## Section geometry (mean over titles)

| section | temporal neighbour cos | NN cos | NN gap (top1-top2) | NN = temporal neighbour | mean pairwise cos | centroid norm | prefix energy 512/256/128 | effective rank | dims for 90 % var | MRL 256 cos error |
|---|---|---|---|---|---|---|---|---|---|---|
| tag | 0.884 | 0.916 | 0.0144 | 0.20 | 0.877 | 0.942 | 0.758 / 0.468 / 0.319 | 7.7 | 8 | 0.0069 |
| audio.30s | 0.920 | 0.937 | 0.0203 | 0.89 | 0.818 | 0.905 | 0.782 / 0.532 / 0.393 | 24.3 | 66 | 0.0168 |
| audio.5s | 0.835 | 0.895 | 0.0096 | 0.14 | 0.779 | 0.882 | 0.781 / 0.520 / 0.397 | 32.0 | 136 | 0.0183 |
| chapter | 0.919 | 0.938 | 0.0176 | 0.66 | 0.890 | 0.949 | 0.784 / 0.543 / 0.385 | 9.3 | 12 | 0.0111 |
| frame | 0.800 | 0.922 | 0.0175 | 0.11 | 0.774 | 0.880 | 0.779 / 0.533 / 0.397 | 21.2 | 70 | 0.0177 |
| video.30s | 0.952 | 0.966 | 0.0178 | 0.91 | 0.844 | 0.919 | 0.782 / 0.538 / 0.394 | 16.4 | 41 | 0.0155 |

## Alignment between sections (the 30 s windows on the same grid; a chapter against the centroid of the windows it contains)

| title | windows | audio->video r@1 / r@1±1 | video->audio r@1 / r@1±1 | diag cos | off-diag cos | cos(audio, video) same window | chapter->video windows r@1 | chapter->audio windows r@1 | title·summaries | modality gap |
|---|---|---|---|---|---|---|---|---|---|---|
| bbb | 42 | 0.02 / 0.07 | 0.05 / 0.10 | 0.651 | 0.648 | 0.651 | 1.00 | 0.20 | summary.frame 0.87, summary.audio.5s 0.76, summary.audio.30s 0.76, summary.video.30s 0.89, summary.chapter 0.93 | 0.737 |
| title-a | 174 | 0.11 / 0.14 | 0.01 / 0.03 | 0.724 | 0.718 | 0.724 | 0.61 | 0.39 | summary.frame 0.89, summary.audio.5s 0.81, summary.audio.30s 0.86, summary.video.30s 0.90, summary.chapter 0.93 | 0.871 |
| title-b | 259 | 0.02 / 0.05 | 0.01 / 0.03 | 0.723 | 0.710 | 0.723 | 0.42 | 0.62 | summary.frame 0.84, summary.audio.5s 0.83, summary.audio.30s 0.87, summary.video.30s 0.88, summary.chapter 0.93 | 0.851 |
| title-c | 177 | 0.08 / 0.23 | 0.12 / 0.30 | 0.740 | 0.698 | 0.740 | 1.00 | 0.78 | summary.frame 0.87, summary.audio.5s 0.82, summary.audio.30s 0.87, summary.video.30s 0.87, summary.chapter 0.92 | 0.878 |

## Text queries per section

| title | section | mean top-1 | margin top1-top2 | top-1 z-score | mean score | score std |
|---|---|---|---|---|---|---|
| bbb | audio.30s | 0.642 | 0.0053 | 1.97 | 0.605 | 0.0189 |
| bbb | audio.5s | 0.648 | 0.0049 | 2.47 | 0.605 | 0.0178 |
| bbb | chapter | 0.686 | 0.0108 | 1.25 | 0.659 | 0.0207 |
| bbb | frame | 0.686 | 0.0075 | 2.49 | 0.620 | 0.0274 |
| bbb | video.30s | 0.699 | 0.0043 | 1.51 | 0.652 | 0.0316 |
| title-a | audio.30s | 0.724 | 0.0094 | 2.56 | 0.627 | 0.0388 |
| title-a | audio.5s | 0.731 | 0.0148 | 3.29 | 0.608 | 0.0384 |
| title-a | chapter | 0.728 | 0.0111 | 1.86 | 0.693 | 0.0192 |
| title-a | frame | 0.725 | 0.0061 | 2.13 | 0.653 | 0.0353 |
| title-a | video.30s | 0.740 | 0.0049 | 2.05 | 0.674 | 0.0340 |
| title-b | audio.30s | 0.724 | 0.0062 | 2.50 | 0.651 | 0.0294 |
| title-b | audio.5s | 0.727 | 0.0099 | 2.89 | 0.633 | 0.0325 |
| title-b | chapter | 0.718 | 0.0069 | 1.94 | 0.686 | 0.0165 |
| title-b | frame | 0.710 | 0.0107 | 2.87 | 0.603 | 0.0372 |
| title-b | video.30s | 0.725 | 0.0056 | 2.61 | 0.645 | 0.0305 |
| title-c | audio.30s | 0.641 | 0.0101 | 3.22 | 0.565 | 0.0238 |
| title-c | audio.5s | 0.664 | 0.0206 | 3.80 | 0.560 | 0.0275 |
| title-c | chapter | 0.651 | 0.0119 | 2.22 | 0.608 | 0.0191 |
| title-c | frame | 0.676 | 0.0072 | 3.17 | 0.588 | 0.0285 |
| title-c | video.30s | 0.670 | 0.0064 | 3.14 | 0.589 | 0.0259 |

### Agreement between sections (fraction of queries whose top-1 moment coincides within ±30 s / ±60 s)

| title | pair | ±30 s | ±60 s | median gap |
|---|---|---|---|---|
| bbb | audio.30s|audio.5s | 0.21 | 0.25 | 230 s |
| bbb | audio.30s|chapter | 0.21 | 0.38 | 128 s |
| bbb | audio.30s|frame | 0.08 | 0.12 | 122 s |
| bbb | audio.30s|video.30s | 0.12 | 0.21 | 225 s |
| bbb | audio.5s|chapter | 0.04 | 0.04 | 252 s |
| bbb | audio.5s|frame | 0.04 | 0.12 | 223 s |
| bbb | audio.5s|video.30s | 0.17 | 0.21 | 158 s |
| bbb | chapter|frame | 0.04 | 0.21 | 127 s |
| bbb | chapter|video.30s | 0.17 | 0.21 | 120 s |
| bbb | frame|video.30s | 0.29 | 0.38 | 137 s |
| title-a | audio.30s|audio.5s | 0.41 | 0.41 | 145 s |
| title-a | audio.30s|chapter | 0.19 | 0.26 | 165 s |
| title-a | audio.30s|frame | 0.04 | 0.07 | 766 s |
| title-a | audio.30s|video.30s | 0.07 | 0.11 | 705 s |
| title-a | audio.5s|chapter | 0.19 | 0.26 | 1,235 s |
| title-a | audio.5s|frame | 0.04 | 0.04 | 809 s |
| title-a | audio.5s|video.30s | 0.00 | 0.04 | 390 s |
| title-a | chapter|frame | 0.00 | 0.00 | 878 s |
| title-a | chapter|video.30s | 0.15 | 0.22 | 255 s |
| title-a | frame|video.30s | 0.11 | 0.15 | 856 s |
| title-b | audio.30s|audio.5s | 0.22 | 0.26 | 770 s |
| title-b | audio.30s|chapter | 0.00 | 0.00 | 705 s |
| title-b | audio.30s|frame | 0.00 | 0.04 | 1,934 s |
| title-b | audio.30s|video.30s | 0.00 | 0.04 | 1,425 s |
| title-b | audio.5s|chapter | 0.04 | 0.19 | 615 s |
| title-b | audio.5s|frame | 0.11 | 0.11 | 472 s |
| title-b | audio.5s|video.30s | 0.11 | 0.15 | 975 s |
| title-b | chapter|frame | 0.04 | 0.04 | 1,431 s |
| title-b | chapter|video.30s | 0.22 | 0.26 | 240 s |
| title-b | frame|video.30s | 0.11 | 0.15 | 1,020 s |
| title-c | audio.30s|audio.5s | 0.15 | 0.15 | 650 s |
| title-c | audio.30s|chapter | 0.19 | 0.30 | 120 s |
| title-c | audio.30s|frame | 0.15 | 0.30 | 524 s |
| title-c | audio.30s|video.30s | 0.26 | 0.44 | 105 s |
| title-c | audio.5s|chapter | 0.04 | 0.07 | 280 s |
| title-c | audio.5s|frame | 0.07 | 0.15 | 629 s |
| title-c | audio.5s|video.30s | 0.15 | 0.19 | 500 s |
| title-c | chapter|frame | 0.07 | 0.22 | 161 s |
| title-c | chapter|video.30s | 0.30 | 0.33 | 120 s |
| title-c | frame|video.30s | 0.19 | 0.37 | 155 s |

### Language pairs (the same query in two languages: same moment?)

| title | section | ±30 s | ±60 s | median gap | cos(q_a, q_b) |
|---|---|---|---|---|---|
| bbb | audio.30s | 0.40 | 0.40 | 90 s | 0.788 (min 0.694) |
| bbb | audio.5s | 0.50 | 0.50 | 68 s | 0.788 (min 0.694) |
| bbb | chapter | 0.40 | 0.40 | 150 s | 0.788 (min 0.694) |
| bbb | frame | 0.60 | 0.70 | 9 s | 0.788 (min 0.694) |
| bbb | video.30s | 0.30 | 0.60 | 60 s | 0.788 (min 0.694) |
| title-a | audio.30s | 0.25 | 0.33 | 225 s | 0.883 (min 0.777) |
| title-a | audio.5s | 0.33 | 0.33 | 990 s | 0.883 (min 0.777) |
| title-a | chapter | 0.58 | 0.58 | 0 s | 0.883 (min 0.777) |
| title-a | frame | 0.25 | 0.25 | 636 s | 0.883 (min 0.777) |
| title-a | video.30s | 0.33 | 0.50 | 75 s | 0.883 (min 0.777) |
| title-b | audio.30s | 0.17 | 0.17 | 592 s | 0.851 (min 0.803) |
| title-b | audio.5s | 0.08 | 0.08 | 1,055 s | 0.851 (min 0.803) |
| title-b | chapter | 0.42 | 0.42 | 225 s | 0.851 (min 0.803) |
| title-b | frame | 0.17 | 0.17 | 1,545 s | 0.851 (min 0.803) |
| title-b | video.30s | 0.08 | 0.08 | 1,005 s | 0.851 (min 0.803) |
| title-c | audio.30s | 0.42 | 0.50 | 68 s | 0.848 (min 0.739) |
| title-c | audio.5s | 0.58 | 0.58 | 0 s | 0.848 (min 0.739) |
| title-c | chapter | 0.67 | 0.67 | 0 s | 0.848 (min 0.739) |
| title-c | frame | 0.33 | 0.33 | 107 s | 0.848 (min 0.739) |
| title-c | video.30s | 0.58 | 0.67 | 8 s | 0.848 (min 0.739) |

## Reading (.ene -> first scores on every section, through the public API)

| title | file | rows | cold: open+sha256 | cold: decode | cold: GEMM | cold total | warm total | open without verification | title part only ("which title?") |
|---|---|---|---|---|---|---|---|---|---|
| bbb | 208,640 B | 356 | 6.46 ms | 1.31 ms | 0.42 ms | **8.61 ms** | 1.51 ms | 0.09 ms | 0.56 ms |
| title-a | 557,056 B | 1,044 | 6.46 ms | 1.49 ms | 0.3 ms | **8.53 ms** | 2.24 ms | 0.1 ms | 0.56 ms |
| title-b | 1,016,000 B | 1,879 | 6.74 ms | 2.28 ms | 0.38 ms | **9.64 ms** | 3.47 ms | 0.15 ms | 0.71 ms |
| title-c | 619,392 B | 1,130 | 6.68 ms | 1.45 ms | 0.25 ms | **8.63 ms** | 2.2 ms | 0.1 ms | 0.55 ms |
