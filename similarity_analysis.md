# Attractor codebook size (k): reliability vs. discriminability

Companion analysis to [similarity_cv.md](similarity_cv.md), using the split-half CV
occupancy and occupancy-bin matrices (`eye_data_plotting.pre_flash_similarities_occupancy`,
`pre_flash_similarities_occupancy_bin`). All numbers below are for `fix_start − 1466 ms → fix_start − 300 ms`, 20 splits, CV seed 0, mazes 1–6.

Two different questions get asked of the same 6×6 CV matrix:

- **Reliability** — the mean of the diagonal. "If I re-measured this maze's state-occupancy
profile from an independent half of trials, would I get the same answer?"
- **Discriminability** — diagonal vs. off-diagonal. "Does this k actually tell mazes apart,
or do different mazes look about as similar to each other as a maze looks to itself?"
Reported here as the **ratio** diagonal-mean ÷ off-diagonal-mean (>1 means the codebook
separates mazes; closer to 1 means it doesn't).

A high reliability score by itself is not evidence of a good k — a degenerate k=1 codebook
(one state covering everything) would score a perfect 1.0 on reliability while carrying zero
information. Discriminability is the more meaningful criterion for choosing k for downstream
analysis.

## 1. Reliability (mean diagonal)


| k   | Faure occ | Faure occ-bin | Nielsen occ | Nielsen occ-bin |
| --- | --------- | ------------- | ----------- | --------------- |
| 2   | 0.822     | 0.942         | 0.954       | **1.000***      |
| 3   | **0.974** | **0.986**     | **0.974**   | 0.984           |
| 4   | 0.962     | 0.971         | 0.916       | 0.968           |
| 5   | 0.971     | 0.979         | 0.935       | 0.973           |
| 6   | 0.973     | 0.980         | 0.900       | 0.957           |
| 12  | 0.891     | 0.937         | 0.888       | 0.948           |


By reliability alone, **k=3** wins for both monkeys on both features.

 Nielsen k=2 occupancy-bin hit an exact 1.000 — a degenerate result (near-zero-variance
visit vector for at least one maze), not a genuine top score. See §2.

## 2. Discriminability (diagonal ÷ off-diagonal)


| k   | Faure occ diag | Faure occ offdiag | **Faure occ ratio** | Faure bin diag | Faure bin offdiag | **Faure bin ratio** |
| --- | -------------- | ----------------- | ------------------- | -------------- | ----------------- | ------------------- |
| 2   | 0.822          | 0.698             | **1.178**           | 0.942          | 0.929             | 1.014               |
| 3   | 0.974          | 0.934             | 1.043               | 0.986          | 0.974             | 1.012               |
| 4   | 0.962          | 0.904             | 1.064               | 0.971          | 0.948             | 1.024               |
| 5   | 0.971          | 0.927             | 1.048               | 0.979          | 0.962             | 1.018               |
| 6   | 0.973          | 0.933             | 1.042               | 0.980          | 0.964             | 1.017               |
| 12  | 0.891          | 0.810             | 1.100               | 0.937          | 0.894             | **1.047**           |
| 24  | 0.885          | 0.822             | 1.077               | 0.924          | 0.888             | 1.040               |



| k   | Nielsen occ diag | Nielsen occ offdiag | **Nielsen occ ratio** | Nielsen bin diag | Nielsen bin offdiag | **Nielsen bin ratio** |
| --- | ---------------- | ------------------- | --------------------- | ---------------- | ------------------- | --------------------- |
| 2   | 0.954            | 0.952               | 1.003                 | 1.000            | 1.000               | 1.000                 |
| 3   | 0.974            | 0.921               | 1.057                 | 0.984            | 0.956               | 1.029                 |
| 4   | 0.916            | 0.698               | 1.311                 | 0.968            | 0.859               | 1.127                 |
| 5   | 0.935            | 0.795               | 1.177                 | 0.973            | 0.897               | 1.084                 |
| 6   | 0.900            | 0.684               | 1.316                 | 0.957            | 0.846               | 1.132                 |
| 12  | 0.888            | 0.716               | 1.241                 | 0.948            | 0.855               | 1.110                 |
| 24  | 0.832            | 0.616               | **1.351**             | 0.899            | 0.775               | **1.160**             |


**Nielsen:** discriminability climbs sharply from k=2/3 (ratio ≈1.0–1.06, essentially no
separation) up through k=4–6 (ratio ≈1.13–1.32) and keeps climbing out to k=24 (ratio 1.35,
the highest tested). k=3's reliability win is misleading here — it's "reliable" mainly
because it isn't resolving much.

**Faure:** a different, non-monotonic pattern. The ratio is highest at the extremes tested —
k=2 (1.178 occ) and k=12 (1.100 occ) — and dips through the k=3–6 middle (ratio ≈1.04–1.06),
where different mazes look nearly as similar to each other as a maze looks to itself. Pushing
further to k=24 pulls the ratio back down from k=12's peak (1.077 vs 1.100 occ; 1.040 vs 1.047
bin) — so more resolution does not straightforwardly help Faure the way it does Nielsen.

**k=12 vs k=24, directly:**


|         | k=12 occ ratio | k=24 occ ratio | k=12 bin ratio | k=24 bin ratio |
| ------- | -------------- | -------------- | -------------- | -------------- |
| Faure   | **1.100**      | 1.077          | **1.047**      | 1.040          |
| Nielsen | 1.241          | **1.351**      | 1.110          | **1.160**      |


k=12 beats k=24 for Faure; k=24 beats k=12 for Nielsen. The two monkeys disagree on whether
more resolution past 12 keeps paying off.

## 3. Case study: maze 3 vs. maze 4 (Faure)

Raw split-half correlation between maze 3 and maze 4 specifically, and that same value
expressed as a **ratio, diagonal mean ÷ pair value** — same direction as §2, so higher is
always better throughout this document. A ratio above 1 means this specific pair sits below
the typical same-maze reliability (genuinely more separable, a discriminability win); below 1
means the pair looks *more* alike than a typical same-maze split-half (a red flag).


| k   | occ maze3-vs-4 | occ diag mean | **occ ratio** | bin maze3-vs-4 | bin diag mean | **bin ratio** |
| --- | -------------- | ------------- | -------------- | -------------- | ------------- | -------------- |
| 2   | 0.863          | 0.822         | 0.953          | 0.915          | 0.942         | **1.030**      |
| 3   | 0.977          | 0.974         | 0.997          | 0.987          | 0.986         | 0.999          |
| 4   | 0.963          | 0.962         | 0.999          | 0.964          | 0.971         | 1.007          |
| 5   | 0.968          | 0.971         | 1.003          | 0.971          | 0.979         | 1.008          |
| 6   | 0.971          | 0.973         | 1.002          | 0.976          | 0.980         | 1.004          |
| 12  | 0.858          | 0.891         | **1.038**      | 0.915          | 0.937         | 1.024          |


**k=12 gives the largest genuine separation of maze 3 from maze 4**, both in raw terms
(lowest correlation, 0.858) and relative to its own diagonal (ratio 1.038, occupancy — the
highest tested). k=2's occupancy ratio (0.953, <1) is the opposite signal: at k=2 these two
mazes look *more* alike than a typical same-maze pair, so k=2 is not a consistent answer
across features (it wins on occ-bin, 1.030, but loses on occupancy).

## 4. Per-maze breakdown

Which individual maze is most reliable (its own diagonal cell) and most discriminable (its
diagonal ÷ the mean of its own off-diagonal row) — computed at each monkey's
best-discriminating k from §2 (Faure k=12, Nielsen k=24). Per the codebook's own maze
grouping (`eye_data_plotting/pre_flash_heatmaps_sum.py`): **Faure** — hierarchical mazes
1–3, sequential 4–6; **Nielsen** — hierarchical 1–4, sequential 5–6.

**Faure, k=12, occupancy**

| Maze | diag | offdiag mean | ratio |
| --- | --- | --- | --- |
| 1 | 0.882 | 0.809 | 1.090 |
| 2 | 0.893 | 0.789 | **1.132** |
| 3 | 0.870 | 0.812 | 1.072 |
| 4 | **0.908** | 0.841 | 1.080 |
| 5 | 0.896 | 0.805 | 1.113 |
| 6 | 0.898 | 0.806 | 1.114 |

**Faure, k=12, occupancy-bin**

| Maze | diag | offdiag mean | ratio |
| --- | --- | --- | --- |
| 1 | 0.930 | 0.893 | 1.042 |
| 2 | 0.942 | 0.886 | **1.062** |
| 3 | 0.926 | 0.893 | 1.037 |
| 4 | **0.945** | 0.910 | 1.039 |
| 5 | 0.938 | 0.893 | 1.050 |
| 6 | 0.939 | 0.892 | 1.052 |

**Nielsen, k=24, occupancy**

| Maze | diag | offdiag mean | ratio |
| --- | --- | --- | --- |
| 1 | 0.819 | 0.617 | 1.326 |
| 2 | 0.801 | 0.644 | 1.244 |
| 3 | 0.850 | 0.655 | 1.297 |
| 4 | 0.782 | 0.674 | 1.161 |
| 5 | 0.863 | 0.628 | 1.374 |
| 6 | **0.880** | 0.478 | **1.841** |

**Nielsen, k=24, occupancy-bin**

| Maze | diag | offdiag mean | ratio |
| --- | --- | --- | --- |
| 1 | 0.887 | 0.766 | 1.159 |
| 2 | 0.876 | 0.789 | 1.110 |
| 3 | 0.907 | 0.798 | 1.137 |
| 4 | 0.876 | 0.812 | 1.079 |
| 5 | **0.925** | 0.793 | 1.166 |
| 6 | 0.923 | 0.691 | **1.335** |

**Summary, with runner-up**

| Monkey | Feature | Reliable (1st) | Reliable (2nd) | Discriminable (1st) | Discriminable (2nd) |
| --- | --- | --- | --- | --- | --- |
| Faure | occupancy | **Maze 4** (0.908) | Maze 6 (0.898) | **Maze 2** (1.132) | Maze 6 (1.114) |
| Faure | occupancy-bin | **Maze 4** (0.945) | Maze 2 (0.942) | **Maze 2** (1.062) | Maze 6 (1.052) |
| Nielsen | occupancy | **Maze 6** (0.880) | Maze 5 (0.863) | **Maze 6** (1.841) | Maze 5 (1.374) |
| Nielsen | occupancy-bin | **Maze 5** (0.925) | Maze 6 (0.923) | **Maze 6** (1.335) | Maze 5 (1.166) |

**Nielsen's pattern is unambiguous:** mazes 5 and 6 — its two sequential-type mazes — take
*both* rank 1 and rank 2 in all four columns; nothing hierarchical (1–4) breaks in anywhere.
Nielsen's maze 6 is the single biggest standout number in this whole analysis (discriminability
ratio 1.84, occupancy).

**Faure shows a different but still real pattern:** hierarchical maze 2 leads discriminability
in both features, with sequential maze 6 the consistent runner-up (2nd place, 3 of 4 columns).
Reliability instead favors maze 4, which sits right at Faure's own hierarchical/sequential
boundary. So Faure doesn't split as cleanly along maze type as Nielsen does — it reads more
as "maze 2 discriminates best, maze 6 is always close behind" than a type-level effect.

## Caveats

- All CV numbers use a single split-half seed (seed=0). The Faure k=12 vs k=24 flip and the
k=3–6 dip are close enough in places (e.g. Faure ratios 1.04–1.06 across k=3–6) that a
multi-seed check would be worth running before treating small rank differences as real
rather than split-sampling noise.
- Nielsen k=2 occupancy-bin (diag=offdiag=1.000) is a degenerate case, not a genuine result —
excluded from any "best k" conclusion.
- Reliability and discriminability can rank k differently; this file exists because they did.
For a question like "which k should I use to distinguish maze types," discriminability
(§2–3) is the relevant criterion, not raw reliability (§1) alone.
- §4's per-maze breakdown is computed at a single k per monkey (each monkey's own
best-discriminating k from §2). It would look different at another k — it is not a
k-independent property of the mazes themselves, and like everything else here it's a
single-seed (seed=0) estimate.

