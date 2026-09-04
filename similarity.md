# Maze-to-maze similarity CV

Every plot here is a **6×6 matrix of Pearson *r* between maze types 1–6**. Cell
`(i, j)` is maze *i* vs maze *j*. It is **not** trial *i* vs trial *j*.

Diagonal entries are split-half reliability of the same maze, estimated from
independent trial halves. Off-diagonal entries are the similarity of two maze
types, estimated without using the same trials twice.

## 1. The procedure

The same everywhere: within a session, split each maze's trials in half, average
each half into one feature vector, correlate held-out halves, repeat, then
average the session matrices.

**Window.** Pre-fixation samples from `fix_start − 1466 ms` through `fix_start`
(`PRE_FIX_START_MS` in `plot_io.py`; `PRE_FIX_END_MS = 0` in each script). Gaze
is warped to unit H (`to_maze`) so exits sit at `(±1, ±1)`.

**Per session, 20 random splits (`--n-splits`, CV seed 0):**

1. For each maze type, randomly cut that maze's trials in half (`n // 2` vs the
   rest).
2. Average the trials in each half into one feature vector. Maze *i* now has
   `A_i` and `B_i`.
3. For every maze pair `(i, j)`:

   \[
   r_{ij} = \mathrm{mean}\big(\mathrm{corr}(A_i, B_j),\; \mathrm{corr}(B_i, A_j)\big)
   \]

   `corr` is Pearson *r* over the entries of the two vectors — bins, states or
   n-grams, never trials or time.
4. Average the 20 split matrices into one 6×6 for the session, then average the
   session matrices.

**What differs between the plots is only the per-trial vector:**

| Plot | Code | Vector for each trial | Maze kept if |
| ---- | ---- | --------------------- | ------------ |
| Heatmap, linear | `pre_flash_similarities_heatmap` | flattened unit-H gaze map, `count / n_trials` over spatial bins (`--bin-w`) | ≥ 4 trials |
| Heatmap, log | `pre_flash_similarities_heatmap_log` | the same, `log1p(count / n_trials)` | ≥ 4 trials |
| Occupancy | `pre_flash_similarities_occupancy` | seconds spent in each of the *K* codebook states | ≥ 2 trials |
| Occupancy, binary | `pre_flash_similarities_occupancy_bin` | length-*K* 0/1 vector: 1 if that state was visited | ≥ 2 trials |
| Transition | `pre_flash_similarities_transition` | bigram + trigram proportions of collapsed state runs (dwell time ignored). Path `k1 → k5 → k2` contributes bigrams `(k1, k5)`, `(k5, k2)` and trigram `(k1, k5, k2)` | ≥ 2 trials |

Heatmap trials additionally require QC-passed fixed geometry (`path_type != -99`,
photodiode OK, `trial_fade == 0`); the state-based plots require
`path_type != -99` and a finite feature vector. Shared CV machinery lives in
`eye_pre_flash/plotting/similarities/common.py`.

```bash
uv run python -m eye_pre_flash.plotting.similarities.heatmap --monkey Faure
uv run python -m eye_pre_flash.plotting.similarities.heatmap_log --monkey Nielsen
uv run python -m eye_pre_flash.plotting.similarities.occupancy --monkey Faure --k 12
uv run python -m eye_pre_flash.plotting.similarities.occupancy_bin --monkey Faure
uv run python -m eye_pre_flash.plotting.similarities.transition --monkey Faure
```

Output: `eye_pre_flash/plotting/out/similarities/<script name>/<monkey>/`.

---

## 2. Equal trial counts: `heatmap_eq` and `heatmap_labels`

Every matrix in §1 is biased by trial count, and the bias runs in exactly the
direction that makes the sequential mazes look like a block.

Split-half CV averages `n // 2` trials into each half map, so a maze with more
trials gets a cleaner half map and a higher diagonal — its own split-half
reliability. Pearson *r* between two noisy means is dragged down by the noise in
**both**, so a maze measured more precisely also correlates more highly with
*every* other maze, and most of all with the other precisely-measured mazes.
Both animals run the most trials on mazes 4–6. That alone is enough to produce a
sequential block.

Two scripts fix this the same way — **every half map averages the same number of
trials** — so the reliabilities are comparable and the raw *r* values can be read
directly against each other:

| Script | Matrix | Question |
| ------ | ------ | -------- |
| `similarities.heatmap_eq` | 6×6, maze × maze | do two mazes elicit the same gaze map, once no maze is measured better than any other? |
| `similarities.heatmap_labels` | 12×12, (maze × decoded strategy) | inside one maze, do trials sharing a decoded strategy produce more similar gaze maps than trials that don't? |

### `heatmap_eq` — the maze matrix with counts equalised

Identical to `similarities.heatmap` except for the half size. In each session it
is `min over included mazes of n // 2`, and every maze's half maps are
subsampled to exactly that many trials. Window, unit-H warp, QC, 20 random
splits and session averaging are unchanged, so the two figures are directly
comparable and **the difference between them is the trial-count bias**.

`--n-half` fixes the same size across sessions too. Sessions still differ in how
much data they have, but that shifts every cell of a session's matrix together
and so cannot create block structure; the per-session default keeps more trials.

Only equalisation is applied — no disattenuation, no rescaling. The diagonal
stays a measured quantity, and a maze whose gaze map is genuinely less
reproducible than another's still shows it.

### `heatmap_labels` — the same matrix split by decoded strategy

The 6×6 matrices cannot separate strategy from geometry. Strategy is close to a
deterministic function of (session, maze), so "mazes 4–6 resemble each other"
and "sequentially-solved trials resemble each other" are the same statement
about the same trials — which is the passive-visual-geometry alternative that
`eye_pre_flash/classifier/classifier.md`'s reviewer response exists to rule out.
Nothing computed across mazes can rule it out.

So each maze's trials are split by their **decoded neural label**, giving a
12×12 matrix over the (maze, strategy) cells: mazes 1–6 labelled hierarchical,
then mazes 1–6 labelled sequential. Cell `(i, j)` is the split-half CV Pearson
*r* between two cells' gaze maps, computed exactly as in §1, so the diagonal is
again each cell's own reliability.

Three readings, in order of how much they can settle:

1. **Within a maze, H vs S** — the boxed cells. Geometry is identical on both
   sides, so `r(H, S)` sitting below that maze's two same-strategy diagonals is
   strategy-dependent sampling with the visual confound removed by
   construction. This is the decisive comparison, and the report tabulates it.
2. **Same strategy, different maze** — inside a quadrant. Does a decoded
   strategy look like itself across geometries?
3. **The two off-diagonal quadrants** — cross-strategy similarity overall.

Counts are balanced across all twelve cells, not just across mazes. Without
that, the majority strategy in each maze would carry the higher reliability and
correlate higher with everything — the same bias `heatmap_eq` removes, running
along the strategy axis and manufacturing the effect being looked for.

**The matrix rests on one common session set.** Cells qualify in different
numbers of sessions, and sessions differ enormously in overall correlation
level — in Faure's `all` scope one session carries roughly double everyone
else's reliability and supplies *only* the six majority cells. Letting each pair
average over whichever sessions hold its two cells therefore inflates every pair
that happens to include that day, and the inflation reads as block structure:
before this was fixed, `4S` looked more similar to `1H`–`3H` (0.425) than to its
own maze's `4H` (0.318). Restricted to the sessions holding both, the ordering
reverses to 0.297 vs 0.316 — same maze, different strategy is the *higher* one,
as shared geometry predicts.

So `common_cell_set` identifies the **comparable core**: the largest cell set
that a single shared set of sessions all supply, maximising cells and then
sessions. Nothing is dropped for it — every cell is still plotted and reported,
because hiding thin cells hides how thin the design is. Instead, entries that do
not rest on the core's shared sessions are marked `*` (and so are the axis
labels of cells outside it). A marked value is a real measurement; it just
averages over different recording days than an unmarked one, so **a marked entry
must not be read against an unmarked one**. The report also gives the core
recomputed on its single session set, where everything is comparable with
everything else.

The within-maze table uses its own, more permissive matching: for each maze,
the sessions where both of *that* maze's cells qualified. That is a per-maze
comparison, so it does not need one global session set and keeps more data.

A cell needs `--min-trials` trials in a session to enter at all, and one thin
cell lowers the half size for every other cell in that session, so raising
`--min-trials` trades cells for cleaner maps.

### Scopes

`heatmap_labels` needs neural labels, so it runs each label scope separately
into its own subfolder, matching `classifier/out/decoding/`:

| Scope | Sessions |
| ----- | -------- |
| `publication` | the 4 the published clustering was defined on (2 per monkey) |
| `all` | SNR-eligible and label-vetted — the primary set (8 in practice) |
| `allplus` | SNR-eligible, vetting skipped (13 in practice) |

`heatmap_eq` needs no labels and runs on every session, like the rest of §1.

```bash
uv run python -m eye_pre_flash.plotting.similarities.heatmap_eq --monkey Faure
uv run python -m eye_pre_flash.plotting.similarities.heatmap_eq --monkey Nielsen --n-half 20
uv run python -m eye_pre_flash.plotting.similarities.heatmap_labels --monkey Nielsen
uv run python -m eye_pre_flash.plotting.similarities.heatmap_labels --monkey Faure --scope all --min-trials 10
```

Output: `heatmap_eq/<monkey>/` and `heatmap_labels/<scope>/<monkey>/` under
`eye_pre_flash/plotting/out/similarities/`, as `.png`; `heatmap_labels` also
writes a `.md` with the census, the within-maze table and the full matrix.

---

# Results: reliability vs. discriminability across k

> **These tables were computed with the window ending at `fix_start − 300 ms`,
> which the scripts no longer use** — the end moved to `fix_start` in commit
> `706c452`. Re-run before quoting them against current figures. Everything else
> below (20 splits, CV seed 0, mazes 1–6) still matches.

Two different questions get asked of the same 6×6 CV matrix:

- **Reliability** — the mean of the diagonal. "If I re-measured this maze's
  state-occupancy profile from an independent half of trials, would I get the
  same answer?"
- **Discriminability** — diagonal vs. off-diagonal, reported as the **ratio**
  diagonal-mean ÷ off-diagonal-mean. "Does this k actually tell mazes apart, or
  do different mazes look about as similar to each other as a maze looks to
  itself?" >1 means the codebook separates mazes; closer to 1 means it doesn't.

Higher is better throughout this document, in every table.

**Reliability alone is not evidence of a good k.** A degenerate k=1 codebook (one
state covering everything) would score a perfect 1.0 while carrying zero
information. Discriminability is the meaningful criterion here.

This is also **not** the criterion the production codebook size was chosen by —
that was codebook stability (held-out coverage × centroid reproducibility, which
peaked at the production K; the selection scripts have since been retired). This
section asks the narrower question of which k separates *maze types*, and the
two need not agree.

## Reliability (mean diagonal)

| k   | Faure occ | Faure occ-bin | Nielsen occ | Nielsen occ-bin |
| --- | --------- | ------------- | ----------- | --------------- |
| 2   | 0.822     | 0.942         | 0.954       | 1.000\*         |
| 3   | **0.974** | **0.986**     | **0.974**   | 0.984           |
| 4   | 0.962     | 0.971         | 0.916       | 0.968           |
| 5   | 0.971     | 0.979         | 0.935       | 0.973           |
| 6   | 0.973     | 0.980         | 0.900       | 0.957           |
| 12  | 0.891     | 0.937         | 0.888       | 0.948           |

By reliability alone, **k=3** wins for both monkeys on both features.

\* Nielsen k=2 occupancy-bin hit an exact 1.000 — a degenerate result
(near-zero-variance visit vector for at least one maze), not a genuine top score.

## Discriminability (diagonal ÷ off-diagonal)

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

**Nielsen:** discriminability climbs sharply from k=2/3 (ratio ≈1.0–1.06,
essentially no separation) up through k=4–6 (≈1.13–1.32) and keeps climbing out
to k=24 (1.35, the highest tested). k=3's reliability win is misleading here —
it's "reliable" mainly because it isn't resolving much.

**Faure:** a different, non-monotonic pattern. The ratio is highest at the
extremes tested — k=2 (1.178 occ) and k=12 (1.100 occ) — and dips through the
k=3–6 middle (≈1.04–1.06), where different mazes look nearly as similar to each
other as a maze looks to itself. Pushing to k=24 pulls the ratio back down from
k=12's peak (1.077 vs 1.100 occ; 1.040 vs 1.047 bin).

So **k=12 beats k=24 for Faure; k=24 beats k=12 for Nielsen.** The two monkeys
disagree on whether more resolution past 12 keeps paying off.

## Case study: maze 3 vs. maze 4 (Faure)

Raw split-half correlation for that specific pair, and the same value as a
**ratio, diagonal mean ÷ pair value**. Above 1 means the pair sits below typical
same-maze reliability (genuinely more separable — a discriminability win); below
1 means the pair looks *more* alike than a typical same-maze split-half, a red
flag.

| k   | occ maze3-vs-4 | occ diag mean | **occ ratio** | bin maze3-vs-4 | bin diag mean | **bin ratio** |
| --- | -------------- | ------------- | -------------- | -------------- | ------------- | -------------- |
| 2   | 0.863          | 0.822         | 0.953          | 0.915          | 0.942         | **1.030**      |
| 3   | 0.977          | 0.974         | 0.997          | 0.987          | 0.986         | 0.999          |
| 4   | 0.963          | 0.962         | 0.999          | 0.964          | 0.971         | 1.007          |
| 5   | 0.968          | 0.971         | 1.003          | 0.971          | 0.979         | 1.008          |
| 6   | 0.971          | 0.973         | 1.002          | 0.976          | 0.980         | 1.004          |
| 12  | 0.858          | 0.891         | **1.038**      | 0.915          | 0.937         | 1.024          |

**k=12 gives the largest genuine separation of maze 3 from maze 4**, both in raw
terms (lowest correlation, 0.858) and relative to its own diagonal (1.038,
occupancy — the highest tested). k=2's occupancy ratio (0.953, <1) is the
opposite signal: at k=2 these two mazes look *more* alike than a typical
same-maze pair, so k=2 is not a consistent answer across features (it wins on
occ-bin, 1.030, but loses on occupancy).

## Per-maze breakdown

Which individual maze is most reliable (its own diagonal cell) and most
discriminable (its diagonal ÷ the mean of its own off-diagonal row), computed at
each monkey's best-discriminating k above — **Faure k=12, Nielsen k=24**. Maze
grouping per `data.labeler.MAZE_GROUPS`: Faure hierarchical
1–3, sequential 4–6; Nielsen hierarchical 1–4, sequential 5–6.

| Maze | F k12 occ diag | offdiag | ratio | F k12 bin diag | offdiag | ratio |
| ---- | -------------- | ------- | ----- | -------------- | ------- | ----- |
| 1 | 0.882 | 0.809 | 1.090 | 0.930 | 0.893 | 1.042 |
| 2 | 0.893 | 0.789 | **1.132** | 0.942 | 0.886 | **1.062** |
| 3 | 0.870 | 0.812 | 1.072 | 0.926 | 0.893 | 1.037 |
| 4 | **0.908** | 0.841 | 1.080 | **0.945** | 0.910 | 1.039 |
| 5 | 0.896 | 0.805 | 1.113 | 0.938 | 0.893 | 1.050 |
| 6 | 0.898 | 0.806 | 1.114 | 0.939 | 0.892 | 1.052 |

| Maze | N k24 occ diag | offdiag | ratio | N k24 bin diag | offdiag | ratio |
| ---- | -------------- | ------- | ----- | -------------- | ------- | ----- |
| 1 | 0.819 | 0.617 | 1.326 | 0.887 | 0.766 | 1.159 |
| 2 | 0.801 | 0.644 | 1.244 | 0.876 | 0.789 | 1.110 |
| 3 | 0.850 | 0.655 | 1.297 | 0.907 | 0.798 | 1.137 |
| 4 | 0.782 | 0.674 | 1.161 | 0.876 | 0.812 | 1.079 |
| 5 | 0.863 | 0.628 | 1.374 | **0.925** | 0.793 | 1.166 |
| 6 | **0.880** | 0.478 | **1.841** | 0.923 | 0.691 | **1.335** |

**Nielsen's pattern is unambiguous:** mazes 5 and 6 — its two sequential-type
mazes — take *both* rank 1 and rank 2 in all four columns; nothing hierarchical
(1–4) breaks in anywhere. Nielsen's maze 6 is the single biggest standout number
in this whole analysis (discriminability 1.841, occupancy).

**Faure shows a different but still real pattern:** hierarchical maze 2 leads
discriminability in both features, with sequential maze 6 the consistent
runner-up (2nd in 3 of 4 columns). Reliability instead favors maze 4, which sits
right at Faure's own hierarchical/sequential boundary. So Faure doesn't split as
cleanly along maze type as Nielsen does — it reads more as "maze 2 discriminates
best, maze 6 is always close behind" than a type-level effect.

## Caveats

- **Single split-half seed (0) throughout.** The Faure k=12 vs k=24 flip and the
  k=3–6 dip are close enough in places (Faure ratios 1.04–1.06 across k=3–6) that
  a multi-seed check is worth running before treating small rank differences as
  real rather than split-sampling noise.
- **Nielsen k=2 occupancy-bin (diag = offdiag = 1.000) is degenerate**, not a
  result — excluded from any "best k" conclusion.
- **Reliability and discriminability rank k differently.** For "which k
  distinguishes maze types," discriminability is the criterion, not raw
  reliability.
- **The per-maze breakdown is computed at a single k per monkey.** It would look
  different at another k — it is not a k-independent property of the mazes.
