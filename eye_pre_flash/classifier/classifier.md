# Eye-data strategy classifier

Predicts hierarchical (`0`) vs sequential (`1`) for a single trial from
pre-fixation eye data alone, with simple feature sets and one easily
interpretable metric: **balanced CV accuracy** (chance = 0.500).

Run everything with one command — it goes from `data/mat/` to every table
below, warming each cache in dependency order:

```bash
uv run python -m eye_pre_flash.classifier.pipeline
uv run python -m eye_pre_flash.classifier.pipeline --from decode-pub   # resume
uv run python -m eye_pre_flash.classifier.pipeline --only decode-all
```

Stages: `convert` → `labels` → `features` → `decode-pub` → `decode-all`. Every
stage is a cache warm, so rerunning is cheap. For the cluster, see
[cloud.md](../cloud.md).

`data.labeler.CLUSTERING_SESSIONS` lists the **23** sessions eligible for a
strategy label — every session whose neural clustering separates maze 1 from
maze 6 at `snr_auc >= 0.95`, from
`zrefs/Dendogram_all/SNR_All_Sessions/all_session_snr_results.csv`. A label
needs that session's neural recording, so the `labels` stage converts, labels
and frees one session at a time. Analyses default to whichever eligible
sessions already have a label built, and print which they skipped.

**All 23 label.** Naming the two clusters hierarchical/sequential needs the
anchor mazes, and `data.labeler.CLUSTER_NAMING` picks how:

| Rule | Names sequential by | Labels |
| ---- | ------------------- | ------ |
| `relative` (default) | larger `p(cluster \| maze 6) − p(cluster \| maze 1)` | 23 |
| `reference` | absolute majority per anchor maze, error when one cluster wins both — what `Single_Trial_Statistics_Clustering_All.m:352` does | 13 |

There are only two ways to map two clusters onto two strategies, and the
relative rule picks whichever better matches "maze 1 hierarchical, maze 6
sequential". That is the best these anchors can do, so it always names and
never refuses.

The reference rule instead assumes a balanced clustering. When `fcluster`
returns a lopsided split — one cluster holding ~75% of all trials — that cluster
holds the majority of maze 1 *and* maze 6 however strong the maze effect is, and
the rule reads that as "the same cluster is both" and errors. That is what
rejected 10 of the 23. `june_12_g0` was refused at 91% cluster-1 occupancy in
maze 1 against 58% in maze 6: a 33-point maze effect, running *opposite* to the
absolute majority.

**The relative rule never contradicts the reference**, which is why it is the
default despite deviating from the published MATLAB. Wherever the reference
succeeds it returns the same answer: the reference succeeds only when cluster
*A* holds more maze-1 trials and *B* more maze-6, i.e. `p1(A) > ½ > p1(B)` and
`p6(B) > ½ > p6(A)`, whence `p6(B) − p1(B) > 0 > p6(A) − p1(A)` and *B* is named
sequential either way. Verified as well as proved — all four sessions whose
neural data is held locally rebuild bit-identical under both rules. Set
`--naming reference` to reproduce the published 13 exactly.

Naming quality varies, and the run says so rather than deciding for you. Each
session prints its maze-6-minus-maze-1 enrichment and a two-sided Fisher exact
p on the cluster × maze{1,6} table; above `WEAK_AXIS_ALPHA = 0.05` it is flagged
`WEAKLY SEPARATED`. Seven sessions are (p = 0.065 to 1.0) — their Ward split ran
along something other than the maze axis. They are still labelled: whether a
session is fit to analyse is decided downstream by the anchor-maze vetting
below, which scores labels against all five anchor mazes rather than two and is
therefore better informed. An individual maze coming out with a seemingly wrong
majority is not on its own a reason to discard a session.

Passing the SNR filter does not imply the clusters separate the anchor mazes:
`snr_auc` comes from a *supervised* maze-1-vs-maze-6 axis, while this 2-cluster
Ward split is unsupervised and need not align with it. The seven weakly named
sessions all score between 0.957 and 0.990.

Of the 23 that label, the anchor-maze vetting below decides which enter `all`;
`allplus` takes all 23. Eligibility was a neural-SNR criterion; the vetting is
what actually certifies the labels.

---

## The claims this analysis exists to support

Reviewer-response text, **verbatim** — the two drafted variants of the same
point. Do not paraphrase or reflow these when editing this file; the bracketed
slots are where the numbers from `out/decoding/` go.

> Third, we have now analyzed the animals' voluntary eye movements during the initial maze-viewing period before fixation. This analysis allows us to ask whether animals actively inspected the maze in a manner related to the subsequent strategy state, and whether the early neural state can be interpreted solely as a passive response to visual geometry. [Insert results here]. 

> Finally, we have now analyzed the animals’ voluntary eye movements during the initial maze-viewing period before fixation. This analysis asks whether animals actively sample the maze differently depending on the subsequently inferred strategy state. [Insert result here.] Such strategy-dependent inspection would provide behavioral evidence that the initial period involves active evaluation of the maze structure, rather than a passive visual response.

Both hinge on the same contrast: **active, strategy-dependent sampling**
versus **a passive response to visual geometry**. The two training regimes
below split exactly along that line — the per-maze tables hold geometry fixed
by construction, and the across-maze tables mitigate it with the unit-H warp
plus a maze-identity reference row.

---

## 1. Data and feature sets

Every feature describes one trial over the pre-fixation free-viewing window,
`fix_start − 1466 ms` through `fix_start`, and is extracted in one of two
coordinate **spaces**:

| Space | Coordinates | Used by | Why |
| ----- | ----------- | ------- | --- |
| `deg` | raw screen degrees | **per-maze** tables | geometry is constant inside a maze; warping would only distort gaze |
| `unith` | unit-H maze space: `data.attractor.to_maze` warps degrees by the trial's `h1`–`h6` arm lengths so every maze becomes the same unit H with exits at `(±1, ±1)` | **across-maze** tables | mitigates the visual-geometry differences a pooled classifier could otherwise exploit |

The warp is exact and invertible per trial, so `deg` coordinates are recovered
losslessly from the unit-H caches.

The **codebook** is a per-monkey k-means over gaze positions, at **K = 6 and
K = 12**, fit separately in each space (the unit-H books come from
`data.attractor`, the degree books are fit the same way in
`eye_pre_flash.classifier.features`).
Samples are assigned fixation-by-fixation to the nearest prototype within a
radius; gaze in no-man's-land is unassigned. Because the codebook is fit per
animal, state `3` for Faure is a different maze location than state `3` for
Nielsen — **all state-indexed features are only comparable within a monkey**,
which is why every table is per-monkey.

**The table rows** — four simple feature families, eight rows:

| Row | Dim | Represents |
| --- | --- | ---------- |
| Codebook occupancy (ms), K=6 / K=12 | 6 / 12 | **where gaze dwelt**: milliseconds in each codebook state |
| Codebook occupancy (binary), K=6 / K=12 | 6 / 12 | **which** states were visited, not how long: `occupancy > 0` as {0, 1} |
| State bigrams, K=6 / K=12 | 36 / 144 | **scan order**: state sequence collapsed to runs (`3 3 3 7 7 3` → `3 7 3`, so dwell time drops out), then the proportion of each ordered pair |
| Gaze heatmap, 5×5 / 10×10 | 25 / 100 | **spatial gaze density**, codebook-free: normalised 2-D histogram of gaze position |

Everything is cached under
`data/processed/<Monkey>_clf2_k<K>_<space>_s1466_e0.npz`; no block has missing
values.

---

## 2. Models

Two models, not eight — one linear, one nonlinear, so a table answers "is
there a linear boundary?" and "does flexibility buy anything?" without
inviting a pick-the-best-of-eight selection effect. Hyperparameters are fixed,
not tuned.

- **Logistic (L2)** — `LogisticRegression(C=1.0, class_weight="balanced")`
  behind a `StandardScaler` (the blocks mix milliseconds, {0,1} flags and
  proportions).
- **Random forest** — `RandomForestClassifier(n_estimators=300,
  min_samples_leaf=2, class_weight="balanced")`. Trees split on thresholds, so
  it skips the scaler. Grown deep, it fits training data almost perfectly —
  its train column near 1.000 with a much lower CV column is what overfitting
  looks like, and that gap is printed on purpose.

`class_weight="balanced"` is load-bearing: the strategy split inside a maze
can be as lopsided as 12/50, and an unweighted fit saturates to the majority
class.

---

## 3. Analysis design

**Grouping, outermost to innermost — two scopes, then monkey, then regime:**

| Scope | Sessions | What it is |
| ----- | -------- | ---------- |
| `publication` | the 4 sessions the published clustering was defined on (`snr_auc` ranks #2–#5 of 61) | the strictest audience-facing set — but only **2 sessions per monkey** |
| `all` | the 23 clearing `snr_auc >= 0.95`, label-vetted (8 in practice) | the primary result |
| `allplus` | the same 23, **without** the label vetting (13 in practice) | robustness check: vetting also removes the sessions with the most within-maze strategy variability, so this shows what those sessions add — at the cost of admitting labels that sit near coin-flip against the anchor map |

Both run, always. An effect present in `all` and absent in `publication` is an
effect that needs the wider pool to see, and saying so is more honest than
picking whichever scope reads better.

**Label-quality vetting.** Eligibility (`snr_auc >= 0.95`) does not guarantee
the labels came out usable, so both scopes vet each labelled session against
the known maze→strategy map: the anchor mazes 1–3 are solved hierarchically
and 5–6 sequentially (maze 4 is the boundary maze and is left unscored — it is
genuinely mixed for Nielsen, both publication sessions included, and
near-mixed for Faure). A session enters the analysis only when its anchor-maze
labels agree with that map on at least `MIN_LABEL_AGREEMENT = 0.70` of trials;
below that the labels sit near coin-flip in every maze, which is what a failed
neural labelling looks like, and such trials are label noise that can only
dilute the tables. The observed distribution is bimodal (0.76–1.00 vs
0.54–0.65), so the threshold sits in the gap and no session is borderline.
Dropped sessions are printed with their agreement at the top of every run. All
four `publication` sessions pass. §6 lists the eight that pass, their per-maze
label composition, and how they compare with the published clustering.

Within a monkey, trials are **concatenated across the scope's sessions** and
two training regimes run:

- **Per-maze** (`permaze_<Monkey>`, `deg` features): one classifier per maze,
  trained and cross-validated on that maze's pooled trials. Maze identity
  cannot help by construction — the model never sees a second maze. A maze
  enters only when both strategies appear ≥ 5 times in it; the census table
  says which. The table's figure is the **trial-weighted average of
  within-maze balanced accuracies** — pooling raw predictions across mazes
  instead would let maze identity back in through each model's class prior and
  inflate the score, which is exactly the confound this regime removes.
- **Across-maze** (`crossmaze_<Monkey>`, `unith` features): one classifier on
  all mazes pooled. The unit-H warp mitigates geometry, and the
  **`Maze identity (reference)`** row — one-hot maze id, no eye data at all —
  shows how much accuracy plain maze identity buys anyway. An eye-feature row
  must clear that reference before it can claim gaze adds information beyond
  which maze was on the screen.

**Cross-validation and metric.** Stratified 5-fold CV (fewer folds only when a
class has under 5 trials), fixed seed. The metric is **balanced accuracy** —
the mean of per-class recall — so chance is exactly 0.500 however lopsided the
split. Each model reports two columns:

- `train` — accuracy on the training folds themselves. High train with low CV
  = overfitting; both near chance = no signal in the feature set.
- `CV` — accuracy on held-out folds; each trial is scored once. The number to
  quote. In the per-maze regime both columns are the trial-weighted average of
  within-maze scores, per the note above.

---

## 4. How to read the output

Everything lands under `out/decoding/<scope>/`, as `.png` plus a combined
`results_raw.csv`. Tables render as PNG only:

| File | Contents |
| ---- | -------- |
| `counts_<Monkey>.png` | the census: per-maze trial counts by strategy, and which mazes support per-maze CV. **Read this first** |
| `permaze_<Monkey>` | per-maze decoding table (degrees features), pooled over mazes |
| `permaze_by_maze_<Monkey>` | the same CV numbers broken out per maze; both models per cell, each bolded against its own best |
| `crossmaze_<Monkey>` | across-maze decoding table (unit-H features), maze-identity reference in the last row |
| `results_raw.csv` | every number above in long format |

Reading order:

1. **`counts_<Monkey>.png`.** Strategy is close to a deterministic function of
   (session, maze), so many mazes are heavily one-sided; the census shows how
   much data each number stands on.
2. **Per-maze CV column.** The cleanest read of strategy-dependent sampling:
   geometry fixed, chance 0.500.
3. **Across-maze CV vs the maze-identity reference.** Accuracy above the
   reference is information beyond visual geometry; accuracy at or below it is
   maze proxying, however high the number.
4. **Train vs CV gap**, per model. The forest's train column near 1.000 is
   expected; what matters is whether its CV column holds up.
5. **Across monkeys last.** One monkey meeting a bar is a result about one
   animal.

**Caveats, stated once.** Trials are pooled across sessions, so a CV fold can
hold trials from a recording day the model trained on. Strategy is strongly
session-clustered, which means these tables read as "is the label decodable
from gaze on these recording days", not as generalisation to unseen days — a
day-level claim would need session-held-out evaluation. And the `publication`
scope is two sessions per monkey; treat it as a consistency check on the
audience-facing sessions, not as the powered result.

---

## 5. Pairwise maze-identity decoding (`pairwise.py`)

A different question, and **no strategy labels are involved**: can a classifier
tell *which maze was on the screen* from where the animal looked? Maze identity
is known for every trial without any neural recording, so this runs on every
trial of a scope's sessions — including the strategy-unlabelled ones the
decoding tables drop. (`--labelled-only` restores their exact trial set when the
two need comparing cell for cell.)

```bash
uv run python -m eye_pre_flash.classifier.pipeline --only pairs-all
uv run python -m eye_pre_flash.classifier.pairwise --scope all --space unith
```

It exists because "gaze does not predict strategy" is only worth reading next to
evidence that these features detect *anything*. If gaze cannot even distinguish
the mazes, the strategy nulls in §3 are uninformative about behaviour and merely
report weak features.

**Pairwise, not 6-way.** Each of the **15 maze pairs** gets its own binary
classifier — maze *i* vs maze *j* — instead of one 6-way multiclass fit. Three
reasons:

1. Chance is **0.500**, the same as every table in §3, so the numbers are
   directly comparable to the strategy results rather than sitting against an
   awkward 0.167.
2. A single multiclass accuracy cannot distinguish "no maze is separable" from
   "only mazes 3 and 4 are confusable". The matrix shows which.
3. The **gradient** across cells is the actual result. Geometrically extreme
   pairs (1 vs 6) should decode far above neighbouring pairs (1 vs 3). Graded
   discriminability tracking geometry is much stronger evidence that gaze
   follows the maze than any one number exceeding chance.

The matrix is symmetric (*i* vs *j* is the same problem as *j* vs *i*). Its
**diagonal is the split-half null**: the same classifier asked to separate one
maze's own trials into two random halves, averaged over 5 halvings. A maze
cannot be told from itself, so this measures the floor the off-diagonal cells
are read against instead of assuming it is 0.500.

Read the diagonal first. Near 0.500 it says the CV and the balanced-accuracy
metric are behaving on that feature set. Well above 0.500 it says a random half
of one maze's trials is separable from the other half, which can only come from
structure the split does not control — session or day effects, drift across the
recording — and that same structure inflates the off-diagonal cells by an
unknown amount. Observed: every feature set lands at **0.488–0.510** in both
monkeys, against off-diagonal means of 0.553–0.659, so the floor is where it
should be. Row means and ranges exclude the diagonal throughout.

**Features are in `deg`, not `unith`** — the one place in this package where
that is the entire point. The question is whether gaze tracks the geometry
actually on the screen, and the unit-H warp exists to remove that geometry.
`--space unith` is therefore the control: a much flatter matrix is the warp
working as intended, and is the direct check on the assumption behind the
across-maze regime in §3.

Metric, CV, models and scopes are unchanged from §3 — balanced accuracy,
stratified 5-fold, fixed seed, the same two models, the same three scopes. The
label-quality vetting still selects the sessions, purely so the pools match the
decoding tables; it is a *label* filter and has no bearing on maze identity.

Accuracies are **uncorrected over the 15 cells**, so read a single marginal cell
with that in mind. The gradient is the claim, not any one pair.

### Output

Everything lands under `out/pairwise/<scope>/`. Tables render as **PNG only** —
the `out/` trees carry figures and `results_raw.csv`, nothing else:

| File | Contents |
| ---- | -------- |
| `counts_<Monkey>.png` | trials per maze in the pool. **Read this first** |
| `pairs_<Monkey>_<feature>.png` | the 6×6 accuracy matrix, **one per feature set**, coloured like the [similarity.md](../../similarity.md) maze matrices |
| `pairs_<Monkey>_<feature>_table.png` | the same matrix as numbers, plus each maze's mean against the other five |
| `summary_<Monkey>` | all eight feature sets collapsed to the mean and min–max of their 15 pairwise accuracies, both models — for ranking the feature sets at a glance and for the forest's train/CV gap |
| `results_raw.csv` | every pair × feature set × model in long format |

`<feature>` is the block and what parameterises it — `occ_ms_k6`, `occ_ms_k12`,
`occ_bin_k6`, `occ_bin_k12`, `bigram_k6`, `bigram_k12`, `heatmap_5x5`,
`heatmap_10x10`. The heatmaps are codebook-free, so they are named by their grid
rather than by a K that does not apply to them.

**The matrices are drawn for the logistic only.** One model keeps each matrix a
plain grid of numbers, and the forest's contribution in §3 is its train/CV gap
as an overfitting readout — something a matrix has no room to show. It still
runs, into `summary_<Monkey>` and the raw csv, next to the logistic.

Two reading aids are deliberate. Every matrix is anchored at **vmin = 0.5**, so a
matrix with no signal reads as washed out instead of being stretched to look
structured. And one monkey's eight matrices share a **single colour ceiling**, so
they can be compared with each other by eye — a weak feature set stays visibly
weak rather than filling its own range.

---

## 6. Label composition by session and maze (`all` scope)

What the labels the `all` scope analyses actually look like, broken out per
session — the census in `counts_<Monkey>.png` pools the scope's sessions, so
the pooled row here is that census and the rows above it are what it hides.
Cut 2026-09-04, the eight sessions that pass vetting: the four `publication`
sessions plus `june_16_g0`, `june_22_g0`, `Nov_6_g0`, `Nov_1_g0` (the other 15
of the 23 drop at agreements 0.486–0.653). Each cell is the **fraction
hierarchical**, with `hierarchical/total` trials; `Agreement` is the
anchor-maze score the §3 vetting thresholds at 0.70.

**Faure**

| Session | m1 | m2 | m3 | m4 | m5 | m6 | Agreement |
|---|---|---|---|---|---|---|---|
| `june_24_g0` | 0.98 (43/44) | 0.70 (35/50) | 0.92 (48/52) | 0.19 (12/62) | 0.20 (18/90) | 0.16 (11/69) | 0.839 |
| `june_8_g0` | 1.00 (44/44) | 0.83 (34/41) | 1.00 (43/43) | 0.02 (1/49) | 0.03 (2/62) | 0.00 (0/54) | 0.963 |
| `june_16_g0` | 0.85 (51/60) | 0.05 (2/37) | 0.74 (34/46) | 0.03 (2/58) | 0.09 (5/55) | 0.04 (3/70) | 0.761 |
| `june_22_g0` | 0.93 (43/46) | 0.80 (39/49) | 0.89 (47/53) | 0.34 (15/44) | 0.40 (21/52) | 0.17 (11/65) | 0.808 |
| **pooled** | **0.93** (181/194) | **0.62** (110/177) | **0.89** (172/194) | **0.14** (30/213) | **0.18** (46/259) | **0.10** (25/258) | |

**Nielsen**

| Session | m1 | m2 | m3 | m4 | m5 | m6 | Agreement |
|---|---|---|---|---|---|---|---|
| `Nov_6_g0` | 0.94 (51/54) | 0.75 (42/56) | 0.86 (59/69) | 0.42 (22/52) | 0.03 (2/62) | 0.00 (0/79) | 0.909 |
| `Oct_22_g0` | 0.71 (44/62) | 0.64 (44/69) | 0.74 (57/77) | 0.26 (21/82) | 0.03 (3/91) | 0.01 (1/78) | 0.822 |
| `Nov_3_g0` | 1.00 (65/65) | 0.99 (85/86) | 1.00 (82/82) | 0.52 (44/85) | 0.01 (1/96) | 0.00 (0/84) | 0.995 |
| `Nov_1_g0` | 0.88 (46/52) | 0.80 (43/54) | 0.76 (48/63) | 0.56 (45/81) | 0.18 (14/79) | 0.06 (5/88) | 0.848 |
| **pooled** | **0.88** (206/233) | **0.81** (214/265) | **0.85** (246/291) | **0.44** (132/300) | **0.06** (20/328) | **0.02** (6/329) | |

### Against the expected maze→strategy map

Mazes 1–3 hierarchical, 5–6 sequential, per session rather than pooled over
the scope. Two deviations, and only one of them is a surprise:

- **`june_16_g0`, maze 2 at 0.05** — a clean majority the wrong way, not a
  near-tie. Its mazes 1 and 3 are ordinary (0.85, 0.74), so this is maze 2
  specifically, and it is the only anchor-maze majority inversion in the eight
  sessions. It still enters `all` because `MIN_LABEL_AGREEMENT` scores pooled
  anchor trials rather than per-maze majorities; at 0.761 it is the lowest of
  the eight, well clear of the 0.70 threshold and of the 0.54–0.65 band the
  dropped sessions occupy. Per §1's note on weak naming, one maze with a wrong
  majority is not on its own a reason to discard a session.
- **Nielsen maze 4 is not hierarchical-majority** — 0.44 pooled, and
  hierarchical in only two of four sessions, both barely (0.52, 0.56) against
  0.42 and 0.26 the other way. Expected: maze 4 is the boundary maze, which is
  exactly why §3 leaves it out of `ANCHOR_MAJORITY`. So a claim that Nielsen
  solves 1–**4** hierarchically is not supported by these labels; 1–3 is.

Everything else lands as expected. Faure 4–6 and Nielsen 5–6 are
sequential-majority in every session, the closest to a tie being Faure
`june_22_g0` maze 5 at 0.40. Faure maze 2 pooled (0.62) is dragged down almost
entirely by `june_16_g0`; the other three sessions sit at 0.70–0.83.

### Agreement with the four-session clustering in `zrefs/`

`zrefs/Dendogram_all/<Monkey>/_Cluster_Distributions_PCA_All_<session>.mat`
holds the published clustering for the four `publication` sessions, as
`cluster_distribution` (2 × 6 counts) and `cluster_trial_ids` (the trial ids
per cluster × maze). Naming the reference's clusters by the repo's rule — the
cluster with more maze-1 trials is hierarchical — the two agree per trial:

| Session | Per-trial agreement | Per-maze counts |
| ------- | ------------------- | --------------- |
| `june_24_g0` | 367/367 | identical in all 6 cells |
| `june_8_g0` | 293/293 | identical in all 6 cells |
| `Nov_3_g0` | 498/498 | identical in all 6 cells |
| `Oct_22_g0` | **448/459** | 11 trials shifted, all reference-hierarchical → ours-sequential |

So three of the four publication sessions reproduce the published clustering
exactly, trial for trial, including the cluster→strategy naming. `Oct_22_g0`'s
11 trials (2.4% of the session) land as m3 −1, m4 −7, m5 −2, m6 −1
hierarchical. **No maze's majority changes**: on the reference counts maze 4
reads 28/82 = 0.34 hierarchical against our 0.26, still sequential-majority,
and every other maze moves by at most 0.03. The Nielsen maze-4 conclusion above
therefore holds under the reference clustering too.

That gap was traced on 2026-09-02 and is **not** tie-breaking, row order,
floating point, the `neuron_ok` filter or a different cut of the same tree —
all tested and ruled out; the reference partition is not the leaf set of any
node in our dendrogram. `SNR_All_Sessions/all_session_snr_results.csv` records
the same neuron and maze-1/maze-6 trial counts as ours (Oct_22: 71 neurons, 62,
78), so the feature matrix has the same shape, trials and neurons with
different *values* — the leading hypothesis is that Oct_22's stored mat was
built from a slightly different export of that session's firing rates. The
decisive test would be rerunning `Single_Trial_Statistics_Clustering_All.m` on
the current Oct_22 neural mat.

When reading these mats, **skip cells whose `MATLAB_empty` attribute is set** —
a v7.3 empty cell stores its dimensions as data, so reading it naively invents
trial ids 0 and 1 and fabricates disagreements. Validate each cell's length
against `cluster_distribution`.
