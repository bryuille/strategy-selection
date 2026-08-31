# Eye-data strategy classifier

Predicts hierarchical (`0`) vs sequential (`1`) for a single trial from
pre-fixation eye data alone, with simple feature sets and one easily
interpretable metric: **balanced CV accuracy** (chance = 0.500).

Run everything with one command — it goes from `data/mat/` to every table
below, warming each cache in dependency order:

```bash
uv run python -m eye_data_classifier.pipeline
uv run python -m eye_data_classifier.pipeline --from decode-pub   # resume
uv run python -m eye_data_classifier.pipeline --only decode-all
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

**Only 13 of the 23 actually label** (full sweep completed on Engaging,
2026-08-31): the other 10 fail the labeler's sanity check — the same neural
cluster comes out as the majority for both maze 1 and maze 6, so there is no
hierarchical/sequential axis to read off. Of the 13 that do label, 5 more fail
the anchor-maze vetting below, leaving **8 analysed sessions: 4 per monkey**
(Faure: june_8, june_24, june_16, june_22; Nielsen: Nov_3, Oct_22, Nov_6,
Nov_1). Eligibility was a neural-SNR criterion; these two checks are what
actually certify the labels.

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
`eye_data_classifier.features`; see [../choosing_k.md](../choosing_k.md)).
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
four `publication` sessions pass.

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

Everything lands under `out/decoding/<scope>/`, as `.png`, `.md`, `.tex` and a
combined `results_raw.csv`:

| File | Contents |
| ---- | -------- |
| `counts_<Monkey>.md` | the census: per-maze trial counts by strategy, and which mazes support per-maze CV. **Read this first** |
| `permaze_<Monkey>` | per-maze decoding table (degrees features), pooled over mazes |
| `permaze_by_maze_<Monkey>` | the same CV numbers broken out per maze; both models per cell, each bolded against its own best |
| `crossmaze_<Monkey>` | across-maze decoding table (unit-H features), maze-identity reference in the last row |
| `results_raw.csv` | every number above in long format |

Reading order:

1. **`counts_<Monkey>.md`.** Strategy is close to a deterministic function of
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
