# Reading these figures

Each figure is one (monkey, maze, feature, variant, label source) cell of
`eye_pre_flash.maze_strategy_pairs`, carrying **both K = 6 and K = 12** as
side-by-side panels. This is the slide version; `METHODS.md` is the full
methodology, including the confounds that are measured rather than removed.

## The matrix

A 2x2 grid per panel, rows/columns `{maze}H` (hierarchical) and `{maze}S`
(sequential):

* **Diagonal** (`HH`, `SS`) — split-half reliability: two disjoint random
  halves of that strategy's own trials correlated against each other. This is
  the ceiling — how similar a strategy's gaze pattern is to *itself* across a
  fresh split, i.e. how much is real structure versus split noise.
* **Off-diagonal** (`HS`, `SH`) — cross-strategy correlation, at the **same
  half size** as the diagonal. If gaze sampling really depends on strategy
  (not just on the maze, held fixed here), this sits below the ceiling.
* **The two off-diagonal cells are not the same number**, because the estimate
  is no longer symmetrised. They are **two exchangeable draws of one
  quantity**, not two directions of an axis: within a split, which half of a
  cell is "a" and which is "b" is an arbitrary coin flip. So how close they sit
  is a QC read on how noisy the estimate is — **not** evidence of asymmetry
  between the strategies.
* **`n = ...`** under each diagonal cell is the pooled trial count for that
  strategy across every session used, not the count behind any one
  correlation. `m` in the footer is the per-split half size, which is what
  actually sets the precision.

Every panel uses a **fixed 0–1 colour scale**, so a big gap looks big and a
small one looks small. Cells below 0 — routine under `mean_removed`, see
below — render in flat grey to mark them off-scale; read the printed number.

## The numbers under each panel

* **Δ (delta)** = `0.5·(r_HH + r_SS) − r_HS`, where `r_HS` is the mean of the
  two cross draws. The raw gap between the reliability ceiling and the
  cross-strategy correlation. **Not comparable across features, variants, or
  K** — different transforms and dimensionalities live on different scales.
* **z** = `(Δ_observed − null_mean) / null_sd`, the null being built by
  reshuffling H/S labels within each (session, maze) and recomputing Δ. This
  **is** comparable across panels, being normalised by each panel's own chance
  variability. It is the number to trust when comparing strength between
  panels.
* **p** — two-sided fraction of shuffles producing a gap at least as extreme.
  A leading `~` means no shuffle beat the observed gap, so `p` is reported at
  the smallest value the permutation count allows (`1/(n_perm+1)`) — a
  statement about the test's resolution, not a measurement. It means "at least
  this significant"; `z` says by how much.
* **beyond floor** = `sqrt(r_HH·r_SS) − r_HS`. **Read this next to Δ.** Because
  the off-diagonal is attenuated by the *geometric* mean of the two
  reliabilities while Δ subtracts it from the *arithmetic* mean, unequal H/S
  reliability produces a positive Δ on its own — even if the two strategies
  share one identical gaze pattern. The permutation null cannot detect this,
  because shuffling makes both pseudo-cells equally reliable. `beyond floor` is
  the part of Δ that survives. If it is near zero while Δ is large, the result
  is explained by the diagonal imbalance, not by strategy. NaN under
  `mean_removed`, where `r_HH−r_SS` is the number to read instead.
* **r_HH−r_SS** — the diagonal imbalance driving that floor.
* **Δ_time: early/late, odd/even** — the same estimator run on a split defined
  by **time alone**, nothing about strategy. `early/late` is a median split on
  trial index; `odd/even` is parity, which is temporally interleaved and so
  measures the estimator's own noise floor on real data. Both use the same `m`
  as the H/S analysis. Near zero means intrasession drift is not driving
  anything. **Comparable to Δ means the result cannot be read as
  strategy-specific** — the label-shuffle null does not cover drift, because
  shuffling destroys the labels' time-clustering before the null is built.
* **|HS−SH|** — spread between the two cross draws. QC only.
* **r_pb** — median |point-biserial| of label against trial index across
  sessions: how much H/S is confounded with time-in-session. Small means the
  drift concern is close to moot for this panel.
* **m** — half size, identical for both cells by construction. Δ's magnitude
  scales with it, which is another reason Δ is not comparable across panels.
* **d** — feature dimensionality. K = 6 and K = 12 give different `d`, so
  their `r` values are not comparable to each other.

## Caveats that don't fit on a slide but matter

* **One maze, one monkey per figure.** A result says nothing about other mazes
  or the other animal — check those panels directly before generalising.
* **The two K panels are not two independent replications.** Same trials, same
  labels, different codebook resolution.
* **`dendro/publication` and `svm/top_ten` are different session pools and a
  different label source** (Ward clustering vs. a linear SVM, both fit on
  neural population activity — see `data/labeler.py`). Agreement between them
  is informative and so is disagreement, but neither is a rerun of the other
  with more data.
* **Multiple panels, one convention.** Every panel is its own permutation test;
  `p <= 0.05` on any single one carries the usual multiple-comparisons caveat
  once you are scanning several side by side.
* **`mean_removed` pushes the off-diagonal negative by arithmetic, not
  biology.** Subtracting the pooled grand-mean profile leaves residuals that
  sum to ~zero across cells, and vectors summing to zero have mean pairwise
  correlation about `−1/(n_cells−1)`. Absolute `r` is not interpretable under
  this variant; the diagonal-vs-off-diagonal read is relative, so it survives.
  Read `no_origin` — a structural rather than fitted way of stripping the same
  common profile — alongside it. If the two agree, the finding is robust; if
  only the fitted one shows an effect, distrust it.
* **These numbers are not comparable to figures from before the estimator
  rewrite.** Every `r` fell and Δ shrank, because the majority cell is now
  thinned to the minority cell's half size, and `null_sd` widened — so **`z`
  may be lower than an older panel even though the estimator is now fair.**
  Coverage also shrank: a session now needs `min(n_H, n_S) >= 10`.
