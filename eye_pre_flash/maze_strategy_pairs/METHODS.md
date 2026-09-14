# Single-maze H-vs-S gaze similarity: methods

The authoritative methodology record for `eye_pre_flash.maze_strategy_pairs`.
`STATS.md` is the shorter guide to reading a finished panel; this file is the
justification, including the things that are *not* controlled and why.

## 1. The question and the estimator

Within one maze, trials are labelled H (hierarchical) or S (sequential) by a
decoder applied to neural population activity (`data/labeler.py`; see
`eye_pre_flash/label_sources.py` for the two label sources). We ask whether
pre-flash gaze sampling depends on that label. Maze geometry is held fixed, so
a difference cannot be a visual confound.

Per session, per split:

1. Draw two **disjoint** random halves of exactly `m` trials from the H cell,
   and two disjoint halves of exactly `m` trials from the S cell.
2. Average each half into one `d`-dimensional feature vector.
3. Take all four Pearson correlations between an "a" half and a "b" half:

   ```
   R[H,H] = pearson(a_H, b_H)      R[H,S] = pearson(a_H, b_S)
   R[S,S] = pearson(a_S, b_S)      R[S,H] = pearson(a_S, b_H)
   ```

Average over 200 splits, then over sessions, and report

```
Δ = 0.5 * (r_HH + r_SS) − r_HS
```

against a within-(session, maze) label-shuffle permutation null.

The diagonal is each strategy's own split-half reliability: an estimate of how
similar a strategy's gaze pattern is to *itself*, which is the ceiling any
cross-strategy correlation could reach. The off-diagonal is the
cross-strategy correlation at the same half size.

## 2. The central design problem

**The diagonal is structurally advantaged.** In many plausible setups it
exceeds the off-diagonal for reasons that have nothing to do with strategy, so
a 2x2 where the diagonal wins is worth nothing unless every such reason has
been removed or measured. Five distinct mechanisms, handled differently:

### 2.1 Count asymmetry — removed

Pearson between two noisy means is attenuated by the noise in **both**
arguments. If `r_HH` correlates two 22-trial means while `r_HS` correlates a
22-trial mean against a 5-trial mean, the off-diagonal is attenuated more and
loses trivially.

Fixed by one half size for both cells, with no exceptions:

```
m = min(n_H, n_S) // 2
```

So all four half-means in every split average exactly `m` trials. The previous
rule (`matrix.session_half_sizes`) allowed a cell below `MIN_STABLE = 8` to
fall back to its own `n // 2`, producing exactly the asymmetry above; it was
flagged in the figure with an asterisk rather than fixed. That exception is
gone.

The cost is real and worth stating: the majority cell is thinned to the
minority cell's size, so it contributes `2m` of its `n_H` trials per split.
Averaging over 200 splits recovers most of the lost precision, since each
split uses a different subset.

### 2.2 Unequal split and session denominators — removed

A zero-variance half-mean in cell *i* NaNs row *i* of `R`. The previous code
accumulated each entry against its own denominator, so `r_ii` would be
averaged over fewer splits than `r_jj`, and `0.5 * (r_HH + r_SS)` would mix
two different split sets — no longer one procedure.

Now a split contributes **all four entries or none**, and a session is
admitted only when its whole 2x2 is finite. Session averaging uses a plain
mean, never `nanmean`, so no NaN can reach the average and the four entries
provably share one session set. Dropped splits and sessions are counted in the
CSV (`n_splits_dropped_total`, `n_sessions_dropped_nan`).

### 2.3 The arithmetic-vs-geometric mean floor — NOT removed, measured

This is the most important limitation, and the permutation null cannot see it.

Under classical test theory, with `rho_H`, `rho_S` the reliabilities of an
`m`-trial mean and `rho_true` the correlation between the two strategies' true
profiles:

```
r_HH ≈ rho_H     r_SS ≈ rho_S     r_HS ≈ sqrt(rho_H * rho_S) * rho_true
```

The off-diagonal is attenuated by the **geometric** mean of the two
reliabilities, while Δ subtracts it from the **arithmetic** mean. So even at
`rho_true = 1` — the two strategies sharing one identical gaze profile, the
very null this analysis argues against —

```
Δ = 0.5 * (rho_H + rho_S) − sqrt(rho_H * rho_S)
  = 0.5 * (sqrt(rho_H) − sqrt(rho_S))^2   ≥ 0
```

`r_HH = 0.8` against `r_SS = 0.4` puts that floor at 0.034; 0.8 against 0.1
puts it at 0.167.

Equal `m` removes the *count* source of `rho_H != rho_S`. It does not remove
the *within-cell variance* source: H and S need not be equally variable trial
to trial. And **the label shuffle destroys exactly the heterogeneity it would
need to calibrate** — shuffling makes both pseudo-cells draws from one
mixture, so `rho_H ≈ rho_S` in the null and the floor sits at ~0 there while
being positive in the data. It is not absorbed by `z` or `p`.

So it is reported instead (`pair.amgm_floor`):

| column | meaning |
|---|---|
| `delta_amgm_floor` | `0.5*(r_HH + r_SS) − sqrt(r_HH * r_SS)` — what unequal reliability explains on its own |
| `delta_beyond_floor` | `sqrt(r_HH * r_SS) − r_HS` — the part of Δ that survives it |
| `diag_imbalance` | `r_HH − r_SS` |

`delta_beyond_floor` is printed on every panel. It is still a raw difference of
correlations within one panel — not a reliability correction and not a
normalised ratio. The first two are NaN when either diagonal is negative,
which is routine under `mean_removed`; there, read `diag_imbalance`.

`checks.check_amgm_floor_is_measured` demonstrates the mechanism on synthetic
data where `rho_true = 1` by construction.

### 2.4 Intrasession drift — NOT removed by the null; separately controlled

Both halves of the H cell are random subsets of H's own trials, so they span
the same session time by construction. H-vs-S does not, if the animal switched
strategy partway through the session. Eye-tracker calibration drift then
inflates the diagonal for reasons unrelated to strategy.

The label-shuffle null cannot calibrate this either, and for a structural
reason: **shuffling H/S without regard to trial order destroys any real
time-clustering of the labels before the null is built.** The null distribution
is therefore assembled almost entirely from time-balanced pseudo-groups, so a
Δ driven entirely by "H and S occupy different parts of the session, and the
signal drifts" can still land far outside it. The permutation test answers
*"is this Δ unusual for any random 2-way split of this size?"*, not *"is it
explained by H/S's actual temporal positions?"*

Three escalating controls in `drift.py`:

**(a) Is the confound plausible here?** `label_time_association` tests, per
(session, maze), whether trial index differs between H and S trials:
Mann-Whitney U (distribution-free, no linearity assumption) and point-biserial
`r_pb` (a signed effect size, so the direction is visible). Reported per
session in `results_raw_pair.csv`, with the median `|r_pb|` on the panel. If H
and S are well interleaved in time in most sessions, (b) and (c) are
confirmatory; if strategy shifts systematically within a block, they are
load-bearing.

**(b) A time-defined surrogate null.** `delta_time` builds a two-group split
from **time alone**, using nothing about strategy, and pushes it through the
*same* estimator (`pair_matrix_from_pools` with substituted labels — the same
hook the permutation null uses, so it is provably the same code path):

* `early_late` — median split on trial index. Detects monotonic drift.
* `odd_even` — trial-index parity. Temporally interleaved, so it detects
  cyclic structure and, more usefully, **measures the estimator's own noise
  floor on real data** at this `m` and `d` with two genuinely exchangeable
  groups. It should sit at ~0; if it does not, the floor is not where the
  permutation null says it is.

*The surrogate is forced to use the H/S half size.* A median or parity split
yields two near-equal groups, so its own `min(n_0, n_1) // 2` would typically
*exceed* the H/S `m`, which the minority strategy caps. A larger `m` raises
every correlation, and `delta_time` would come out inflated for a reason
having nothing to do with drift. Holding `m` fixed is the same
match-the-counts discipline of §2.1, applied between the analysis and its
control. For `early_late` the forced size always fits (with `n = n_H + n_S`,
the smaller median group has `floor(n/2) >= min(n_H, n_S) >= 2m`); for
`odd_even` the groups can skew, so a session that cannot supply `2m` in both
parity groups is dropped and counted.

How to read it:

* `delta_time ≈ 0` — drift is not a meaningful force at these half sizes; the
  observed H/S Δ stands.
* `delta_time` comparable to the observed Δ — **the observed result cannot be
  read as strategy-specific** without (c).

**(c) Detrend and recompute.** `detrend_pools` regresses each trial's feature
vector on trial index within (session, maze) — the scope where the confound
lives — and keeps the residuals. Behind `--detrend`, default off, because it
is only needed when (a) and (b) say the confound is non-trivial. The decisive
pattern is **the H/S Δ surviving detrending while `delta_time` on the same
residuals collapses toward zero**: that says the effect is strategy-specific
rather than a drift artifact. Two caveats: the design includes an intercept,
so this also removes each session's own mean profile and is partly a
per-session `mean_removed` rather than orthogonal to the variant axis; and it
consumes `order + 1` degrees of freedom per dimension, which is not negligible
at `d = 5`.

**A rejected alternative, recorded so it is not re-proposed.** Stratifying the
half draw to sample evenly across session time was considered and rejected.
Two random halves of one cell's own trials are already temporally
indistinguishable in expectation, so stratifying reduces the split-to-split
variance of the *diagonal* but cannot fix a between-cell temporal mismatch: if
H's whole trial pool occupies different session time than S's, there is no
"late half of H" to sample. The confound lives at the H-vs-S comparison level
and has to be addressed there.

### 2.5 Fisher-z inflation — measured, and reported both ways

Correlations are averaged in Fisher-z space: `arctanh`, mean over kept splits,
unweighted mean over sessions, `tanh`. That is the correct aggregation rule for
correlations, but it is not neutral here. Expanding about the raw mean with
split variance `s^2`:

```
tanh(mean(arctanh r)) − mean(r)  ≈  s^2 * r / (1 − r^2)
```

odd in `r` and steeply increasing in `|r|`. The diagonal sits higher, so **it
gets the larger upward push and Fisher-z inflates Δ — in the direction of the
hypothesis**, by roughly 4–10% in plausible regimes. `z` and `p` largely
cancel it because the null uses the same rule, but not exactly (the inflation
is larger at larger `|r|`, and observed correlations exceed null ones).

So `r_raw` and `delta_obs_raw` — arithmetic means over the identical kept-split
and session sets — are carried alongside everywhere, with
`delta_fisher_minus_raw` in the CSV. **If the two ever disagree qualitatively,
believe the raw one**, which has no mechanism favouring the diagonal.

Δ is formed in **r-space**: back-transform each cell, then difference. A
z-space Δ would manufacture the artifact this whole file is about — `arctanh`
blows up near 1, so `0.5*(z_HH + z_SS) − z_HS` would report how close the
diagonal sits to the ceiling rather than the size of the gap (`r_diag`
0.90 → 0.95 moves the r-space Δ by 0.05 and the z-space Δ by 0.36).

### 2.6 Residual asymmetries, with their direction

* **Half disjointness.** `a_H` and `b_H` are drawn without replacement from
  one pool, so their noise components are negatively dependent, while `a_H`
  and `b_S` are independent. This **depresses the diagonal** — it works
  against the hypothesis. Not corrected.
* **Minority-cell exhaustiveness.** The minority cell uses ~all its trials
  (nearly an exhaustive partition, so its halves are anti-correlated and
  `r_SS` is depressed) while the majority subsamples freely. This **is**
  calibrated by the null, because the shuffle preserves both counts exactly.

The general rule, worth remembering: **the permutation null calibrates
everything that depends on the counts, and nothing that depends on the
within-cell variances or on trial timing.**

## 3. The unsymmetrised off-diagonal

`R[H,S]` and `R[S,H]` are reported separately rather than averaged into one
number written into both cells, so the 2x2 shows four distinct values.

**They are two exchangeable draws of the same population quantity, not two
directions of a meaningful axis.** Within a split, which half of a cell is "a"
and which is "b" is an arbitrary random assignment, drawn independently per
cell. So `|R[H,S] − R[S,H]|` (`cross_gap`, on every panel) is a statement about
how noisy the estimate is at this `m` — nothing more. **It must not be read as
evidence of real asymmetry between the strategies.**
`checks.check_cross_draws_exchangeable` asserts their means agree.

Reporting them separately buys transparency: the spread is visible instead of
hidden inside a mean. Δ still uses a single off-diagonal, the Fisher-z mean of
both draws, so it stays comparable to earlier runs; `delta_ab` and `delta_ba`
in the CSV give Δ from each single draw.

One consequence to be honest about: Δ's off-diagonal term averages two draws
while each diagonal term is a single correlation, so the off-diagonal has lower
split variance — which slightly compounds the Fisher-z inflation of §2.5. That
was invisible before; `cross_gap` now measures it.

## 4. Parameters

| | value | why |
|---|---|---|
| `N_SPLITS` | 200 | A single split is a noisy estimate of a noisy quantity. Deliberately package-local, **not** `plotting.similarities.common.N_SPLITS` (which is 20 and has four other consumers). |
| `R_CLIP` | `1 − 1e-6` | Two-sided. `cov/(a_sd*b_sd)` reaches `|r| = 1 + 2e-16`, which `arctanh` turns into NaN, and at `d = 5` exact `|r| = 1` is reachable. `|z| <= 7.25` against ~2.65 for the largest plausible real `r`, so it engages only on near-degeneracy. Two-sided because `mean_removed` makes negative `r` routine. Clipping caps the high-`|r|` diagonal more often, so it *reduces* Δ — conservative. Counted per entry. |
| `MIN_TRIALS` | 4 | Arithmetic floor for a cell to exist. |
| `MIN_HALF` | 5 | A 2-trial half-mean at `d = 5` is noise, and sessions are averaged unweighted so it would count as much as an `m = 20` session. A **session-inclusion** rule, not an exception to equal counts: H and S always share one `m`. `--min-half 2` for a sensitivity run. |
| `n_perm` | 1000 | `p` floor is `1/(n_perm+1)`; a tilde on the panel marks the floor. |
| sessions | unweighted | A session with more trials gets cleaner half-means and hence a higher diagonal, so n-weighting would amplify exactly the count bias equal `m` exists to remove. A pooled z-mean would do the same by the back door. |

## 5. Split-to-split spread

`split_deltas` returns a per-split, session-averaged Δ for the observed run
(not for the 1000 permutations, which would answer no question). From it:
`delta_split_sd`, `delta_split_p025/p975`, and
`delta_split_sem = sd / sqrt(n_splits_full)`.

**`split_sem_over_null_sd < 0.10` is the acceptance criterion** for "is 200
splits enough?" — residual split-draw noise in the headline Δ, relative to the
null's own width. If it is not met, raise `--n-splits`.

Two caveats:

* `mean_i(Δ_i) != Δ_headline`. The headline averages first and differences
  after a nonlinear `tanh`.
* Split index `i` is an arbitrary pairing across sessions. Harmless for the
  distribution (the draws are i.i.d. and independent across sessions, so any
  pairing has the same law), but the resulting SD is **not** a confidence
  interval on Δ across sessions or animals — it holds the data fixed and
  varies only the split draw. The between-session error model is deliberately
  not what this analysis tests.

## 6. Variants, and why `mean_removed` goes negative

`full`, `no_origin` and `mean_removed` (`variants.py`) are all swept. `full`
correlations pile up near 1 because every cell shares a dominant central-
fixation profile; the other two strip that profile, structurally
(`no_origin` drops the codebook state nearest the screen origin) or by fitting
(`mean_removed` subtracts the pooled grand-mean profile).

`mean_removed` makes the off-diagonal negative **arithmetically**. The grand
mean is essentially the shared profile, so what survives subtraction is each
cell's small residual; residuals sum to ~zero across cells by construction,
and a set of vectors summing to zero has mean pairwise correlation about
`−1/(n_cells − 1)`. It is **not** evidence that H and S gaze patterns oppose
each other, and absolute `r` stops being interpretable under this variant. The
diagonal-vs-off-diagonal read is relative, so it survives.

Its grand mean is fitted per `(monkey, feature, k, source)` — keyed on the
source because the fit mask comes from `WIDEST_SCOPE[source]`, which is
`top_four` for dendro and `top_ten` for svm. Keying one field short would
silently apply one source's grand mean to the other's panels.

Δ is **not comparable across features, variants, or K** — different transforms
and dimensionalities live on different scales. `z` is the comparable number,
being normalised by each panel's own null.

## 7. Figures

One figure per (monkey, maze, source, feature, variant), with K = 6 and K = 12
as side-by-side panels and independently computed statistics. Colour is a fixed
**0 → 1** scale on every panel and every variant, so a large
diagonal-vs-off-diagonal gap looks large and a small one looks small —
per-panel autoscaling would render a 0.02 span on a `full` panel as the entire
colour ramp. Values below 0 (routine under `mean_removed`) render in a flat
grey `under` swatch rather than clamping to the darkest blue; the printed
number still states the value.

Output: `out/<source>/<monkey>/<variant>/maze<M>_<feature>.png`, with
`results_pair.csv` (one row per maze × feature × K) and
`results_raw_pair.csv` (one row per session, including the drift association
statistics) beside them.

## 8. Verification

`python -m eye_pre_flash.maze_strategy_pairs.checks` — synthetic-data invariant
checks, no feature cache needed. No output regression is possible: the
half-size rule, NaN policy, symmetrisation, averaging rule and RNG consumption
order all changed, so nothing is bit-comparable to a previous run.

The load-bearing check is `check_relabel_exchangeable`: relabelling H<->S must
reverse both matrix axes and leave Δ *exactly* unchanged. If the estimator had
any built-in preference for the diagonal over the off-diagonal, or for one cell
over the other, it would fail there. `check_zero_signal_unequal_n` is the one
that would have caught the old unequal-half-size handicap.

## 9. Expected differences from the previous run

* **Every `r` falls and Δ shrinks**, because the majority cell is now thinned
  to the minority cell's half size. `null_sd` widens. **`z` may go down even
  though the estimator is now fair.** This is the fix working, not a
  regression.
* **`min_half = 5` requires `min(n_H, n_S) >= 10`**, well above the old
  `MIN_TRIALS = 4` floor. Expect visible coverage loss, worst for
  `dendro/publication` (2 sessions per monkey) and for Faure's thin sequential
  cells; some panels may go empty. Every drop is counted and printed.
* The thin-cell asterisk is gone from the figures, because the exception it
  flagged no longer exists. `m` is printed instead.
