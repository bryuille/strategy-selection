# Maze x decoded-strategy occupancy similarity

A 12x12 split-half CV correlation matrix over `(maze, strategy)` cells —
mazes 1-6 hierarchical, then mazes 1-6 sequential — built from codebook
occupancy vectors rather than raw gaze maps. Not to be confused with the
`part_corr` name in older docs: see §2.

```bash
uv run python -m eye_pre_flash.corr.run
uv run python -m eye_pre_flash.corr.run --source svm --scope allplus --monkey Faure
```

## 1. The claim this exists to support

The 6x6 maze-only similarity matrices (`similarity.md`) cannot separate
strategy from geometry: strategy is close to a deterministic function of
`(session, maze)`, so "mazes 4-6 resemble each other" and "sequentially-solved
trials resemble each other" are the same statement about the same trials.
Holding maze type fixed and splitting each maze's trials by a *decoded neural*
strategy label removes that confound by construction. The decisive read is
the boxed cells — `(maze m, H)` vs `(maze m, S)` — where geometry is identical
on both sides, so anything below that maze's own split-half reliability is
strategy-dependent gaze sampling, not a visual confound.

This revives the retired `eye_pre_flash.plotting.similarities.heatmap_labels`
(`git show HEAD:eye_pre_flash/plotting/similarities/heatmap_labels.py`), which
did the same thing against 102,400-dim spatial gaze maps and was dropped for
a coverage problem (§7). `corr` uses low-dimensional codebook-occupancy
vectors instead, computes every cell regardless of count rather than
requiring both strategies to clear a coverage floor, and adds a second,
supervised label source.

## 2. Not `part_corr`

Earlier drafts of `README.md` and `similarity.md` referenced a planned
`eye_pre_flash/part_corr/` package — single trials scored against a
half-split template of their group, column-centred to remove a
template-quality confound. That package was never built; `corr` absorbed its
doc references (see `similarity.md`'s "Splitting a maze by decoded strategy"
section). The method here is the group-vs-group split-half CV described
throughout this document, not the template-scoring method those older drafts
described.

## 3. Data and features

Three feature blocks, all from
`eye_pre_flash.classifier.features.load_features` (already cached, no new
extraction):

| Feature (`--feature`) | Cache block | d |
| --- | --- | --- |
| `occupancy` | `occ_ms` — seconds per codebook state | K |
| `occupancy_bin` | `occ_bin` — visited/not-visited per state | K |
| `bigram` | ordered state-run pair proportions | K² |

`K in {6, 12}` (`--k`). Window: `fix_start - 1466 ms -> fix_start`
(`eye_pre_flash.plotting.plot_io.window_label`).

## 4. The dynamic-range problem and the two variants

The existing 6x6 occupancy matrices sit compressed against 1.0
(`similarity.md`, k=12):

| | diag | off-diag | total maze effect |
| --- | --- | --- | --- |
| Faure occupancy | 0.891 | 0.810 | 0.081 |
| Faure occ-bin | 0.937 | 0.894 | 0.043 |
| Nielsen occupancy | 0.888 | 0.716 | 0.172 |
| Nielsen occ-bin | 0.948 | 0.855 | 0.093 |

Every cell shares a dominant common gaze profile, so all r pile up in a
narrow band, and a within-maze strategy effect must be smaller than the
between-maze effect above it. The sampling SD of a Pearson over 12 points is
roughly +-0.06, so for Faure occ-bin the noise on one cell can exceed the
whole effect being differenced. (These particular tables carry a stale-window
banner in `similarity.md` — computed before the window's end moved to
`fix_start` — so treat the numbers as indicative; the compression itself is
structural, not a window artifact.)

Two variants (`--variant`) test whether central fixation is the cause:

- **`full`** — the cached vectors, unmodified. Comparable to the existing 6x6
  figures.
- **`no_origin`** — drop the codebook state nearest the screen origin (0, 0):
  `occ_ms`/`occ_bin` lose one dimension, `bigram` loses every ordered pair
  touching it.
- **`pc1_removed`** — project out the leading principal component, fit once
  per (monkey, feature, k, space) over the widest scope's trials (not per
  session or per cell — a direction that differed between sessions would put
  each session's matrix in a different subspace, exactly the cross-session
  comparability problem `heatmap_labels`' comparable-core machinery existed
  to fix). A removed direction makes the mean off-diagonal r go negative;
  that is arithmetic, not a finding.

`examples_dims.csv`'s `pc1_explained_var` row is the direct answer to "was it
central fixation": in one spot check (Nielsen occupancy k=12 unith) PC1
explained only ~16% of the variance and removing it widened the off-diagonal
spread only modestly (std 0.053 -> 0.071) — central fixation is present but
is not the whole story, worth checking per (monkey, feature, k, space) rather
than assumed.

## 5. `deg` vs `unith`

Both are emitted (`--space`). The boxed cells hold geometry fixed, so `deg`
(raw screen degrees) is the honest space there — the unit-H warp
(`data.attractor.to_maze`) is a cross-maze device, and applying it inside one
maze only distorts gaze. The quadrant reads (same strategy, different maze)
need `unith`, or they partly measure geometry rather than strategy. This is
the same argument as `eye_pre_flash.classifier.decoding.REGIMES`
(`per-maze -> deg`, `across-maze -> unith`).

## 6. The two label sources

`--source dendro` (default alongside `svm`, i.e. both run unless `--source`
is given) — the existing Ward-clustering labels
(`data.labeler.build_strategy_choices`), vetted against all five anchor
mazes (`eye_pre_flash.classifier.labels.ANCHOR_MAJORITY`,
`MIN_LABEL_AGREEMENT = 0.70`).

`--source svm` — a maze-1-vs-6 linear SVM (`data.labeler.build_svm_choices`)
fit on the same 3-PC neural space the dendrogram clusters, under the
assumption that mazes 1 and 6 anchor the ends of the strategy spectrum. Trials
from those two mazes get 5-fold stratified CV with an out-of-fold prediction;
every other trial (mazes 2-5) gets the **mode** across the 5 fold models'
predictions. This is a supervised alternative that can track a nonlinear
session shape the dendrogram's 2-cluster Ward split cannot.

**Grading the SVM by agreement with the anchor map is circular** (it was
*trained* to say maze 1 = H, maze 6 = S), so it is not anchor-vetted at all.
Instead: the anchor-minority cells `1S` and `6H` are **skipped outright**
(`eye_pre_flash.corr.labels.SOURCE_SKIP_CELLS`), since the classifier's own
training axis makes them near-empty by construction even out-of-fold; and the
only external validity check reported per session
(`dendro_agreement_2345` in `results_raw.csv`) is the SVM's agreement with the
dendrogram labels on mazes 2, 3, 5 — the mazes neither fold ever trains or
tests on. State this plainly when reading SVM results: the labels are
validated (via `svm_auc`, held-out ROC-AUC on the anchor split) exactly where
the analysis does not need them, and have only a soft cross-check exactly
where it does.

## 7. Session selection

Three scopes (`--scope`), same names for both sources but **different session
lists** for `svm`:

| Scope | `dendro` sessions | `svm` sessions |
| --- | --- | --- |
| `publication` | the 4 original-clustering sessions | same 4 |
| `all` | 23, anchor-agreement vetted | **top 4 per monkey by `snr_auc`, 8 total** — a strict superset of the publication 4 |
| `allplus` | same 23, unvetted | same 23, unvetted |

`svm`'s `all` is a session *list*, not a quality gate, because
`MIN_SVM_AUC >= 0.95` would drop nobody — `data.labeler.CLUSTERING_SESSIONS`
(the pool both sources' `all`/`allplus` draw from) was *defined* by that same
threshold, so a gate on it would leave `all` and `allplus` byte-identical for
`svm`.

`--top-n` truncates each monkey's scope to its top-N sessions by
`data.labeler.SNR_AUC`, applied **after** vetting (so a session vetting would
reject cannot consume a slot). Off by default (`0`) — the three scopes above
already define the session sets.

## 8. CV design

Split-half CV, `equal_halves` (`eye_pre_flash.plotting.similarities.heatmap_eq`):
each qualifying cell contributes two disjoint equal-size halves per split,
meaned into d-dimensional vectors, correlated
`r_ij = mean(pearson(A_i, B_j), pearson(B_i, A_j))`, averaged over
`--n-splits` (default 20) splits and then over sessions.

**Every cell is computed.** `--min-trials` (default 4) is the arithmetic
floor — a cell must split into two halves of >= 2. Below `--min-stable`
(default 8) a cell is still computed, at its own smaller half size, and
marked thin (`*`) rather than dropped. The shared half size for a session is
set by its **stable** cells only (`min` over cells with n >= min-stable of
`n // 2`), not by the thinnest cell in the session — a single 2-trial cell no
longer drags every other cell in that session down to 1-trial halves.
Unbalanced halves are not merely noisier, they are biased in the direction
being tested: the majority-strategy cell gets cleaner halves and hence a
higher diagonal, so confining that bias to cells marked thin matters. High
variance in the few-session scopes (`publication`, and often `all`) is
accepted for now rather than excluded.

The exhaustive-subset "comparable core" restriction the deleted
`heatmap_labels` used (`common_cell_set`, `unreliable_mask`, `matched_matrix`)
is deliberately **not** carried into the plotting path here — every
per-session correlation still lands in `results_raw.csv`, so that
restriction remains reconstructable later without re-running anything, at the
cost of not being applied automatically.

## 9. The permutation null

`--n-perm` (default 1000) label-shuffle resamples per maze, shuffling
strategy labels **within each (session, maze)** — which preserves every
cell's trial count exactly — and recomputing the full within-maze delta each
time. A permutation null rather than a bootstrap CI: the question is not "how
much does this vary between recording days" but "is this delta
distinguishable from zero at d as low as 6-12 with these counts", and the
null measures that empirically at the actual dimensionality and coverage
rather than assuming a bootstrap's between-session resampling is the right
error model at this dimensionality. Reported per maze in `nulls.csv`: the
observed delta, the null mean and SD, a two-sided p-value (+1 Laplace
smoothing, so p is never exactly 0), and the null's 2.5-97.5th percentiles.

## 10. How to read the output

1. **Census first** (the report's first table) — trials pooled per cell, how
   many sessions each cell appeared in, and which cells are thin.
2. **The within-maze table** — `r(H,H)`, `r(S,S)`, `r(H,S)` on matched
   session sets per maze, `delta = 0.5*(r_HH + r_SS) - r_HS`, and the
   permutation p-value. This is the decisive read.
3. **The full matrix** — every cell, thin ones marked `*`. Never compare a
   marked entry against an unmarked one as if they carried equal weight.
4. Beyond the boxed cells: same-strategy-different-maze (does a decoded
   strategy look like itself across geometries?), then the two cross-strategy
   quadrants.

## 11. Output files

```
eye_pre_flash/corr/out/
  <source>/                dendro | svm
    <feature>/              occupancy | occupancy_bin | bigram
      <variant>/            full | no_origin | pc1_removed
        examples.csv
        examples_dims.csv
        <scope>/            publication | all | allplus
          matrix_<Monkey>_k<K>_<space>.png
          results_raw.csv
          nulls.csv
```

`results_raw.csv` — one row per (leaf config, monkey, session, cell i, cell
j): every per-session correlation plus its trial counts, half sizes, and
stability flags, so the plotted matrix, the within-maze table and (with
`presence`/`stable` reconstructed) a comparable-core restriction are all
rebuildable without recomputation. `nulls.csv` — one row per (leaf config,
monkey, maze): the permutation summary from §9.

`examples.csv` (long, one row per sampled row x dimension) — `row_kind` in
`{trial, half_mean}`. Both matter: the skew question has a different answer
before and after averaging a half's worth of trials, and it is the half-mean
the Pearson actually sees. `examples_dims.csv` (one row per cell x dimension,
plus a `dim="__vector__"` summary row per cell and a `dim="__pc1__"` row for
the `pc1_removed` variant) — `frac_zero`, `n_distinct` and `top_value_frac`
are the direct "heavily skewed to a few values" read; `mean_n_nonzero_per_trial`
says whether a nominally K-dimensional vector is effectively lower-dimensional
in practice.

## 12. Caveats

- **A Pearson over 6-144 points.** The smaller `K`/feature combinations put
  real weight on the permutation null in §9 — do not read a bare delta
  without it.
- **Coverage in the thin scopes is accepted, not solved.** `publication` (2
  sessions/monkey) and often `all` will show many thin (`*`) cells; `allplus`
  is where this analysis has the coverage to be informative.
- **Zero-variance `occ_bin` cells** (every trial visited every state) produce
  a degenerate NaN Pearson, counted in `n_degenerate` and noted on the figure
  — an em-dash from this cause means "no variance", not "no coverage".
- **The SVM source's anchor asymmetry** (§6): validated where it matters
  least, soft-checked where it matters most.
- **144 cells across up to 54 leaves, uncorrected.** The within-maze delta on
  one pre-declared (source, feature, variant, k, space) is the read; the rest
  is a robustness surface, not an independent search.
- **Trials are pooled across sessions**, and strategy is strongly
  session-clustered — this reads as "on these recording days", not as
  generalisation to unseen days.

## 13. Reproduce

```bash
uv run python -m eye_pre_flash.corr.run --source dendro --feature occupancy \
    --variant full --scope allplus --monkey Faure --k 12 --space unith
```

Full sweep on the cluster: `mkdir -p logs && sbatch slurm/run_corr.sbatch`,
then `rsync -av engaging:strategy-selection/eye_pre_flash/corr/out/
./eye_pre_flash/corr/out/`.
