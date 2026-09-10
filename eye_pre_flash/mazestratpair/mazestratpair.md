# Single-maze H-vs-S occupancy similarity

A 2x2 split-half similarity matrix over the two cells `{(maze, hierarchical),
(maze, sequential)}` of **one** maze: the diagonal is each strategy's own
split-half reliability, the off-diagonal is the cross-strategy correlation at
the same half size, and the statistic is how far the second falls below the
first. Defaults to Nielsen maze 4. Each figure carries all three feature
variants as three 2x2 panels; significance is a within-(session, maze)
label-shuffle permutation null.

```bash
uv run python -m eye_pre_flash.mazestratpair.run                      # Nielsen, maze 4
uv run python -m eye_pre_flash.mazestratpair.run --source svm --k 6 --feature bigram
uv run python -m eye_pre_flash.mazestratpair.run --monkey Faure --maze 2
uv run python -m eye_pre_flash.mazestratpair.run --all-monkeys --all-mazes --n-perm 200
```

Figures and CSVs land in
`eye_pre_flash/mazestratpair/out/<source>/<feature>/<scope>/`. On the cluster:
`sbatch slurm/run_mazestratpair.sbatch`.

## 1. The claim this exists to support

`eye_pre_flash.corr` builds a 12x12 (maze x decoded strategy) matrix and argues
(`corr.md` §1, §10) that its decisive read is the **same-maze** H-vs-S
comparison. Geometry is identical on both sides of that comparison — the same
maze was on screen for both cells — so a cross-strategy correlation below that
maze's own split-half reliability cannot be a visual confound. It has to be
strategy-dependent gaze sampling. Every other cell in the 12x12 mixes strategy
with geometry, because strategy is close to a deterministic function of
(session, maze).

In the 12x12 that read is four numbers inside a 144-cell grid, and it is also
handicapped by the estimator — see §3. This package is the same test at the
scale of the claim: two cells, one maze, one statistic.

**Nielsen maze 4 is the default because it is the only place the read is
clean.** Maze 4 is deliberately unscored in `classifier.labels.ANCHOR_MAJORITY`
(the anchor map runs mazes 1, 2, 3 → hierarchical and 5, 6 → sequential), so
nothing about its labels is circular, and it is the one maze where both label
sources independently agree the split is genuinely mixed: dendro 332 H / 410 S
(sequential share 0.553), SVM 352 H / 390 S (share 0.526). See
`eye_pre_flash/counts/counts.md`.

**The pre-declared read**, so the sweep is a robustness surface and not a
search: `dendro / top_four / occupancy / full / K=12 / Nielsen maze 4`,
permutation two-sided p. Everything else the sweep produces — 2 sources x 3
features x 2 K x up to 2 scopes x 3 variants x 6 mazes x 2 monkeys — is there
to show the answer does not depend on the knobs, and carries no independent
p-value budget. Same discipline `corr.md` §12 applies to its 144 cells.

## 2. What it is not

Not a re-crop of corr's output. The numbers here differ from corr's maze-4
cells and are meant to — §3. Not a decoding analysis: this is a correlation
between half-means, and `classifier.decoding` is where accuracy lives. Not a
session-level test: see §5.

## 3. Why the numbers differ from corr, and why that is a gain

`corr.matrix.session_half_sizes` picks **one** shared half size per session:
the minimum, over every cell in that session that clears `min_stable`, of
`n // 2`. Across twelve cells that minimum is set by whichever maze is thinnest
on its minority strategy — usually maze 1 sequential or maze 6 hierarchical.
Maze 4's own comfortable counts never get to choose.

Measured from corr's own `results_raw_k6.csv` on 2026-09-10:

| Session | n(4H) | n(4S) | corr's half | this package's half |
|---|---|---|---|---|
| `Nov_1_g0` | 45 | 36 | 5 | **18** |
| `Nov_3_g0` | 44 | 41 | 20 | 20 |
| `Nov_6_g0` | 22 | 30 | 5 | **11** |
| `Oct_22_g0` | 21 | 61 | 9 | 10 |

`Nov_1_g0` has 45 and 36 trials and corr correlates half-means built from
**five** of them. Restricted to these two cells it is eighteen. It is the same
estimator applied to more trials per half, so every `r` here — diagonal and
off-diagonal alike — is higher than corr's. **That is precision, not an
effect.** Two figures of the same cells with different numbers invite exactly
the wrong reading, so:

* `results_raw_*.csv` carries `half_corr_rule_H` / `half_corr_rule_S`, what
  corr's unrestricted rule would have chosen on that same session.
* The printed report tables both, session by session.
* `--n-half <corr's size>` forces corr's half size back. Δ should land within
  Monte-Carlo noise of corr's; if it does not, the half size was not the only
  difference and something is wrong.

### `--half-rule`

`stable_shared` (default) is corr's rule verbatim. It has a corner case that is
harmless across twelve cells and is not harmless across two: when only one cell
clears `min_stable`, the shared size comes from that cell and the other falls
back to its own `n // 2`. Measured on a Faure-like session with 5 H and 50 S
trials, that gives halves of **2 and 25** — so `r(H,H)` is built from 2-trial
means while `r(H,S)` pairs a 2-trial mean against a 25-trial mean, and Δ is
exactly their difference. `min_pair` instead takes `min(n_H, n_S) // 2` for
both cells, always balanced.

The two rules **coincide whenever both cells clear `min_stable`**, which is
every Nielsen maze-4 session in every scope. They diverge only where one
strategy is genuinely thin — Faure maze 4, or any maze under SVM labels away
from that animal's boundary. The permutation null absorbs the bias either way
(§5), but `equal_n` in `results_raw` flags it and the figure stars the thin
cell.

## 4. Data, features and variants

Everything comes from `classifier.features.load_features(monkey, k=, space=)`:
one row per trial, with `session`, `trial_indices_all`, `maze_id` and the three
feature blocks. Window is `fix_start − 1466 ms → fix_start`, I-DT fixations
only.

| CLI name | block | unit |
|---|---|---|
| `occupancy` | `occ_ms` | seconds per state (the *time* occupancy) |
| `occupancy_bin` | `occ_bin` | visit fraction per state |
| `bigram` | `bigram` | bigram proportion, `k x k` ordered state pairs |

Names are corr's, unchanged, so the two packages' CLIs and output trees read
the same way.

Variants are `corr.variants`, imported not reimplemented: `full`,
`no_origin` (drops the codebook state nearest the screen origin, and for
`bigram` every ordered pair touching it), `mean_removed` (subtracts a grand-mean
profile). **The grand mean is fitted exactly where corr fits it** — once per
(monkey, feature, K) over the widest scope's labelled trials across *all*
mazes, never on the single maze. Refitting it per maze would let the profile
absorb that maze's own geometry, which is the confound this analysis exists to
exclude, and would make the panel incomparable with corr's.

Unit-H only, no `deg`. A degree-space codebook is fit on pooled screen degrees,
so a given state is on-arm for one geometry and off-arm for another and `deg`
occupancy encodes which maze was on screen (`corr.md` §5).

## 5. CV design and the permutation null

Per session, restricted to the maze: the two cells are split into two disjoint
halves of the shared size 20 times
(`plotting.similarities.heatmap_eq.equal_halves`), half-means are correlated
pairwise (`corr.matrix.pairwise_pearson_matrix`), and the result is symmetrised
as `0.5 * (R_ab + R_ab.T)`. The diagonal is therefore a genuine split-half
reliability — its two halves are disjoint — and the off-diagonal is the
cross-strategy correlation **at the same half size on both sides**, so Δ is not
an artefact of one cell having more trials.

Sessions are averaged unweighted, as corr does. Weighting by trial count would
amplify exactly the count bias `equal_halves` exists to remove: a session with
more trials gets cleaner half-means and hence a higher diagonal. Every
per-session matrix reaches `results_raw_*.csv`, so a weighted version stays
reconstructable without a rerun.

A session is dropped unless **both** cells clear `min_trials` (4). With two
cells a single cell yields no correlation at all, so there is nothing to mark
thin and nothing to average.

**Δ = 0.5·(r_HH + r_SS) − r_HS**, and its null is a label shuffle confined to
each (session, maze) stratum, 1000 resamples by default. That stratum is the
point: shuffling inside it preserves every cell's trial count exactly, and
therefore both half sizes, both thin flags and the presence matrix as well — the
shuffled matrices carry the identical estimator, down to an unequal half size
where one strategy is thin. So the null is a test of the **labels**, not of the
coverage, and it absorbs the §3 half-size bias rather than being fooled by it.

A permutation null rather than a bootstrap CI, for corr's reason (`corr.md`
§9): the question is not how much Δ varies between recording days but whether
it is distinguishable from its own noise floor at d as low as 6 with these
counts. Hence **Δ's reference is `null_mean`, not zero.** Reported per variant:
`delta_obs`, `null_mean`, `null_sd`, `z_perm = (obs − null_mean) / null_sd`,
`p_two_sided` (+1 Laplace-smoothed, centred on the null mean — corr's
convention verbatim, so the two packages' p-values mean the same thing),
`p_one_sided_greater`, and the null's 2.5–97.5th percentiles.

Two departures from `corr.matrix.permutation_null`, both deliberate:

1. **The pool is built once.** Cell membership depends only on the label, so
   trial counts and half sizes are invariant under the shuffle; `pair.build_pools`
   builds the per-session records once and only `y` is resampled, rather than
   rebuilding a `(session, trial_id)` lookup and re-running `labels_for_rows`
   over the whole feature table 1000 times.
2. **Sessions are iterated in sorted order.** corr iterates
   `set(keep_sessions)`, and Python randomises string hashing per process, so
   its strata order — and hence every p-value in its `nulls_k*.csv` — varies
   between runs despite `--seed`. This package's null is reproducible; verified
   bit-identical across two calls at one seed.

**No session-level test is reported.** The per-session Δ table and an
"N/M sessions positive" count are printed as description, but a one-sample *t*
or a Wilcoxon on 2 sessions (`publication`) or 4 (`top_four`, which
`corr/labels.py` documents *is* the entire vetted pool) would not be evidence,
and the permutation null already conditions on exactly these recording days.
Every per-session Δ is in `results_raw_*.csv` if a session-level test is wanted
later at `svm/top_ten`.

### The null's own limits

Two columns exist because a single maze can make them bind where pooling six
mazes never does:

* `n_perm_distinct_log10` — log10 of the number of distinct label assignments
  the strata admit. This is the p-value's true floor. If it approaches
  log10(`n_perm`), the reported p is limited by the strata and not by
  `--n-perm`.
* `n_sessions_unexchangeable` — sessions whose maze trials were all one
  strategy. Invariant under every permutation, so they would dilute the null
  toward the observed and make the test conservative; `build_pools` drops them,
  and this column says the exclusion happened.

## 6. Label sources and session selection

Both imported from `corr.labels` unchanged, and both are a path component in
the output tree:

* `dendro` — Ward clustering on the strategy PC scores
  (`data.labeler.build_strategy_choices`), anchor-agreement vetted at 0.70.
  Scopes `publication` (2 sessions per monkey) and `top_four` (4).
* `svm` — a maze-1-vs-6 linear SVM on the same PC space
  (`data.labeler.build_svm_choices`), unvetted, because grading it by agreement
  with a map that *defines* mazes 1 and 6 would be circular. Scope `top_ten`
  (up to 10).

`corr.labels.SOURCE_SKIP_CELLS` blanks 1S and 6H under the SVM source as
anchor-minority cells. Translated here by `pair.skip_pair_cells`: under `svm`,
mazes 1 and 6 have no H/S pair at all, so those leaves are skipped with a
message rather than half-computed. Adding `dendro → top_ten` to
`corr.labels.SOURCE_SCOPES` is a one-line change if more sessions are ever
wanted on the dendro side; it was deliberately not done, because those sessions
failed the 0.70 anchor gate.

## 7. How to read the output

1. **Census first**, in the report: trials pooled per strategy, sessions each
   cell appeared in, thin flags, and how many of the scope's sessions carried
   both strategies at all.
2. **The half-size table** — this package's half next to corr's. If they are
   equal, this figure and corr's maze-4 cells should agree; if they differ,
   expect every `r` here to be higher and do not read that as an effect.
3. **The three-variant table** — `r(H,H)`, `r(S,S)`, `r(H,S)`, Δ, `null_mean`,
   `null_sd`, `z`, both p's. Δ against `null_mean`, never against zero.
4. **`z`, not Δ, across panels.** `full` sits in a compressed high-*r* band
   (0.95–0.99 on Nielsen maze 4) while `mean_removed` is mean-centred, so a
   bigger Δ on `mean_removed` is a scale artefact. `z` is on a common scale by
   construction. This is also why each panel gets its own colour scale, and why
   `mean_removed` gets a diverging map with white pinned at zero when its
   values straddle zero.
5. **Two different n's on the figure.** The `n = 132` in a diagonal cell is
   trials pooled across sessions. The n behind any single Pearson is `2 x the
   half size` within one session — perhaps 20. The panel title carries the
   half, and the footer states the distinction.

## 8. Output files

```
eye_pre_flash/mazestratpair/out/
  <source>/                       dendro | svm
    <feature>/                    occupancy | occupancy_bin | bigram
      <scope>/                    dendro: publication, top_four;  svm: top_ten
        pair_<Monkey>_maze<M>_k<K>.png          three 2x2 panels, one per variant
        results_raw_<Monkey>_maze<M>_k<K>.csv   one row per (variant, session)
        nulls_<Monkey>_maze<M>_k<K>.csv         one row per variant
        null_draws_<Monkey>_maze<M>_k<K>.csv    every permutation draw
```

The variant is a panel and a CSV column, never a directory — the three
variants of one maze *are* the comparison, and splitting them across
directories makes that a file-management exercise. Every stem carries monkey,
maze and K because `pair_io.write_rows` truncates: a stem shared between two
mazes would silently overwrite. A run narrowing `--variant` appends the variant
names to the stem, so a one-panel debugging rerun cannot clobber the canonical
three-panel outputs.

`results_raw_*.csv` is the reconstruction record: per session, both trial
counts, both half sizes, corr's half sizes, `equal_n`, both stable flags, all
three `r`s, that session's Δ, and the session's label-quality diagnostics
(`snr_auc`, `svm_auc`, `oof_balanced_acc`, `dendro_agreement_2345`, from
`corr.run._session_diagnostics`). `nulls_*.csv` is the summary — one row per
variant, everything in §5. `null_draws_*.csv` holds every draw, so the null
histogram can be replotted and the tail redefined without a rerun; disable with
`--no-null-draws`.

The report is **printed to the run log, not written beside the png** —
corr's discipline. `out/` carries figures and CSVs only.

## 9. Caveats

* **Faure maze 4 is thin, and under SVM labels nearly empty.** Measured from
  corr's `results_raw_k6.csv`: under `dendro/top_four`, 2 of Faure's 4 sessions
  have **zero** maze-4 hierarchical trials; under `svm/top_ten`, 6 of 10 do.
  Those sessions are dropped, so `--monkey Faure --maze 4` runs on 2 sessions
  (dendro) and the SVM version is not worth reading. Faure's near-balanced
  maze under SVM labels is maze 2 (373 H / 310 S).
* **`publication` is 2 sessions.** Δ there is an average of two numbers. The
  permutation null is still exact but its resolution is coarse; `top_four` and
  `top_ten` are the ones to read.
* **`mean_removed` absolute *r* is not interpretable.** Subtracting a common
  profile makes centred vectors sum to about zero across cells, so the mean
  off-diagonal goes negative by arithmetic (`corr/variants.py`). Only Δ against
  that panel's own ceiling reads.
* **Correlations over 6 to 144 points.** At K=6 the Pearson is over six
  numbers. That is the headline caveat of the whole approach and the reason the
  null is measured rather than assumed; the figure prints `d` per panel.
* **Averaging raw *r*, not Fisher-z.** At *r* ≈ 0.95 raw averaging over few
  sessions is biased low, and the bias is larger for the more reliable cell —
  i.e. it acts on the diagonal more than the off-diagonal, the direction of Δ.
  corr accepts this and matching it preserves comparability; per-session `r`s
  are in `results_raw_*.csv` if a Fisher-averaged Δ is wanted.
* **Effective n is set by the minority strategy.** `equal_halves` subsamples
  both cells to the shared size, so a 45/36 cell discards trials from the
  larger side on every split. Correct by design — that balance is the point —
  but it means precision is governed by the strategy with fewer trials, which
  under a decoded label is also the strategy whose labels are most likely
  wrong.

## 10. Reproduce

```bash
# Local: import and CLI only. `data/processed/` is empty on the laptop, so
# nothing can actually run here.
uv run python -m eye_pre_flash.mazestratpair.run --help

# Cluster.
rsync -av --delete --exclude-from=.rsync-exclude ./ engaging:strategy-selection/
ssh engaging 'cd strategy-selection && mkdir -p logs \
    && sbatch slurm/run_mazestratpair.sbatch --n-perm 50'   # ~1 min smoke run
ssh engaging 'cd strategy-selection && sbatch slurm/run_mazestratpair.sbatch'

# Pull the figures back.
rsync -av engaging:strategy-selection/eye_pre_flash/mazestratpair/out/ \
    ./eye_pre_flash/mazestratpair/out/
```

Cross-check against corr: compare `nulls_Nielsen_maze4_k12.csv` here with
maze 4's row in
`eye_pre_flash/corr/out/<source>/<feature>/<variant>/<scope>/nulls_k12.csv`.
Expect the same sign, a larger `|Δ|` here, and — with `--n-half` set to corr's
`half_i` for maze 4 — agreement within Monte-Carlo noise.
