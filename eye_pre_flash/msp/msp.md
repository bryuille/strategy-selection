# msp — maze-strategy pairs with a fixed five-state codebook

A self-contained re-cut of `eye_pre_flash/maze_strategy_pairs/`. Same
question, same estimator, one deliberate change: the gaze codebook is **not
fitted**. Its five prototypes sit at the screen origin (0, 0) and the four
maze exits (−1, 1), (−1, −1), (1, 1), (1, −1) in unit-H coordinates, for both
monkeys. K = 5 is a statement about the maze, not a number chosen from the
data, so there is no K sweep and no elbow.

Everything else is pinned too: SVM labels at the top-ten scope, mazes 2–5,
binary occupancy, the three variants `full` / `no_origin` / `mean_removed`.
Nothing here imports from another `eye_pre_flash` folder; the shared
`data/` layer is the only dependency.

## The question

Within one maze, is the gaze of two groups of trials that share a decoded
strategy (H or S, from the neural SVM) more similar than the gaze of two
groups that do not? Maze geometry is identical on both sides, so a
difference cannot be a visual confound.

## Pipeline

1. **Clip** each trial to the pre-fixation window
   `[fix_start − 1466 ms, fix_start]`.
2. **Warp** gaze to unit H (`data.attractor.to_maze`, cached): fixation point
   at (0, 0), exits at (±1, ±1), stem top at (0, 1).
3. **Codebook**: `codebook.CODEBOOK_XY`, fixed. No fitting.
4. **Fixations** by I-DT on the clipped, unwarped data (cached events),
   re-tested against the 100 ms minimum after clipping.
5. **Assign** each fixation's mean warped position to the nearest prototype
   within `ASSIGN_RADIUS = 1.0`; fixations outside every ball are dropped.
6. **Feature**: binary occupancy, a 5-vector of visited / not visited per
   state.
7. **Estimate** per (monkey, variant, maze): pool every labelled trial of
   that maze from every in-scope session; cut into four disjoint groups of
   `m = min(n_H, n_S) // 2` (two H, two S); average each group into one
   vector; correlate the means for each of the four pairings; repeat over
   100 fresh groupings. `Δ = 0.5·(r_HH + r_SS) − 0.5·(r_HS + r_SH)`, tested
   against 1000 label shuffles **within each session**.

Steps 1–2 and 4–7 are copied from the fitted pipeline unchanged. In
particular the rule for which trials become rows (QC, maze 1–6, a usable
window, at least two valid in-window on-screen samples) is the one the
k-means pool used, so `n_H` / `n_S` per maze equal those of every other
maze-strategy-pairs run at the svm/top_ten scope.

## Reading a panel

* **Diagonal** (`HH`, `SS`): similarity of two group means of the same
  strategy. **Off-diagonal** (`HS`, `SH`): two draws of one quantity.
* **`n = …`** under each diagonal cell: trials of that strategy pooled over
  the sessions in scope. No trial is excluded, so it is the same across the
  three variants.
* **Δ, z, p** above the panel. `z = (Δ − null_mean) / null_sd`; `p` is
  two-sided; a leading `~` means no shuffle beat the observed gap, so `p` is
  at the floor `1/(n_perm+1)` and `z` says by how much.
* **Colour is a within-figure read only.** The scale spans the panel's own
  min-to-max. Judge strength by `z`; `--vmax 1.0` fixes the scale if panels
  must be compared by eye.
* **Compare `z` across panels, never raw `r`.** Averaging `m` trials
  suppresses noise before the correlation, so cells rise with `m`, and `m`
  differs by maze and monkey.

The remaining estimator caveats (unequal H/S noise inflating Δ, drift within
a session, non-independent rounds, multiplicity) are unchanged from
`maze_strategy_pairs/CAVEATS.md` and apply verbatim.

## Caveats specific to the fixed codebook

**Coverage.** A fitted codebook puts prototypes where gaze actually dwells;
this one puts them where the maze's landmarks are. Gaze on the middle of an
arm (around (±1, 0)), on the stem, or beyond an exit is more than 1.0 from
every prototype and is dropped. The dropped fraction is therefore higher than
under any fitted codebook, and it can differ between mazes and between
strategies. `build --dry-run` prints, per maze, the share of in-window
fixations assigned, each state's visit rate, and the share of all-zero rows;
`codebook.png` shows the balls over every fixation centroid:
one dot is one I-DT fixation, drawn at the mean warped position of its
valid in-window samples, which is the point the assignment used. `codebook_maze<M>.png` is the same
view restricted to one maze's fixations. Read those before the panels.

Measured at radius 1.0 on the svm/top_ten pooled rows (laptop caches,
2026-09-23):

| monkey | maze | fixations assigned | visit rate origin / LU / LD / RU / RD | all-zero rows: full / no_origin |
|---|---|---|---|---|
| Faure | 2 | 88% | 0.93 / 0.70 / 0.72 / 0.08 / 0.02 | 0.2% / 4.5% |
| Faure | 3 | 87% | 0.94 / 0.54 / 0.66 / 0.19 / 0.01 | 0.4% / 11.4% |
| Faure | 4 | 91% | 0.97 / 0.65 / 0.72 / 0.11 / 0.05 | 0.3% / 8.5% |
| Faure | 5 | 85% | 0.94 / 0.55 / 0.52 / 0.21 / 0.05 | 0.8% / 15.8% |
| Nielsen | 2 | 71% | 0.93 / 0.40 / 0.05 / 0.35 / 0.17 | 2.4% / 30.6% |
| Nielsen | 3 | 71% | 0.92 / 0.53 / 0.07 / 0.30 / 0.14 | 2.1% / 25.9% |
| Nielsen | 4 | 67% | 0.92 / 0.32 / 0.04 / 0.33 / 0.16 | 2.2% / 38.2% |
| Nielsen | 5 | 70% | 0.96 / 0.48 / 0.06 / 0.23 / 0.24 | 1.8% / 31.3% |

The same at radius 0.5 (`--radius 0.5`, under `out/r0.5/`):

| monkey | maze | fixations assigned | visit rate origin / LU / LD / RU / RD | all-zero rows: full / no_origin |
|---|---|---|---|---|
| Faure | 2 | 67% | 0.82 / 0.60 / 0.63 / 0.05 / 0.01 | 3.2% / 12.7% |
| Faure | 3 | 62% | 0.86 / 0.38 / 0.56 / 0.14 / 0.01 | 4.3% / 23.7% |
| Faure | 4 | 71% | 0.92 / 0.57 / 0.63 / 0.08 / 0.03 | 2.2% / 15.8% |
| Faure | 5 | 63% | 0.88 / 0.42 / 0.42 / 0.17 / 0.03 | 4.9% / 27.6% |
| Nielsen | 2 | 49% | 0.87 / 0.29 / 0.05 / 0.18 / 0.13 | 5.3% / 49.3% |
| Nielsen | 3 | 51% | 0.88 / 0.43 / 0.05 / 0.10 / 0.11 | 6.6% / 44.4% |
| Nielsen | 4 | 48% | 0.88 / 0.22 / 0.04 / 0.18 / 0.14 | 6.8% / 55.6% |
| Nielsen | 5 | 52% | 0.92 / 0.36 / 0.05 / 0.08 / 0.22 | 2.8% / 43.9% |

Halving the radius drops a fifth of Faure's assigned fixations and a third of
Nielsen's, and roughly doubles the all-zero rows. The balls no longer overlap
(origin-to-exit distance sqrt(2) > 2 * 0.5), so every assignment is
unambiguous, at the price of coverage. Estimates at the two radii, for the
two mazes with balanced strategy use, `full` / `no_origin` / `mean_removed`:

| cell | radius 1.0 | radius 0.5 |
|---|---|---|
| Faure maze 2 | z +3.4 / +0.5 / +2.4 | z +2.4 / −0.8 / +1.8 |
| Nielsen maze 4 | z +8.8 / +9.2 / +3.1 | z +9.8 / +8.2 / +2.9 |
| Nielsen maze 2 | z +1.0 / +1.6 / +1.5 | z +2.7 / +2.9 / +1.6 |

Nielsen maze 4 is unchanged; Faure maze 2 weakens and still fails
`no_origin`; Nielsen maze 2 becomes marginal (p 0.03) under `full` and
`no_origin`.

Two things stand out. Faure's fixations sit on the landmarks (85-91%
assigned, and the codebook figure shows the fixed prototypes on top of the
dense clusters); Nielsen's do not to the same degree (67-71%), and Nielsen
almost never has a fixation within 1.0 of the lower-left exit. Nielsen's
unassigned fixations are mostly a diffuse off-maze cloud in the upper-right
quadrant beyond (1, 1), not gaze along the arms (see `codebook.png`). So for
Nielsen, `no_origin` leaves a quarter to a third of trials as all-zero
4-vectors, which are kept (see below) but carry no information about that
trial. A Nielsen `no_origin` panel is therefore a much coarser measurement
than the same panel for Faure, and its `z` should be read with that in mind.

**Boundary tie at the stem top.** (0, 1) is at distance exactly 1.0 from the
origin and from both upper exits. The codebook lists the origin first, so an
exact tie assigns to origin; in practice float32 rounding decides fixations
that sit on the boundary. Nothing is unstable across runs (the computation
is deterministic), but a centroid there is a coin the geometry flipped, not
the animal.

**Overlap.** Origin-to-exit distance is √2, less than two radii, so
neighbouring balls overlap and nearest-prototype decides, exactly as for
fitted codebooks.

**Five binary dimensions.** Each trial is a 5-vector of zeros and ones, and
under `no_origin` a 4-vector. Zero-variance rows (nothing visited, or
everything visited, or under `no_origin` only the origin visited) are common
and are **kept**: they enter their group's mean and are counted in
`n_degenerate` in `results.csv`. Group means are continuous, so the
correlations are well defined; the coarseness shows up as a wider null, not
a broken statistic.

**`mean_removed` is fitted per (monkey, maze)** over exactly the rows the
estimator pools (every in-scope labelled trial of that maze), the choice the
pooled-per-maze pipeline also makes. Every read here is within one maze and
across panels by `z`, so a cross-maze grand mean would buy nothing and would
blend maze-specific exit profiles into a mean that fits no maze.
Subtracting a common profile pushes mean off-diagonal `r` negative by
arithmetic; only the diagonal-vs-off-diagonal gap survives. Read `no_origin`
alongside it.

**`--radius`** changes what "at a landmark" means and is carried in the cache
stem, so a different radius costs one extraction pass and never mixes with
the default's cache. The codebook itself is verified on load; editing
`codebook.py` invalidates the cache automatically.

## Fixation concentration by strategy

`strategy_maze<M>.png` asks the descriptive version of the
question the 2x2 tests: where do H trials' fixations land, where do S trials'
land, and how does the share at each fixed state differ? Four panels per
maze, using every in-scope labelled trial of that maze (the rows the
estimator pools):

* top row: 2-D density of H fixation centroids and of S fixation centroids,
  each expressed as the **fraction of that strategy's fixations** per bin
  (each panel sums to 1) on one shared colour scale, with the five balls
  overlaid;
* bottom left: each state's share of that strategy's fixations, H and S side
  by side, with `S−H` printed under each state;
* bottom right: the fraction of that strategy's **trials** that visit the
  state at least once (the mean of `occ_bin` over that strategy's rows), which
  is what the estimator actually correlates.

Everything is normalised within strategy because `n_H` and `n_S` differ by up
to an order of magnitude in some mazes (Faure maze 3: 517 vs 36; maze 4: 49 vs
577). Raw fixation counts would show only that imbalance. The trial and
fixation counts, and fixations per trial, are printed in the panel titles so
the imbalance stays visible; a strategy with 36 trials has a noisy density
panel, and the bars carry no error bars, so read them as description, not as
a test -- the 2x2 and its `z` are the test.

## Scope

Ten sessions per monkey from `data.labeler.CLUSTERING_SESSIONS`, ranked by
maze-1-vs-6 neural `snr_auc`, no anchor vetting — `label_sources`'
`svm/top_ten`, reproduced in `labels.py`. Nielsen has exactly ten; Faure
drops june_17_g0, April_8_g0 and june_11_g0. A missing label file raises
rather than triggering a neural conversion inside the job.

## Running

```bash
uv run python -m eye_pre_flash.msp.checks              # estimator invariants, synthetic, no cache
uv run python -m eye_pre_flash.msp.features            # build both feature caches (one pass each)
uv run python -m eye_pre_flash.msp.build --dry-run     # counts + coverage per maze
uv run python -m eye_pre_flash.msp.build               # full: 2 monkeys x 3 variants x 4 mazes
uv run python -m eye_pre_flash.msp.build --monkey Faure --maze 4 --n-perm 200   # smoke test
```

`--dry-run` needs the feature cache and builds it if absent (an extraction
pass over the attractor, behavioral and clean-event caches; minutes). The
estimator itself is seconds per panel.

On Engaging:

```bash
mkdir -p logs
sbatch slurm/run_msp.sbatch --dry-run
sbatch slurm/run_msp.sbatch
# results leave ~/strategy-selection/eye_pre_flash/msp/out/ (run on the laptop):
rsync -av --exclude='__pycache__' engaging:strategy-selection/eye_pre_flash/msp/out/ ./eye_pre_flash/msp/out/
```

## Output

```
out/r<radius>/<Monkey>/codebook.png                 prototypes, balls, every fixation centroid by state
out/r<radius>/<Monkey>/codebook_maze<M>.png         the same for one maze's fixations (all sessions)
out/r<radius>/<Monkey>/strategy_maze<M>.png         H vs S fixation density and per-state shares, one per maze
out/r<radius>/<Monkey>/<variant>/maze<M>.png        one 2x2 per maze (2-5)
out/r<radius>/<Monkey>/<variant>/results.csv        one row per maze
```

`r1/` is the default radius; `r0.5/` was produced with `--radius 0.5`. The
radius is a directory level so the two never overwrite each other.

```
```

`results.csv` columns: `monkey, variant, maze, radius, r_HH, r_SS, r_HS,
r_SH, delta, z, p, p_at_floor, null_mean, null_sd, n_H, n_S, m, d,
n_sessions, n_degenerate, n_rounds, n_perm, reason_skipped`. A maze with
fewer than 10 pooled trials of either strategy gets a row with
`reason_skipped` and no figure (a stale figure at that path is removed).

Cache: `$STRATEGY_DATA_ROOT/processed/<Monkey>_msp_fixed5_r1_unith_s1466_e0.npz`
(`occ_ms`, `occ_bin`, row metadata, per-trial `n_fix` / `n_fix_assigned`,
every fixation centroid with its state and trial row (`fix_xy`, `fix_state`,
`fix_row`), and the codebook, radius and window it was built under; a cache
missing any of these is rebuilt on load).

## Files

| file | purpose |
|---|---|
| `codebook.py` | the five fixed prototypes, radius, screen limit |
| `features.py` | extraction against the fixed codebook; cache; helpers copied from `classifier/features.py` and `data/attractor.py` |
| `labels.py` | svm/top_ten scope and label lookup, reproduced |
| `variants.py` | `full` / `no_origin` / `mean_removed` |
| `estimator.py` | the 2x2, Δ, within-session shuffle null (verbatim copy) |
| `figures.py` | panel figure, codebook figure, `save_figure` |
| `colormap.py` | parula (verbatim copy) |
| `paths.py` | `out/` layout |
| `build.py` | entry point |
| `checks.py` | synthetic estimator checks |
