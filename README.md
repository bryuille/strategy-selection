# PFC strategy selection

Gaze during the pre-fixation window, scored against neural strategy labels
(hierarchical **H** vs sequential **S**). Two monkeys (Faure, Nielsen), several
recording sessions each, six mazes per session.

Start with the data (next section), then environment setup, then the analyses.
Every runnable command, with its flags, is in
[README_COMMANDS.md](README_COMMANDS.md).

## Data

You provide two folders; the code needs nothing else.

- **`MAT_ROOT`** — the raw `.mat` files (behavioral, eye, neural, single-trial). Required: there is no default, so set it before anything runs. The code only ever *reads* from it (every file is opened read-only, and `PROCESSED_ROOT` is refused if it sits inside `MAT_ROOT`). Not reproducible, so keep a backup.
- **`PROCESSED_ROOT`** — everything the code computes: strategy labels, eye concatenations, clean events, unit-H warp, msp / regression features. Defaults to `./data/processed` (created on first use) and is rebuilt on demand. No raw data is ever copied or cached here.

Nothing under `data/` except code is tracked in git.

### Setting the data paths

Open `data/config.py` and edit the two lines at the top (there are no environment variables):

```python
MAT_ROOT = None  # e.g. Path("/Volumes/archive/mat")
PROCESSED_ROOT = Path("./data/processed")
```

`MAT_ROOT` must contain:

```
<MAT_ROOT>/
├── Behavioral_Data/{Faure,Nielsen}/<session>_good_trials_concat.mat
├── Eye_Data/{Faure,Nielsen}/Eye_Data_<session>.mat
├── Neural_Data/{Faure,Nielsen}/<session>_Whole_Trial_FR_Causal.mat
└── Single_Trial_List/{Faure,Nielsen}/...
```

Check what the code sees:

```bash
uv run python -c "import data.config as c; print(c.MAT_ROOT, c.PROCESSED_ROOT)"
```

If `MAT_ROOT` is still unset, any data access stops with a message saying so.

### File naming

Sessions are named `<month>_<day>_g<n>` (e.g. `june_24_g0`, `Oct_22_g0`). The
month prefix determines the monkey: `june` / `April` → Faure; `Oct` / `Nov` /
`Dec` → Nielsen. The authoritative path functions are in `data/config.py`.

### Building the data

Once `MAT_ROOT` is set:

1. `data.labels`: strategy labels (H / S) per session, and the SVM labels used
   by msp. Reads the large neural `.mat` files directly, one session at a time.
2. Run an analysis. Eye and feature caches under `PROCESSED_ROOT` build on first
   load, reading the eye / behavioral `.mat` files.

Commands and flags: [README_COMMANDS.md](README_COMMANDS.md#data). Raw field
reference: [DATA_DICTIONARY.md](DATA_DICTIONARY.md). Processed cache formats:
[data/data.md](data/data.md).

Loaders only check that cache keys exist, not that the settings used to build
them match the current code. After changing detector or QC constants, delete the
affected derived caches and rebuild (see
[README_COMMANDS.md](README_COMMANDS.md#clearing-derived-caches)).

## Environment setup

Python ≥ 3.14 (see `.python-version`).

**With uv** (recommended; `uv.lock` pins the environment):

```bash
uv sync
uv run python -m msp.checks        # sanity check, needs no data
```

**Without uv:**

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install h5py matplotlib numpy pandas pymovements scikit-learn scipy
python -m msp.checks               # sanity check, needs no data
```

The dependency list lives in `pyproject.toml`; the `pip install` line mirrors
it. Anywhere a doc says `uv run python -m X`, the non-uv equivalent (venv
active) is `python -m X`.

## Repo layout

- **`data/`** — Shared data layer, not an analysis: path config, `.mat` → `.npz` conversion, loaders, cache builders, strategy-label generation.
- **`counts/`** — Analysis 1: how many H and S trials exist per session and maze.
- **`msp/`** — Analysis 2 (**main**): maze-strategy pairs, pooled across sessions; `msp.per_session` runs it one session at a time. `msp/legacy/` keeps the retired `pre1466` saccade plots.
- **`msp_trends/`** — Analysis 3: msp on a coarser, merged codebook, plus a pooled-over-mazes estimate.
- **`regression/`** — Analysis 4: logistic regression of strategy on gaze, with maze held fixed.
- **`scatter/`** — Analysis 5: mean gaze per maze over `fix_start` → `flash_one` (after fixation, before the flash).
- **`slurm/`** — Batch scripts for a Slurm cluster: `run_msp.sbatch`, `run_regression.sbatch`.

Each analysis package imports the shared `data/` layer read-only and writes
only to its own `out/` folder. `out/` is generated and can be rebuilt. When a
default changes, rename old outputs to an explicit tag (e.g. `r0.5_<old>/`)
instead of deleting them.

## The analyses

All analyses ask the same underlying question: **does where the monkey looked
before it started moving carry information about which strategy it used?** They
differ in how the question is posed and what is controlled.

Shared vocabulary:

- **H / S**: hierarchical / sequential strategy, the per-trial label.
- **Labels** come from one of two sources. *SVM*: a linear classifier of neural
  activity trained on mazes 1 vs 6, applied to the top-ten sessions per monkey
  (ranked by `snr_auc`). *Dendrogram*: Ward clustering of neural activity,
  vetted down to the publication sessions.
- **Unit-H coordinates**: each maze is warped so the fixation point is at
  (0, 0), the four exits are at (±1, ±1) and the top of the stem is at (0, 1).
  This puts all mazes in one frame.
- **Landmarks / codebook states**: the origin (0, 0) and the four exits LU, LD,
  RU, RD. A fixation is assigned to the nearest landmark within a radius, or
  dropped if none is close enough.
- **Window**: `geofix` runs from maze onset to fixation onset, about 1.5 s.
  The retired fixed window (`pre1466`, the 1466 ms before fixation onset)
  survives only in `msp/legacy/` (below).

### `counts/`: how much data is there?

A census, not a test. For each (session, maze) it counts H and S trials and
draws a heatmap: one cell per (session, maze), text `H/S`, fill showing the
sequential share (blue = all H, orange = all S, grey = balanced). A cell whose
minority strategy is rare is marked red and bold: it has trials, but not on both
sides, so it cannot support an H-vs-S comparison.

Use it to see which sessions and mazes can support any analysis below. For the
SVM labels, mazes 1 and 6 are almost single-strategy by construction (they are
the classifier's training axis), so expect them to be marked red.

Two label sources (`dendro`, `svm`) and several session scopes (`publication`,
`all`, `allplus`, `top_ten`). `counts.census` makes the working heatmaps;
`counts.build` makes slide-ready copies of the same plots with the publication
sessions boxed. Output: `counts/out/<source>/hs_<monkey>_<scope>.png`.

### `msp/`: maze-strategy pairs (main analysis)

**Question.** Within one maze, is gaze more similar across two groups of trials
that share a decoded strategy (H–H or S–S) than across two groups that do not
(H–S)? Both groups see identical maze geometry, so a difference cannot be a
visual confound.

**Codebook.** Fixed, not fitted: five states at the origin and four exits, in
unit-H coordinates, the same for both monkeys. There is no choice of K to sweep.

**Pipeline.**

1. Clip each trial to the window and warp gaze to unit-H.
2. Detect fixations (I-DT) and assign each to the nearest of the five states
   within `ASSIGN_RADIUS` (1.0 by default; `--radius` changes it). Fixations
   outside every ball are dropped.
3. Each trial becomes a 5-vector of visited / not-visited per state (binary
   occupancy).
4. Per (monkey, variant, maze): pool the labelled trials of that maze across
   in-scope sessions. Split each strategy into two disjoint random halves,
   average each half into one vector, and correlate the four pairings (HH, SS,
   HS, SH). Repeat over 100 random splits.
5. The statistic is `Δ = 0.5·(r_HH + r_SS) − 0.5·(r_HS + r_SH)`: positive when
   same-strategy groups look more alike than different-strategy groups. It is
   tested against 1000 label shuffles *within each session*, giving `z` and `p`.

**Variants.** `full` (all five states), `no_origin` (drop the origin state),
`mean_removed` (subtract the maze's grand-mean gaze vector first, so what is left
is how a trial deviates from the average trial of that maze).

**Reading the output.** Each figure is a 2×2 correlation matrix with Δ, z and p
above it. Compare `z` across panels, never raw `r`: averaging trials suppresses
noise, so `r` rises with group size, and group sizes differ by maze, monkey and
strategy. Colour scales are per-figure unless `--vmax` is fixed. Under
`mean_removed` a small strategy's diagonal has a floor near −1 from the
split-half mirroring, so those cells are not a consistency measure.

**Caveats.** A fixed codebook drops gaze that lands mid-arm or off the exits, so
the dropped fraction is higher than under a fitted codebook and can differ
between mazes. Estimator caveats (unequal H/S noise inflating Δ, drift within a
session, non-independent splits, multiple comparisons) apply across all the msp
packages.

Output: `msp/out/<tag>/<Monkey>/`, where the tag is `r<radius>` (`r0.5`, `r1`).
`msp.saccades` additionally plots pre-fixation gaze for one session, maze and
trial under the same assignment, into `msp/out/saccades/<tag>/`.

**Legacy saccade plots (`msp/legacy/`).** `msp.legacy.saccades` draws the same
gaze panels under the retired `pre1466` window
(`[fix_start − 1466 ms, fix_start]`), reading the attractor warp and the s1466 clean-event caches
instead of re-detecting per trial. It is the only pre1466 code left; its plots
are in `msp/legacy/out/saccades/r<radius>_pre1466/` (radii 0.5 and 1).

### `msp.per_session`: one recording at a time

The msp estimator with the pooling removed: each (monkey, session, maze, variant)
cell uses only that session's trials, and the label-shuffle null is a shuffle
within that single session.

Why: pooled msp averages over sessions, so an effect that exists in one session
(or is driven by one) is diluted or hidden. This shows which sessions carry the
effect and whether it is consistent. The cost is far fewer trials per cell, so
individual cells are noisy and many are skipped; with dozens of cells, a few
p < 0.05 are expected by chance.

A module of `msp` (formerly the `msp_per_session` package), reusing its
features, labels, estimator and figures. Output:
`msp/out/per_session/<tag>/<Monkey>/`, grouped by plot type
(`<variant>/maze<M>_<session>.png`, `codebook/`, `strategy/`).

### `msp_trends/`: msp on a coarser codebook

Motivation: pooled msp finds the H-vs-S effect cleanly only in a few cells
(Faure maze 2, Nielsen mazes 2 and 4), but the state profiles hint at a
consistent direction per monkey: H trials go to the left exits, S trials to the
origin (Faure) or the right exits (Nielsen). This package tests that directly
on a codebook that states it: three states shared by both monkeys,

- **`left`** — combines LU, LD
- **`origin`** — origin
- **`right`** — combines RU, RD

Why three states and not two: Pearson correlation between two 2-vectors is
always ±1 (each is centred on its own mean), so a two-state codebook cannot
measure magnitude. The third state is kept for that reason (and msp's
`no_origin` variant is dropped).

What it adds over msp:

- **Merged codebooks** (`--codebook`): `balls` (above), plus `quads` and `halves`
  (origin ball plus maze quadrants, or maze halves).
- **A pooled-over-mazes estimate.** Each maze keeps its own 2×2 (block means
  never mix mazes, so the very different H/S balance per maze cannot leak in).
  Pooled Δ is the trial-count-weighted mean of the per-maze Δs, tested against
  the same weighted mean of independent per-maze shuffles. One figure per monkey
  can then cover mazes 2–5.
- **Variants** `full` and `mean_removed`.
- **Label scope** (`--labels`): `svm` (top ten) or `dendro` (publication
  sessions).

Per-maze results are msp's estimator run unchanged on the merged states.
Output: `msp_trends/out/<tag>/<Monkey>/`.

### `regression/`: does gaze predict strategy beyond the maze?

**Question.** Some mazes are mostly S and some mostly H, and gaze differs by
maze too, so a raw gaze-vs-strategy association could just reflect the maze.
This asks whether gaze tells you anything about H vs S *within* a maze.

**Model.** One logistic regression per monkey, S = 1 and H = 0, fit by maximum
likelihood:

```
log-odds(S) = α[maze] + γ · window_length + Σ over landmarks and predictors  β · gaze_feature
```

- `α[maze]`: one offset per maze, the baseline S-ness of that maze, which absorbs
  everything constant within a maze (arm positions, what is drawn). Mazes with
  fewer than 5 H or 5 S trials are dropped. Sessions are pooled within a maze.
- `γ`: a window-length control, since dwell and visit counts grow with window
  length.
- `β`: the effect of each gaze feature with the maze held fixed. Positive
  means looking like this goes with S, negative with H. Features are centred
  and z-scored, so bars are comparable.

**Gaze features** per landmark (origin, LU, LD, RU, RD; only those visited on
5–95% of trials): *occupied* (visited at all vs not), *visit count*, and *mean
fixation duration*. Total dwell is left out because it equals visits × duration.
A feature is dropped for a landmark (shown as × in the figure, reason in the CSV)
if it barely varies, is nearly explained by the landmark's other features, or has
zero variance.

**Output.** `regression/out/<window>/coefficients.png`: one panel per monkey,
maze offsets on the left, gaze coefficients on the right with 95% confidence
intervals. Solid bar = interval excludes 0. ★ = still significant after Holm
correction over all ~15 coefficients. Also `coefficients.csv` and `mazes.csv`.

**Limits.** Associational, not causal: gaze may follow the strategy rather than
cause it. Trials are not independent (shared session and neighbours), so
intervals are somewhat optimistic. Offsets are shown for scale but not tested.
Most mazes are nearly all-H or all-S and are dropped, so the estimates rest on
the few mixed mazes.

Labels here are the dendrogram labels; the window is `geofix`.

### `scatter/`: where the eyes sit before the flash

A descriptive check, not an H-vs-S test. Each trial is reduced to its mean
gaze (degrees, unwarped) over `fix_start` → `flash_one`: the animal is already
fixating and the flash has not appeared. Each maze is then one dot at the mean
of those trial means, with ±1 SD bars in x and y, inside a dashed 2.5° circle
around the fixation target.

A second figure is a table of directional p-values: for maze pair (i, j), how
far maze i's mean sits from maze j's, scaled by maze j's own spread
(uncorrected for multiple comparisons). Output: `scatter/out/<Monkey>/by_maze.png`
and `maze_pvalues.png`. Brought back from `zarchive/` on 2026-10-02; it reads
only the pooled eye and behavioral caches.
