# PFC strategy selection

Python implementation of left/right choice and strategy decision-variable (DV)
traces from neural firing rates, plus an eye-data classifier for the
hierarchical/sequential strategy state.

| Document | Covers |
| -------- | ------ |
| [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | every field in the raw `.mat` files |
| [eye_pre_flash/classifier/classifier.md](eye_pre_flash/classifier/classifier.md) | strategy decoding from gaze: feature sets, models, CV design, how to read the tables |
| [alt_defense.md](alt_defense.md) | the reviewer-response framing the decoding results support, with the numbers behind it |
| [similarity.md](similarity.md) | maze-to-maze split-half similarity CV, and results across k |
| [eye_pre_flash/clusters/clusters.md](eye_pre_flash/clusters/clusters.md) | choosing the codebook size K per maze by how much strategy information it carries |
| [cloud.md](cloud.md) | running the pipeline on Engaging (Slurm, quota, rsync) |

## Setup

```bash
uv sync
```

### Data layout

| Tree | Contents |
| ---- | -------- |
| `data/mat/{Behavioral_Data,Eye_Data,Neural_Data,Single_Trial_List}/{Faure,Nielsen}/` | Raw `.mat` files |
| `data/npz/...` | Converted `.npz` caches (same tree) |
| `data/processed/` | Derived products (`{session}_trial_timebins.npz`, `{Monkey}_eye_data.npz`, `{Monkey}_clean_eye_data.npz`, …) |
| `data/archive/` | Prior `raw/` / `processed/` trees |

Set `MONKEY` / `SESSION` in `data/config.py`. All data-tree path helpers live
there.

### Convert mat → npz

```bash
uv run -m data.convert all
uv run -m data.convert eye --monkey Faure
uv run -m data.convert behavioral neural
uv run -m data.convert neural --sessions june_24_g0
```

`behavioral` / `neural` / `eye` / `single_trial` with no `--monkey` convert every
session under `data/mat/` for both animals. Neural conversion is the large step
(multi-GB `.mat` per session) — see [cloud.md](cloud.md) for why the cluster
pipeline converts it selectively.

### Neural strategy labels (per session)

Clustering runs **per session** (neuron IDs are session-local; do not stack
`trial_timebins` across days). The four *publication* sessions are:

| Monkey | Session |
| ------ | ------- |
| Faure | `june_24_g0`, `june_8_g0` |
| Nielsen | `Nov_3_g0`, `Oct_22_g0` |

`data.labeler.CLUSTERING_SESSIONS` widens this to the **23** sessions whose
clustering separates maze 1 from maze 6 at `snr_auc >= 0.95`
(`zrefs/Dendogram_all/SNR_All_Sessions/all_session_snr_results.csv`; the four
above rank #2-#5 of 61). Adding sessions does not change any existing label,
since each is clustered alone. Labels are built one session at a time by
`python -m data.labeler --type strategy --sweep`, which frees each multi-GB
neural npz once its ~5 KB label is written.

Each session is first reduced to the `min_value:max_value` window from
`data/mat/Single_Trial_List/`. Kept trials are visible (`trial_fade == 0`), pass
photodiode QC, and have `path_type != -99`; kept neurons were recorded on every
trial of the window and clear `FR_THRESH = 1` on at least one maze.

The first 500 ms after flash_one is then averaged per trial and neuron,
z-scored, reduced to 3 PCs, and Ward-clustered (`maxclust=2`). The cluster with
more maze-1 trials is hierarchical (`0`); more maze-6 trials is sequential (`1`).

```bash
uv run -m data.convert behavioral neural single_trial --monkey Faure
uv run python -m data.labeler --type strategy
uv run python -m data.labeler --type strategy --session june_24_g0
uv run python -m data.labeler --type lr --session june_24_g0
```

Writes `{session}_trial_timebins.npz`, `{session}_trial_metadata.npz`, and
`{session}_strategy_choices.npz` under `data/processed/`.

### Clean eye data

```bash
uv run python -c "from data.loader import load_clean_eye_data; load_clean_eye_data('Faure')"
```

Writes `data/processed/<Monkey>_clean_eye_data.npz` as a pymovements event
table. Each row is a saccade or an I-DT fixation (`name`, `onset`/`offset`/
`duration` in ms from `geo_present`, `location_x`/`location_y`, `peak_velocity`,
`amplitude`, `dispersion`, plus `session` and `trial_indices_all`). Blinks are
detected from `pupil_size` and removed (with padding) before labeling;
out-of-bounds gaze is also dropped.

Fixations come back **exactly as I-DT detected them**. A slow drift that crosses
`IDT_DISPERSION_THRESHOLD_DEG` (2.0°) ends one fixation and starts the next, and
nothing stitches the pieces back together — so every row's `dispersion` really
is the spread of the event it describes, and no fixation spans more gaze travel
than the threshold allows.

### Unit-H warp

```bash
uv run python -m data.attractor --monkey Faure
```

Warps gaze by `h1`–`h6` so every maze becomes the same unit H, with exits at
`(-1, 1)`, `(-1, -1)`, `(1, 1)` and `(1, -1)`. Writes
`data/processed/<Monkey>_attractor_eye_data.npz`: warped `(x, y)` per sample, a
validity mask restricted to fixation samples, and a timebase.

**No codebook, no state IDs.** This cache is K-independent — one per monkey,
serving every K — because the codebook is fit downstream by
`eye_pre_flash.classifier.features`, on samples already clipped to the analysis
window. There is exactly one codebook in the analysis and that is where it
lives.

### Gaze codebook and features (`eye_pre_flash/classifier/features.py`)

```bash
uv run python -m eye_pre_flash.classifier.features
```

The five steps, in order:

1. Clip each trial to `[fix_start - 1466 ms, fix_start]`.
2. Warp to unit H (step above).
3. K-means over the pooled in-window fixation samples, per monkey, across
   sessions — K = 6 and 12.
4. Locate fixations with I-DT only, on the clipped but **unwarped** data, where
   a dispersion threshold in degrees is physically meaningful.
5. Assign each fixation's **mean** warped position to the nearest prototype whose
   unit ball (radius 1.0) contains it. Saccades, blinks and fixations outside
   every ball are omitted.

Writes `data/processed/<Monkey>_clf2_k{K}_unith_s1466_e0.npz` with three feature
blocks per trial: `occ_ms` (time occupancy), `occ_bin` (binary occupancy) and
`bigram` (transition proportions between clusters).

Consecutive fixations landing on the same cluster need no merge step: runs are
collapsed over the assigned-only sample subsequence, so they already count once.

## Pre-flash eye data (`eye_pre_flash/`)

Everything about gaze during the pre-fixation viewing period lives under
`eye_pre_flash/`: the codebook, feature blocks and strategy classifier in
`classifier/`, the label-source registry in `label_sources.py`, the single-maze
H vs S similarity in `maze_strategy_pairs/`, the per-maze codebook-size sweep in
`clusters/`, the H/S trial census in `counts/`, the fix-start scatter in
`scatter/`, descriptive plots in `plotting/`.

### Strategy classifier (`eye_pre_flash/classifier/`)

Tests whether pre-fixation gaze predicts the trial's strategy label —
behavioural evidence that the pre-fixation period involves active evaluation
rather than a passive response to visual geometry.

Simple feature sets (codebook occupancy in ms and binary, state bigrams, each
at K=6 and K=12; gaze heatmaps at two resolutions), two fixed models
(logistic regression and a random forest), and one metric: **balanced CV
accuracy** (chance = 0.500), with training-set accuracy printed beside it to
expose overfitting. Tables are grouped by session scope (`publication` = the 4
clustering sessions, `all` = the 23 clearing `snr_auc >= 0.95`), split by
monkey, and run in two training regimes:

- **per-maze** — one classifier per maze, raw degree coordinates (geometry is
  constant inside a maze, so no unit-H warp).
- **across-maze** — one classifier over all mazes, unit-H coordinates to
  mitigate visual geometry, with a maze-identity reference row showing how much
  accuracy maze id alone buys.

```bash
uv run python -m eye_pre_flash.classifier.pipeline            # everything, in order
uv run python -m eye_pre_flash.classifier.decoding --scope publication
uv run python -m eye_pre_flash.classifier.decoding --scope all
```

Output under `eye_pre_flash/classifier/out/decoding/<scope>/`. Full reference:
[eye_pre_flash/classifier/classifier.md](eye_pre_flash/classifier/classifier.md).

### Single-maze H vs S similarity (`eye_pre_flash/maze_strategy_pairs/`)

Tests whether, **within one maze**, gaze looks less like itself across the two
decoded strategies than it does across split halves of either strategy alone —
the one form of the question where visual geometry is held identical on both
sides and so cannot explain the answer.

A 2x2 matrix per maze. Trials for that maze are **pooled across every
in-scope session** (per-session cells are too thin to estimate from), then cut
into four disjoint groups of equal size `m = min(n_H, n_S) // 2` — two from H,
two from S. Each of the four cross-group pairings is scored, and the whole
thing is repeated over 100 fresh random groupings and averaged. The diagonal
is within-strategy similarity, the off-diagonal cross-strategy. The statistic
is `Δ = 0.5·(r_HH + r_SS) − 0.5·(r_HS + r_SH)`, tested against a label-shuffle
null that reshuffles **within each session**, which keeps every session's H/S
counts fixed so the null carries the same mix of same-session trial pairs as
the data.

Two ways of scoring a pairing are swept into parallel output trees:
`trial_by_trial` averages the correlations between individual trials, while
`block_means` averages each group into one vector first and correlates those.
Their cells are on different scales — averaging suppresses trial noise before
the correlation, so `block_means` runs much higher and rises with group size —
so compare `z` between the trees, never the raw correlations.

One figure per (method, monkey, maze, source, feature, variant), carrying both
K = 6 and K = 12 as side-by-side panels; all three variants (`full`,
`no_origin`, `mean_removed`) are swept. Every axis takes a list. Runs on a
laptop in about 15 minutes — no cluster job needed, only the cached labels and
feature blocks.

```bash
uv run python -m eye_pre_flash.maze_strategy_pairs.build                    # full sweep, both methods
uv run python -m eye_pre_flash.maze_strategy_pairs.build --dry-run          # targets + pooled trial counts
uv run python -m eye_pre_flash.maze_strategy_pairs.build --monkey Faure --maze 2
uv run python -m eye_pre_flash.maze_strategy_pairs.build --estimator trial_by_trial
uv run python -m eye_pre_flash.maze_strategy_pairs.checks                   # invariant checks
```

Output under
`eye_pre_flash/maze_strategy_pairs/out/<method>/<source>/<monkey>/<variant>/`,
with `results.csv` beside the figures. A maze needs at least 10 pooled trials
of each strategy or it gets no figure.
How to read a panel, and what it does not control for:
[eye_pre_flash/maze_strategy_pairs/CAVEATS.md](eye_pre_flash/maze_strategy_pairs/CAVEATS.md).

### Codebook size per maze (`eye_pre_flash/clusters/`)

Asks which codebook size K carries the most information about the strategy
label, separately for each (monkey, maze), with the k-means fitted **per maze**
rather than once per monkey. The K = 6 and K = 12 used everywhere else are
inherited values; the study that chose them is gone from the tree, and what
survives in [similarity.md](similarity.md) ranks K by reliability and by
maze-identity discriminability, pooled over mazes.

The score is held-out information in bits per trial (`CE_prior - CE_model`)
with cross-validated R² beside it. It has to be held-out: in-sample variance
explained and in-sample mutual information both rise monotonically with K, so
an uncorrected version answers "largest K" by construction. Folds are grouped
by session by default, because session-blind folds let the model fingerprint
the session and recover its class prior — a leak that grows with K and would
bias the winner upward.

```bash
uv run python -m eye_pre_flash.clusters.checks                # equivalence + invariants
uv run python -m eye_pre_flash.clusters.build --dry-run       # target cells + trial counts
uv run python -m eye_pre_flash.clusters.build
sbatch slurm/run_clusters.sbatch                              # the full sweep
```

Output under `eye_pre_flash/clusters/out/<cv>/<source>/<monkey>/`, with
`best_k.png` as the headline table and `results.csv` beside the figures. Unlike
`maze_strategy_pairs`, this needs a cluster job: it builds a per-monkey fixation
table off the attractor and clean-event caches first.

**Read the minority-class column before any winning K.** Per-cell sample size
spans 17 to 386 trials, and in the thin cells the winner is set by n rather
than by information; `d_ceiling` flags those. The census and the ten caveats
are in
[eye_pre_flash/clusters/clusters.md](eye_pre_flash/clusters/clusters.md).

### Fix-start scatter (`eye_pre_flash/scatter/`)

One (x, y) per usable trial — the mean gaze over the window from `fix_start`
to `flash_one`, i.e. where the eyes sit once pre-flash fixation is established
but before anything has appeared — averaged per maze into one mean and SD,
across trials, of that per-trial mean.

`by_maze.png` plots the six per-maze means as dots with ±1 SD bars.
`maze_pvalues.png` puts a number on how separated those dots are: for each
ordered pair of mazes `(i, j)`, it standardizes maze `i`'s mean by maze `j`'s
own SD on each axis (`zx`, `zy`), combines them as `zx**2 + zy**2` — a
chi-square(2 df) statistic under the null that maze `i`'s mean is a typical
draw from maze `j`'s per-trial spread — and reports the closed-form p-value
`exp(-chi2/2)`. This is **directional** (`p[i][j] != p[j][i]` in general,
since each cell uses a different maze's SD) and **uncorrected** for the 30
comparisons per monkey — read raw p-values, not a family-wise significance
call.

```bash
uv run python -m eye_pre_flash.scatter.build
uv run python -m eye_pre_flash.scatter.build --monkey Faure
```

Output under `eye_pre_flash/scatter/out/<monkey>/`. Must run where the
eye/behavioral caches live (the cluster) — see [cloud.md](cloud.md).

## Decision-variable traces (`neural_traces/`)

DV decoders refit from neural firing rates live in `neural_traces/decoders/`;
the scripts below plot their single-trial and averaged traces.

```bash
uv run -m neural_traces.decoders.lr                              # test the lr DV model
uv run -m neural_traces.plotting.lr_trial_traces
uv run -m neural_traces.plotting.lr_pre_flash_traces
uv run -m neural_traces.plotting.strategy_pre_flash_traces
uv run -m neural_traces.plotting.strategy_pre_flash_per_trial_traces
uv run -m neural_traces.plotting.strategy_post_flash_per_trial_traces
```

Saved to `neural_traces/plotting/out/` as `<decoder>_<window>_traces.png`:

- `lr_trial_traces.png` — left/right DV during the trial (flash_one to ~200 ms
  after flash_three)
- `lr_pre_flash_traces.png` — left/right DV prior to flash one (geo_present to
  flash_one)
- `strategy_pre_flash_traces.png` — hierarchical/sequential strategy DV prior to
  flash one (geo_present to flash_one)

### Plots (`eye_pre_flash/plotting/`)

Four packages, each owning its own `out/`, so figures land beside the code that
makes them. All take `--monkey`; the per-trial viewer takes `--session`,
`--maze`, `--view` (`average` / `trials` / `trial`) and `--trial-id`.

| Package | Writes to | Contents |
| ------- | --------- | -------- |
| `saccades/` | `saccades/out/` | per-trial gaze with codebook state assignments |
| `heatmaps/` | `heatmaps/out/` | gaze heatmaps, by maze regime (`maze/`) or decoded label (`label/`) |
| `similarities/` | `similarities/out/` | maze-to-maze similarity matrices |
| `movie/` | `movie/out/` | eye-tracking QC movies |

**Per-trial gaze** (`saccades/`, `movie/`)

`saccades.attractor` is the one per-trial viewer: raw gaze, the prototype it
snapped to, and the codebook, written to
`saccades/out/<session>/k<K>/maze_<n>/`. Shaded spans mark each assigned
fixation; unshaded stretches are gaze that was moving, blinking, or outside
every cluster's unit ball.
| `movie.qc` | eye-tracking QC movies for the june_24 session |

```bash
uv run python -m eye_pre_flash.plotting.saccades.attractor --session june_24_g0 --maze 3
uv run python -m eye_pre_flash.plotting.saccades.attractor --session june_24_g0 --maze 3 --view trials --k 6
uv run python -m eye_pre_flash.plotting.saccades.attractor --session june_24_g0 --maze 3 --view trial --trial-id 308
```

**Gaze heatmaps by maze regime** (`heatmaps/maze/`). `core` is per maze across
all sessions (±20°, linear count/trial). The `sum` family pools hierarchical
mazes against sequential mazes (`data.labeler.MAZE_GROUPS`); the suffixes
compose:

| Suffix | Effect |
| ------ | ------ |
| *(none)* | raw counts, degrees |
| `_log` | `log1p` count scale |
| `_norm` | normalized to unit H |
| `_diff` | hierarchical minus sequential, unit H only |

giving `sum`, `sum_log`, `sum_norm`, `sum_log_norm`, `sum_norm_diff`, and
`sum_log_norm_diff`.

```bash
uv run python -m eye_pre_flash.plotting.heatmaps.maze.core --monkey Faure --maze 1
uv run python -m eye_pre_flash.plotting.heatmaps.maze.sum_log_norm --monkey Nielsen
```

**Gaze heatmaps by decoded label** (`heatmaps/label/`). The same six
renderings, with the same module names, but the two pools are the trials the
neural clustering *labelled* hierarchical and sequential rather than the mazes
assumed to be solved that way — all six mazes mix into both pools. Only
sessions carrying a neural label contribute, so these take `--scope`
(`publication` / `all` / `allplus`, exactly as in `classifier/`) and default to
generating all three:

```bash
uv run python -m eye_pre_flash.plotting.heatmaps.label.sum
uv run python -m eye_pre_flash.plotting.heatmaps.label.sum_log_norm --scope publication
uv run python -m eye_pre_flash.plotting.heatmaps.label.sum_norm_diff --monkey Faure --scope all
```

Output is `heatmaps/out/label/<scope>/<script>/<monkey>/`, one level deeper than
the maze family. Compare a pair side by side to see how much of a maze-grouped
contrast survives when the grouping stops assuming the maze fixes the strategy.

**Maze-to-maze similarity** (`similarities/`, 6×6 session-averaged split-half
Pearson *r*). Procedure and results: [similarity.md](similarity.md).

| Script | Correlates |
| ------ | ---------- |
| `similarities.heatmap` | unit-H gaze maps, linear `count/trial` |
| `similarities.heatmap_log` | the same, `log1p(count/trial)` |
| `similarities.heatmap_fixations` | the same, fixation samples only |
| `similarities.heatmap_saccades` | the same, saccade samples only |
| `similarities.occupancy` | seconds per codebook state |
| `similarities.occupancy_bin` | visited / not visited per state |
| `similarities.transition` | state bigram paths |

```bash
uv run python -m eye_pre_flash.plotting.similarities.heatmap --monkey Faure
uv run python -m eye_pre_flash.plotting.similarities.occupancy --monkey Nielsen --k 12
```

## Running on the cluster

Every cache rebuilds from cold with one job (`slurm/run_rebuild.sbatch`), and
the full classifier pipeline runs end to end under Slurm
(`slurm/run_classifier.sbatch`). Push code up, run, pull figures back;
`data/mat/` lives on the cluster and never moves. See [cloud.md](cloud.md).
