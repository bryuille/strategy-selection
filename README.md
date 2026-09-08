# PFC strategy selection

Python implementation of left/right choice and strategy decision-variable (DV)
traces from neural firing rates, plus an eye-data classifier for the
hierarchical/sequential strategy state.

| Document | Covers |
| -------- | ------ |
| [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | every field in the raw `.mat` files |
| [eye_pre_flash/classifier/classifier.md](eye_pre_flash/classifier/classifier.md) | strategy decoding from gaze: feature sets, models, CV design, how to read the tables |
| [eye_post_flash/post_flash.md](eye_post_flash/post_flash.md) | post-feedback counterfactual saccades by strategy regime: the measure, the alternative-ranking model, alignment modes |
| [alt_defense.md](alt_defense.md) | the reviewer-response framing the decoding results support, with the numbers behind it |
| [similarity.md](similarity.md) | maze-to-maze split-half similarity CV, and results across k |
| [eye_pre_flash/corr/corr.md](eye_pre_flash/corr/corr.md) | the 12×12 (maze × strategy) occupancy similarity matrix: dendro vs. SVM labels, the origin/PC1 variants, the permutation null |
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
table. Each row is a saccade or fixation (`name`, `onset`/`offset`/`duration` in
ms from `geo_present`, `location_x`/`location_y`, `peak_velocity`, `amplitude`,
`dispersion`, plus `session` and `trial_indices_all`). Blinks are detected from
`pupil_size` and removed (with padding) before saccade/fixation labeling;
out-of-bounds gaze is also dropped.

### Gaze codebook (k-means)

Fit k-means prototypes on `{Monkey}_eye_data.npz`. Gaze is warped by `h1`–`h6`
so every maze is a unit H with exits at `(-1, 1)`, `(-1, -1)`, `(1, 1)`, and
`(1, -1)`. Prototypes are clipped to the `[-3, 3]²` maze box. Each pymovements
fixation event is assigned as a whole to the nearest prototype of its maze-space
centroid (radius 1.0); otherwise `state_id = -1`. Blinks, saccades, and
out-of-maze samples stay unassigned.

```bash
uv run python -m data.attractor
uv run python -m data.attractor --monkey Faure --k 12
uv run python -m data.attractor --sweep
```

Writes `data/processed/<Monkey>_attractor_eye_data.npz` (default K=12; `--sweep`
writes `<Monkey>_attractor_k{K}_eye_data.npz` for K in 4, 5, 6, 7, 8, 10, 12).
Each trial stores snapped `(x, y)`, a state ID per sample, a validity mask, and a
state-run table (`transition_*`).

K = 12 was selected by codebook stability (held-out coverage × centroid
reproducibility); the selection scripts have since been retired.

## Pre-flash eye data (`eye_pre_flash/`)

Everything about gaze during the pre-fixation viewing period lives under
`eye_pre_flash/`: the strategy classifier in `classifier/`, descriptive plots
in `plotting/`.

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

Five families, one subpackage each; every script writes to
`eye_pre_flash/plotting/out/<family>/<script>/`. All take `--monkey`; the
per-trial viewers take `--session`, `--maze`, `--view` (`maze` / `trials` /
`trial`), `--trial-id`, and `--mode`.

| Family | Contents |
| ------ | -------- |
| `saccades/` | per-trial and per-maze gaze traces |
| `heatmap_maze/` | gaze heatmaps grouped by maze regime |
| `heatmap_label/` | the same renderings grouped by decoded neural label |
| `similarities/` | maze-to-maze similarity matrices |
| `movie/` | eye-tracking QC movies |

**Per-trial and per-maze gaze** (`saccades/`, `movie/`)

| Script | Shows |
| ------ | ----- |
| `saccades.traces` | pre-fixation gaze aligned to `fix_start`, per session and maze |
| `saccades.labeled` | the same, with pymovements event labels |
| `saccades.attractor` | the same, with codebook state assignments (`out/.../k<K>/maze_<n>/`) |
| `movie.qc` | eye-tracking QC movies for the june_24 session |

2D plots save by default; 3D plots (`--mode 3d`) display interactively unless
`--save` is passed.

```bash
uv run python -m eye_pre_flash.plotting.saccades.traces --session june_24_g0 --maze 1
uv run python -m eye_pre_flash.plotting.saccades.traces --session june_24_g0 --maze 1 --view trials
uv run python -m eye_pre_flash.plotting.saccades.attractor --session june_24_g0 --maze 3 --view trial --trial-id 141
```

**Gaze heatmaps by maze regime** (`heatmap_maze/`). `core` is per maze across
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
uv run python -m eye_pre_flash.plotting.heatmap_maze.core --monkey Faure --maze 1
uv run python -m eye_pre_flash.plotting.heatmap_maze.sum_log_norm --monkey Nielsen
```

**Gaze heatmaps by decoded label** (`heatmap_label/`). The same six
renderings, with the same module names, but the two pools are the trials the
neural clustering *labelled* hierarchical and sequential rather than the mazes
assumed to be solved that way — all six mazes mix into both pools. Only
sessions carrying a neural label contribute, so these take `--scope`
(`publication` / `all` / `allplus`, exactly as in `classifier/`) and default to
generating all three:

```bash
uv run python -m eye_pre_flash.plotting.heatmap_label.sum
uv run python -m eye_pre_flash.plotting.heatmap_label.sum_log_norm --scope publication
uv run python -m eye_pre_flash.plotting.heatmap_label.sum_norm_diff --monkey Faure --scope all
```

Output is `out/heatmap_label/<scope>/<script>/<monkey>/`, one level deeper than
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
| `similarities.transition` | bigram + trigram state-run paths |

```bash
uv run python -m eye_pre_flash.plotting.similarities.heatmap --monkey Faure
uv run python -m eye_pre_flash.plotting.similarities.occupancy --monkey Nielsen --k 12
```

**One 12×12 matrix over (maze × strategy)** (`eye_pre_flash/corr/`). The 6×6
matrices above cannot separate strategy from geometry, because strategy is
close to a deterministic function of (session, maze). This splits each maze's
trials by a decoded neural strategy label and correlates the resulting
codebook-occupancy vectors, in the same split-half CV style as
`similarities.heatmap_eq` extended to the label axis — independent trial
halves throughout, so the diagonal is a cell's own split-half reliability. It
replaced `similarities.heatmap_labels`.

| | Fixed | Varied | Where |
| --- | --- | --- | --- |
| **boxed cells** | maze | strategy | same maze, H vs S — the decisive read |
| **quadrant** | strategy | maze | inside one strategy block |

Read as `(r(H,H)+r(S,S))/2 − r(H,S)`, which self-normalises: a group whose
trials are less correlatable lowers its own diagonal too. Two label sources —
the existing dendrogram (Ward) clustering, and a maze-1-vs-6 linear SVM
projected onto every trial, for sessions whose neural strategy shape the
dendrogram's 2-cluster split can't separate — and three feature variants
(`full`, `no_origin`, `pc1_removed`) that test whether central fixation is
what compresses the 6×6 matrices' dynamic range. A within-(session, maze)
label-shuffle permutation null stands in for a bootstrap CI, since the
estimator's noise floor at d = 6–144 should be measured, not assumed. Full
reference: [eye_pre_flash/corr/corr.md](eye_pre_flash/corr/corr.md).

```bash
uv run python -m eye_pre_flash.corr.run --scope allplus
uv run python -m eye_pre_flash.corr.run --source svm --feature occupancy --monkey Faure
```

Output: `eye_pre_flash/corr/out/<source>/<feature>/<variant>/<scope>/`.

## Post-flash eye data (`eye_post_flash/`)

Quantifies voluntary post-feedback saccades toward the most likely unchosen
alternative exit, and asks which decision model explains where those saccades
go. `counterfactual` tests whether the looked-at alternative tracks the
hierarchical/sequential regime, by maze regime (`data.labeler.MAZE_GROUPS`)
and by per-trial neural strategy label; `model_comparison` fits the
manuscript's full model set (optimal, lapse, total-time, hierarchical,
postdictive, revision) plus single-interval, proximity and empirical
references to the same trials and ranks them by AIC/BIC. Full reference:
[eye_post_flash/post_flash.md](eye_post_flash/post_flash.md).

```bash
uv run python -m eye_post_flash.counterfactual                   # both monkeys
uv run python -m eye_post_flash.counterfactual --align response  # no cluster step needed
uv run python -m eye_post_flash.model_comparison --align response
uv run python -m eye_post_flash.model_comparison --plots-only  # rebuild figures, no refit
```

`--align feedback` (the reviewer's definition) needs behavioral npz that carry
`feedback_time` — one light cluster job, `sbatch slurm/run_post_flash.sbatch`.
Output under `eye_post_flash/out/counterfactual/<align>/<monkey>/`.

## Running on the cluster

The full classifier pipeline runs end to end on Engaging under Slurm
(`slurm/run_classifier.sbatch`), and the post-flash analysis has its own light
job (`slurm/run_post_flash.sbatch`). Push code up, run, pull figures back;
`data/mat/` lives on the cluster and never moves. See [cloud.md](cloud.md).
