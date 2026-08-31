# PFC strategy selection

Python implementation of left/right choice and strategy decision-variable (DV)
traces from neural firing rates, plus an eye-data classifier for the
hierarchical/sequential strategy state.

| Document | Covers |
| -------- | ------ |
| [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | every field in the raw `.mat` files |
| [eye_data_classifier/classifier.md](eye_data_classifier/classifier.md) | strategy decoding from gaze: feature sets, models, CV design, how to read the tables |
| [alt_defense.md](alt_defense.md) | the reviewer-response framing the decoding results support, with the numbers behind it |
| [choosing_k.md](choosing_k.md) | how the codebook size K is selected |
| [similarity.md](similarity.md) | maze-to-maze split-half similarity CV, and results across k |
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

**Choosing K** is its own analysis — three scripts, only one of which can
actually select it. See [choosing_k.md](choosing_k.md).

## Eye-data strategy classifier

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
uv run python -m eye_data_classifier.pipeline            # everything, in order
uv run python -m eye_data_classifier.decoding --scope publication
uv run python -m eye_data_classifier.decoding --scope all
```

Output under `eye_data_classifier/out/decoding/<scope>/`. Full reference:
[eye_data_classifier/classifier.md](eye_data_classifier/classifier.md).

## Decision-variable traces

```bash
uv run -m decoders.lr                              # test the lr DV model
uv run -m trace_plotting.lr_trial_traces
uv run -m trace_plotting.lr_pre_flash_traces
uv run -m trace_plotting.strategy_pre_flash_traces
uv run -m trace_plotting.strategy_pre_flash_per_trial_traces
uv run -m trace_plotting.strategy_post_flash_per_trial_traces
```

Saved to `trace_plotting/out/` as `<decoder>_<window>_traces.png`:

- `lr_trial_traces.png` — left/right DV during the trial (flash_one to ~200 ms
  after flash_three)
- `lr_pre_flash_traces.png` — left/right DV prior to flash one (geo_present to
  flash_one)
- `strategy_pre_flash_traces.png` — hierarchical/sequential strategy DV prior to
  flash one (geo_present to flash_one)

## Eye-data plots

Every script writes to `eye_data_plotting/out/<script name>/`. All take
`--monkey`; the per-trial viewers take `--session`, `--maze`, `--view`
(`maze` / `trials` / `trial`), `--trial-id`, and `--mode`.

**Per-trial and per-maze gaze**

| Script | Shows |
| ------ | ----- |
| `pre_flash_saccades` | pre-fixation gaze aligned to `fix_start`, per session and maze |
| `pre_flash_labeled_saccades` | the same, with pymovements event labels |
| `pre_flash_attractor_saccades` | the same, with codebook state assignments (`out/.../k<K>/maze_<n>/`) |
| `pre_flash_movie` | eye-tracking QC movies for the june_24 session |

2D plots save by default; 3D plots (`--mode 3d`) display interactively unless
`--save` is passed.

```bash
uv run python -m eye_data_plotting.pre_flash_saccades --session june_24_g0 --maze 1
uv run python -m eye_data_plotting.pre_flash_saccades --session june_24_g0 --maze 1 --view trials
uv run python -m eye_data_plotting.pre_flash_attractor_saccades --session june_24_g0 --maze 3 --view trial --trial-id 141
```

**Gaze heatmaps.** `pre_flash_heatmaps` is per maze across all sessions (±20°,
linear count/trial). The `_sum` family pools hierarchical mazes against
sequential mazes; the suffixes compose:

| Suffix | Effect |
| ------ | ------ |
| *(none)* | raw counts, degrees |
| `_log` | `log1p` count scale |
| `_norm` | normalized to unit H |
| `_diff` | hierarchical minus sequential, unit H only |

giving `pre_flash_heatmaps_sum`, `_sum_log`, `_sum_norm`, `_sum_log_norm`,
`_sum_norm_diff`, and `_sum_log_norm_diff`.

```bash
uv run python -m eye_data_plotting.pre_flash_heatmaps --monkey Faure --maze 1
uv run python -m eye_data_plotting.pre_flash_heatmaps_sum_log_norm --monkey Nielsen
```

**Maze-to-maze similarity** (6×6 session-averaged split-half Pearson *r*).
Procedure and results: [similarity.md](similarity.md).

| Script | Correlates |
| ------ | ---------- |
| `pre_flash_similarities_heatmap` | unit-H gaze maps, linear `count/trial` |
| `pre_flash_similarities_heatmap_log` | the same, `log1p(count/trial)` |
| `pre_flash_similarities_heatmap_fixations` | the same, fixation samples only |
| `pre_flash_similarities_heatmap_saccades` | the same, saccade samples only |
| `pre_flash_similarities_occupancy` | seconds per codebook state |
| `pre_flash_similarities_occupancy_bin` | visited / not visited per state |
| `pre_flash_similarities_transition` | bigram + trigram state-run paths |

```bash
uv run python -m eye_data_plotting.pre_flash_similarities_heatmap --monkey Faure
uv run python -m eye_data_plotting.pre_flash_similarities_occupancy --monkey Nielsen --k 12
```

**Codebook size.** `pre_flash_centroid_stability_better`,
`pre_flash_centroid_stability`, `pre_flash_mse_error` — see
[choosing_k.md](choosing_k.md).

## Running on the cluster

The full classifier pipeline runs end to end on Engaging under Slurm. Push code
up, run, pull figures back; `data/mat/` lives on the cluster and never moves.
See [cloud.md](cloud.md).
