# PFC strategy selection

Python implementation of left/right choice and strategy decision-variable (DV) traces from neural firing rates.

## Setup

```bash
uv sync
```

### Data layout


| Tree                                                                                 | Contents                                                                                                     |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `data/mat/{Behavioral_Data,Eye_Data,Neural_Data,Single_Trial_List}/{Faure,Nielsen}/` | Raw `.mat` files                                                                                             |
| `data/npz/...`                                                                       | Converted `.npz` caches (same tree)                                                                          |
| `data/processed/`                                                                    | Derived products (`{session}_trial_timebins.npz`, `{Monkey}_eye_data.npz`, `{Monkey}_clean_eye_data.npz`, …) |
| `data/archive/`                                                                      | Prior `raw/` / `processed/` trees                                                                            |


Set `MONKEY` / `SESSION` in `data/config.py`. All data-tree path helpers live there.

### Convert mat → npz

```bash
uv run -m data.convert all
uv run -m data.convert eye --monkey Faure
uv run -m data.convert behavioral neural
```



### Clean eye data

```bash
uv run python -c "from data.loader import load_clean_eye_data; load_clean_eye_data('Faure')"
```

Writes `data/processed/<Monkey>_clean_eye_data.npz` as a pymovements event table. Each row is a saccade or fixation (`name`, `onset`/`offset`/`duration` in ms from `geo_present`, `location_x`/`location_y`, `peak_velocity`, `amplitude`, `dispersion`, plus `session` and `trial_indices_all`). Blinks are detected from `pupil_size` and removed (with padding) before saccade/fixation labeling; out-of-bounds gaze is also dropped.

### Gaze codebook (k-means)

Fit k-means prototypes on `{Monkey}_eye_data.npz`. Gaze is warped by `h1`–`h6` so every maze is a unit H with exits at `(-1, 1)`, `(-1, -1)`, `(1, 1)`, and `(1, -1)`. Prototypes are clipped to the `[-3, 3]²` maze box. Each pymovements fixation event is assigned as a whole to the nearest prototype of its maze-space centroid (radius 1.0); otherwise `state_id = -1`. Blinks, saccades, and out-of-maze samples stay unassigned.

```bash
uv run python -m data.attractor
uv run python -m data.attractor --monkey Faure --k 12
uv run python -m data.attractor --sweep
uv run python -m data.attractor --monkey Faure --k 12 --refit
```

Writes `data/processed/<Monkey>_attractor_eye_data.npz` (default K=12; `--sweep` writes `<Monkey>_attractor_k{K}_eye_data.npz` for K in 4, 5, 6, 7, 8, 10, 12). Prototype centers are saved to `<Monkey>_attractor_k{K}_codebook.npz`. Each trial stores snapped `(x, y)`, a state ID per sample, a validity mask, and a state-run table (`transition_*`).

```bash
uv run python -m eye_data_plotting.pre_flash_attractor_saccades --session june_24_g0 --maze 1
uv run python -m eye_data_plotting.pre_flash_attractor_saccades --session june_24_g0 --maze 1 --view trials
uv run python -m eye_data_plotting.pre_flash_attractor_saccades --session june_24_g0 --maze 3 --view trial --trial-id 141
```

Output: `eye_data_plotting/out/pre_flash_attractor_saccades/<session>/k<K>/maze_<n>/`

### Rsync without big data

```bash
rsync -av --exclude-from=.rsync-exclude ./ user@host:path/strategy-selection/
```



## Run

To test the lr dv model:

```bash
uv run -m decoders.lr
```

To generate all decision variable plots:

```bash
uv run -m trace_plotting.lr_trial_traces
uv run -m trace_plotting.lr_pre_flash_traces
uv run -m trace_plotting.strategy_pre_flash_traces
```

Pre-fixation eye saccade plots (per session and maze):

```bash
uv run python -m eye_data_plotting.pre_flash_saccades --session june_24_g0 --maze 1
uv run python -m eye_data_plotting.pre_flash_saccades --session june_24_g0 --maze 1 --view trials
uv run python -m eye_data_plotting.pre_flash_saccades --session june_24_g0 --maze 1 --view trial --trial-id 141
uv run python -m eye_data_plotting.pre_flash_saccades --session june_24_g0 --maze 1 --mode 3d --view trial --trial-id 141 --save
```

2D plots save by default. 3D plots display interactively unless `--save` is passed.
Output: `eye_data_plotting/out/pre_flash_saccades/<session>/maze_<n>/`

Pre-fixation gaze heatmaps (all sessions, by maze, ±20°, linear count/trial):

```bash
uv run python -m eye_data_plotting.pre_flash_heatmaps --monkey Faure
uv run python -m eye_data_plotting.pre_flash_heatmaps --monkey Faure --maze 1
```

Output: `eye_data_plotting/out/pre_flash_heatmaps/<monkey>/bin<deg>_maze_<n>.png`

Pooled raw-count heatmaps (hierarchical mazes vs sequential mazes):

```bash
uv run python -m eye_data_plotting.pre_flash_heatmaps_sum --monkey Faure
uv run python -m eye_data_plotting.pre_flash_heatmaps_sum --monkey Nielsen
```

Output: `eye_data_plotting/out/pre_flash_heatmaps_sum/<monkey>/`

Same pooled maps on a log1p count scale:

```bash
uv run python -m eye_data_plotting.pre_flash_heatmaps_sum_log --monkey Faure
uv run python -m eye_data_plotting.pre_flash_heatmaps_sum_log --monkey Nielsen
```

Output: `eye_data_plotting/out/pre_flash_heatmaps_sum_log/<monkey>/`

Pooled raw-count heatmaps normalized to unit H (hierarchical vs sequential):

```bash
uv run python -m eye_data_plotting.pre_flash_heatmaps_sum_norm --monkey Faure
uv run python -m eye_data_plotting.pre_flash_heatmaps_sum_norm --monkey Nielsen
```

Output: `eye_data_plotting/out/pre_flash_heatmaps_sum_norm/<monkey>/`

Same unit-H pooled maps on a log1p count scale:

```bash
uv run python -m eye_data_plotting.pre_flash_heatmaps_sum_log_norm --monkey Faure
uv run python -m eye_data_plotting.pre_flash_heatmaps_sum_log_norm --monkey Nielsen
```

Output: `eye_data_plotting/out/pre_flash_heatmaps_sum_log_norm/<monkey>/`

Unit-H maze-to-maze heatmap similarities (session-averaged split-half Pearson *r*, 6×6 grid). Linear `count/trial`:

```bash
uv run python -m eye_data_plotting.pre_flash_similarities_heatmap --monkey Faure
uv run python -m eye_data_plotting.pre_flash_similarities_heatmap --monkey Nielsen
```

Output: `eye_data_plotting/out/pre_flash_similarities_heatmap/<monkey>/`

Same CV on `log1p(count/trial)` maps:

```bash
uv run python -m eye_data_plotting.pre_flash_similarities_heatmap_log --monkey Faure
uv run python -m eye_data_plotting.pre_flash_similarities_heatmap_log --monkey Nielsen
```

Output: `eye_data_plotting/out/pre_flash_similarities_heatmap_log/<monkey>/`

k12 codebook occupancy similarities (session-averaged split-half Pearson *r*, 6×6 grid):

```bash
uv run python -m eye_data_plotting.pre_flash_similarities_occupancy --monkey Faure
uv run python -m eye_data_plotting.pre_flash_similarities_occupancy --monkey Nielsen
```

Output: `eye_data_plotting/out/pre_flash_similarities_occupancy/<monkey>/`

k12 binary occupancy similarities (visited=1 / not visited=0, session-averaged split-half Pearson *r*):

```bash
uv run python -m eye_data_plotting.pre_flash_similarities_occupancy_bin --monkey Faure
uv run python -m eye_data_plotting.pre_flash_similarities_occupancy_bin --monkey Nielsen
```

Output: `eye_data_plotting/out/pre_flash_similarities_occupancy_bin/<monkey>/`

k12 transition-order similarities (bigram + trigram run paths, session-averaged split-half Pearson *r*):

```bash
uv run python -m eye_data_plotting.pre_flash_similarities_transition --monkey Faure
uv run python -m eye_data_plotting.pre_flash_similarities_transition --monkey Nielsen
```

Output: `eye_data_plotting/out/pre_flash_similarities_transition/<monkey>/`

### Figures

Trace figures are saved to `trace_plotting/out/`. The naming scheme is `<decoder>_<window>_traces.png`:

- `lr_trial_traces.png` — left/right DV during the trial (flash_one to ~200 ms after flash_three)
- `lr_pre_flash_traces.png` — left/right DV prior to flash one (geo_present to flash_one)
- `strategy_pre_flash_traces.png` — hierarchical/sequential strategy DV prior to flash one (geo_present to flash_one)

