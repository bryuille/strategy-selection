# PFC strategy selection

Does where a monkey looks before it moves predict which strategy it uses?
Gaze in the pre-fixation window is scored against neural strategy labels,
hierarchical (**H**) vs sequential (**S**), for two monkeys (Faure, Nielsen),
several sessions each, six mazes per session.

Every command and flag is in [README_COMMANDS.md](README_COMMANDS.md).

## Setup

Python ≥ 3.14. With uv (`uv.lock` pins the environment):

```bash
uv sync
uv run python -m msp.checks        # sanity check, needs no data
```

Without uv: create a venv, `pip install h5py matplotlib numpy pandas
pymovements scikit-learn scipy` (mirrors `pyproject.toml`), and drop the
`uv run` prefix from every command.

## Data

Set two paths at the top of `data/config.py` (there are no environment
variables):

```python
MAT_ROOT = None  # raw .mat files; required, only ever read
PROCESSED_ROOT = Path("./data/processed")  # everything computed; rebuildable
```

```
<MAT_ROOT>/
├── Behavioral_Data/{Faure,Nielsen}/<session>_good_trials_concat.mat
├── Eye_Data/{Faure,Nielsen}/Eye_Data_<session>.mat
├── Neural_Data/{Faure,Nielsen}/<session>_Whole_Trial_FR_Causal.mat
└── Single_Trial_List/{Faure,Nielsen}/...
```

`PROCESSED_ROOT` may not sit inside `MAT_ROOT`, and needs room for about
50 GB to run every analysis. Most of that (~38 GB) is the per-session
`trial_timebins` caches that `data.labels` leaves behind;
`data.labels --sweep --reclaim-timebins` deletes each one once its labels are
built.

Sessions are `<month>_<day>_g<n>`; `june` / `April` are Faure, `Oct` / `Nov` / `Dec` Nielsen.

To build from scratch:

1. `data.labels`: per-session strategy labels (dendrogram and SVM), from the
   neural mats. About a minute per session.
2. `data.loader`: pooled per-monkey eye caches. They also build on first use,
   but `slurm/run_msp.sbatch` checks for them.
3. Run any analysis; feature caches build on first use.

Loaders check that cache keys exist, not that they match the current code:
after changing detector or QC constants, delete the derived caches
([how](README_COMMANDS.md#clearing-derived-caches)). Raw fields:
[DATA_DICTIONARY.md](DATA_DICTIONARY.md). Cache formats:
[data/data.md](data/data.md). Cluster runs: [README_COMMANDS.md](README_COMMANDS.md#slurm).

## Layout

| Folder        | Contents                                                                 |
| ------------- | ------------------------------------------------------------------------ |
| `data/`       | Shared layer: paths, `.mat` readers, cache builders, strategy labels     |
| `counts/`     | H/S trial census per session and maze                                    |
| `msp/`        | **Main analysis**, maze-strategy pairs; also `msp.per_session`, `msp.saccades`, and `msp/legacy/` |
| `msp_trends/` | msp on coarser merged codebooks, with a pooled-over-mazes estimate       |
| `regression/` | Logistic regression of strategy on gaze, maze held fixed                 |
| `scatter/`    | Mean gaze per maze between fixation and the flash                        |
| `slurm/`      | Batch scripts for Engaging                                               |

Each package reads `data/` and writes only to its own `out/`. When a default
changes, rename old outputs to an explicit tag rather than deleting them.

## Analyses

Labels are either *SVM* (neural classifier trained on mazes 1 vs 6; top-ten
sessions per monkey by `snr_auc`) or *dendrogram* (Ward clustering; the
publication sessions, two per monkey). Gaze is warped to "unit H" (fixation point at the
origin, exits at (±1, ±1)), and each fixation is assigned to the nearest of
five states (origin, LU, LD, RU, RD) within `--radius`, giving tags such as
`r0.5`. The window runs from maze onset to fixation onset (`geofix`).

### `counts/`

H/S trial counts per (session, maze) as heatmaps; cells with a minority
strategy under 10% are marked red. Counts every labelled trial that passes QC.
Output: `counts/out/<source>/hs_<monkey>_<scope>.png`.

### `msp/` (main)

Within a maze, is gaze more alike between same-strategy trial groups than
between H and S groups? SVM labels, mazes 2–5, at least 10 trials per
strategy.

Each trial is a binary visited/not vector over the five states. Each strategy
is split into random halves (100 times), the half-means are correlated, and
`Δ = ½(r_HH + r_SS) − ½(r_HS + r_SH)` is tested against 1000 within-session
label shuffles. Variants: `full`, `no_origin`, `mean_removed`.

Read `z`, not raw `r`, which rises with group size. Under `mean_removed`, a
diagonal can be pulled toward −1 when centring removes most of that strategy's
signal; on imbalanced mazes this hits the majority. Output:
`msp/out/r<radius>/<Monkey>/`.

- `msp.per_session`: the same estimator one session at a time
  (`msp/out/per_session/`).
- `msp.saccades`: gaze traces per session, maze and trial (`msp/out/saccades/`).
- `msp.legacy.saccades`: those plots under the retired
  `[fix_start − 1466 ms, fix_start]` window (`msp/legacy/out/`).

### `msp_trends/`

msp on merged codebooks, testing each monkey's apparent direction (H looks
left; S looks at the origin for Faure, right for Nielsen):

- `balls` (default): `left` (LU+LD), `origin`, `right` (RU+RD). Three states,
  since two 2-vectors always correlate at ±1.
- `quads` / `halves`: the origin ball plus maze quadrants or halves.

Adds a pooled-over-mazes Δ (trial-weighted), `binary` and `dwell` blocks, and
`--labels svm | dendro`. Output: `msp_trends/out/<tag>/`.

### `regression/`

Does gaze predict strategy *within* a maze? One logistic regression per
monkey (dendrogram labels):

```
log-odds(S) = α[maze] + γ·log(window length) + Σ β·gaze feature
```

Maze offsets absorb everything constant within a maze; mazes with fewer than
5 trials of either strategy are dropped. Gaze features, per state visited on
5–95% of trials, are *occupied*, *visit count* and *mean fixation duration*,
all z-scored. Unestimable columns are dropped and marked × in the figure. In
the figure, solid bars have a 95% CI excluding 0, and ★ survives Holm
correction. The result is associational, and its intervals are optimistic.
Output: `regression/out/geofix/`.

### `scatter/`

Descriptive: mean gaze (degrees) from `fix_start` to `flash_one`, one dot per
maze with ±1 SD bars, plus a table of directional, uncorrected p-values for
each maze pair. Output: `scatter/out/<Monkey>/`.
