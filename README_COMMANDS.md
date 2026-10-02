# Commands

Every runnable entry point, with its flags. Setup and repo layout are in
[README.md](README.md). Run from the repo root.

Commands use `uv run`. Without uv, activate the venv and drop the prefix:
`uv run python -m msp.build` becomes `python -m msp.build`.

Every `build` takes `--dry-run` (counts and coverage, no statistics) and
`--refresh` (rebuild the feature cache). Always dry-run first.

Contents: [Data](#data) · [msp](#msp) · [msp.per_session](#mspper_session) ·
[msp_trends](#msp_trends) · [regression](#regression) · [scatter](#scatter) · [counts](#counts) ·
[Slurm](#slurm) · [Clearing derived caches](#clearing-derived-caches)

## Data

### `data.loader`: pooled eye caches

```bash
uv run python -m data.loader                    # eye_behavioral + eye_data, both monkeys
uv run python -m data.loader --monkey Faure
uv run python -m data.loader --attractor        # also the attractor warp (msp.legacy.saccades only)
```

### `data.labels`: strategy labels

```bash
uv run python -m data.labels --type strategy --sweep                  # all sessions, one at a time
uv run python -m data.labels --type strategy --sweep --with-svm       # plus the maze-1-vs-6 SVM label
uv run python -m data.labels --type strategy --sweep --reclaim-timebins # delete the trial-timebin cache after each session
uv run python -m data.labels --type strategy --session june_24_g0     # one session
uv run python -m data.labels --type strategy --naming relative        # alternate cluster naming
```

| Flag               | Meaning                                                                                                   |
| ------------------ | --------------------------------------------------------------------------------------------------------- |
| `--type strategy`  | Required; the only label type                                                                             |
| `--session`        | One or more sessions (default: all clustering sessions)                                                   |
| `--sweep`          | Label each session in turn                                                                                |
| `--with-svm`       | Also build the SVM label off the same read                                                                |
| `--reclaim-timebins` | With `--sweep`, delete the trial-timebin cache after each session to save disk |
| `--overwrite`      | Rebuild existing label caches                                                                             |
| `--naming`         | `reference` (default, reproduces the published MATLAB) or `relative` (labels 3 sessions `reference` rejects) |

## msp

Needs the pooled eye / behavioral / SVM-label caches in `data/processed/`
(eye caches build on first load; SVM labels from `data.labels`).

```bash
uv run python -m msp.checks                  # estimator invariants, synthetic, no data
uv run python -m msp.features                # feature caches for both monkeys
uv run python -m msp.build --dry-run         # pooled counts + codebook coverage
uv run python -m msp.build                   # full: 2 monkeys × 3 variants × 4 mazes
uv run python -m msp.build --monkey Faure --maze 4 --n-perm 200   # smoke test
uv run python -m msp.build --radius 0.5                           # tighter assignment balls
uv run python -m msp.build --variant full no_origin --vmax 0.3    # subset of variants, fixed colour scale
```

`msp.build` flags:

| Flag                | Default            | Meaning                                                                |
| ------------------- | ------------------ | ---------------------------------------------------------------------- |
| `--monkey`          | both               | `Faure`, `Nielsen`                                                     |
| `--maze`            | 2–5                | Mazes to run                                                           |
| `--variant`         | all                | `full`, `no_origin`, `mean_removed`                                    |
| `--radius`          | config default     | Assignment radius around each fixed prototype (unit H); part of the cache stem, so a new value triggers one extraction pass |
| `--n-rounds`        | estimator default  | Split-half rounds                                                      |
| `--n-perm`          | estimator default  | Label-shuffle permutations                                             |
| `--min-trials`      | estimator default  | Per strategy, pooled; below this the maze is skipped                   |
| `--vmax`            | per-figure         | Fixed 0-to-VMAX colour scale                                           |
| `--dpi`             | 300                | Figure resolution                                                      |
| `--out-root`        | `msp/out`          | Output directory                                                       |
| `--seed`            | 0                  | RNG seed                                                               |
| `--refresh`         | off                | Rebuild the feature cache                                              |
| `--dry-run`         | off                | Counts and coverage only                                               |

`msp.features` takes `--monkey`, `--radius`, `--refresh`.

Figures land in `msp/out/r<radius>/<Monkey>/`. Each trial runs from maze
onset to fixation onset, re-detected from raw eye.

### `msp.saccades`: pre-fixation gaze plots

```bash
uv run python -m msp.saccades --tag r1 --session june_24_g0 --maze 4                    # session average
uv run python -m msp.saccades --tag r1 --session june_24_g0 --maze 4 --view trials      # every trial
uv run python -m msp.saccades --tag r1 --session june_24_g0 --maze 4 --view trial --trial-id 105
uv run python -m msp.saccades --all-analyses                                            # every tag × session × maze 2–5
```

| Flag             | Meaning                                                                          |
| ---------------- | -------------------------------------------------------------------------------- |
| `--tag`          | Analysis tag (e.g. `r1`, `r0.5`); required unless `--all-analyses`               |
| `--session`      | Session id; required unless `--all-analyses`                                     |
| `--maze`         | 1–6; required unless `--all-analyses`                                            |
| `--monkey`       | Inferred from the session if omitted                                             |
| `--view`         | `average` (default), `trials`, `trial`                                           |
| `--trial-id`     | With `--view trial`                                                              |
| `--dpi`          | Default 200                                                                      |
| `--all-analyses` | Run every tag × publication session × mazes 2–5                                  |

### `msp.legacy.saccades`: the retired `pre1466` window

The same gaze panels over `[fix_start − 1466 ms, fix_start]`, from the
attractor and `*_clean_eye_data_s1466_e0` caches. Output:
`msp/legacy/out/saccades/r<radius>_pre1466/`.

```bash
uv run python -m msp.legacy.saccades --radius 0.5 --session june_24_g0 --maze 4
uv run python -m msp.legacy.saccades --radius 1 --session june_24_g0 --maze 4 --view trials
uv run python -m msp.legacy.saccades --all-analyses                                     # radii 0.5, 1 × session × maze 2–5
```

Flags as for `msp.saccades`, with `--radius` in place of `--tag`.

## msp.per_session

Same estimator, one recording session at a time. Shares msp's feature caches.
Output: `msp/out/per_session/<tag>/<Monkey>/`.

```bash
uv run python -m msp.per_session --dry-run
uv run python -m msp.per_session --radius 0.5
uv run python -m msp.per_session --monkey Faure --session june_24_g0 --maze 4
```

Flags as for `msp.build`, plus `--session` (one or more ids; default is the
monkey's top-ten SVM scope).

## msp_trends

msp on merged codebooks.

```bash
uv run python -m msp_trends.checks                           # synthetic, no data
uv run python -m msp_trends.build --dry-run --all-tags
uv run python -m msp_trends.build                            # r1
uv run python -m msp_trends.build --radius 0.5
uv run python -m msp_trends.build --radius 0.5 --labels dendro
uv run python -m msp_trends.build --radius 0.5 --codebook quads   # also: halves
uv run python -m msp_trends.build --all-tags                 # every tag in config.TAGS
uv run python -m msp_trends.build --monkey Faure --n-perm 200     # smoke test
```

Flags as for `msp.build`, plus:

| Flag         | Meaning                                                                                       |
| ------------ | --------------------------------------------------------------------------------------------- |
| `--block`    | Which blocks to run (default: all)                                                            |
| `--labels`   | `svm` (top-ten sessions) or `dendro` (dendrogram labels, publication sessions; tag suffix `_dendro`) |
| `--codebook` | `balls` (merged exit balls), `quads`, `halves` (origin ball + quadrants / halves)             |
| `--all-tags` | Run every tag in `config.TAGS`                                                                |

## regression

```bash
uv run python -m regression.checks                  # synthetic invariants, seconds
uv run python -m regression.build --dry-run         # mazes kept, visit rates, no fit
uv run python -m regression.build                   # fit both monkeys, write CSVs + figure
uv run python -m regression.build --monkey Faure --sessions june_24_g0
```

| Flag             | Meaning                                              |
| ---------------- | ---------------------------------------------------- |
| `--monkey`       | One or more monkeys (default: both)                  |
| `--sessions`     | Subset of the publication sessions                   |
| `--dry-run`      | Report only; fit nothing                             |
| `--refresh`      | Rebuild the fixation caches                          |

## scatter

Mean gaze per maze over `fix_start` → `flash_one`, plus the maze-pair p-value
table. Needs the pooled eye and behavioral caches.

```bash
uv run python -m scatter.build                  # both monkeys
uv run python -m scatter.build --monkey Faure
```

## counts

H/S trial census per (session, maze), as heatmaps.

```bash
uv run python -m counts.census --source svm --monkey Faure
uv run python -m counts.census --source svm --monkey Nielsen --scope publication
uv run python -m counts.build                        # presentation copies, all sources and monkeys
uv run python -m counts.build --source svm --monkey Faure --scope top_ten
```

| Flag        | Meaning                                                                                                        |
| ----------- | -------------------------------------------------------------------------------------------------------------- |
| `--source`  | Label source (`census`: one; `build`: one or more)                                                             |
| `--monkey`  | `Faure` or `Nielsen`                                                                                           |
| `--scope`   | `census`: `publication`, `all` (default). `build` also accepts `top_ten` (default)                             |

## Slurm

Batch scripts live in `slurm/`. Submit from the repo root (logs go to
`logs/`). All request one core; `MAT_ROOT` / `PROCESSED_ROOT` come from
`data/config.py`, so set them to the cluster paths before pushing.

| Script                 | Runs                                                                 |
| ---------------------- | -------------------------------------------------------------------- |
| `run_labels.sbatch`    | `data.labels --sweep --with-svm`, one array task per `CLUSTERING_SESSIONS` entry (64 GB) |
| `run_module.sbatch`    | Any `python -m <module> [args]` (32 GB, 4 h; override with `--mem` / `-t`) |
| `run_msp.sbatch`       | `msp.build`, after checking the eye and SVM-label caches exist       |
| `run_regression.sbatch`| `regression.checks`, then `regression.build`                         |

```bash
mkdir -p logs
sbatch slurm/run_labels.sbatch                                   # all 23 sessions
sbatch -J eyecaches --mem=64G slurm/run_module.sbatch data.loader
sbatch slurm/run_msp.sbatch --radius 0.5
sbatch -J persess slurm/run_module.sbatch msp.per_session --radius 0.5
sbatch -J trends  slurm/run_module.sbatch msp_trends.build --radius 0.5 --codebook quads
sbatch slurm/run_regression.sbatch
sbatch -J scatter slurm/run_module.sbatch scatter.build
sbatch -J counts  slurm/run_module.sbatch counts.build
```

From an empty `PROCESSED_ROOT`: run labels and eye caches first (they are
independent), then everything else with `--dependency=afterok:<jobid>`. Jobs
that share a feature cache must not run at once on a cold cache:
`msp.per_session` after `msp.build` at the same radius, and `msp_trends` runs
at the same radius and codebook (e.g. `r0.5` and `r0.5_dendro`) one after
the other.

## Clearing derived caches

Loaders check that keys exist, not that settings match. After changing
detector or QC constants, delete the derived eye and feature caches and rebuild:

```bash
cd /path/to/your/PROCESSED_ROOT         # as set in data/config.py (default ./data/processed)
rm -f *_clean_eye_data_s*.npz *_attractor_eye_data.npz \
      *_msp_fixed5_*.npz *_msp_trends_*.npz *_reg_fixed5_*.npz
```

Never delete anything under `MAT_ROOT`; it is the only copy of the raw data.
