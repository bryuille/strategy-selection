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
MAT_ROOT = None  # raw .mat files; required, read-only, keep a backup
PROCESSED_ROOT = Path("./data/processed")  # everything computed; rebuildable
```

```
<MAT_ROOT>/
├── Behavioral_Data/{Faure,Nielsen}/<session>_good_trials_concat.mat
├── Eye_Data/{Faure,Nielsen}/Eye_Data_<session>.mat
├── Neural_Data/{Faure,Nielsen}/<session>_Whole_Trial_FR_Causal.mat
└── Single_Trial_List/{Faure,Nielsen}/...
```

`PROCESSED_ROOT` may not sit inside `MAT_ROOT`. Sessions are
`<month>_<day>_g<n>`; `june` / `April` are Faure, `Oct` / `Nov` / `Dec` Nielsen.

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

## Vocabulary

- **Labels.** *SVM*: a linear classifier of neural activity trained on mazes
  1 vs 6, over the top-ten sessions per monkey by `snr_auc`. *Dendrogram*:
  Ward clustering of neural activity, vetted against the anchor mazes down to
  four sessions per monkey.
- **Unit H.** Each maze is warped so the fixation point is at (0, 0), the exits
  at (±1, ±1) and the top of the stem at (0, 1), putting all mazes in one frame.
- **States.** The origin plus the four exits (LU, LD, RU, RD). Each fixation
  goes to the nearest state within a radius (`--radius`, tag `r<radius>`), or
  is dropped.
- **Window.** `geofix`: maze onset to fixation onset, about 1.5 s.

## Analyses

### `counts/`

A census, not a test. One heatmap cell per (session, maze), showing `H/S`
counts with fill for the sequential share. Cells whose minority strategy is
under 10% are marked red, since they can't support an H-vs-S comparison. The
pool is every labelled trial that passes QC in mazes 1–6. Expect SVM mazes 1
and 6 to be red: they are the classifier's training axis. Scopes:
`publication`, `all`, `top_ten`. Output: `counts/out/<source>/hs_<monkey>_<scope>.png`.

### `msp/` (main)

Within one maze, is gaze more alike between trial groups that share a
strategy (H–H, S–S) than between groups that don't (H–S)? The geometry is
identical, so a difference can't be a visual confound.

Each trial becomes a binary 5-vector (visited each state or not). Per
(monkey, variant, maze), each strategy is split into two random halves 100
times. Each half is averaged, and the four pairings are correlated. The
statistic, `Δ = ½(r_HH + r_SS) − ½(r_HS + r_SH)`, is tested against 1000
within-session label shuffles, giving `z` and `p`.

- **Variants:** `full`; `no_origin` (drop the origin state); `mean_removed`
  (subtract the maze's mean vector first).
- **Read `z`, not raw `r`.** `r` rises with group size, which varies by maze,
  monkey and strategy. Under `mean_removed`, a small strategy's diagonal sits
  near −1 by construction.
- **Caveats.** The fixed codebook drops gaze off the landmarks, by an amount
  that varies with the maze. Unequal H/S noise, within-session drift,
  overlapping splits and multiple comparisons apply to every msp package.

Output: `msp/out/r<radius>/<Monkey>/`. Related modules:

- `msp.per_session`: the same estimator, one session at a time, to show which
  sessions carry the effect. Cells are noisy and many are skipped. Output:
  `msp/out/per_session/`.
- `msp.saccades`: gaze traces per session, maze and trial. Output:
  `msp/out/saccades/`.
- `msp.legacy.saccades`: the same plots under the retired fixed window
  (`[fix_start − 1466 ms, fix_start]`), the only remaining code for that
  window. Output: `msp/legacy/out/`.

### `msp_trends/`

Pooled msp finds clean effects in only a few cells, but the state profiles
suggest a direction per monkey: H trials look left, and S trials look at the
origin (Faure) or right (Nielsen). This package tests that on merged codebooks:

- `balls` (default): `left` (LU+LD), `origin`, `right` (RU+RD). Three states
  rather than two, because the correlation of two 2-vectors is always ±1.
- `quads` / `halves`: the origin ball plus maze quadrants, or halves.

It adds a pooled-over-mazes Δ (trial-weighted mean of per-maze Δs, tested
against the same mean of per-maze shuffles), `binary` and `dwell` blocks,
variants `full` and `mean_removed`, and `--labels svm | dendro`. Output:
`msp_trends/out/<tag>/`.

### `regression/`

Mazes differ in both strategy mix and gaze, so does gaze predict strategy
*within* a maze? One logistic regression per monkey on the dendrogram labels:

```
log-odds(S) = α[maze] + γ·log(window length) + Σ β·gaze feature
```

- **Maze offsets:** `α` absorbs everything constant within a maze. Mazes with
  fewer than 5 trials of either strategy are dropped.
- **Gaze features:** per landmark visited on 5–95% of trials, *occupied*,
  *visit count* and *mean fixation duration*, z-scored. Collinear or
  near-constant columns are dropped (× in the figure, reason in the CSV).
- **Figure:** solid bars have a 95% CI that excludes 0, and ★ marks
  significance after Holm correction.

The result is associational. Trials aren't independent, so the intervals are
optimistic, and the fit rests on the few mixed mazes. Output:
`regression/out/geofix/`.

### `scatter/`

Descriptive. Each trial's mean gaze (degrees) from `fix_start` to
`flash_one` is averaged per maze and drawn as a dot with ±1 SD bars. A table
gives directional, uncorrected p-values for each maze pair. Output:
`scatter/out/<Monkey>/`.
