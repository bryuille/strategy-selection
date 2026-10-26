# Commands

Every runnable entry point, with its flags. Setup and repo layout are in
[README.md](README.md). Run from the repo root.

Commands use `uv run`. Without uv, activate the venv and drop the prefix:
`uv run python -m msp.build` becomes `python -m msp.build`.

Every `build` takes `--dry-run` (counts and coverage, no statistics) and
`--refresh` (rebuild the feature cache). Always dry-run first.

Contents: [Data](#data) · [msp](#msp) · [msp.per_session](#mspper_session) ·
[classifier](#classifier) · [scatter](#scatter) · [counts](#counts) ·
[counterfactual](#counterfactual) ·
[Slurm](#slurm) · [Clearing derived caches](#clearing-derived-caches)

## Data

### `data.loader`: pooled eye caches

```bash
uv run python -m data.loader                    # eye_behavioral + eye_data, both monkeys
uv run python -m data.loader --monkey Faure
uv run python -m data.loader --attractor        # also the attractor warp (msp.legacy.saccades only)
```

### `data.looks`: look caches (the `timestamp_preprocessing` package)

Every analysis reads its eye events from here. `preprocess_trial` of the
root-level `timestamp_preprocessing/` package (never edited) runs on the raw
eye mats and the result is cached per monkey and epoch as
`<Monkey>_looks_<epoch>.npz`: `fix` (maze onset to `fix_start`; msp,
classifier, msp.saccades) and `trial` (maze onset to `trial_end`;
counterfactual). Needs the sessions' eye mats under `MAT_ROOT`.

```bash
uv run python -m data.looks --check                              # synthetic, no data
uv run python -m data.looks --epoch fix trial                    # every session with a local eye mat, both monkeys
uv run python -m data.looks --monkey Faure --epoch fix --sessions june_8_g0 june_24_g0
uv run python -m data.looks --compare Oct_22_g0                  # looks vs the legacy pymovements fixations
sbatch -J looks slurm/run_module.sbatch data.looks --epoch fix trial   # cluster
```

| Flag          | Meaning                                                                 |
| ------------- | ----------------------------------------------------------------------- |
| `--monkey`    | `Faure`, `Nielsen` (default both)                                       |
| `--epoch`     | `fix`, `trial` (default both)                                           |
| `--sessions`  | Subset (default: every session with an eye mat); missing ones are appended to the cache |
| `--refresh`   | Rebuild the requested sessions                                          |
| `--check`     | Synthetic adapter checks only                                           |
| `--compare`   | Before/after report on one session                                      |

A cache is rebuilt automatically when the package source (`core.py`,
`current.py`, `stable_support.py`) or its config changes.

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
uv run python -m msp.build --radius 0.5 --codebook capsules       # also: quads, halves (tag r0.5capsules)
uv run python -m msp.build --variant full no_origin --vmax 0.3    # subset of variants, fixed colour scale
```

`msp.build` flags:

| Flag                | Default            | Meaning                                                                |
| ------------------- | ------------------ | ---------------------------------------------------------------------- |
| `--monkey`          | both               | `Faure`, `Nielsen`                                                     |
| `--maze`            | 2–5                | Mazes to run                                                           |
| `--variant`         | all                | `full`, `no_origin`, `mean_removed`                                    |
| `--radius`          | config default     | Assignment radius around each fixed prototype (unit H); part of the cache stem, so a new value triggers one extraction pass |
| `--codebook`        | `balls`            | `balls` (origin + four exit balls), `quads` / `halves` (origin ball + maze quadrants / halves), `capsules` (origin-to-exit capsules, no origin state); tag suffix, and one cache per (radius, rule) |
| `--n-rounds`        | estimator default  | Split-half rounds                                                      |
| `--n-perm`          | estimator default  | Label-shuffle permutations                                             |
| `--min-trials`      | estimator default  | Per strategy, pooled; below this the maze is skipped                   |
| `--vmax`            | per-figure         | Fixed 0-to-VMAX colour scale                                           |
| `--dpi`             | 300                | Figure resolution                                                      |
| `--out-root`        | `msp/out`          | Output directory                                                       |
| `--seed`            | 0                  | RNG seed                                                               |
| `--refresh`         | off                | Rebuild the feature cache                                              |
| `--dry-run`         | off                | Counts and coverage only                                               |

`msp.features` takes `--monkey`, `--radius`, `--codebook`, `--refresh`.

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

Flags as for `msp.build` (balls only, no `--codebook`), plus `--session` (one
or more ids; default is the monkey's top-ten SVM scope).

## classifier

Strategy-label classifier. Both scopes build their fixation table on first
use (`classifier.fixations`, needs pymovements; caches keep the legacy name
`<Monkey>_msp_revised_fix[_svm].npz`). The `svm` scope (top-ten sessions,
SVM labels, mazes 2–5) also needs the `<session>_strategy_svm.npz` labels. The raw branch builds one 1 kHz trace cache per
monkey and scope (`<Monkey>_clf_raw[_svm].npz`) on first use and derives
every bin width from it. Pupil tables for the non-publication sessions come
from `classifier.pupil_extract --session ...` on the cluster
(`classifier/out/pupil/`, pull it down before building fixation tables
locally); without them those sessions' pupil features are NaN.

```bash
uv run python -m classifier.checks                            # synthetic, no data
uv run python -m classifier.build                             # both branches, both models, dendro scope
uv run python -m classifier.build --branch regions_dwell --model logreg   # → out/dendro/regions/dwell/
uv run python -m classifier.build --branch regions            # dwell share + visited, nine regions
uv run python -m classifier.build --branch regions regions_plus --per-session   # → out/dendro/regions/both_per_session/
uv run python -m classifier.build --branch exits --model logreg   # the four exit balls alone → out/dendro/exits/both/logreg/
uv run python -m classifier.build --branch lor --model logreg     # origin ball + left / right half of the maze → out/dendro/lor/both/logreg/
uv run python -m classifier.build --branch balls --model logreg   # msp's r0.5 codebook in degrees → out/dendro/balls/both/logreg/
uv run python -m classifier.codebook --branch lor balls regions exits   # region drawings over all six mazes → out/dendro/<branch>/<branch>_by_maze.png
uv run python -m classifier.pres                              # best per-session region / exits run → <run dir>/session_auc.{png,svg}: one bar per (monkey, session)
uv run python -m classifier.pres --merged --variant exits/both/logreg   # → out/dendro/exits/both/logreg/merged_auc.{png,svg}: one bar per monkey
uv run python -m classifier.pres --merged --labels svm --model gbm   # best pooled svm-scope gbm run → <run dir>/merged_auc.{png,svg}
uv run python -m classifier.build --labels svm                # msp's scope: ten sessions, SVM labels
uv run python -m classifier.build --n-perm 20                 # smoke test
uv run python -m classifier.fixations                         # fixation tables (both scopes)
uv run python -m classifier.pupil_extract --session Oct_22_g0 Nov_18_g0   # cluster: pupil tables
```

| Flag        | Meaning                                                            |
| ----------- | ------------------------------------------------------------------ |
| `--labels`  | `dendro` (default): dendrogram labels, publication sessions, mazes 1–6; `svm`: SVM labels, top-ten sessions, mazes 2–5. Top level of the output: `out/<dendro|svm>/<raw|regions|regions_plus|exits|lor|balls>/<variant>/<model>/` |
| `--branch`  | `raw` (binned x/y/pupil traces; variant `b<ms>`), `raw_plus` (the same traces + the 6×6 time-collapsed gaze map over ±15°; family `raw_plus/`, variant `b<ms>`), `regions` (dwell share + visited over nine regions; variant `both`), `regions_dwell` / `regions_binary` (one set; variants `dwell` / `binary`); `<family>_all` (dwell share + visited + visit count; variant `all`); `regions_plus[_dwell\|_binary]` (same, with "elsewhere" split at 10° and pupil change; family `regions_plus/`); `exits[_dwell\|_binary\|_all]` (the four exit balls alone, no origin quadrants or elsewhere; family `exits/`); `lor[_dwell\|_binary\|_all]` (3° origin ball + left / right half of the maze inside a perimeter 3° out from the exit quadrilateral; family `lor/`); `balls[_dwell\|_binary\|_all]` (msp's r0.5 codebook in degrees: 3° origin ball + the four 3° exit balls, nothing else; family `balls/`). All regions are in degrees around each maze's exits |
| `--model`   | `logreg` (L2 logistic regression), `gbm` (shallow boosted trees)   |
| `--bin-ms`  | Raw branch: trace bin width in ms (`1` = the whole 1 kHz trace; default 50; the variant is `b<ms>`) |
| `--interactions` | Add gaze × maze interaction terms (per-maze weights shrunk toward the shared ones); variant gets `_x_maze` |
| `--per-session` | Fit, cross-validate and permutation-test each session on its own (`<Monkey>_<session>.png`, `eval_<Monkey>_<session>.npz`), plus one `combined` row per monkey: the pair-weighted mean of the per-session AUCs against the same weighted mean of their nulls; variant gets `_per_session` |
| `--replot`  | Redraw figures from the saved `eval_<Monkey>.npz` without refitting |
| `--n-perm`  | Label shuffles for the within-maze AUC null (default 200; any other count adds `_p<n>` to the variant, so it never overwrites the default run) |
| `--refresh` | Rebuild the raw-trace cache (and the svm fixation cache)           |


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

## counterfactual

Counterfactual looks (looks at the unchosen exits after the choice) scored
against each strategy's choice-conditioned belief, top-two SNR sessions per
monkey, looks from the `trial` looks cache. Method and results:
[counterfactual/README.md](counterfactual/README.md).

```bash
uv run python -m counterfactual.checks                 # synthetic, no data
uv run python -m counterfactual.extract [--refresh]    # trial / look cache
uv run python -m counterfactual.beliefs [--refresh]    # per-maze tables (chosen exit x exit looked at), assigned to trials
uv run python -m counterfactual.compare                # relative metric -> counterfactual/out/<Monkey>/compare_sessions.png
uv run python -m counterfactual.logp                   # absolute metric -> counterfactual/out/<Monkey>/logp_sessions.png
uv run python -m counterfactual.compare --uncond       # both metrics with unconditioned (Figure Five) q -> uncond_* (also counterfactual.logp --uncond)
uv run python -m counterfactual.best_model             # all behavioral models ranked by gain over chance -> best_model_summary.md
uv run python -m counterfactual.tables                 # the tables next to the observed first looks -> belief_tables.md
uv run python -m counterfactual.descriptive            # descriptive models of the first look (counterfactual/DESCRIPTIVE.md) -> descriptive_results.csv
uv run python -m counterfactual.movie --session june_24_g0 --trial-id 1 2      # one movie per trial
uv run python -m counterfactual.movie --session june_8_g0 june_24_g0 Nov_3_g0 Nov_6_g0 --batch --start 50 --n-trials 20
```

`counterfactual.movie` writes `counterfactual/out/movies/<Monkey>/<session>/trial_<id>.mp4`, or with
`--batch` one `batch_<first>-<last>.mp4` per session (`--labelled` keeps labelled trials only, `--speed`
sets playback, default 0.25). Gaze is the `timestamp_preprocessing` sample record; looks and saccades come
from the same caches the analysis uses.

Every command except `movie` takes `--monkey` (`Faure`, `Nielsen`; default both).

## Slurm

Batch scripts live in `slurm/`. Submit from the repo root (logs go to
`logs/`). All request one core; `MAT_ROOT` / `PROCESSED_ROOT` come from
`data/config.py`, so set them to the cluster paths before pushing.

| Script                 | Runs                                                                 |
| ---------------------- | -------------------------------------------------------------------- |
| `run_labels.sbatch`    | `data.labels --sweep --with-svm`, one array task per `CLUSTERING_SESSIONS` entry (64 GB) |
| `run_module.sbatch`    | Any `python -m <module> [args]` (32 GB, 4 h; override with `--mem` / `-t`) |
| `run_msp.sbatch`       | `msp.build`, after checking the eye and SVM-label caches exist       |

```bash
mkdir -p logs
sbatch slurm/run_labels.sbatch                                   # all 23 sessions
sbatch -J eyecaches --mem=64G slurm/run_module.sbatch data.loader
sbatch slurm/run_msp.sbatch --radius 0.5
sbatch -J persess slurm/run_module.sbatch msp.per_session --radius 0.5
sbatch slurm/run_msp.sbatch --radius 0.5 --codebook capsules
sbatch -J scatter slurm/run_module.sbatch scatter.build
sbatch -J counts  slurm/run_module.sbatch counts.build
```

From an empty `PROCESSED_ROOT`: run labels and eye caches first (they are
independent), then everything else with `--dependency=afterok:<jobid>`. Jobs
that share a feature cache must not run at once on a cold cache:
`msp.per_session` after `msp.build` at the same radius, and `msp.build`
`quads` after `halves` (or vice versa) at the same radius, since they share
the `_region` cache.

## Clearing derived caches

Loaders check that keys exist, not that settings match. After changing
detector or QC constants, delete the derived eye and feature caches and rebuild:

```bash
cd /path/to/your/PROCESSED_ROOT         # as set in data/config.py (default ./data/processed)
rm -f *_looks_*.npz *_msp_looks5_*.npz *_msp_revised_fix*.npz \
      *_cfv2_trials.npz *_cfv2_q.npz *_clf_raw*.npz
```

**Event source change, 2026-10-22.** Eye events now come from the
`timestamp_preprocessing` through `data.looks`; the pymovements detector in
`data.builder` is legacy (read only by `msp.legacy.saccades` through the
`*_clean_eye_data_s1466_e0` / attractor caches). The `*_msp_fixed5_*` caches are orphaned (no code reads them; safe to delete). The
classifier (SCHEMA 4), msp and counterfactual caches record the
package source hash and rebuild themselves when it changes.

Never delete anything under `MAT_ROOT`; it is the only copy of the raw data.

**Detector change, 2026-10-16.** Fixations are no longer pymovements I-DT
(2° dispersion); `data.builder.trial_events` now ends a fixation only at a
detected saccade or when gaze moves more than 2° from the fixation's anchored
centroid, and the event name is `fixation` (was `fixation_idt`). Every cache
in the `rm` line above that was built before this date encodes the old
detector and must be deleted before msp or classifier are rerun
(`classifier.fixations` self-invalidates through its `schema`; the msp
caches do not). `*_clf_raw*` bins raw gaze around `fix_start` and
is unaffected. The caches were deliberately **not** rebuilt with the change.
