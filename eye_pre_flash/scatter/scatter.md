# Gaze at `fix_start`

Where the eyes are at the single instant the fixation requirement begins — one
`(x, y)` per trial — and whether that point depends on the maze or on the
decoded strategy.

Every other pre-flash analysis in this repo integrates the free-viewing
*window* ending at `fix_start` (`eye_pre_flash/plotting/plot_io.py`:
`fix_start − 1466 ms → fix_start`). This package takes the endpoint only, so a
trial is a point rather than a cloud and group structure shows up as a
translation of a point cloud.

```bash
uv run python -m eye_pre_flash.scatter.plots --monkey Faure
uv run python -m eye_pre_flash.scatter.plots --all                # both monkeys × both spaces × raw and centred
uv run python -m eye_pre_flash.scatter.stats --all
```

Figures land in `eye_pre_flash/scatter/out/<monkey>/` for the default `deg`
space, scope `all`, uncentred. A non-default value adds its own folder level
instead of the default staying implicit: a non-`deg` space prefixes
`<space>/`, a non-`all` scope adds `<scope>/`, and `--center session` adds
`centered/` (or suffixes an existing scope folder as `<scope>_centered/`).

## What a trial is

| Step | Rule |
| ---- | ---- |
| Trial filters | mazes 1–6, `path_type != -99`, `photodiode_qc_bad` false, `trial_fade == 0` — the same pool as the `heatmap_maze` / `similarities` family |
| The sample | eye traces are 1 ms, so the sample nearest `fix_start` is taken and the trial is dropped if it is more than `TOL_MS = 2` ms away |
| Sample QC | the sample must be finite and inside `POSITION_LIMIT_DEG` (±30°). No pupil-based blink rejection: a blink at `fix_start` shows up as a NaN or an out-of-range sample and is dropped by those two tests |
| Session pool | by default the scope's labelled sessions (`--scope publication\|all\|allplus`), so the maze views and the strategy views describe the *same* trials. `--all-sessions` widens the maze views to every session |

Coordinates come in two frames, and the difference between them is the point:

- **`deg`** — raw degrees from the fixation target. The arms of maze 1 and maze
  6 are different lengths, so a maze difference here can be geometry rather
  than behaviour.
- **`unith`** — each trial warped onto the unit H by its own arm lengths
  (`data.attractor.to_maze`), which divides that geometry out.

`--center session` additionally subtracts each session's own mean gaze, since
tracker calibration drifts between sessions and part of the raw spread is that
offset rather than anything about mazes.

## The views

| Figure | Variable held fixed | Reads |
| ------ | ------------------- | ----- |
| `pooled_maze` | none | every trial coloured by maze, with x and y marginals (Gaussian KDE per maze) |
| `pooled_strategy` | none | the same panel by decoded label, plus each label's central-50% contour |
| `maze_facets` | maze | six panels; each maze's contour against the greyed pooled contour |
| `strategy_facets` | strategy | the same for the two labels |
| `maze_x_strategy` | both | the 6 × 2 grid — does the label still move the cloud *inside* a maze? |
| `session_facets` | session | one panel per session, so calibration cannot carry the maze effect |
| `centroids` | none | group means with 95% CIs, mazes joined 1 → 6. **The effect sizes are ~0.1–0.3° against a ~0.8° spread, so this is the only view they are legible in** |
| `marginals` | none | the 1-D version: per-group KDEs of x and y |

`plots.py` also writes `trials.csv` -- the per-trial `x, y, maze, label,
session, trial_id, choice` table every view above is drawn from -- so a figure
can be redrawn, or re-aggregated a different way, without re-running
`collect_fix_start` against the raw eye/behavioral data. `stats.py` writes
`group_means.csv` (per maze, per label, per maze × label cell, per session)
and `tests.txt` beside them.

Plot conventions: maze is *ordered* (1–3 hierarchical, 5–6 sequential, 4 the
boundary maze), so it gets a single-hue ordinal ramp; strategy is two
identities and gets two categorical hues with direct labels. Contours are the
50%-probability-mass level of a Gaussian KDE and are drawn only for groups of
at least `MIN_CONTOUR_N = 40` trials — below that a single contour traces the
few points it has rather than the group's shape.

## What the figures show

Both animals, `--scope all` (4 labelled sessions each; Faure n=1843, Nielsen
n=2448 trials):

**The maze moves the eye horizontally, monotonically, and it is not
calibration.** Spearman ρ of maze index against eye x:

| Monkey | `deg` | `deg`, within session | `unith` |
| ------ | ----- | --------------------- | ------- |
| Faure | +0.157 (p=1.4e−11) | +0.165 | +0.187 |
| Nielsen | +0.203 (p=3.4e−24) | +0.207 | +0.268 |

Maze 1 → 6 walks the mean gaze rightward by ~0.22° for Faure (−0.24° →
−0.02°) and ~0.29° for Nielsen (−0.10° → +0.20°). Per-session ρ matches the pooled ρ, and
`--center session` leaves it unchanged, so it is a within-session effect. It is
*stronger* in unit-H than in degrees, so it is not the arm-length geometry
either — the animals sit further toward the side the upcoming maze needs.
Vertically there is at most a weak negative trend (ρ ≈ −0.04 Faure, −0.06
Nielsen).

**The decoded strategy separates the pooled clouds, but mostly because it
tracks the maze.** Sequential − hierarchical in eye x:

| Monkey | pooled Δ | Cohen's *d* | maze-stratified Δ |
| ------ | -------- | ----------- | ----------------- |
| Faure | +0.174° (p=6e−9) | +0.33 | **+0.130°** |
| Nielsen | +0.160° (p=6e−11) | +0.31 | **−0.022°** |

The pooled difference is the maze trend wearing a strategy label: maze 1–3 are
mostly decoded hierarchical and 5–6 mostly sequential, so the two label pools
are two different maze mixtures. Averaging the *within-maze* label difference
(maze weights, cells with ≥10 trials per label) keeps three quarters of the
effect for Faure and erases it for Nielsen. `maze_x_strategy` shows why the
stratified number is fragile: the off-diagonal cells are thin (maze 1
sequential n=13, maze 6 hierarchical n=25 for Faure), so the within-maze
contrast rests on a few dozen trials per cell.

Read together: **at `fix_start` the gaze already carries the maze, and carries
the strategy only to the extent the strategy is the maze.** Faure leaves a
residual within-maze strategy shift worth following up; Nielsen does not.

## Caveats

- The whole effect lives in a ±1° neighbourhood of the fixation target — this
  is the tail of the free-viewing period, not a saccade target. Do not read
  the panels at the scale of the maze.
- The maze × strategy cells are strongly unbalanced by construction, so the
  stratified label effect is the least well-powered number here and the one
  most worth recomputing on `--scope allplus`.
- Trials with no gaze sample within 2 ms of `fix_start` are dropped silently
  in the figures but counted in the `dropped` tally that `plots.py` prints
  (it was 0 for both animals at the time of writing; the losses are all
  photodiode/fade QC).
