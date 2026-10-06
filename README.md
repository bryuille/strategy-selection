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
| `msp_revised/`| Isolated re-run on the publication sessions: codebook search, all six mazes, pupil |
| `classifier/`  | Predict the strategy label from pre-fixation gaze: raw traces vs engineered features, within-maze AUC; dendrogram or SVM scope |
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
`r0.5`. Each trial is analysed from maze onset to fixation onset.

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

- `balls` (default): the origin ball plus the four exit balls (LU, LD, RU, RD),
  five states.
- `balls3`: the retired merge, `left` (LU+LD), `origin`, `right` (RU+RD). Three
  states, since two 2-vectors always correlate at ±1.
- `quads` / `halves`: the origin ball plus maze quadrants or halves.
- `capsules`: one capsule per exit (all points within the radius of the
  origin-to-exit segment), four states and no origin state; the overlap
  around the origin goes to the nearest segment, i.e. by quadrant.

Adds a pooled-over-mazes Δ (trial-weighted), `binary` and `dwell` blocks, and
`--labels svm | dendro`. Output: `msp_trends/out/<tag>/`.

### `msp_revised/`

msp's split-half 2x2, re-implemented in isolation (copies of the msp /
msp_trends logic; imports only `data/`), on a fixed scope: the four
publication sessions, dendrogram labels, all six mazes. Every in-window
fixation is cached once with its centroid in **unit H and in degrees**, so a
codebook is a cheap relabelling (`msp_revised/codebooks.py`, 41 of them: exit
balls in either frame, with or without the point above the origin, quadrants,
halves, capsules with and without an origin ball, origin-ball quadrants and
halves, nearest H segment, direction sectors, eccentricity rings, vertical
bands, inside/outside). Feature kinds: `binary`, `dwell`, `count`, `first`,
`last`, `both`; `--pupil` appends within-session z of the window's mean pupil
size and its early-to-late change (pupil traces for the four sessions come
from the cluster, `msp_revised.pupil_extract`). Variants: `full`,
`mean_removed` (maze grand mean).

Strategy is almost a function of maze here (maze 1 is all H, maze 6 all S),
and only the right crossbar arm differs between mazes (h4 = 10, 7.5, 5 deg),
so unit H stretches right-side x by up to 2x exactly where the H/S mix flips.
Every maze is therefore tested on its own, and the pooled estimate is the
trial-weighted mean of the per-maze Δs with aligned nulls: no block mean ever
mixes mazes, so a unit-H scale difference between mazes cannot enter it. The
degree-frame codebooks are the second check. `msp_revised.search` screens
every configuration and ranks by the smaller of the two monkeys' pooled z,
penalised by any solid maze (≥ 10 of each strategy) with Δ < 0; after ~900
comparisons the ranking is a shortlist, so finalists are rebuilt at the full
null and checked session by session. Output:
`msp_revised/out/<codebook>__<kind>[+pupil]/` (one codebook map per monkey,
`paper/<variant>/<Monkey>.png`, `results.csv`) and `msp_revised/out/search/`.

**Findings (2026-10-05; 936 configurations screened at n_perm 250, finalists
at 1000).** No configuration reaches pooled p < 0.05 in both monkeys with the
same sign; the best min pooled z is ≈ 1.7, all with pupil features. Within
the mazes that have both strategies, no region of gaze separates H from S
consistently across mazes and monkeys. The only recurring lead is binary
occupancy of origin + exits + the point above the origin (`balls6_*`, `full`),
which reaches z ≈ 2 in both monkeys when all trials are pooled into one 2x2,
in both frames, but not under the per-maze pooled estimate. Pupil features
raise Faure's scores broadly and do nothing for Nielsen. The right-of-origin
trend visible across mazes is maze geometry: the right-side share within the
origin ball rises monotonically with maze number in degrees too, and flips
direction within maze.

### `classifier/`

Can gaze predict the strategy label beyond what the maze already tells you?
Two label scopes (`--labels`): `dendro` (default), the dendrogram labels on
the publication sessions, mazes 1–6, the msp_revised scope; and `svm`, the
SVM labels on the ten best sessions per monkey, mazes 2–5, the msp scope
(mazes 1 and 6 are the SVM's anchors). Two feature branches, one evaluation:

- `raw`: the eye trace itself, gaze x / y (degrees) and pupil over the last
  1.5 s before fixation onset, block-averaged from the 1 kHz trace to a bin
  width (default 50 ms → 90 features; `--bin-ms 1` is the whole 1 kHz
  trace), each bin session-z-scored. A bin with no gaze sample (blink,
  off-screen) stays NaN, never interpolated: the tree model takes NaN
  natively and the logistic pipeline imputes it to the column mean. Trials
  with under half the window covered are dropped. Nothing hand-picked.
- `engineered`: 26 quantities the msp_revised search pointed at: origin-disc
  (5°) dwell and its four quadrants, exit-ball (3°) occupancy and dwell,
  eccentricity rings, fixation count, first / last fixation region, first
  fixation onset, pupil mean and change.

- `regions`: the two families the engineered fits lean on, over one
  partition of the screen in degrees: the four quadrants of the 5° origin
  disc, the four 3° exit balls, and everything else. Two sets over the same
  nine regions, msp's binary / dwell pair: `dwell` (share of in-window
  fixation time per region, sums to 1) and `binary` (region visited or
  not); 18 features, no ordering, latency or pupil terms. `regions_dwell`
  and `regions_binary` run one set alone. Binary is nested in dwell
  (visited ⇔ dwell share > 0), so the tree model gives `regions` and
  `regions_dwell` identical fits; for the logistic model the pair adds a
  little or nothing over dwell alone.

Pupil comes from the per-session tables `msp_revised.pupil_extract` writes
on the cluster; a session without one gets NaN pupil features (only the
four publication sessions have tables so far).

Models: L2 logistic regression and a shallow gradient-boosted tree. Every
model also gets the maze one-hot. Since maze is nearly a function of
strategy here and is readable from gaze, the ordinary AUC is inflated (≈ 0.9)
and is shown only for contrast. The headline is the **within-maze pairwise
AUC**: over (H, S) pairs of trials *from the same maze*, the fraction ranked
correctly by out-of-fold probability (4 × 5-fold CV stratified on session,
maze and label). Its chance level comes from 200 refits with labels shuffled
within (session, maze). A leave-one-session-out fit (train on the other
sessions, score the held-out one; pooled over held-out predictions) is
reported alongside, which with two sessions is train one day, test the
other. Output: `classifier/out/<branch>[_svm]/<model>/`
(`<Monkey>.png`, `results.csv`, `features.csv`).

**Findings (2026-10-05, dendro scope).** Within-maze AUC, out of fold, with the shuffle
null's mean ≈ 0.50 and sd ≈ 0.05 (200 shuffles). Empty gaze bins are kept as
NaN (no interpolation). Counts below include the (later dropped) valid-gaze
coverage channel:

| Branch / model | Faure | Nielsen |
| --- | --- | --- |
| raw, logistic, 50 ms bins (156 features) | 0.68 (p = 0.005) | 0.51 (chance) |
| raw, logistic, 1 ms = whole trace (6,036) | 0.67 (p = 0.005) | 0.56 (p = 0.050) |
| raw, boosted trees, 50 ms | 0.64 (p = 0.005) | 0.49 (chance) |
| engineered, logistic (26 features) | 0.67 (p = 0.005) | 0.58 (p = 0.035) |
| engineered, boosted trees | 0.67 (p = 0.005) | 0.60 (p = 0.070) |
| engineered, logistic + gaze × maze terms | 0.68 (p = 0.005) | 0.57 (p = 0.095) |

Faure's gaze carries strategy information beyond maze in every branch and
transfers across sessions (train one day, test the other: 0.58–0.70).
Nielsen is at chance on the raw traces (the whole-trace 0.56 does not
transfer: 0.45 one way, 0.66 the other) and marginal on the engineered
features (0.58–0.60; cross-session 0.51–0.61). Bin width does not matter much
in the fixed-width sweeps. Per-maze gaze weights change nothing for the
engineered features, so the within-maze signal is shared across mazes, not
maze-specific; they weaken cross-session transfer (Faure 0.56–0.58).
Features shared by the raw and engineered models: lower pupil → S in Faure
(through most of the window), pupil rising in the last 250 ms → S in
Nielsen; dwell below-right of the fixation point → S and below-left → H in
Faure; dwell near the RU exit → S in Nielsen; and in both monkeys the
largest single raw weight is gaze height about 275 ms before fixation, with
opposite signs, which is why the monkeys' models do not transfer to each
other. Trial order within a session predicts S on its own (Faure within-maze
AUC 0.62, Nielsen 0.44), so the mechanistic reading of the time-related
features (first-fixation latency) is open; the predictive result is
unaffected. Ordinary AUC including maze is 0.87–0.96 everywhere: the maze
confound, not gaze.

Nielsen's null here is a power limit, not evidence of absence: within-maze
pairs come almost entirely from maze 4 of two days (66 H, 71 S), whereas
msp's Nielsen maze-4 result (z ≈ 9) rests on 351 H and 388 S over ten
SVM-labelled sessions, and msp_revised on the publication sessions is also
null for Nielsen (maze 4 z ≈ 1.3). The `svm` scope runs the classifier on
msp's sessions and labels to settle whether that group-mean difference is
decodable trial by trial.

**Findings (2026-10-06, regions branch, dendro scope).** Within-maze AUC,
logistic / boosted trees: Faure 0.60 / 0.53 (`regions`), 0.61 / 0.53
(dwell only), 0.56 / 0.55 (binary only); Nielsen 0.59 / 0.61, 0.56 / 0.61,
0.56 / 0.54. The nine dwell shares recover most of what the 26 engineered
features gave on the publication sessions, at the cost of the pupil and
latency terms that carried Faure.

**Findings (2026-10-06, svm scope: ten sessions per monkey, SVM labels,
mazes 2–5, pupil tables for every session).** Within-maze AUC, logistic /
boosted trees; null sd ≈ 0.02, p from 200 shuffles; the leave-one-session-out
pooled AUC sits within 0.01–0.04 of every entry (largest gap: Nielsen
engineered trees, 0.64 CV vs 0.55 held-out):

| Branch | Faure (2,384–2,438 trials) | Nielsen (2,935–3,028 trials) |
| --- | --- | --- |
| raw, 50 ms bins (90) | 0.57 (p = 0.015) / 0.61 (p = 0.005) | 0.52 (p = 0.075) / 0.55 (p = 0.055) |
| engineered (26) | 0.56 (p = 0.005) / 0.59 (p = 0.010) | 0.58 (p = 0.005) / 0.64 (p = 0.005) |
| regions, dwell + binary (18) | 0.55 (p = 0.015) / 0.54 (p = 0.055) | 0.57 (p = 0.005) / 0.60 (p = 0.005) |
| regions, dwell only (9) | 0.56 (p = 0.005) / 0.54 (p = 0.055) | 0.57 (p = 0.005) / 0.60 (p = 0.005) |
| regions, binary only (9) | 0.55 (p = 0.015) / 0.52 (p = 0.100) | 0.56 (p = 0.005) / 0.55 (p = 0.005) |

With msp's sessions Nielsen's gaze does carry strategy beyond maze: the nine
dwell shares alone clear chance by 3.4 null sd, every maze sits between 0.53
and 0.59 (maze 4, 351 H / 388 S, is the strongest), and eight of ten
held-out sessions score above 0.5. The direction is the one msp_trends saw:
more dwell at the right exits (RD, then RU) → S, more at the left exit LU
and the origin's upper-left quadrant → H. Faure's svm-scope signal is
weaker than on the publication sessions and sits at the origin rather than
the exits (origin upper-left → S, left exits and time outside every region
→ H), so the two monkeys' maps still do not transfer to each other. Faure
raw traces decode better under trees than under the linear model, Nielsen
raw traces decode under neither. The svm-scope session-held-out numbers
match the cross-validated ones throughout, so none of this rests on mixing
days between training and test.

*Across monkeys (svm scope).* The weight signs agree far more than on the
publication sessions: 18 of 26 engineered coefficients and 6 of 9 dwell
shares share a sign, and the six largest engineered weights all do: more
origin-disc dwell, more RD-exit dwell, less LD-exit dwell, a rising pupil,
more time beyond 10°, and a visited LD exit all → S in both monkeys. The
disagreement is the origin's upper-left quadrant (→ S in Faure, → H in
Nielsen) and the two latency / first-fixation terms. A model fitted on one
monkey and scored on the other reaches only 0.49–0.54 within maze, against
0.55–0.58 within monkey, so the shared part is a direction, not a decoder:
the monkey-specific weights carry most of the magnitude.

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
Output: `regression/out/`.

### `scatter/`

Descriptive: mean gaze (degrees) from `fix_start` to `flash_one`, one dot per
maze with ±1 SD bars, plus a table of directional, uncorrected p-values for
each maze pair. Output: `scatter/out/<Monkey>/`.
