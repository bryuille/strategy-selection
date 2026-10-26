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
| `classifier/`  | Predict the strategy label from pre-fixation gaze: raw traces vs region dwell, within-maze AUC; dendrogram or SVM scope |
| `scatter/`    | Mean gaze per maze between fixation and the flash                        |
| `counterfactual/` | Post-choice looks at the unchosen exits vs each strategy's choice-conditioned belief (top-two SNR sessions) |
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
Eye events are *looks* from the `timestamp_preprocessing` package
(repo root, consumed through `data.looks`, never edited here): samples keep
their own timestamps, saccades are Engbert-style, and a look is a stable
interval between saccades or data gaps with at least 75 ms of support within a
2° pairwise spread; its position is the time-weighted raw centroid and its
dwell the stable support (see `timestamp_preprocessing/HANDOFF.md`). The
pymovements detector in `data.builder` is legacy (`msp/legacy` only).

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

`--codebook` swaps the five balls for a region codebook over the same
extraction (tag suffix, e.g. `r0.5capsules`):

- `balls` (default): the origin ball plus the four exit balls (LU, LD, RU, RD).
- `quads` / `halves`: the origin ball plus the rest of the maze (the r1 exit
  balls' hull) split into quadrants (5 states) or left / right (3).
- `capsules`: one capsule per exit, every point within the radius of the
  origin-to-exit segment; 4 states, no origin state. The overlap around the
  origin goes to the nearest segment, i.e. by quadrant, so central gaze
  before maze onset is spread over all four.

`no_origin` runs only for `balls` and `quads` (the others would have no
origin to drop, or leave two states, which always correlate at ±1).

Read `z`, not raw `r`, which rises with group size. Under `mean_removed`, a
diagonal can be pulled toward −1 when centring removes most of that strategy's
signal; on imbalanced mazes this hits the majority. Output:
`msp/out/<tag>/<Monkey>/`.

- `msp.per_session`: the same estimator one session at a time
  (`msp/out/per_session/`).
- `msp.saccades`: gaze traces per session, maze and trial (`msp/out/saccades/`).
- `msp.legacy.saccades`: those plots under the retired
  `[fix_start − 1466 ms, fix_start]` window (`msp/legacy/out/`).

**Findings (2026-10-22, looks-based caches).** Rerun of the `balls` codebook
at `r1` (default) and `r0.5`, both monkeys, mazes 2–5, 1000 shuffles, after
the switch from the pymovements detector to `data.looks`. The outputs from
before the looks switch are under `msp/out/pre_looks_2026-10-22/`. Nothing
changes: the looks QC admits a handful more trials per maze (Faure maze 2
322 → 323 H / 210 → 211 S, Nielsen maze 4 351 → 352 / 388 → 390), and every
z moves by less than one unit. `full` variant z, before → after the looks
switch (p where either side is below 0.05):

| Maze | Faure `r1` | Faure `r0.5` | Nielsen `r1` | Nielsen `r0.5` |
| --- | --- | --- | --- | --- |
| 2 | 3.2 → 3.8 (p 0.011 → 0.006) | 2.1 → 3.3 (p 0.042 → 0.014) | 0.3 → 0.9 | 3.2 → 3.3 (p 0.019 → 0.016) |
| 3 | 1.1 → 0.1 | −0.6 → −0.6 | −1.1 → −0.8 | −1.2 → −0.9 |
| 4 | 0.3 → 0.1 | 0.1 → −0.1 | 7.4 → 7.7 (p 0.001) | 9.0 → 8.8 (p 0.001) |
| 5 | 0.5 → 0.4 | −0.3 → −0.6 | −0.3 → −0.2 | 0.0 → −0.1 |

`no_origin` and `mean_removed` behave the same way: Nielsen maze 4 stays at
z 8.6 → 8.6 (`no_origin`, `r1`), 7.8 → 7.7 (`r0.5`), 2.8 → 3.0 and 2.8 → 2.7
(`mean_removed`); Faure maze 2 `mean_removed` is 2.0 → 2.2 (`r1`) and
1.6 → 2.0 (`r0.5`, p 0.087 → 0.022), the one cell that crosses 0.05.

**Earlier codebook search (2026-10-05; the `msp_revised` package, removed
2026-10-06, is in git history).** 41 codebooks in unit H and degrees on the
publication sessions, mazes 1–6, 936 configurations at n_perm 250 with
finalists at 1000, pooled as the trial-weighted mean of per-maze Δ. None
reached pooled p < 0.05 in both monkeys with the same sign (best min pooled
z ≈ 1.7, all with pupil features). The only recurring lead was binary
occupancy of origin + exits + the point above the origin, z ≈ 2 in both
monkeys with all mazes in one 2x2 but not per maze. The right-of-origin trend
across mazes is maze geometry: it rises with maze number in degrees too and
flips within maze.

### `classifier/`

Can gaze predict the strategy label beyond what the maze already tells you?
Two label scopes (`--labels`): `dendro` (default), the dendrogram labels on
the publication sessions, mazes 1–6; and `svm`, the
SVM labels on the ten best sessions per monkey, mazes 2–5, the msp scope
(mazes 1 and 6 are the SVM's anchors). Two feature branches, one evaluation:

- `raw`: the eye trace itself, gaze x / y (degrees) and pupil over the last
  1.5 s before fixation onset, block-averaged from the 1 kHz trace to a bin
  width (default 50 ms → 90 features; `--bin-ms 1` is the whole 1 kHz
  trace), each bin session-z-scored. A bin with no gaze sample (blink,
  off-screen) stays NaN, never interpolated: the tree model takes NaN
  natively and the logistic pipeline imputes it to the column mean. Trials
  with under half the window covered are dropped. Nothing hand-picked.
- `regions`: dwell and visits over one
  partition of the screen in degrees: the four quadrants of the 5° origin
  disc, the four 3° exit balls, and everything else. Two sets over the same
  nine regions, msp's binary / dwell pair: `dwell` (share of in-window
  fixation time per region, sums to 1) and `binary` (region visited or
  not); 18 features, no ordering, latency or pupil terms. `regions_dwell`
  and `regions_binary` run one set alone. Binary is nested in dwell
  (visited ⇔ dwell share > 0), so the tree model gives `regions` and
  `regions_dwell` identical fits; for the logistic model the pair adds a
  little or nothing over dwell alone.
- `regions_plus`: the same, with "elsewhere" split into near (5–10°) and far
  (> 10°) and a pupil-change term (late minus early, z within session)
  added: the two engineered terms whose sign agreed across monkeys and held
  under a trial-order control.
- `exits`: the four 3° exit balls alone, dropping the origin quadrants and
  "elsewhere" (8 features for dwell + binary). Dwell share is still of all
  in-window fixation time, so the four shares sum to the time spent at any
  exit, not to 1.
- `lor`: msp's `halves` codebook in degrees: a 3° origin ball, then the left
  and right half of the maze, where "inside the maze" is within 3° of the
  quadrilateral the exits span (the perimeter hugs a 3° ball at each exit,
  not msp's wide r1 margin). 6 features for dwell + binary; fixations
  outside the perimeter are unassigned. `classifier.codebook` draws it.
- `balls`: msp's `r0.5` codebook in degrees: a 3° origin ball plus the four 3°
  exit balls, nearest ball wins, everything else unassigned (10 features for
  dwell + binary). No quadrants, no "elsewhere".

Every classifier region is in degrees around each maze's own exit locations;
the classifier never warps gaze to unit H (the 3° balls stand in for msp's
r0.5 unit-H balls, which span 2.5–4.5° depending on maze and direction).

Pupil comes from the per-session tables `classifier.pupil_extract` writes
on the cluster (`classifier/out/pupil/`); the fixation table is
`classifier.fixations`; a session without one gets NaN pupil features (only the
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
other. Output: `classifier/out/<scope>/<raw|regions|regions_plus|exits|lor|balls>/<variant>/<model>/`
(`<Monkey>.png`, `results.csv`, `features.csv`, plus `classifier.pres`'s
`merged_auc.*` / `session_auc.*` bar chart where it has been run): scope `dendro` / `svm`;
variant `b<bin ms>` for raw, `both` / `dwell` / `binary` for regions, with
`_p<n>` for a non-default shuffle count (the `_p1000` runs) and `_x_maze` for
interactions. How each figure is built: [classifier/README.md](classifier/README.md).

**Findings (2026-10-05, dendro scope; before the looks switch).** Within-maze AUC, out of fold, with the shuffle
null's mean ≈ 0.50 and sd ≈ 0.05 (200 shuffles). Empty gaze bins are kept as
NaN (no interpolation). Counts below include the (later dropped) valid-gaze
coverage channel:

| Branch / model | Faure | Nielsen |
| --- | --- | --- |
| raw, logistic, 50 ms bins (156 features) | 0.68 (p = 0.005) | 0.51 (chance) |
| raw, logistic, 1 ms = whole trace (6,036) | 0.67 (p = 0.005) | 0.56 (p = 0.050) |
| raw, boosted trees, 50 ms | 0.64 (p = 0.005) | 0.49 (chance) |

Faure's gaze carries strategy information beyond maze in every branch and
transfers across sessions (train one day, test the other: 0.58–0.70).
Nielsen is at chance on the raw traces (the whole-trace 0.56 does not
transfer: 0.45 one way, 0.66 the other). Bin width does not matter much
in the fixed-width sweeps. Raw-trace weights: lower pupil → S in Faure
(through most of the window), pupil rising in the last 250 ms → S in
Nielsen; and in both monkeys the
largest single raw weight is gaze height about 275 ms before fixation, with
opposite signs, which is why the monkeys' models do not transfer to each
other. Trial order within a session predicts S on its own (Faure within-maze
AUC 0.62, Nielsen 0.44), so the mechanistic reading of the time-related
features is open; the predictive result is unaffected. Ordinary AUC including maze is 0.87–0.96 everywhere: the maze
confound, not gaze.

Nielsen's null here is a power limit, not evidence of absence: within-maze
pairs come almost entirely from maze 4 of two days (66 H, 71 S), whereas
msp's Nielsen maze-4 result (z ≈ 9) rests on 351 H and 388 S over ten
SVM-labelled sessions, and the earlier codebook search on the publication sessions was also
null for Nielsen (maze 4 z ≈ 1.3). The `svm` scope runs the classifier on
msp's sessions and labels to settle whether that group-mean difference is
decodable trial by trial.

**Findings (2026-10-06, regions branch, dendro scope; before the looks switch).** Within-maze AUC,
logistic / boosted trees: Faure 0.60 / 0.53 (`regions`), 0.61 / 0.53
(dwell only), 0.56 / 0.55 (binary only); Nielsen 0.59 / 0.61, 0.56 / 0.61,
0.56 / 0.54. The regions carry no pupil or latency terms, which carried
Faure's raw-trace result.

**Findings (2026-10-06, svm scope: ten sessions per monkey, SVM labels,
mazes 2–5, pupil tables for every session; before the looks switch).** Within-maze AUC, logistic /
boosted trees; null sd ≈ 0.02, p from 200 shuffles; the leave-one-session-out
pooled AUC sits within 0.01–0.04 of every entry:

| Branch | Faure (2,384–2,438 trials) | Nielsen (2,935–3,028 trials) |
| --- | --- | --- |
| raw, 50 ms bins (90) | 0.57 (p = 0.015) / 0.61 (p = 0.005) | 0.52 (p = 0.075) / 0.55 (p = 0.055) |
| regions, dwell + binary (18) | 0.55 (p = 0.015) / 0.54 (p = 0.055) | 0.57 (p = 0.005) / 0.60 (p = 0.005) |
| regions, dwell only (9) | 0.56 (p = 0.005) / 0.54 (p = 0.055) | 0.57 (p = 0.005) / 0.60 (p = 0.005) |
| regions, binary only (9) | 0.55 (p = 0.015) / 0.52 (p = 0.100) | 0.56 (p = 0.005) / 0.55 (p = 0.005) |

With msp's sessions Nielsen's gaze does carry strategy beyond maze: the nine
dwell shares alone clear chance by 3.4 null sd, every maze sits between 0.53
and 0.59 (maze 4, 351 H / 388 S, is the strongest), and eight of ten
held-out sessions score above 0.5. The direction is the one the msp codebooks show:
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
publication sessions: 6 of 9 dwell shares share a sign (more RD-exit dwell
and less LD-exit dwell → S in both monkeys). The disagreement is the
origin's upper-left quadrant (→ S in Faure, → H in Nielsen). A model fitted on one
monkey and scored on the other reaches only 0.49–0.54 within maze, against
0.55–0.58 within monkey, so the shared part is a direction, not a decoder:
the monkey-specific weights carry most of the magnitude.

**Findings (2026-10-22, looks-based caches, both scopes).** The default
branches (`raw`, 50 ms bins; `regions`, dwell + binary) rerun with eye events
from `data.looks` (SCHEMA 4 fixation caches, RAW_SCHEMA 3 raw caches, 200
shuffles, pupil tables rebuilt from the eye mats for all twenty sessions).
The outputs from before the looks switch, the pymovements detector, are
under `classifier/out/pre_looks_2026-10-22/`. Within-maze AUC, before →
after, logistic / boosted trees:

| Scope, branch | Faure | Nielsen |
| --- | --- | --- |
| dendro, raw (636 → 628 / 849 → 845 trials) | 0.63 → 0.65 (p 0.005) / 0.64† → 0.63 (p 0.010) | 0.48 → 0.47 (chance) / 0.49† → 0.48 (chance) |
| dendro, regions (652 → 660 / 865 → 870) | 0.60 → 0.52 (p 0.020 → 0.29) / 0.53 → 0.56 (p 0.20 → 0.11) | 0.59 → 0.59 (p 0.035 → 0.030) / 0.61 → 0.55 (p 0.005 → 0.12) |
| svm, raw (2,384 → 2,359 / 2,935 → 2,924) | 0.57 → 0.57 (p 0.015 → 0.010) / 0.61 → 0.63 (p 0.005) | 0.52 → 0.52 (p 0.075 → 0.14) / 0.55 → 0.55 (p 0.055 → 0.10) |
| svm, regions (2,438 → 2,453 / 3,028 → 3,053) | 0.55 → 0.54 (p 0.015 → 0.040) / 0.54 → 0.53 (p 0.055 → 0.11) | 0.57 → 0.58 (p 0.005) / 0.60 → 0.60 (p 0.005) |

† the 2026-10-05 run with the coverage channel (156 features); the
moved-aside tree has no later trees fit for the dendro raw branch.

The raw branch reads the eye trace directly, so only its trial set changes
with the looks QC, and its results are untouched: Faure decodes in both
scopes (dendro leave-one-session-out 0.68 / 0.65; svm 0.57 / 0.60), Nielsen
does not. The regions branch is where the detector enters, and on ten
sessions it holds: Nielsen's dwell + binary stays at 0.58 / 0.60 (p 0.005,
leave-one-session-out 0.57 / 0.59) and Faure's at 0.54 / 0.53, with the
logistic p slipping from 0.015 to 0.040. On the two publication sessions,
where the null sd is 0.05, the regions entries move by up to 1.5 null sd in
both directions (Faure logistic 0.60 → 0.52, Nielsen trees 0.61 → 0.55), so
the dendro-scope regions numbers above should be read as noise around 0.55
rather than as a result either way; the svm scope carries the regions
claim.


### `counterfactual/`

After the choice, do the looks at the *unchosen* exits follow what the
monkey's strategy believes? Each behavioral model (Hierarchical, Second-only
and the other Figure One models; published Weber fractions 0.165 / 0.153)
stores one row per (maze, chosen exit): its belief over the other three
exits, from noise draws on the chosen exit's path kept when the model's rule
picks that exit. The looks from `flash_three` to `trial_end`
(`timestamp_preprocessing` via `data.looks`) are scored with Plackett–Luce:
a relative metric (H − S within each label group, one-sided paired t-test)
and an absolute one (gain over chance). Top-two SNR sessions per monkey,
dendrogram labels. Output: `counterfactual/out/<Monkey>/`; method, results
and the descriptive models of the looks in
[counterfactual/README.md](counterfactual/README.md) and
[counterfactual/DESCRIPTIVE.md](counterfactual/DESCRIPTIVE.md).

### `scatter/`

Descriptive: mean gaze (degrees) from `fix_start` to `flash_one`, one dot per
maze with ±1 SD bars, plus a table of directional, uncorrected p-values for
each maze pair. Output: `scatter/out/<Monkey>/`.
