# `classifier/`: how the figures are made

Question: on a single trial, does gaze before fixation predict the neural
strategy label (H = 0, S = 1) **beyond which maze the trial was**?

Commands and flags: [README_COMMANDS.md](../README_COMMANDS.md#classifier).
Findings: [README.md](../README.md#classifier).

## Inputs

- **Trials**: QC-passing trials from one label scope, one monkey at a time.
  - `dendro`: the 2 publication sessions per monkey, dendrogram labels, mazes 1–6.
  - `svm`: the top-10 sessions per monkey, SVM labels, mazes 2–5. Mazes 1 and
    6 are left out because they are the SVM's training anchors.
- **Window**: maze onset to fixation onset (about 1.5 s).
- **Features** (one set per branch):
  - `raw`: gaze x, gaze y and pupil, averaged in 50 ms bins over the last
    1.5 s. Each bin is z-scored within session, and empty bins stay NaN.
  - `engineered` (26):
    - origin-disc dwell share and its four quadrant shares;
    - for each of the four exit balls, visited (yes / no) and dwell share;
    - eccentricity-ring shares, fixation count;
    - first and last fixation location (origin / left / right);
    - first-fixation onset;
    - pupil mean and change (z within session).
  - `regions`: the same nine regions (4 origin quadrants, 4 exit balls,
    elsewhere), as dwell share (`regions_dwell`), visited (`regions_binary`),
    or both (`regions`). No pupil, latency or order terms.
  - `order`: trial position in the session (linear and squared) and nothing
    else. It shows how much the trial-order confound alone predicts.
- Fixations are I-DT detections. Regions are measured in degrees around
  maze-specific exit locations.

## What is trained and what is tested

1. **Model input**: the features plus a **maze one-hot**, in every model.
   Models are `logreg` (L2 logistic, C = 0.1, NaN imputed to 0 after
   standardising) and `gbm` (shallow boosted trees, which take NaN natively).
2. **Splits**: 5-fold cross-validation, repeated 4 times with different
   shuffles. Folds are stratified on (session, maze, label), so every test
   fold has the same mix as the whole pool. Each fold trains on 4/5 of the
   trials and predicts P(S) for the held-out 1/5.
3. **Averaging**: every trial gets one out-of-fold P(S) per repeat. The
   4 values are averaged into one out-of-fold P(S) per trial.
4. **Score: within-maze AUC.** Take every (H, S) pair of trials **from the
   same maze**. The score is the fraction of pairs where the S trial got the
   higher P(S); ties count ½. Pairs are pooled over mazes, so a maze with
   more pairs weighs more. The maze one-hot cannot help here, because it is
   identical within a pair. Per-maze bars in the figure are the same score
   for one maze.
5. **Null and p**: the whole fit is repeated `n_perm` times (1 repeat each)
   with labels shuffled within (session, maze). p is one-sided,
   p = (1 + #null ≥ observed) / (n_perm + 1). z is measured against the null
   mean and sd, and the grey band in the figure is the null's 95 % range.
6. **Held-out session**: train on all other sessions and predict the left-out
   one; repeat for each session. Only pairs from the same (session, maze)
   are scored, because predictions from different fits are not comparable.
   This tests transfer across days.
7. **Weights** (right panel): one fit on all trials. For `logreg`, these are
   standardised coefficients (red → S, blue → H). For `gbm`, they are
   permutation importances (the drop in within-maze AUC when a feature is
   shuffled). They are fitted in-sample and are for ranking features only.

## Trial-order control (`--order-control`)

Strategy labels drift over a session, so a feature that also drifts (pupil,
fatigue, latency) can look predictive. With this flag:

- Trial position (linear and squared) is added to every model's input.
- The null keeps the labels and trial order fixed and changes the gaze
  instead. Each feature is split into the part predicted by trial order
  (fitted within session) and the residual. Only the residuals are shuffled
  within (session, maze).

Order therefore earns the same credit in the real fit and in the null, and p
asks whether gaze adds anything beyond drift. The AUC itself includes order's
share, so read it against the null mean, not against 0.5.

## Design choices worth knowing

- **Why within-maze AUC**: maze nearly determines strategy on these sessions,
  and maze can be read from gaze. The ordinary AUC (about 0.9, printed in
  the title) is mostly maze.
- **Shuffle within (session, maze)**: this keeps each session's label
  balance per maze, so the null has the same structure as the data.
- **Tree nulls sit above 0.5** (up to 0.57). Always compare tree AUCs with
  their null.
- **`regions` and `regions_dwell` give identical trees**: "visited" is just
  "dwell > 0", so the trees make the same splits.
- **Missing pupil** → NaN features, not dropped trials.

## Outputs

`out/<branch>[_svm][_x_maze][_order]/<model>/`, or under `--out-root`:

- `<Monkey>.png`: left panel, AUC per maze, pooled, and the null band; right
  panel, the top 20 weights.
- `results.csv`: one row per maze, one per held-out session, the
  `leave-one-session-out pooled` row, and the `within-maze pooled` row
  (AUC, null mean and sd, z, p, ordinary AUC, n_H, n_S, n_perm).
- `features.csv`: weights or importances.
- `eval_<Monkey>.npz`: everything `--replot` needs to redraw the figure.

Existing result sets: `out/` (n_perm 200), `out/nperm1000/`, and
`out/order_control/` (n_perm 1000 with the trial-order control, plus the
`order` branch alone).
