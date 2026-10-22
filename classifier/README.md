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
  - `raw_plus`: the `raw` traces plus a 6 × 6 gaze map (share of in-window
    samples per 5° cell over ±15°), so timing and static position together.
  - `regions`: nine regions (4 origin quadrants, 4 exit balls,
    elsewhere), as dwell share (`regions_dwell`), visited (`regions_binary`),
    or both (`regions`). No pupil, latency or order terms.
  - `regions_plus` (same three variants): the regions with "elsewhere"
    split into near (5–10°, outside the exit balls) and far (> 10°), so
    dwell shares still sum to 1, plus one pupil-change term (mean of the
    last 300 ms minus the first 300 ms, z within session). These are the
    two engineered terms that kept their sign in both monkeys and survived a
    trial-order control.
  - `all` (a variant of every regions family: `regions_all`, `regions_plus_all`,
    `exits_all`, `lor_all`, `balls_all`): dwell share + visited + visit count,
    where a visit is one entry into the region (consecutive fixations inside
    it count once).
  - `exits` (same variants: `exits`, `exits_dwell`, `exits_binary`,
    `exits_all`): the four exit balls alone, no origin quadrants and no
    "elsewhere". Dwell share is still of all in-window fixation time, so the
    four shares sum to the time spent at any exit rather than to 1.
  - `lor` (same variants): msp's `halves` codebook in degrees: a 3° origin
    ball, then the left (x < 0) and right half of "inside the maze", every
    point within 3° of the quadrilateral the four exits span, so the
    perimeter hugs a 3° ball at each exit (`LOR_ORIGIN_DEG`,
    `LOR_PERIMETER_DEG`). Fixations outside are unassigned.
  - `balls` (same variants): msp's r0.5 codebook in degrees: a 3° origin ball
    (`BALLS_ORIGIN_DEG`) plus the four 3° exit balls, nearest ball wins,
    everything else unassigned. Five regions, no quadrants, no "elsewhere".
  - `capsules` (same variants): a 3° origin ball, four capsules and elsewhere.
    A capsule is every point within 3° of the segment from the origin to one
    exit, the two balls joined by parallel tangents; the origin ball wins
    where they overlap. `capsules_tight` keeps those capsules and shrinks the
    origin ball to 1.5°.
- Everything in `classifier` is in degrees around each maze's own exit
  locations; nothing is warped to unit H. Where a codebook mirrors one of
  msp's (whose balls are r0.5 unit H), a 3° ball stands in: 0.5 unit H is
  2.5–4.5° depending on maze and direction.
- `classifier.codebook` draws any of `regions`, `exits`, `lor`, `balls` over
  the six mazes with every fixation centroid coloured by its region
  (`out/<scope>/<branch>/<branch>_by_maze.png`), to check a codebook against the data.
- Fixations are looks from `data.looks` (the advisor's `timestamp_preprocessing`): >= 75 ms stable support, bounded by saccades and gaps; dwell = stable support. Regions are measured in degrees around
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
   mean and sd, and the grey band in the figure is the null's 5th–95th
   percentile, so its top edge is the p = 0.05 cutoff.
6. **Held-out session**: train on all other sessions and predict the left-out
   one; repeat for each session. Only pairs from the same (session, maze)
   are scored, because predictions from different fits are not comparable.
   This tests transfer across days.
   With `--per-session`, steps 1–5 run on each session alone instead (no
   held-out-session fit), and a `combined` row per monkey pools them: the
   pair-weighted mean of the per-session within-maze AUCs, tested against the
   same weighted mean of the per-session nulls, draw by draw.
7. **Weights** (right panel): one fit on all trials. For `logreg`, these are
   standardised coefficients (red → S, blue → H). For `gbm`, they are
   permutation importances (the drop in within-maze AUC when a feature is
   shuffled). They are fitted in-sample and are for ranking features only.

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
- **Missing pupil** → NaN features, not dropped trials (raw and
  `regions_plus`).

## Outputs

`out/<scope>/<family>/<variant>/<model>/`, where:

- scope is `dendro` or `svm`;
- family is `raw` (variant `b<bin ms>`, e.g. `b50`, `b1`), `regions`,
  `regions_plus`, `exits`, `lor` or `balls` (variant `both`, `dwell`,
  `binary` or `all`);
- the variant gets `_p<n>` for a non-default shuffle count (`both_p1000`) and
  `_x_maze` with `--interactions`, so no run overwrites another;
- model is `logreg` or `gbm`.

Each folder holds:

- `<Monkey>.png`: left panel, AUC per maze, pooled, and the null band; right
  panel, the top 20 weights.
- `results.csv`: one row per maze, one per held-out session, the
  `leave-one-session-out pooled` row, and the `within-maze pooled` row
  (AUC, null mean and sd, z, p, ordinary AUC, n_H, n_S, n_perm).
- `features.csv`: weights or importances.
- `eval_<Monkey>.npz`: everything `--replot` needs to redraw the figure.
- `merged_auc.{png,svg}` (pooled runs) or `session_auc.{png,svg}`
  (`_per_session` runs), where `classifier.pres` has been run on the folder:
  the presentation bar chart of within-maze AUC against the shuffle null,
  one bar per monkey or per (monkey, session).

`out/pupil/` holds the per-session pupil tables (inputs, not results).
