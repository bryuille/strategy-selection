# regression — does where the monkey looked predict its strategy?

One question, one model, one figure.

**Question.** On a given trial, the clustering says the monkey used a
*hierarchical* (H) or a *sequential* (S) strategy. Before the monkey moved, its
eyes visited some of the five landmarks (origin, left-up, left-down, right-up,
right-down). Does that gaze tell us anything about H vs S **beyond what the maze
alone already tells us**?

Self-contained: imports only the shared `data/` layer (read-only).

## The idea in plain language

Logistic regression predicts a yes/no outcome (here: S = 1, H = 0) from
several numbers. It works on the **log-odds** of S:

> log-odds of S = **maze offset** + (effect of A) × A + (effect of B) × B + …

* **Log-odds** are just a scale where 0 means "50/50", positive means "S more
  likely", negative means "H more likely". Each +1 multiplies the odds of S by
  e ≈ 2.7; +0.7 doubles them; −0.7 halves them. The figure's right-hand axis
  shows this directly (×2, ×0.5, …).
* **Maze offset**: one number per maze. Some mazes are mostly S, some mostly H.
  The offset soaks up all of that: "how S-ish is this maze before I look at the
  eyes at all". Anything that is the same on every trial of a maze (where its
  arms are, where the eye is drawn) is absorbed here and cannot masquerade as a
  gaze effect.
* **Effects (coefficients, β)**: how much the log-odds of S moves when a gaze
  number goes up, **with the maze held fixed**. Positive β = looking like this
  goes with S; negative β = goes with H; β near 0 = no relationship.

So the offsets answer "how S-ish is each maze", and the gaze coefficients answer
"within a maze, does this gaze behaviour tip the trial toward S or H".

## The figure

`out/<window>/coefficients.png`: one panel per monkey (monkeys are fit
separately).

* The **horizontal line at 0** runs through the middle: no effect.
* Bars up = toward S, bars down = toward H.
* **Left of the gap: maze offsets** (the baseline of each maze). They are the
  S log-odds of that maze for a trial with average gaze, so they track each
  maze's S rate. They are usually large; that is expected.
* **Right of the gap: the gaze predictors**, grouped by kind of measurement,
  one bar per landmark. Thin black lines are 95% confidence intervals.
* **Solid bar**: the interval excludes 0 (the effect is distinguishable from
  nothing). **Faded**: it does not.
* **★**: still significant after correcting for having looked at ~15
  coefficients at once (Holm). Without that correction, about 1 in 20 faded-out
  coefficients would look significant by luck, so lean on stars.
* **×** at 0: this predictor could not be fitted for that landmark (reason in
  the CSV, and see "Predictors that cannot be separated" below).

## The four predictors (per landmark)

| predictor | what the number is | what β means |
|---|---|---|
| bin occupancy | 1 if the eyes landed on the landmark at all in the window, else 0 | log-odds change for *visited vs not* |
| time occupancy | total time spent on the landmark (log ms) | change per 1 SD more time, among trials that visited it |
| visit count | how many separate visits (an I-DT drift split into fragments counts once) | change per 1 SD more visits, among visited trials |
| mean fixation duration | time ÷ visits (log ms) | change per 1 SD longer fixations, among visited trials |

"Per 1 SD" means per one standard deviation of that measure (a typical spread),
so bars are comparable across predictors. For the last three, an unvisited
trial gets 0 (= the average visited trial), and the bin-occupancy bar carries
visited-vs-not, so the two do not double count.

## Predictors that cannot be separated

Time, visits and duration are tied by arithmetic: **time = visits × duration**
(so log time = log visits + log duration). Any two of them nearly determine the
third, so the model cannot give each its own independent effect. The code tries
the predictors in the order above and drops a predictor, saying why, when:

* **visits barely vary** (fewer than 10 trials off the usual count, which at the
  exits is usually "exactly one visit"). Then mean duration *is* time, and
  duration is dropped as "same as time occupancy". With so few revisits, a
  handful of trials could also separate the labels perfectly and send the
  estimate to infinity, which is the other reason visits are dropped.
* **the rest of the state's predictors already explain it** (> 90% of its
  variance, or the leftover comes from fewer than ~10 trials). This is what
  usually happens to mean duration where visits vary: it is time ÷ visits.

In practice expect **bin occupancy and time occupancy for every landmark, visit
counts at the origin (and wherever revisits are common), and mean duration
rarely**. That is a property of these measurements, not a bug. If you would
rather keep duration and drop time, swap the order in `config.PREDICTORS`.
The `vif` column in the CSV says how entangled each fitted coefficient still is
(1 = independent; above ~5 = its bar is hard to pin down, expect a wide
interval).

## Procedure

1. **Trials**: unfaded (`trial_fade == 0`), photodiode QC pass, `path_type !=
   -99`, maze 1–6, labelled (Ward dendrogram: 0 = H, 1 = S).
2. **Window**: `geofix`, from geo_present to fix_start (variable length,
   median about 1.5 s). Dwell and visit counts grow with window length, so the
   model includes a window-length term (z of log window) to soak that up; it is
   fitted but not plotted (it is the `window` row of `coefficients.csv`). `--window pre1466` (the fixed 1466 ms before fixation start,
   the classifier window) is still available.
3. **Gaze → landmarks**: detect fixations, warp the maze to unit-H, assign each
   fixation to the nearest of five landmarks within radius 1 (`features.py`;
   parity with msp's cache is checked by `--check-parity`).
4. **Mazes**: a maze keeps its offset only if it has at least 5 H and 5 S
   trials (otherwise its offset alone would explain it perfectly). Sessions are
   **pooled within a maze**: one offset per maze, not per maze × session.
5. **Landmarks**: a landmark is used only if the eyes visited it on 5–95% of
   trials.
6. **Fit** one logistic regression (unpenalized Newton-IRLS; it matches
   scikit-learn to 1e-6) with the offsets and every surviving predictor at once.
   Predictors are centred, which makes each offset the maze's baseline.
7. **Intervals**: 95% Wald intervals; Holm-adjusted p-values over the gaze
   coefficients.

## What this does *not* show

* **Not causal.** A gaze pattern that goes with S may be a consequence of the
  strategy rather than a cause.
* **Trials are not independent** (same session, neighbouring trials, same
  monkey). Intervals assume independence and are somewhat optimistic.
* **Offsets are not tested**; they are shown for scale.
* Most mazes are nearly all-H or all-S for a given monkey and are dropped at
  step 4. The estimates rest on the few mixed mazes, listed in `mazes.csv`.
* Predictors interact in a landmark (they are bars from one joint fit), so
  read each bar as "holding the others in that panel fixed".

## Running

```bash
python -m regression.checks                          # synthetic invariants, seconds
python -m regression.build --dry-run                 # mazes kept, visit rates, no fit
python -m regression.build                           # fit both monkeys, write CSVs + figure
python -m regression.build --window pre1466
python -m regression.build --check-parity            # pre1466 vs msp cache
```

Detection runs on the raw eye npz, which for three of the four sessions exists
only on the cluster; fixation tables are cached as
`<Monkey>_reg_fixed5_r1_<window>.npz` and are unchanged by this revision, so
existing cluster caches are reused.

```bash
mkdir -p logs
sbatch regression/run_regression.sbatch --dry-run
sbatch regression/run_regression.sbatch
rsync -av engaging:strategy-selection/regression/out/ ./regression/out/
```

The fit itself is seconds; only a cold cache build is slow.

## Output

```
out/<window>/coefficients.png   THE figure
out/<window>/coefficients.csv   every coefficient: beta, SE, 95% CI, z, p, Holm p, VIF, status
out/<window>/mazes.csv          per maze: H and S counts, whether it was kept
out/<window>/summary.csv        per monkey: rows, params, convergence, landmarks kept/dropped
out/legacy/                     everything from the previous design (per maze × session
                                offsets, block tests, bootstrap, CV), kept for reference
```

## Files

| file | purpose |
|---|---|
| `config.py` | sessions, windows, codebook, predictor order, inclusion thresholds |
| `features.py` | detect → warp → assign per window; processed cache; `trial_measures` |
| `labels.py` | dendrogram label lookup (never builds) |
| `model.py` | design (offsets + four predictors), IRLS, Wald table, Holm |
| `build.py` | entry point |
| `figures.py` | `coefficients.png` |
| `checks.py` | synthetic checks (offsets remove geometry, planted effects recovered, CI coverage, redundancy rules) |
| `run_regression.sbatch` | cluster job, `-c 1` |
