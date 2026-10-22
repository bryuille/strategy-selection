# `counterfactual/`: post-choice saccades vs the strategy's exit ranking

Question: after the monkey commits to an exit, do its saccades to the
*unchosen* exits follow the ranking that its strategy predicts? Trials are
neurally labelled hierarchical (H) or sequential (S); each strategy, as a
behavioral model, ranks the four exits per trial. The test asks whether the
first counterfactual saccade of H-labelled trials is better explained by the
H ranking and that of S-labelled trials by the S ranking.

Commands: [README_COMMANDS.md](../README_COMMANDS.md#counterfactual).

## Inputs

- **Sessions**: the top-two SNR sessions per monkey (`config.SESSIONS`:
  Faure june_8_g0, june_24_g0; Nielsen Nov_3_g0, Nov_6_g0).
- **Trials**: QC-passing (`data.builder.trial_qc_ok`) with exactly one
  `trial_answer*` flag set and an estimated saccade threshold.
- **Labels**: the dendrogram strategy labels (`<session>_strategy_choices.npz`,
  0 = H, 1 = S), read, never built here.
- **Looks**: `data.looks` (the advisor's `timestamp_preprocessing`), epoch
  `trial` (maze onset to `trial_end`). A look is a qualified stable interval
  (>= 75 ms of stable support, max pairwise spread <= 2°) bounded by detected
  saccades and data gaps; its position is the time-weighted raw centroid.
- **Window**: looks overlapping `[saccade_init, trial_end]` (the look on the
  chosen exit usually begins before `saccade_init`, so looks straddling the
  start count; `n_straddle` tallies them).

## Codes and the exit list (`extract.py`)

`1 = LU, 2 = LD, 3 = RU, 4 = RD` (DATA_DICTIONARY.md), `0` = origin, `-1` =
elsewhere. Each look's raw centroid (degrees) is assigned to the nearest exit
within `EXIT_RADIUS_DEG` (3°), with the exits at `(-h1, h2)`, `(-h1, -h3)`,
`(h4, h5)`, `(h4, -h6)`; otherwise origin within 3°, otherwise elsewhere.

The **chosen exit** is decoded from the active `trial_answer{1..4}` flag and
`LR`, `LR2` (`exits.chosen_exit`; `checks.py` verifies it against the archived
16-entry table). The **exit list** is the chosen exit followed by every
post-choice look whose state is an exit, in time order, repeats kept, e.g.
`[3, 3, 1, 2, 3]` = `RU RU LU LD RU`. The first look in the window is where
the choice saccade landed; when it is on the chosen exit (`landing_on_chosen`,
true on 99% / 96% of trials, the sanity check of the decoding) item 0 already
stands for it and the post-choice items start with the next look.

## The two models (`models.py`)

Ported from the advisor's Figure Five simulation
(`zarchive/Figure_Five/03_Likelihood_Analysis/`, `simulate_hs_models` and
`Model_Helpers/{Hierarchical,Sequential}_Model_Physiology.m`). Per trial the
true path's two interval durations (arm length / 10°/s) are measured with
Weber noise, `one = t1 + N(0, 1) · scalar · t1`, and arms are scored with
`normpdf(measured, arm, scalar · arm)`:

- **H** compares `one` against h1 vs h4 and commits left or right (fair coin if
  equal), then compares `two` against the two vertical arms on that side.
- **S** (the paper's "second-only") compares `two` against all four vertical
  arms; h1 and h4 are never consulted.

10,000 noise draws per trial give each model's choice probability over LU, LD,
RU, RD. `scalar` is the published mean Weber fraction per monkey, **Faure
0.165, Nielsen 0.153** (`MR_AG_MJ/load_timing_scalar.m` reading
`Data/Pre_Physiology/Model_Results/Behavioral_Psychometrics/timing_scalars.mat`).
The counterfactual distribution `q` is the model's probability renormalised
over the three unchosen exits (with `MODEL_EPS = 1e-3` smoothing). Cache
`<Monkey>_cf_models.npz`.

## The test (`analysis.py`, `build.py`)

- **Primary.** For the first counterfactual exit `e1`:
  `delta = log q_matched(e1) − log q_mismatched(e1)`, matched = the trial's
  label. Per monkey: mean, bootstrap CI, and a permutation p from 10,000 label
  shuffles within session × maze (Figure Five's label-shuffle control). Labels
  are nearly a function of maze (maze 1 all H, maze 6 all S), so this null
  sits near the observed value: the CI tests delta against 0 across trials,
  the permutation tests the label effect *beyond maze*. The **mixed strata**
  rows (session × maze with >= 10 trials of each label) are where the two can
  be told apart.
- **Secondary.** Top-counterfactual accuracy (`e1` equals each model's
  highest-ranked unchosen exit), by label, on the trials where H and S rank a
  different exit first; a Plackett–Luce likelihood of the ordered distinct
  unchosen exits.
- **Controls.** Uniform `q = 1/3`; a spatial-proximity model
  `q ∝ exp(−d / 5°)`, because H predicts the same-side exit, which is also the
  nearest; the same-side rate of `e1` by label; correct vs error trials;
  sensitivity to the smoothing and to the Weber scalar (×0.7, ×1.3).

## Outputs (`counterfactual/out/<Monkey>/`)

`exit_sequences.csv` (one row per trial), `trials.csv` (per labelled trial:
label, `e1`, `q_H`, `q_S`, log likelihoods), `results.csv` (long format, CIs
and p values), `summary.txt`, `delta_by_label.png`, `top_accuracy.png`.
Movies: `counterfactual.movie` renders one `.mp4` per trial or a `--batch`
video under `out/movies/<Monkey>/<session>/`, reading looks and saccades from
the exit cache (clock from flash three, the analysed window's start marked,
the strategy label in the title).

## Findings (2026-10-22, first run; 10,000 label shuffles)

| | Faure | Nielsen |
| --- | --- | --- |
| labelled trials (H / S) | 660 (291 / 369) | 870 (453 / 417) |
| with a counterfactual exit look | H 31%, S 37% | H 71%, S 69% |
| delta, all trials (95% CI) | +0.59 [+0.42, +0.77], n 224 | +0.13 [+0.01, +0.26], n 608 |
| within-maze permutation p (null mean) | 0.75 (+0.62) | 0.12 (+0.10) |
| H trials: log q_H − log q_S | +1.02 | +0.47 |
| S trials: log q_S − log q_H | +0.31 | −0.24 |
| mixed strata delta (p) | +0.13, n 94 (0.64) | +0.25, n 172 (0.11) |
| disagreement trials, H-labelled: e1 = H top / S top | 82% / 13% (n 39) | 36% / 28% (n 169) |
| disagreement trials, S-labelled: e1 = H top / S top | 5% / 69% (n 55) | 37% / 32% (n 123) |

Across all trials the matched model wins in both monkeys, strongly in Faure,
and in Faure the first counterfactual look picks the matched model's top exit
on the disagreement trials (82% for H trials, 69% for S trials). But the
within-maze permutation does not separate this from the maze effect: Faure's
labels are almost fixed by maze, and in the four mixed strata the effect is
small and not significant. In Nielsen the H trials favour H (+0.47) while the
S trials favour H too (−0.24 for S), so the all-trials delta is carried by
the H-labelled trials, and the mixed strata show the same pattern (H +0.42,
S −0.07). The proximity model scores as well as or better than either
strategy model in Nielsen (mean log q −1.22 / −1.33 vs H −1.97 / −1.83), so a
"look at the nearest exit" account is not excluded there. Error trials show
no matched-model advantage in either monkey (Faure −0.13, n 44; Nielsen
−0.10, n 137).
