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
- **Looks**: `data.looks` (the `timestamp_preprocessing` package), epoch
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

Ported from the Figure Five simulation
(`zarchive/MR_AG_MJ/Figure_Five/03_Likelihood_Analysis/`, `simulate_hs_models` and
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

## The test (`compare.py`)

Within each label group, does the label's model explain the counterfactual
saccades better than the other model? The hierarchical model H is compared
with the sequential model S. (`--alt total` / `--alt optimal` swap in Figure
Five's Total Time or Optimal model as an experiment; those outputs are not
kept.)

Per trial the ordered looks at unchosen exits are scored under both models
and differenced: `log P_H(sequence) − log P_S(sequence)`, where `P` is the
Plackett–Luce probability of drawing those exits in that order without
replacement from the model's unchosen-exit probabilities (the model's choice
probability renormalised over the three unchosen exits, `MODEL_EPS`
smoothing). With one counterfactual saccade this is `log q_H(e) − log q_S(e)`.

Groups: H-labelled trials (is H − S above 0?), S-labelled trials (is S − H
above 0?), and all trials of the unit (matched − mismatched above 0). Units:
each session and the whole monkey (the `sessions` figures) and each
session × maze stratum (the `strata` figures). Both variants are produced:
all counterfactual saccades and the first one only.

**How the p values are computed.** Let `d_i` be trial *i*'s difference,
oriented so that positive favours the label's model. The headline p is a
one-sided paired t-test of the `d_i` against 0, the test Figure Five uses for
neural-label versus comparator NLL: `t = mean(d) / (sd(d) / √n)`,
`p = P(T_{n−1} ≥ t)`. In brackets is a sign-flip permutation p that assumes
no normality: each `d_i` has its sign flipped at random, 10,000 times, the
null distribution of the mean is built, and `p = (#{null ≥ observed} + 1) /
(10,000 + 1)`. The CI is the 2.5 to 97.5 percentile interval of 2,000
bootstrap resamples of the trials. "Favour" is the share of trials where the
label's model wins. This compares two models conditional on the label; it
does not ask whether the label adds information beyond the maze (labels are
nearly a function of maze here).

### Result (2026-10-22)

Monkey level, all counterfactual saccades; mean per-trial difference
[95% CI], one-sided paired t-test p. Positive favours the label's model. Full
tables (sessions, session × maze strata, first-saccade variant) in
`out/<Monkey>/compare_s_summary.txt`.

| group | Faure | Nielsen |
| --- | --- | --- |
| H-labelled (H − S) | +1.03 [+0.69, +1.36], p < 1e-4 (n 89) | +0.57 [+0.38, +0.77], p < 1e-4 (n 322) |
| S-labelled (S − H) | +0.31 [+0.11, +0.49], p 0.002 (n 135) | −0.25 [−0.44, −0.06], p 0.99 (n 286) |
| all (matched − mismatched) | +0.59 [+0.41, +0.77], p < 1e-4 (n 224) | +0.19 [+0.05, +0.33], p 0.005 (n 608) |

Faure: the label's model wins in both groups, in both sessions. Nielsen:
H-labelled trials are better explained by H, but S-labelled trials are also
better explained by H (−0.25, significant in the wrong direction); the pooled
result is carried by the H group alone. The first-saccade-only variant gives
the same signs and significance throughout. Because labels track maze, the
Faure result is also consistent with "trials in H mazes look H-like and
trials in S mazes look S-like"; the comparison does not separate the two.

## Absolute fit (`logp.py`)

The paired test says which model is less wrong, not whether either explains
the looks. `counterfactual.logp` scores each model on its own: mean
Plackett–Luce log P of the counterfactual saccades per trial under H, S and
a uniform reference (1/3 on each unchosen exit, scored the same way, so it is
each unit's chance level), per label group, for the monkey and each session,
with a 95% bootstrap CI.

Monkey level, all counterfactual saccades (2026-10-23):

| group | H model | S model | uniform |
| --- | --- | --- | --- |
| Faure H-labelled (n 89) | −1.25 | −2.28 | −1.12 |
| Faure S-labelled (n 135) | −1.81 | −1.51 | −1.13 |
| Nielsen H-labelled (n 322) | −2.33 | −2.90 | −1.28 |
| Nielsen S-labelled (n 286) | −2.21 | −2.45 | −1.25 |

Neither model reaches chance in any group: a minority of looks land on exits
the model gives under 1%, and they dominate the mean.

## Outputs

- `counterfactual/out/<Monkey>/exit_sequences.csv` (one row per trial: the
  exit list as codes and names) and `summary.txt` (extraction tallies, the
  landing sanity check, look states) from `counterfactual.build`.
- `counterfactual/out/<Monkey>/compare_s_results.csv`, `compare_s_summary.txt`
  and `compare_s_{sessions,strata}_{all,first}.png` from
  `counterfactual.compare`.
- `counterfactual/out/<Monkey>/logp_results.csv` and
  `logp_sessions_{all,first}.png` from `counterfactual.logp`.
- `counterfactual/out/movies/<Monkey>/<session>/`: `counterfactual.movie`
  renders one `.mp4` per trial or a `--batch` video, reading looks and
  saccades from the exit cache (clock from flash three, the analysed window's
  start marked, the strategy label in the title).
