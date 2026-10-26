# `counterfactual/`: counterfactual looks vs choice-conditioned strategy beliefs

## Question

After the monkey commits to an exit, it often looks at one or more of the other three exits. Trials are neurally labelled hierarchical (H) or sequential (S). The question is whether, **on H-labelled trials, the hierarchical strategy explains those counterfactual looks better than the second-only strategy, and on S-labelled trials the reverse.**

This package replaces v1 (the earlier `counterfactual/` package, last at commit `09e6fea`). It asks the same question and gives the same two outputs, a relative and an absolute metric, but builds each strategy's prediction differently (see [How this differs from v1](#how-this-differs-from-v1)).

Each strategy has a **lookup table per maze: chosen exit × exit looked at**. If the monkey chose RU, read the RU row: it says how much probability the strategy puts on each of LU, LD and RD.

The RU row is built by simulating the path that ends at RU and keeping the noisy measurements under which the strategy would itself have chosen RU. The row is then applied to every trial that chose RU, correct or error.

Nothing is fitted to the looks; every number in the tables is fixed before the looks are scored.

Commands: [README_COMMANDS.md](../README_COMMANDS.md#counterfactual). Where the models diverge from the looks, and which simple descriptive model does explain them: [DESCRIPTIVE.md](DESCRIPTIVE.md).

## Data and looks (`extract.py`)

- **Sessions.** The top-two SNR sessions per monkey: Faure june_8_g0 and june_24_g0; Nielsen Nov_3_g0 and Nov_6_g0.
- **Trials.** A trial is kept if it:
  - passes `data.builder.trial_qc_ok` (fixed geometry, no fade, photodiode QC);
  - has exactly one `trial_answer*` flag set;
  - has a usable look record (`data.looks.usable_rows`).
- **Labels.** The dendrogram labels, `<session>_strategy_choices.npz` (0 = H, 1 = S).
- **Chosen exit.** Decoded from the active `trial_answer{1..4}` flag together with `LR` and `LR2` (DATA_DICTIONARY.md). Codes: 1 LU, 2 LD, 3 RU, 4 RD.
- **Correct exit.** `(path_type − 1) mod 4 + 1`, which always agrees with the exit given by (`LR`, `LR2`). It enters the analysis only to split correct from error trials.
- **Looks.** From `data.looks` (the `timestamp_preprocessing` package, epoch `trial`). A look is ≥ 75 ms of stable gaze with spread ≤ 2°, positioned at its raw centroid in degrees.
  - The window keeps every look overlapping `[flash_three, trial_end]`.
  - A look is assigned to the nearest exit within 3°, with the exits at (−h1, h2), (−h1, −h3), (h4, h5), (h4, −h6).
- **Counterfactual items.** The exits other than the chosen one that the monkey looked at, in time order, repeats kept. For example, after choosing LU, the window `LU LU LD origin RU LD` gives the items `LD RU LD`.
  - Starting the window at flash_three instead of saccade_init adds one first look (one Faure trial). Every other counterfactual look begins after the choice.

Labelled trials with at least one counterfactual look:

| session | H-labelled | S-labelled |
| --- | --- | --- |
| Faure june_8_g0 | 39 | 63 |
| Faure june_24_g0 | 50 | 72 |
| Nielsen Nov_3_g0 | 203 | 164 |
| Nielsen Nov_6_g0 | 119 | 122 |

Both monkeys run the same six mazes (ball speed 10°/s). In mazes 5 and 6, h1 = h4, so the first interval cannot tell the sides apart.

## Building one row of the table (`beliefs.py`, `noise.py`)

Take a maze and a chosen exit c.

**1. Draws on c's path.** The path that ends at c sets two durations:
- t1 = its horizontal arm (h1 / vel if c is on the left, h4 / vel if on the right);
- t2 = its vertical arm (h2, h3, h5 or h6, divided by vel).

100,000 noisy measurement pairs are drawn with scalar (Weber) noise:

```
tm1 = t1 · (1 + w · z1),   tm2 = t2 · (1 + w · z2),   z ~ N(0, 1)
```

so tm ~ N(t, (w t)²). w is the published Weber fraction (Faure 0.165, Nielsen 0.153), from `timing_scalars.mat` via `load_timing_scalar`, the value the Figure Five MATLAB uses. The likelihood of an arm of duration a given a measurement is `p(tm | a) = N(tm; a, (w a)²)`. All models use the same draws.

**2. Keep the draws on which the strategy chooses c.** Each strategy has a choice rule that maps a draw to an exit:
- **H (hierarchical).** Choose the side whose horizontal arm has the larger p(tm1 | arm). If h1 = h4 the two sides are tied and each wins half the time. Then, on that side, choose the vertical arm with the larger p(tm2 | arm).
- **S (second-only).** Choose the vertical arm, among all four, with the larger p(tm2 | arm). tm1 is ignored.

Exact ties (two arms of equal length) split evenly. A draw is kept with weight equal to the probability that the rule picks c on it: 1 or 0, or 1/k on a k-way tie. The kept draws are the measurements under which the strategy would have made this choice. The acceptance rate is the share of draws kept; medians over trials: Faure H 0.70, S 0.64; Nielsen H 0.74, S 0.67.

**3. Average the strategy's belief over the kept draws.** On each draw:
- **H:** belief(e) = [p(tm1 | side of e) / Σ over the two sides] × [p(tm2 | arm of e) / Σ over the two arms on that side].
- **S:** belief(e) = p(tm2 | arm of e) / Σ over all four arms.

The weighted average over the kept draws, with c set to 0, 0.001 added to each of the other three exits and renormalised, is **row c: q_M(e)**. It is what strategy M believes about the exits it did not choose, given that it chose c.

Each (maze, chosen exit) row is simulated once and shared exactly by every trial in that cell (24 rows per monkey).

**Why a belief and not a choice frequency.** On the kept draws the rule picks c by construction, so "how often would M pick LD" is 0 on every kept draw. The belief is graded: it says how plausible LD still looks to an M-monkey whose measurements led it to c.

**Worked example.** Faure, maze 5: h = (5, 7, 11, 5, 9, 5), so durations are 0.5 / 0.5 for the sides and 0.7, 1.1, 0.9, 0.5 s for the LU, LD, RU, RD arms. Row LU:

| chose LU | LD (sibling) | RU | RD |
| --- | --- | --- | --- |
| q_H | 0.08 | 0.56 | 0.36 |
| q_S | 0.12 | 0.60 | 0.28 |
| observed first looks (n 12) | 0.83 | 0.08 | 0.08 |

- **H:** h1 = h4, so tm1 cannot pick a side and H's choice of the left side is a coin flip. Its kept draws are those where, on the left, tm2 (drawn around 0.7) fit LU's arm better than LD's (1.1). On those draws the side belief stays 50/50, LD's arm is far from tm2, and RU (0.9) and RD (0.5) take the rest.
- **S:** keeps the draws where tm2 fit 0.7 best of the four arms (0.60–0.81 s). From there RU (0.9) and RD (0.5) look closer than LD (1.1).
- **The monkey looks at the sibling LD**, the exit both strategies consider least plausible.

Every row, next to the observed looks, is in `out/<Monkey>/belief_tables.md` (`counterfactual.tables`):
- **Section 1:** cross-maze aggregates by relation to the chosen exit (sibling, horizontal, diagonal), conditioned and unconditioned, per label × outcome.
- **Section 2:** one table per maze. Under each chosen exit, the H and S rows, then the observed share of first looks (all labelled, H-labelled, S-labelled).

**Why S can give the sibling more than H.** The aggregates in section 1 mix mazes. S ignores sides, so its sibling probability is a coincidence of arm lengths:
- In mazes 2, 4 and 6 (vertical arms 9 / 11 left, 7 / 5 right), the sibling's arm is often the closest to the chosen one, and S gives it 0.28–0.98.
- In mazes 1, 3 and 5 (7 / 11, 9 / 5), it never is, and S gives it ≤ 0.12.

H's sibling probability follows how well tm1 identifies the side:
- 0.64–0.93 in mazes 1–2 (h1 = 5 vs h4 = 10);
- 0.23–0.69 in mazes 3–4 (5 vs 7.5);
- ≤ 0.22 in mazes 5–6 (h1 = h4).

S-labelled trials come mostly from mazes 4–6, where H has little or no side information and S's arm-length match happens to favour the sibling in mazes 4 and 6. So over S-labelled trials S's average sibling probability (0.58) exceeds H's (0.30). Within a maze the comparison can go either way.

## Scoring (`compare.py`)

The items are scored with Plackett–Luce, i.e. sampling without replacement from q:

```
P(LD, RU) = q(LD) · q(RU) / (1 − q(LD))
```

A revisit of an exit already scored adds nothing, and the last remaining exit is forced (probability 1, adds 0). So a trial contributes one or two informative factors.

- **Relative metric.** H and S score the same items on the same trial, so the number of factors is the same under both models and the difference needs no length correction. Trials with two informative looks do carry more weight in the mean than trials with one.
- **Absolute metric.** A raw log P falls with every extra factor, and so does chance: log 1/3 for one look, log 1/6 for two or more. So the absolute metric is the **gain over chance**, log P_M − log P_uniform, where uniform puts 1/3 on each unchosen exit and is scored the same way. Chance is 0 whatever the number of looks.

## The two graphs

**`out/<Monkey>/compare_sessions.png`: the relative metric.**
- Per trial, d = log P_H − log P_S, sign-flipped on S-labelled trials, so positive always favours the label's model.
- Three markers per unit (the monkey, then each session): H-labelled (H − S), S-labelled (S − H), and all labelled trials pooled (matched − mismatched).
- Each marker shows the mean with a 95% bootstrap CI (2,000 resamples of trials). Beside it: n, the one-sided paired t-test p for d > 0, and "wins", the share of trials where the label's model scores higher.

**`out/<Monkey>/logp_sessions.png`: the absolute metric.**
- Mean gain over chance of H and of S, per label group and unit, with a 95% bootstrap CI.
- The dashed line at 0 is chance; right of it means the model predicts the looks better than chance.

## How this differs from v1

v1 built q from each model's **unconditioned choice probability on the trial's true path**: how often the model would choose each exit when the ball takes that path, renormalised over the three unchosen exits.

Here q is a row of the per-maze table: the model's **belief given that it chose c**, built on c's own path. The two differ in two ways:
- **Conditioning on the choice:** belief on the kept draws, against picks on all draws.
- **Which path:** c's path, against the true path. The two agree on correct trials and differ on error trials.

As a check, on correct trials this package's unconditioned choice frequencies match v1's cached `prob_h`, `prob_s`, `prob_optimal` and `prob_total` within Monte Carlo error (mean absolute difference ≤ 0.003, max ≤ 0.02). So the draws and the choice rules are the same as there. (The v1 caches, `<Monkey>_cf_models.npz`, are no longer built by any code.)

## Results (2026-10-26)

**Relative**, monkey level: mean d per trial [95% CI], one-sided p, wins.

| group | Faure | Nielsen |
| --- | --- | --- |
| H-labelled (H − S) | +1.10 [+0.72, +1.49], p < 0.001, 65% (n 89) | +0.96 [+0.71, +1.20], p < 0.001, 56% (n 322) |
| S-labelled (S − H) | +0.36 [+0.17, +0.53], p < 0.001, 67% (n 135) | −0.65 [−0.85, −0.46], p 1.00, 34% (n 286) |
| all (matched − mismatched) | +0.65 [+0.45, +0.85], p < 0.001, 67% (n 224) | +0.20 [+0.02, +0.37], p 0.012, 45% (n 608) |

- **Faure:** the label's model wins in both groups.
- **Nielsen:** H-labelled trials are better explained by H. S-labelled trials are also better explained by H, significantly in the wrong direction. The pooled result rests on the H group.

**Absolute**, monkey level: mean gain over chance per trial [95% CI].

| group | H model | S model |
| --- | --- | --- |
| Faure H-labelled | −0.08 [−0.36, +0.18] | −1.18 [−1.52, −0.82] |
| Faure S-labelled | −0.59 [−0.78, −0.41] | −0.23 [−0.50, +0.03] |
| Nielsen H-labelled | −0.97 [−1.15, −0.79] | −1.93 [−2.19, −1.66] |
| Nielsen S-labelled | −0.87 [−1.11, −0.63] | −1.52 [−1.80, −1.25] |

- **Faure:** the matched model is at chance (H on H-labelled, S on S-labelled); the mismatched model is below.
- **Nielsen:** both models score below chance in both groups.

**Why both models are near or below chance.** Faure looks at the sibling about 80% of the time under both labels. The strategies rank the unchosen exits by how plausible they are as the ball's path given the measurements, and in these mazes a cross-side arm is always at least as close in length to the chosen arm as the sibling's is. So both models give the sibling too little, and a few confident misses dominate the mean (see `belief_tables.md`, section 1).

## Unconditioned (Figure Five) models: the `uncond_` figures

`counterfactual.compare --uncond` and `counterfactual.logp --uncond` redraw both graphs with each row set to the model's **unconditioned choice probability** on c's path. That is the share of all draws on which the rule picks each exit, with c zeroed, 0.001 added to the other three, and renormalised: the Figure Five model's choice frequencies when c is the correct exit. Only q changes; looks, scoring and statistics are the same.

Outputs: `out/<Monkey>/uncond_{compare,logp}_{results.csv,sessions.png}`.

Monkey level, conditioned vs unconditioned (mean per trial):

| | Faure conditioned | Faure unconditioned | Nielsen conditioned | Nielsen unconditioned |
| --- | --- | --- | --- | --- |
| H-labelled, H − S | +1.10 | +1.11 | +0.96 | +0.69 |
| S-labelled, S − H | +0.36 | +0.38 | −0.65 | −0.31 |
| Matched model, gain over chance, H-labelled | −0.08 | −0.17 | −0.97 | −1.03 |
| Matched model, gain over chance, S-labelled | −0.23 | −0.42 | −1.52 | −1.25 |

Conditioning is slightly better for the matched model in Faure, and worse for S in Nielsen. Keeping only the draws that chose c sharpens S's belief against exits whose arms are far from c's, and Nielsen often looks at those.

## Best model per label (`best_model.py`)

v1's `out/<Monkey>/best_model_summary.md` (in git at commit `09e6fea`) ranks behavioral models by how well they predict the **choice** (held-out choice NLL).

This table ranks them by the **objective log P of the counterfactual looks** instead. Every model gets the same treatment as H and S above: its own per-maze table (draws on c's path, kept when its rule picks c, belief averaged), scored as the mean gain over chance per label group. Nothing is fitted, so there is no cross-validation. The revision and lapse variants are left out.

The models (`noise.py`; the rules follow the Figure One MATLAB helpers in `zarchive/MR_AG_MJ/Figure_One/H_Maze_Behavioral_Analysis/Model_Results/Models/`):

| Model | Rule (what is chosen on a draw) | Belief over the four exits |
| --- | --- | --- |
| Hierarchical | side by p(tm1 \| side), then the arm on that side by p(tm2 \| arm) | p(side \| tm1) · p(arm \| tm2, side) |
| Second-only | arm by p(tm2 \| arm) over all four | p(arm \| tm2) |
| Optimal | argmax p(tm1 \| side) p(tm2 \| arm) | that product, normalised |
| Postdictive | side by p(tm1 \| side) · Σ p(tm2 \| arm on that side), then the arm by p(tm2 \| arm) | the joint posterior (as Optimal) |
| Total Time | argmax p(total \| side arm + vertical arm), total a separate noisy measurement of t1 + t2 | that, normalised |
| Total Time (convolution) | as Total Time, on tm1 + tm2 with sd w·√(a1² + a2²) | that, normalised |
| First-only | side by p(tm1 \| side), then a fair coin between its arms | p(side \| tm1) · ½ |

The source of v1's "Total Time (convolution)" is not in the repo. The definition above is this package's: the sum of the two interval measurements, scored with the exact convolution of their two Gaussians.

### Results (2026-10-26)

Mean gain over chance per trial, the best model first. Full tables, with Δ vs best and per-session top four, are in `out/<Monkey>/best_model_summary.md`.

| | H-labelled | S-labelled |
| --- | --- | --- |
| Faure | First-only +0.13; Optimal +0.09; Postdictive +0.09; Hierarchical −0.08; … Second-only −1.18 (last) | Total Time +0.04; First-only −0.05; … Second-only −0.23 (4th); Hierarchical −0.59 (last) |
| Nielsen | Total Time −0.51; Total Time (conv.) −0.85; Hierarchical −0.97; … Second-only −1.93 (last) | First-only −0.09; Total Time −0.75; Hierarchical −0.87; … Second-only −1.52 (last) |

**The ranking rewards flat beliefs.** A mean log P is dominated by the few looks at exits a model gave a very small probability.

- **First-only** never makes the within-side call, so it never puts a near-zero probability on the sibling. Its median gain is exactly 0, and only 9% (Faure) and 21% (Nielsen) of its trials lose more than 2 nats.
- **Optimal and Postdictive** have Faure's best median gain (+0.62 and +0.71 over all labelled trials), but 16% of their trials lose more than 2 nats.

Read the table as "which model's belief is best calibrated to where the monkey looks", not "which model is sharpest where it is right".

## Outputs and commands

- `out/<Monkey>/compare_results.csv` and `compare_sessions.png` (`counterfactual.compare`), plus the `uncond_` versions.
- `out/<Monkey>/logp_results.csv` and `logp_sessions.png` (`counterfactual.logp`), plus the `uncond_` versions.
- `out/<Monkey>/best_model_results.csv` and `best_model_summary.md` (`counterfactual.best_model`).
- `out/<Monkey>/belief_tables.md` (`counterfactual.tables`).
- Caches in `data/processed/`:
  - `<Monkey>_cfv2_trials.npz`: trials and window looks (`counterfactual.extract`);
  - `<Monkey>_cfv2_q.npz`: q for every model per trial (its (maze, chosen exit) row), the acceptance rates and the unconditioned choice frequencies (`counterfactual.beliefs`).

```bash
uv run python -m counterfactual.checks     # synthetic: decoding, beliefs, acceptance vs exact, Plackett–Luce
uv run python -m counterfactual.extract    # trial / look cache
uv run python -m counterfactual.beliefs    # per-maze tables (chosen exit x exit looked at), assigned to trials
uv run python -m counterfactual.compare    # relative metric
uv run python -m counterfactual.logp       # absolute metric
uv run python -m counterfactual.compare --uncond; uv run python -m counterfactual.logp --uncond   # same with unconditioned (Figure Five) rows -> uncond_*
uv run python -m counterfactual.best_model # every behavioral model ranked by gain over chance, per label
uv run python -m counterfactual.tables     # the tables, next to the observed first looks
uv run python -m counterfactual.descriptive # descriptive models of the first look, CV gain over chance (DESCRIPTIVE.md)
```
