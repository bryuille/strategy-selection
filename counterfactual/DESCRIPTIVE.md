# What explains the counterfactual looks? (2026-10-26)

Companion to [README.md](README.md). The strategy models (H, S and the other behavioral models in `best_model.py`) score near or below chance on the counterfactual looks. This note records:
- where they diverge from the data;
- which simple descriptive model does explain the looks, maze by maze.

## Where the strategy models diverge from the looks

Numbers are for the matched model (H on H-labelled trials, S on S-labelled), first counterfactual look, labelled trials (Faure n 224, Nielsen n 608). They come from one-off probes on the counterfactual caches, not a saved script.

**1. Overconfidence.** Each unchosen exit on each trial counts once:

| Model probability for an exit | Faure: how often it is looked at | Nielsen: how often it is looked at |
| --- | --- | --- |
| < 0.05 (mean 0.013) | 14% | 34% |
| 0.05–0.2 | 32% | 27% |
| 0.2–0.4 | 34% | 38% |
| > 0.6 (mean 0.83 / 0.79) | 60% | 34% |

- **Faure:** the ranking carries signal, but both ends are too extreme. The 15% of looks at exits given under 0.05 cost −3.0 nats each and cancel all the gains: the mean gain over chance is −0.15, and +0.36 without those looks.
- **Nielsen:** calibration is flat. Looks do not depend on the model's belief.

**2. The horizontal neighbour is nearly ruled out.** Gain per first look by target:

| Target | Faure: model q, gain | Nielsen: model q, gain |
| --- | --- | --- |
| sibling | 0.59, +0.08 | 0.48, −0.36 |
| diagonal | 0.41, −0.39 | 0.34, −0.93 |
| horizontal | 0.21, −1.33 | 0.12, −2.29 |

**3. Error trials.**
- The models put 0.59 (Faure) and 0.53 (Nielsen) on the exit that was actually correct, because errors go to the exit most confusable with it.
- The monkey's first look goes there only at chance: 36% and 34%, against 33%.
- Error trials therefore score worse: −0.57 against −0.05 for Faure, −1.47 against −0.92 for Nielsen.
- There's no sign that the monkey re-checks the most plausible alternative, or that feedback steers it to the correct exit.

**4. The second look is anti-predictive.** The model's ranking of the remaining two exits scores −0.79 nats against a coin on 146 Nielsen trials, and −0.63 on Faure's 9.

**5. Not screen distance.** The first look goes to the unchosen exit nearest the chosen one on screen only 17% of the time for Faure (below chance) and 30% for Nielsen (chance).

**6. The sibling.** Both models under-predict the sibling for reasons tied to the maze:
- **H's sibling probability follows how well tm1 identifies the side.** It is near 0 in mazes 5–6, where h1 = h4.
- **S's sibling probability is a coincidence of arm lengths.** It is high in mazes 2, 4 and 6, and ≤ 0.12 in mazes 1, 3 and 5.

See `README.md` and `out/<Monkey>/belief_tables.md`.

## Descriptive models (`descriptive.py`)

Each model is a multinomial logit over the three unchosen exits:
- fitted to the first counterfactual looks of every trial that has one (Faure n 305, Nielsen n 825);
- scored by **10-fold cross-validated gain over chance per trial**, log P(look) − log ⅓, so a model is always scored on trials it was not fitted on.

The **saturated reference** is the add-one-smoothed frequency of each target within its (maze, chosen exit) cell, scored the same way. It is close to the most any model of (maze, chosen exit) can reach.

Per unchosen exit, the features are:
- **relation** to the chosen exit (sibling, horizontal, diagonal);
- **screen identity** (LU, LD, RU, RD);
- **vertical-arm similarity** to the chosen exit (S-like);
- **total-path similarity** (Total Time-like);
- **log q_H, log q_S**, the strategy beliefs with free weights.

| Model (CV gain over chance per trial) | Faure | Nielsen |
| --- | --- | --- |
| Vertical-arm similarity | +0.07 | 0.00 |
| Total-path similarity | 0.00 | 0.00 |
| Strategy beliefs, free weights | +0.09 | +0.03 |
| Relation | +0.30 | +0.02 |
| Screen exit | +0.16 | +0.08 |
| **Relation + screen exit (5 parameters)** | **+0.43** | +0.10 |
| Relation + screen exit + strategy beliefs | +0.42 | +0.10 |
| Relation + screen exit, separate per maze | +0.43 | **+0.16** |
| Saturated (maze, chosen exit) table | +0.43 | +0.20 |

Per maze (mazes 1–6), gain of the best compact model against the saturated table:

| | Model | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Faure | relation + screen exit | +0.56 | +0.40 | +0.40 | +0.38 | +0.30 | +0.55 |
| Faure | saturated | +0.50 | +0.34 | +0.41 | +0.46 | +0.32 | +0.57 |
| Nielsen | relation + screen exit, per maze | +0.22 | −0.01 | +0.27 | +0.13 | +0.29 | +0.01 |
| Nielsen | saturated | +0.21 | +0.02 | +0.27 | +0.16 | +0.43 | +0.08 |

## Findings

**Faure: "sibling first, avoid the lower-right exit."**
- Five parameters match the saturated table (+0.43 each), so nothing systematic is left.
- Fitted log-odds: horizontal −1.8 and diagonal −1.8 relative to the sibling; LD −0.2, RU −0.6 and RD −2.2 relative to LU.
- It holds in every maze (+0.30 to +0.56; maze 5 weakest).
- RD avoidance explains why Faure, after choosing RU, goes to LD rather than its sibling RD (100%, 64% and 54% in mazes 1, 5 and 6).
- **It does not depend on the label.** The H-labelled vs S-labelled difference (+0.59 vs +0.42) follows the maze mix.
- **The strategy beliefs add nothing on top of it** (+0.42 vs +0.43).

Faure's counterfactual looks are a fixed spatial routine, not a readout of either strategy. In the relative metric of `compare.py`, the label's model wins because the other model ranks the exits worse against this routine, not because the looks change with the label.

**Nielsen: weak, maze-specific structure.**
- **The ceiling is low.** The saturated table reaches +0.20 overall, and the structure is uneven across mazes:
  - **real in mazes 1, 3 and 5** (+0.21 to +0.43). In maze 1, LU, LD and RU send 59–83% of first looks to RD; in maze 5, RD → LU is 89% and LU → LD 78%;
  - **little in mazes 2, 4 and 6** (+0.02 to +0.16).
- **The favoured target changes with the maze.** RD dominates in mazes 1, 3 and 4. RU is almost never looked at (5–10%) except in maze 6 (33%).
- **Global models miss it** (+0.10). Separate per-maze relation and screen preferences recover most of what exists (+0.16 of +0.20), but that is 30 parameters for 825 trials: a description, not an explanation.
- **Timing-based features explain nothing** (arm and path similarity ≈ 0).

## Caveats

- **These models are fitted to the looks.** Cross-validation protects against overfitting, not against explaining by description.
- **Folds mix sessions.** Whether Nielsen's per-maze preferences are stable across its two sessions has not been checked.
- **First looks only.**
- **The divergence numbers above (sections 1–5) are from one-off probes.** The model comparison is reproducible with `uv run python -m counterfactual.descriptive` (output `out/<Monkey>/descriptive_results.csv`).
