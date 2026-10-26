# Nielsen: first counterfactual look and the maze geometry

Dendrogram labels, top-two sessions; looks from `counterfactual.extract`, repeats of the chosen exit removed. Measured 2026-10-23.

*Same side* = other vertical arm on the chosen side; *mirror* = other side, same up/down; *diagonal* = other side, other up/down.

## Where the first look goes

Uniform would put about a third in each column.

| Label | n | Same side | Mirror | Diagonal | χ² vs uniform p |
|---|---|---|---|---|---|
| H | 322 | 41% | 29% | 30% | 0.01 |
| S | 286 | 38% | 21% | 41% | 5e-5 |

## How well geometry predicts it

Conditional logit over the three unchosen exits; 5-fold CV mean log P per trial (higher is better; uniform −1.099). Features: same side, mirror, distance from the chosen exit, vertical and horizontal arm-length mismatch to the chosen exit (arm in degrees = 10°/s × seconds).

| Label | Relation only | Arm-length mismatch only | All features |
|---|---|---|---|
| H | -1.10 | -1.09 | -1.07 |
| S | -1.07 | -1.07 | -1.00 |

## Why H, S and Optimal score below uniform

Each model's unchosen-exit distribution q (`counterfactual.models` cache, `MODEL_EPS` smoothing), all labelled trials. Actual looks same side / mirror / diagonal: 40% / 25% / 35%.

| Model | Top guess = look | Mean log q | Looks with q < 1% | Their share of the loss | Mean log q without them | Mean q same / mirror / diagonal |
|---|---|---|---|---|---|---|
| H | 44% | -1.90 | 17% | 49% | -1.17 | 0.41 / 0.20 / 0.39 |
| S | 40% | -2.26 | 26% | 58% | -1.30 | 0.35 / 0.25 / 0.40 |
| Optimal | 42% | -2.01 | 18% | 46% | -1.31 | 0.54 / 0.18 / 0.29 |

q is each model's choice-error distribution (arm-length similarity), while the looks are mostly spatial. Confident misses (q < 1%) put H, S and Optimal below uniform, so flatter alternatives beat H in `counterfactual.compare` whatever the label. Read the `compare_s` p values alongside a uniform baseline.
