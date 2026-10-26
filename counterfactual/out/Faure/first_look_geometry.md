# Faure: first counterfactual look and the maze geometry

Dendrogram labels, top-two sessions; looks from `counterfactual.extract`, repeats of the chosen exit removed. Measured 2026-10-23.

*Same side* = other vertical arm on the chosen side; *mirror* = other side, same up/down; *diagonal* = other side, other up/down.

## Where the first look goes

Uniform would put about a third in each column.

| Label | n | Same side | Mirror | Diagonal | χ² vs uniform p |
|---|---|---|---|---|---|
| H | 89 | 80% | 11% | 9% | 2e-19 |
| S | 135 | 70% | 12% | 18% | 6e-19 |

## How well geometry predicts it

Conditional logit over the three unchosen exits; 5-fold CV mean log P per trial (higher is better; uniform −1.099). Features: same side, mirror, distance from the chosen exit, vertical and horizontal arm-length mismatch to the chosen exit (arm in degrees = 10°/s × seconds).

| Label | Relation only | Arm-length mismatch only | All features |
|---|---|---|---|
| H | -0.67 | -0.74 | -0.62 |
| S | -0.82 | -0.95 | -0.72 |

## Why H, S and Optimal score below uniform

Each model's unchosen-exit distribution q (`counterfactual.models` cache, `MODEL_EPS` smoothing), all labelled trials. Actual looks same side / mirror / diagonal: 74% / 12% / 14%.

| Model | Top guess = look | Mean log q | Looks with q < 1% | Their share of the loss | Mean log q without them | Mean q same / mirror / diagonal |
|---|---|---|---|---|---|---|
| H | 39% | -1.59 | 7% | 26% | -1.26 | 0.39 / 0.13 / 0.48 |
| S | 43% | -1.80 | 18% | 51% | -1.09 | 0.42 / 0.13 / 0.45 |
| Optimal | 59% | -1.33 | 11% | 42% | -0.86 | 0.57 / 0.11 / 0.32 |

q is each model's choice-error distribution (arm-length similarity), while the looks are mostly spatial. Confident misses (q < 1%) put H, S and Optimal below uniform, so flatter alternatives beat H in `counterfactual.compare` whatever the label. Read the `compare_s` p values alongside a uniform baseline.
