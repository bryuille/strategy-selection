# Which model explains the counterfactual saccade? -- Faure

align = response, window = 1500 ms, radius = 3.0 deg, quadrature nodes = 21
**The question.** After feedback, the animal sometimes looks at an exit it did not choose. There are three such exits. Each model says which one it should be. How often is it right?
3304 trials, grouped into 24 cells (one per path type -- the maze geometry is identical within a path type, so all its trials make the same prediction).

## How to read the numbers

| column | plain meaning |
| ------ | ------------- |
| **correct** | Share of saccades that went to the exit the model named. This is the headline: bigger is better. |
| **vs chance** | Same number minus 33.3%, the rate you get by guessing among three exits. Zero means the model knows nothing. |
| **of achievable** | Where the model sits between guessing (33.3% = 0%) and the best any model could do (88.9% = 100%). The ceiling is set by always naming each cell's most common target. |
| **whole distribution** | The same 0-100% scale, but scoring the model on the *proportions* it predicts across all three exits, not just on whether its single best guess is right. This is the stricter number: naming the most common target is easy when a cell is lopsided, so a model can score well on `of achievable` while getting the split badly wrong. Where the two columns disagree, believe this one. |
| **params** | Free parameters the model was allowed to fit. Two models with equal `correct` are not equal if one used five parameters and the other one. |
| **flags** | A fitted parameter that hit its allowed limit, or a timing-noise value too large to be real. Either means the number was not estimated -- the fit was pushed into a degenerate shape to buy accuracy, so its rank is not evidence for that model. |

## Ranking

| model | correct | vs chance | of achievable | whole distribution | params | flags |
|---|---|---|---|---|---|---|
| saturated *(ceiling)* | 88.9% | +55.6% | 100% | 100% | 48 |  |
| sibling | 88.7% | +55.4% | 100% | 84% | 1 |  |
| only_first | 77.1% | +43.7% | 79% | 50% | 2 | wm=0.0394 at limit |
| revision[XLR_max] | 73.5% | +40.1% | 72% | 55% | 5 |  |
| optimal | 71.0% | +37.7% | 68% | 43% | 2 |  |
| optimal_lapse | 71.0% | +37.7% | 68% | 43% | 3 |  |
| postdictive | 71.0% | +37.7% | 68% | 43% | 2 |  |
| only_second | 52.3% | +19.0% | 34% | 12% | 2 |  |
| total_time | 43.2% | +9.9% | 18% | 2% | 2 | wm=1.93 implausible |
| hierarchical | 41.8% | +8.4% | 15% | 6% | 2 | wm=1.88 implausible |
| uniform | 33.3% | +0.0% | 0% | 0% | 0 |  |
| proximity | 17.1% | -16.2% | -29% | 0% | 1 |  |

`uniform` is the guess-at-random floor and `saturated` the per-cell ceiling; every other model sits between them. Log-likelihood, AIC and BIC are in `fits.csv` -- `whole distribution` is the log-likelihood on a 0-100% scale, and AIC's penalty for extra parameters is what the `params` column stands in for.

**The favoured exit is not the nearest one.** `proximity` names whichever unchosen exit sits closest to the chosen one, and it is right only 17.1% of the time -- *below* the 33.3% you get by guessing. So these saccades are not simply going to the shortest hop, and the sibling-is-usually-nearer worry does not explain them.

**Best model: sibling** -- names the right exit on 88.7% of trials (+55.4% over chance, 100% of what is achievable), using 1 parameter(s).

## Per-cell detail

One row per path type. `looked at most` is where the saccades actually went most often; `model says` is the exit the best model named.

| path type | chosen exit | trials | looked at most (share) | model says | match |
|---|---|---|---|---|---|
| 1 | LU | 64 | LD (86%) | LD | yes |
| 2 | LD | 261 | LU (99%) | LU | yes |
| 3 | RU | 38 | RD (47%) | RD | yes |
| 4 | RD | 153 | RU (88%) | RU | yes |
| 5 | LU | 30 | LD (97%) | LD | yes |
| 6 | LD | 247 | LU (100%) | LU | yes |
| 7 | RU | 62 | LU (50%) | RD | NO |
| 8 | RD | 124 | RU (84%) | RU | yes |
| 9 | LU | 46 | LD (80%) | LD | yes |
| 10 | LD | 234 | LU (99%) | LU | yes |
| 11 | RU | 25 | RD (56%) | RD | yes |
| 12 | RD | 181 | RU (87%) | RU | yes |
| 13 | LU | 28 | LD (89%) | LD | yes |
| 14 | LD | 308 | LU (99%) | LU | yes |
| 15 | RU | 34 | RD (47%) | RD | yes |
| 16 | RD | 190 | RU (87%) | RU | yes |
| 17 | LU | 38 | LD (74%) | LD | yes |
| 18 | LD | 235 | LU (97%) | LU | yes |
| 19 | RU | 20 | RD (60%) | RD | yes |
| 20 | RD | 249 | RU (59%) | RU | yes |
| 21 | LU | 29 | LD (93%) | LD | yes |
| 22 | LD | 323 | LU (100%) | LU | yes |
| 23 | RU | 45 | RD (71%) | RD | yes |
| 24 | RD | 340 | RU (92%) | RU | yes |
