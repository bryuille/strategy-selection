# Which model explains the counterfactual saccade? -- Nielsen

align = response, window = 1500 ms, radius = 3.0 deg, quadrature nodes = 21
**The question.** After feedback, the animal sometimes looks at an exit it did not choose. There are three such exits. Each model says which one it should be. How often is it right?
11353 trials, grouped into 24 cells (one per path type -- the maze geometry is identical within a path type, so all its trials make the same prediction).

## How to read the numbers

| column | plain meaning |
| ------ | ------------- |
| **correct** | Share of saccades that went to the exit the model named. This is the headline: bigger is better. |
| **vs chance** | Same number minus 33.3%, the rate you get by guessing among three exits. Zero means the model knows nothing. |
| **of achievable** | Where the model sits between guessing (33.3% = 0%) and the best any model could do (66.5% = 100%). The ceiling is set by always naming each cell's most common target. |
| **whole distribution** | The same 0-100% scale, but scoring the model on the *proportions* it predicts across all three exits, not just on whether its single best guess is right. This is the stricter number: naming the most common target is easy when a cell is lopsided, so a model can score well on `of achievable` while getting the split badly wrong. Where the two columns disagree, believe this one. |
| **params** | Free parameters the model was allowed to fit. Two models with equal `correct` are not equal if one used five parameters and the other one. |
| **flags** | A fitted parameter that hit its allowed limit, or a timing-noise value too large to be real. Either means the number was not estimated -- the fit was pushed into a degenerate shape to buy accuracy, so its rank is not evidence for that model. |

## Ranking

| model | correct | vs chance | of achievable | whole distribution | params | flags |
|---|---|---|---|---|---|---|
| saturated *(ceiling)* | 66.5% | +33.2% | 100% | 100% | 48 |  |
| only_second | 62.4% | +29.1% | 88% | 40% | 2 | wm=1.94 implausible |
| revision[XUD] | 57.4% | +24.1% | 73% | 48% | 5 | theta=0.991 at limit, alpha=0.988 at limit |
| optimal | 53.4% | +20.1% | 61% | 29% | 2 | wm=1.96 at limit |
| optimal_lapse | 53.4% | +20.1% | 61% | 29% | 3 | wm=2 at limit, lam=0.985 at limit |
| postdictive | 53.4% | +20.1% | 61% | 29% | 2 | wm=1.96 at limit |
| total_time | 48.7% | +15.3% | 46% | 12% | 2 | wm=1.89 implausible |
| hierarchical | 42.9% | +9.6% | 29% | 14% | 2 | wm=2 at limit |
| only_first | 41.7% | +8.3% | 25% | 4% | 2 |  |
| sibling | 38.9% | +5.6% | 17% | 2% | 1 |  |
| proximity | 35.9% | +2.6% | 8% | 6% | 1 |  |
| uniform | 33.3% | +0.0% | 0% | 0% | 0 |  |

`uniform` is the guess-at-random floor and `saturated` the per-cell ceiling; every other model sits between them. Log-likelihood, AIC and BIC are in `fits.csv` -- `whole distribution` is the log-likelihood on a 0-100% scale, and AIC's penalty for extra parameters is what the `params` column stands in for.

**The two columns disagree here.** `only_second` names the right exit most often (62.4%), but `revision[XUD]` predicts the three-way split best (48% vs 40% on the whole-distribution scale). When a cell is lopsided, naming its most common target is easy and says little; treat the accuracy ranking as the readable summary and the whole-distribution one as the verdict.

**Best model: only_second** -- names the right exit on 62.4% of trials (+29.1% over chance, 88% of what is achievable), using 2 parameter(s). Flagged: wm=1.94 implausible -- read the caveat in the table above.

## Per-cell detail

One row per path type. `looked at most` is where the saccades actually went most often; `model says` is the exit the best model named.

| path type | chosen exit | trials | looked at most (share) | model says | match |
|---|---|---|---|---|---|
| 1 | LU | 611 | RD (77%) | RD | yes |
| 2 | LD | 533 | RD (71%) | RD | yes |
| 3 | RU | 608 | RD (85%) | RD | yes |
| 4 | RD | 482 | LU (48%) | LU | yes |
| 5 | LU | 412 | RD (44%) | RD | yes |
| 6 | LD | 388 | RD (75%) | RD | yes |
| 7 | RU | 577 | RD (69%) | RD | yes |
| 8 | RD | 360 | LU (48%) | RU | NO |
| 9 | LU | 602 | RD (61%) | RD | yes |
| 10 | LD | 518 | RD (58%) | RD | yes |
| 11 | RU | 520 | RD (83%) | RD | yes |
| 12 | RD | 446 | LU (80%) | LU | yes |
| 13 | LU | 372 | RU (39%) | RD | NO |
| 14 | LD | 264 | RD (51%) | RD | yes |
| 15 | RU | 599 | RD (85%) | RD | yes |
| 16 | RD | 495 | LU (63%) | RU | NO |
| 17 | LU | 340 | LD (47%) | RD | NO |
| 18 | LD | 328 | RD (66%) | RD | yes |
| 19 | RU | 381 | RD (81%) | RD | yes |
| 20 | RD | 879 | LU (89%) | LU | yes |
| 21 | LU | 402 | RU (42%) | RD | NO |
| 22 | LD | 410 | RD (63%) | RD | yes |
| 23 | RU | 216 | RD (94%) | RD | yes |
| 24 | RD | 610 | LU (43%) | RU | NO |
