# Post-feedback counterfactual looks -- Nielsen

align = response, window = 1500 ms, radius = 3.0 deg, trial_fade == 0
trials analysed: 22014 (16369 correct, 5645 error); skipped: {'no_response_fixation': 2, 'no_events': 0, 'no_feedback': 0}

### By maze regime -- correct trials (primary)

| regime | trials | P(look ML alt) | 95% CI | P(look per other alt) | first-look = ML (chance 1/3) |
|---|---|---|---|---|---|
| hierarchical | 11001 | 3360/11001 = 0.305 | [0.297, 0.314] | 0.286 | 2604/7787 = 0.334 (p=0.85) |
| sequential | 5368 | 1375/5368 = 0.256 | [0.245, 0.268] | 0.260 | 1255/3566 = 0.352 (p=0.019) |

hierarchical vs sequential P(look ML alt), Fisher exact: p = 5.47e-11

### By maze regime -- error trials

| regime | trials | P(look ML alt) | 95% CI | P(look per other alt) | first-look = ML (chance 1/3) |
|---|---|---|---|---|---|
| hierarchical | 2651 | 637/2651 = 0.240 | [0.224, 0.257] | 0.197 | 574/1450 = 0.396 (p=6.9e-07) |
| sequential | 2994 | 464/2994 = 0.155 | [0.142, 0.168] | 0.221 | 406/1560 = 0.260 (p=4.3e-10) |

hierarchical vs sequential P(look ML alt), Fisher exact: p = 7.29e-16

### Per session (correct trials)

| session | regime(s) | trials | P(look ML alt) |
|---|---|---|---|
| Dec_01_g0 | hierarchical/sequential | 571 | 188/571 = 0.329 |
| Dec_02_g0 | hierarchical/sequential | 490 | 129/490 = 0.263 |
| Dec_03_g0 | hierarchical/sequential | 511 | 157/511 = 0.307 |
| Dec_05_g0 | hierarchical/sequential | 468 | 151/468 = 0.323 |
| Dec_06_g0 | hierarchical/sequential | 738 | 184/738 = 0.249 |
| Nov_10_g0 | hierarchical/sequential | 517 | 140/517 = 0.271 |
| Nov_11_g0 | hierarchical/sequential | 525 | 148/525 = 0.282 |
| Nov_17_g0 | hierarchical/sequential | 433 | 107/433 = 0.247 |
| Nov_18_g0 | hierarchical/sequential | 467 | 135/467 = 0.289 |
| Nov_19_g0 | hierarchical/sequential | 448 | 164/448 = 0.366 |
| Nov_1_g0 | hierarchical/sequential | 428 | 95/428 = 0.222 |
| Nov_22_g0 | hierarchical/sequential | 524 | 193/524 = 0.368 |
| Nov_23_g0 | hierarchical/sequential | 473 | 178/473 = 0.376 |
| Nov_24_g0 | hierarchical/sequential | 467 | 144/467 = 0.308 |
| Nov_25_g0 | hierarchical/sequential | 471 | 174/471 = 0.369 |
| Nov_27_g0 | hierarchical/sequential | 544 | 158/544 = 0.290 |
| Nov_28_g0 | hierarchical/sequential | 429 | 116/429 = 0.270 |
| Nov_29_g0 | hierarchical/sequential | 475 | 155/475 = 0.326 |
| Nov_2_g0 | hierarchical/sequential | 408 | 117/408 = 0.287 |
| Nov_30_g0 | hierarchical/sequential | 500 | 142/500 = 0.284 |
| Nov_3_g0 | hierarchical/sequential | 475 | 116/475 = 0.244 |
| Nov_4_g0 | hierarchical/sequential | 473 | 132/473 = 0.279 |
| Nov_6_g0 | hierarchical/sequential | 438 | 121/438 = 0.276 |
| Nov_7_g0 | hierarchical/sequential | 476 | 130/476 = 0.273 |
| Nov_9_g0 | hierarchical/sequential | 458 | 151/458 = 0.330 |
| Oct_21_g0 | hierarchical/sequential | 460 | 134/460 = 0.291 |
| Oct_22_g0 | hierarchical/sequential | 501 | 146/501 = 0.291 |
| Oct_23_g0 | hierarchical/sequential | 508 | 116/508 = 0.228 |
| Oct_24_g0 | hierarchical/sequential | 518 | 145/518 = 0.280 |
| Oct_25_g0 | hierarchical/sequential | 489 | 127/489 = 0.260 |
| Oct_26_g0 | hierarchical/sequential | 444 | 116/444 = 0.261 |
| Oct_27_g0 | hierarchical/sequential | 443 | 114/443 = 0.257 |
| Oct_28_g0 | hierarchical/sequential | 331 | 83/331 = 0.251 |
| Oct_31_g0 | hierarchical/sequential | 468 | 129/468 = 0.276 |

### By per-trial neural strategy label -- correct trials (2022 labelled)

| regime | trials | P(look ML alt) | 95% CI | P(look per other alt) | first-look = ML (chance 1/3) |
|---|---|---|---|---|---|
| hierarchical | 1062 | 286/1062 = 0.269 | [0.243, 0.297] | 0.265 | 236/706 = 0.334 (p=0.97) |
| sequential | 960 | 254/960 = 0.265 | [0.238, 0.293] | 0.239 | 229/614 = 0.373 (p=0.04) |

hierarchical vs sequential P(look ML alt), Fisher exact: p = 0.84
