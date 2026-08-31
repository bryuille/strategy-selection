# Post-feedback counterfactual looks -- Faure

align = response, window = 1500 ms, radius = 3.0 deg, trial_fade == 0
trials analysed: 14030 (11283 correct, 2747 error); skipped: {'no_response_fixation': 1, 'no_events': 0, 'no_feedback': 0}

### By maze regime -- correct trials (primary)

| regime | trials | P(look ML alt) | 95% CI | P(look per other alt) | first-look = ML (chance 1/3) |
|---|---|---|---|---|---|
| hierarchical | 5353 | 1072/5353 = 0.200 | [0.190, 0.211] | 0.037 | 1070/1465 = 0.730 (p=2e-211) |
| sequential | 5930 | 1297/5930 = 0.219 | [0.208, 0.229] | 0.049 | 1289/1839 = 0.701 (p=7e-227) |

hierarchical vs sequential P(look ML alt), Fisher exact: p = 0.0171

### By maze regime -- error trials

| regime | trials | P(look ML alt) | 95% CI | P(look per other alt) | first-look = ML (chance 1/3) |
|---|---|---|---|---|---|
| hierarchical | 931 | 80/931 = 0.086 | [0.070, 0.106] | 0.078 | 77/222 = 0.347 (p=0.67) |
| sequential | 1816 | 192/1816 = 0.106 | [0.092, 0.121] | 0.073 | 189/448 = 0.422 (p=0.00011) |

hierarchical vs sequential P(look ML alt), Fisher exact: p = 0.106

### Per session (correct trials)

| session | regime(s) | trials | P(look ML alt) |
|---|---|---|---|
| April_11_2_g0 | hierarchical/sequential | 479 | 110/479 = 0.230 |
| April_13_g0 | hierarchical/sequential | 388 | 89/388 = 0.229 |
| April_20_g0 | hierarchical/sequential | 434 | 124/434 = 0.286 |
| April_25_g1 | hierarchical/sequential | 388 | 99/388 = 0.255 |
| April_8_g0 | hierarchical/sequential | 429 | 117/429 = 0.273 |
| june_09_09_g0 | hierarchical/sequential | 405 | 92/405 = 0.227 |
| june_10_g0 | hierarchical/sequential | 431 | 98/431 = 0.227 |
| june_11_g0 | hierarchical/sequential | 429 | 84/429 = 0.196 |
| june_12_g0 | hierarchical/sequential | 418 | 86/418 = 0.206 |
| june_13_g0 | hierarchical/sequential | 437 | 92/437 = 0.211 |
| june_14_g0 | hierarchical/sequential | 445 | 81/445 = 0.182 |
| june_15_g0 | hierarchical/sequential | 409 | 85/409 = 0.208 |
| june_16_g0 | hierarchical/sequential | 406 | 73/406 = 0.180 |
| june_17_g0 | hierarchical/sequential | 423 | 83/423 = 0.196 |
| june_18_g0 | hierarchical/sequential | 463 | 103/463 = 0.222 |
| june_19_g0 | hierarchical/sequential | 413 | 76/413 = 0.184 |
| june_20_g0 | hierarchical/sequential | 439 | 73/439 = 0.166 |
| june_21_g0 | hierarchical/sequential | 373 | 78/373 = 0.209 |
| june_22_g0 | hierarchical/sequential | 348 | 85/348 = 0.244 |
| june_23_g0 | hierarchical/sequential | 429 | 88/429 = 0.205 |
| june_24_g0 | hierarchical/sequential | 379 | 67/379 = 0.177 |
| june_25_g0 | hierarchical/sequential | 408 | 65/408 = 0.159 |
| june_27_g0 | hierarchical/sequential | 399 | 108/399 = 0.271 |
| june_28_g0 | hierarchical/sequential | 460 | 77/460 = 0.167 |
| june_29_g0 | hierarchical/sequential | 456 | 66/456 = 0.145 |
| june_30_g0 | hierarchical/sequential | 455 | 89/455 = 0.196 |
| june_8_g0 | hierarchical/sequential | 340 | 81/340 = 0.238 |

### By per-trial neural strategy label -- correct trials (2045 labelled)

| regime | trials | P(look ML alt) | 95% CI | P(look per other alt) | first-look = ML (chance 1/3) |
|---|---|---|---|---|---|
| hierarchical | 1006 | 201/1006 = 0.200 | [0.176, 0.226] | 0.038 | 200/276 = 0.725 (p=4.6e-40) |
| sequential | 1039 | 219/1039 = 0.211 | [0.187, 0.237] | 0.040 | 217/300 = 0.723 (p=5.4e-43) |

hierarchical vs sequential P(look ML alt), Fisher exact: p = 0.547
