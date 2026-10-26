# Nielsen: best behavioral choice model per neural strategy label

Dendrogram labels, top-two sessions. The chosen exit is scored under each model: 10,000 noise draws per trial (seed 45), published Weber fraction; lapse and revision parameters fit per group. Ranked by 5-fold CV x 5 repeats held-out NLL per trial (lower is better). Δ vs best with a bootstrap 95% CI over trials. Chance = ln 4 = 1.386.

## H-labelled trials (n = 453)

| Model | CV NLL / trial | Δ vs best [95% CI] | AIC | BIC | Zero-count trials | Fitted parameters |
|---|---|---|---|---|---|---|
| H + Revision X_max | 0.5364 | best | 487.5 | 499.8 | – | slope=3 off=0s θ=0.825 |
| H + Revision X_U_D | 0.5405 | +0.0041 [+0.0000, +0.0084] | 490.4 | 502.8 | – | slope=5 off=0s θ=0.664 |
| H + Revision X_LR_Max | 0.5447 | +0.0083 [-0.0009, +0.0185] | 494.1 | 506.4 | – | slope=5 off=0s θ=0.974 |
| H + Revision X_LR_UD | 0.5487 | +0.0123 [+0.0004, +0.0257] | 497.1 | 509.4 | – | slope=5 off=0s θ=0.837 |
| Optimal + Lapse | 0.5523 | +0.0159 [+0.0036, +0.0296] | 500.3 | 504.4 | – | lapse=0.066 |
| Hierarchical | 0.5819 | +0.0455 [-0.0037, +0.1037] | 527.2 | 527.2 | 2 | – |
| Optimal | 0.5922 | +0.0558 [+0.0142, +0.1018] | 536.5 | 536.5 | 0 | – |
| Postdictive | 0.5943 | +0.0579 [+0.0160, +0.1046] | 538.5 | 538.5 | 0 | – |
| Second-only (Sequential) | 0.6582 | +0.1218 [+0.0844, +0.1607] | 596.4 | 596.4 | 0 | – |
| Total Time (convolution) | 0.9261 | +0.3897 [+0.3180, +0.4588] | 839.1 | 839.1 | 0 | – |
| First-only | 0.9993 | +0.4629 [+0.4183, +0.5064] | 905.3 | 905.3 | – | – |
| Total Time | 1.0364 | +0.5000 [+0.4230, +0.5741] | 939.0 | 939.0 | 0 | – |

Top four by session (CV NLL / trial):

- Nov_3_g0 (n=277): H + Revision X_max 0.536, H + Revision X_U_D 0.542, H + Revision X_LR_Max 0.551, H + Revision X_LR_UD 0.556
- Nov_6_g0 (n=176): H + Revision X_LR_Max 0.534, H + Revision X_max 0.537, H + Revision X_LR_UD 0.537, H + Revision X_U_D 0.539

## S-labelled trials (n = 417)

| Model | CV NLL / trial | Δ vs best [95% CI] | AIC | BIC | Zero-count trials | Fitted parameters |
|---|---|---|---|---|---|---|
| Second-only (Sequential) | 0.8252 | best | 688.2 | 688.2 | 0 | – |
| H + Revision X_max | 0.8341 | +0.0089 [-0.0192, +0.0341] | 691.4 | 703.5 | – | slope=1.5 off=0s θ=1.3 |
| Optimal + Lapse | 0.8346 | +0.0095 [-0.0298, +0.0451] | 696.7 | 700.8 | – | lapse=0.072 |
| H + Revision X_U_D | 0.8364 | +0.0113 [-0.0198, +0.0411] | 692.4 | 704.5 | – | slope=1.5 off=0s θ=1.15 |
| H + Revision X_LR_Max | 0.8413 | +0.0161 [-0.0185, +0.0483] | 698.7 | 710.8 | – | slope=1.5 off=0.02s θ=2.81 |
| H + Revision X_LR_UD | 0.8414 | +0.0162 [-0.0193, +0.0492] | 698.7 | 710.8 | – | slope=1.5 off=0.02s θ=2.91 |
| Optimal | 0.8765 | +0.0513 [+0.0141, +0.0955] | 731.0 | 731.0 | 0 | – |
| Postdictive | 0.8778 | +0.0526 [+0.0141, +0.0982] | 732.1 | 732.1 | 0 | – |
| Total Time (convolution) | 0.8888 | +0.0636 [+0.0068, +0.1149] | 741.2 | 741.2 | 0 | – |
| Total Time | 0.9591 | +0.1339 [+0.0649, +0.1969] | 799.9 | 799.9 | 0 | – |
| Hierarchical | 1.0224 | +0.1972 [+0.1396, +0.2533] | 852.7 | 852.7 | 0 | – |
| First-only | 1.3630 | +0.5378 [+0.4508, +0.6212] | 1136.7 | 1136.7 | – | – |

Top four by session (CV NLL / trial):

- Nov_3_g0 (n=221): Second-only (Sequential) 0.733, H + Revision X_max 0.766, Optimal 0.768, H + Revision X_U_D 0.777
- Nov_6_g0 (n=196): Optimal + Lapse 0.891, H + Revision X_U_D 0.904, H + Revision X_max 0.911, H + Revision X_LR_UD 0.912

## Seed check

Winner with a second Monte Carlo seed (7): H-labelled H + Revision X_max; S-labelled Second-only (Sequential).
