# Faure: best behavioral choice model per neural strategy label

Dendrogram labels, top-two sessions. The chosen exit is scored under each model: 10,000 noise draws per trial (seed 45), published Weber fraction; lapse and revision parameters fit per group. Ranked by 5-fold CV x 5 repeats held-out NLL per trial (lower is better). Δ vs best with a bootstrap 95% CI over trials. Chance = ln 4 = 1.386.

## H-labelled trials (n = 291)

| Model | CV NLL / trial | Δ vs best [95% CI] | AIC | BIC | Zero-count trials | Fitted parameters |
|---|---|---|---|---|---|---|
| H + Revision X_LR_Max | 0.5960 | best | 346.0 | 357.0 | – | slope=1 off=0.2s θ=0.816 |
| H + Revision X_LR_UD | 0.5962 | +0.0002 [-0.0021, +0.0027] | 346.7 | 357.8 | – | slope=1 off=0.2s θ=0.883 |
| H + Revision X_U_D | 0.6039 | +0.0080 [-0.0038, +0.0209] | 349.9 | 360.9 | – | slope=1 off=0.2s θ=0.581 |
| Optimal + Lapse | 0.6055 | +0.0095 [-0.0219, +0.0410] | 351.5 | 355.2 | – | lapse=0.053 |
| H + Revision X_max | 0.6072 | +0.0113 [-0.0003, +0.0238] | 349.6 | 360.7 | – | slope=1 off=0.2s θ=0.56 |
| Hierarchical | 0.6100 | +0.0141 [-0.0062, +0.0369] | 355.0 | 355.0 | 0 | – |
| Optimal | 0.6236 | +0.0276 [-0.0076, +0.0679] | 362.9 | 362.9 | 0 | – |
| Postdictive | 0.6303 | +0.0343 [-0.0038, +0.0779] | 366.8 | 366.8 | 0 | – |
| Second-only (Sequential) | 0.7422 | +0.1463 [+0.0782, +0.2225] | 432.0 | 432.0 | 1 | – |
| Total Time (convolution) | 0.9272 | +0.3313 [+0.2343, +0.4273] | 539.6 | 539.6 | 0 | – |
| Total Time | 1.0026 | +0.4066 [+0.3074, +0.5046] | 583.5 | 583.5 | 0 | – |
| First-only | 1.0512 | +0.4552 [+0.3944, +0.5096] | 611.8 | 611.8 | – | – |

Top four by session (CV NLL / trial):

- june_24_g0 (n=167): H + Revision X_LR_Max 0.651, H + Revision X_LR_UD 0.652, Optimal + Lapse 0.662, H + Revision X_U_D 0.670
- june_8_g0 (n=124): Hierarchical 0.513, H + Revision X_U_D 0.516, H + Revision X_max 0.520, H + Revision X_LR_UD 0.520

## S-labelled trials (n = 369)

| Model | CV NLL / trial | Δ vs best [95% CI] | AIC | BIC | Zero-count trials | Fitted parameters |
|---|---|---|---|---|---|---|
| Optimal + Lapse | 0.7113 | best | 523.9 | 527.8 | – | lapse=0.015 |
| Optimal | 0.7147 | +0.0034 [-0.0054, +0.0174] | 527.4 | 527.4 | 0 | – |
| Second-only (Sequential) | 0.7149 | +0.0036 [-0.0257, +0.0379] | 527.6 | 527.6 | 0 | – |
| Postdictive | 0.7207 | +0.0094 [-0.0041, +0.0262] | 531.9 | 531.9 | 0 | – |
| H + Revision X_U_D | 0.7393 | +0.0280 [-0.0084, +0.0616] | 549.2 | 560.9 | – | slope=1 off=0s θ=2.14 |
| H + Revision X_max | 0.7417 | +0.0304 [+0.0044, +0.0527] | 549.5 | 561.3 | – | slope=1 off=0s θ=1.51 |
| H + Revision X_LR_Max | 0.7653 | +0.0540 [+0.0243, +0.0798] | 565.0 | 576.7 | – | slope=1.25 off=0s θ=2.43 |
| H + Revision X_LR_UD | 0.7664 | +0.0551 [+0.0255, +0.0812] | 565.3 | 577.1 | – | slope=1.25 off=0s θ=2.74 |
| Total Time (convolution) | 0.8764 | +0.1651 [+0.0984, +0.2262] | 646.8 | 646.8 | 0 | – |
| Hierarchical | 0.9042 | +0.1929 [+0.1476, +0.2356] | 667.3 | 667.3 | 0 | – |
| Total Time | 1.0093 | +0.2980 [+0.2254, +0.3655] | 744.8 | 744.8 | 0 | – |
| First-only | 1.2918 | +0.5805 [+0.5233, +0.6335] | 953.3 | 953.3 | – | – |

Top four by session (CV NLL / trial):

- june_24_g0 (n=200): Optimal + Lapse 0.802, Optimal 0.813, H + Revision X_U_D 0.817, Second-only (Sequential) 0.821
- june_8_g0 (n=169): Second-only (Sequential) 0.589, Optimal 0.598, Postdictive 0.600, Optimal + Lapse 0.604

## Seed check

Winner with a second Monte Carlo seed (7): H-labelled H + Revision X_LR_Max; S-labelled Optimal + Lapse.
