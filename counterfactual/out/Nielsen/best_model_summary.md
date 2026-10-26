# Nielsen: best behavioral model per neural strategy label, by counterfactual looks

Dendrogram labels, top-two sessions. Each model's belief over the unchosen exits is conditioned on the chosen exit (10,000 draws from the correct path, published Weber fraction, draws kept when the model's rule picks the chosen exit; counterfactual/README.md). Score: mean gain over chance of the counterfactual looks per trial, log P(model) − log P(uniform), Plackett–Luce; 0 = chance, higher is better. Δ vs best with a 95% paired bootstrap CI over trials. Nothing is fitted.

## H-labelled trials (n = 322)

| Model | Gain over chance / trial | Δ vs best [95% CI] | Acceptance (median) |
|---|---|---|---|
| Total Time | -0.508 | best | 0.42 |
| Total Time (convolution) | -0.852 | -0.345 [-0.413, -0.279] | 0.57 |
| Hierarchical | -0.969 | -0.461 [-0.712, -0.210] | 0.74 |
| Postdictive | -1.009 | -0.502 [-0.778, -0.246] | 0.84 |
| Optimal | -1.012 | -0.505 [-0.779, -0.248] | 0.84 |
| First-only | -1.210 | -0.703 [-1.017, -0.390] | 0.44 |
| Second-only (Sequential) | -1.926 | -1.418 [-1.671, -1.162] | 0.67 |

Top four by session (gain over chance / trial):

- Nov_3_g0 (n=203): Total Time -0.462, Total Time (convolution) -0.790, Hierarchical -1.086, Postdictive -1.154
- Nov_6_g0 (n=119): Total Time -0.586, Postdictive -0.762, Optimal -0.763, Hierarchical -0.768

## S-labelled trials (n = 286)

| Model | Gain over chance / trial | Δ vs best [95% CI] | Acceptance (median) |
|---|---|---|---|
| First-only | -0.085 | best | 0.44 |
| Total Time | -0.749 | -0.665 [-0.875, -0.457] | 0.42 |
| Hierarchical | -0.869 | -0.785 [-1.016, -0.543] | 0.74 |
| Total Time (convolution) | -1.207 | -1.122 [-1.374, -0.863] | 0.57 |
| Postdictive | -1.415 | -1.330 [-1.599, -1.064] | 0.84 |
| Optimal | -1.422 | -1.338 [-1.603, -1.070] | 0.84 |
| Second-only (Sequential) | -1.524 | -1.439 [-1.713, -1.160] | 0.67 |

Top four by session (gain over chance / trial):

- Nov_3_g0 (n=164): First-only -0.058, Total Time -0.642, Hierarchical -0.801, Total Time (convolution) -1.106
- Nov_6_g0 (n=122): First-only -0.121, Total Time -0.894, Hierarchical -0.962, Total Time (convolution) -1.343
