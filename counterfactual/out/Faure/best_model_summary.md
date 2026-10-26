# Faure: best behavioral model per neural strategy label, by counterfactual looks

Dendrogram labels, top-two sessions. Each model's belief over the unchosen exits is conditioned on the chosen exit (10,000 draws from the correct path, published Weber fraction, draws kept when the model's rule picks the chosen exit; counterfactual/README.md). Score: mean gain over chance of the counterfactual looks per trial, log P(model) − log P(uniform), Plackett–Luce; 0 = chance, higher is better. Δ vs best with a 95% paired bootstrap CI over trials. Nothing is fitted.

## H-labelled trials (n = 89)

| Model | Gain over chance / trial | Δ vs best [95% CI] | Acceptance (median) |
|---|---|---|---|
| First-only | +0.128 | best | 0.43 |
| Optimal | +0.092 | -0.036 [-0.263, +0.197] | 0.77 |
| Postdictive | +0.088 | -0.040 [-0.265, +0.189] | 0.77 |
| Hierarchical | -0.083 | -0.211 [-0.445, +0.027] | 0.70 |
| Total Time | -0.572 | -0.700 [-1.115, -0.275] | 0.39 |
| Total Time (convolution) | -1.090 | -1.218 [-1.724, -0.725] | 0.53 |
| Second-only (Sequential) | -1.182 | -1.310 [-1.839, -0.780] | 0.64 |

Top four by session (gain over chance / trial):

- june_24_g0 (n=50): First-only +0.038, Postdictive -0.022, Optimal -0.026, Hierarchical -0.247
- june_8_g0 (n=39): First-only +0.243, Optimal +0.242, Postdictive +0.228, Hierarchical +0.127

## S-labelled trials (n = 135)

| Model | Gain over chance / trial | Δ vs best [95% CI] | Acceptance (median) |
|---|---|---|---|
| Total Time | +0.039 | best | 0.39 |
| First-only | -0.046 | -0.085 [-0.274, +0.101] | 0.43 |
| Total Time (convolution) | -0.154 | -0.192 [-0.265, -0.124] | 0.53 |
| Second-only (Sequential) | -0.232 | -0.271 [-0.459, -0.085] | 0.64 |
| Postdictive | -0.282 | -0.321 [-0.521, -0.130] | 0.77 |
| Optimal | -0.289 | -0.328 [-0.528, -0.134] | 0.77 |
| Hierarchical | -0.593 | -0.632 [-0.803, -0.457] | 0.70 |

Top four by session (gain over chance / trial):

- june_24_g0 (n=72): Total Time +0.042, Total Time (convolution) -0.137, First-only -0.215, Second-only (Sequential) -0.320
- june_8_g0 (n=63): First-only +0.147, Total Time +0.036, Second-only (Sequential) -0.132, Postdictive -0.154
