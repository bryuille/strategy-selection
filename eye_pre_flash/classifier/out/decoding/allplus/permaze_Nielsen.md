Nielsen -- per-maze decoding (degrees features, scope: allplus). Balanced accuracy, chance = 0.500

| Feature set                       | d   | Trials | Mazes | Logistic (L2) train | Logistic (L2) CV | Random forest train | Random forest CV |
|-----------------------------------|-----|--------|-------|---------------------|------------------|---------------------|------------------|
| Codebook occupancy (ms), K=6      | 6   | 2635   | 6     | 0.577               | **0.540**        | 0.924               | 0.525            |
| Codebook occupancy (ms), K=12     | 12  | 2635   | 6     | 0.588               | 0.517            | 0.940               | 0.503            |
| Codebook occupancy (binary), K=6  | 6   | 2635   | 6     | 0.558               | 0.496            | 0.605               | 0.515            |
| Codebook occupancy (binary), K=12 | 12  | 2635   | 6     | 0.586               | 0.516            | 0.689               | 0.517            |
| State bigrams, K=6                | 36  | 2635   | 6     | 0.621               | 0.523            | 0.665               | 0.508            |
| State bigrams, K=12               | 144 | 2635   | 6     | 0.717               | 0.497            | 0.749               | 0.504            |
| Gaze heatmap, 5x5                 | 25  | 2635   | 6     | 0.604               | 0.512            | 0.866               | 0.515            |
| Gaze heatmap, 10x10               | 100 | 2635   | 6     | 0.705               | 0.523            | 0.927               | **0.534**        |

Bold: best CV accuracy per model.
