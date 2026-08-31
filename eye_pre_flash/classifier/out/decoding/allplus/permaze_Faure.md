Faure -- per-maze decoding (degrees features, scope: allplus). Balanced accuracy, chance = 0.500

| Feature set                       | d   | Trials | Mazes | Logistic (L2) train | Logistic (L2) CV | Random forest train | Random forest CV |
|-----------------------------------|-----|--------|-------|---------------------|------------------|---------------------|------------------|
| Codebook occupancy (ms), K=6      | 6   | 2448   | 6     | 0.610               | 0.585            | 0.941               | 0.536            |
| Codebook occupancy (ms), K=12     | 12  | 2448   | 6     | 0.636               | **0.589**        | 0.967               | 0.562            |
| Codebook occupancy (binary), K=6  | 6   | 2448   | 6     | 0.608               | 0.582            | 0.632               | **0.578**        |
| Codebook occupancy (binary), K=12 | 12  | 2448   | 6     | 0.633               | 0.582            | 0.698               | 0.559            |
| State bigrams, K=6                | 36  | 2448   | 6     | 0.639               | 0.573            | 0.700               | 0.560            |
| State bigrams, K=12               | 144 | 2448   | 6     | 0.736               | 0.555            | 0.794               | 0.553            |
| Gaze heatmap, 5x5                 | 25  | 2448   | 6     | 0.632               | 0.542            | 0.959               | 0.541            |
| Gaze heatmap, 10x10               | 100 | 2448   | 6     | 0.750               | 0.557            | 0.960               | 0.547            |

Bold: best CV accuracy per model.
