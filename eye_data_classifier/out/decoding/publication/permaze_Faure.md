Faure -- per-maze decoding (degrees features, scope: publication). Balanced accuracy, chance = 0.500

| Feature set                       | d   | Trials | Mazes | Logistic (L2) train | Logistic (L2) CV | Random forest train | Random forest CV |
|-----------------------------------|-----|--------|-------|---------------------|------------------|---------------------|------------------|
| Codebook occupancy (ms), K=6      | 6   | 477    | 4     | 0.661               | **0.558**        | 0.979               | 0.491            |
| Codebook occupancy (ms), K=12     | 12  | 477    | 4     | 0.720               | 0.515            | 0.985               | 0.488            |
| Codebook occupancy (binary), K=6  | 6   | 477    | 4     | 0.658               | 0.512            | 0.695               | **0.547**        |
| Codebook occupancy (binary), K=12 | 12  | 477    | 4     | 0.711               | 0.515            | 0.762               | 0.501            |
| State bigrams, K=6                | 36  | 477    | 4     | 0.760               | 0.520            | 0.764               | 0.484            |
| State bigrams, K=12               | 144 | 477    | 4     | 0.910               | 0.533            | 0.842               | 0.530            |
| Gaze heatmap, 5x5                 | 25  | 477    | 4     | 0.746               | 0.511            | 0.988               | 0.495            |
| Gaze heatmap, 10x10               | 100 | 477    | 4     | 0.945               | 0.536            | 0.978               | 0.497            |

Bold: best CV accuracy per model.
