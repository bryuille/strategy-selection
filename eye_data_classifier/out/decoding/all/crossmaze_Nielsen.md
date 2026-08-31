Nielsen -- across-maze decoding (unit-H features, scope: all). Balanced accuracy, chance = 0.500

| Feature set                       | d   | Trials | Logistic (L2) train | Logistic (L2) CV | Random forest train | Random forest CV |
|-----------------------------------|-----|--------|---------------------|------------------|---------------------|------------------|
| Codebook occupancy (ms), K=6      | 6   | 1746   | 0.613               | 0.598            | 0.891               | 0.585            |
| Codebook occupancy (ms), K=12     | 12  | 1746   | 0.625               | 0.613            | 0.914               | 0.611            |
| Codebook occupancy (binary), K=6  | 6   | 1746   | 0.623               | 0.620            | 0.627               | 0.613            |
| Codebook occupancy (binary), K=12 | 12  | 1746   | 0.634               | **0.627**        | 0.670               | 0.599            |
| State bigrams, K=6                | 36  | 1746   | 0.629               | 0.607            | 0.659               | 0.607            |
| State bigrams, K=12               | 144 | 1746   | 0.662               | 0.598            | 0.713               | 0.629            |
| Gaze heatmap, 5x5                 | 25  | 1746   | 0.637               | 0.621            | 0.893               | 0.627            |
| Gaze heatmap, 10x10               | 100 | 1746   | 0.665               | 0.623            | 0.903               | **0.636**        |
| Maze identity (reference)         | 6   | 1746   | 0.837               | 0.837            | 0.837               | 0.837            |

Bold: best CV accuracy per model, maze-identity reference excluded.
