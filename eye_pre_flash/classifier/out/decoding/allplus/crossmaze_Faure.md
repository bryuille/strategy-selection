Faure -- across-maze decoding (unit-H features, scope: allplus). Balanced accuracy, chance = 0.500

| Feature set                       | d   | Trials | Logistic (L2) train | Logistic (L2) CV | Random forest train | Random forest CV |
|-----------------------------------|-----|--------|---------------------|------------------|---------------------|------------------|
| Codebook occupancy (ms), K=6      | 6   | 2448   | 0.585               | 0.579            | 0.934               | 0.553            |
| Codebook occupancy (ms), K=12     | 12  | 2448   | 0.587               | 0.579            | 0.969               | 0.542            |
| Codebook occupancy (binary), K=6  | 6   | 2448   | 0.545               | 0.541            | 0.551               | 0.521            |
| Codebook occupancy (binary), K=12 | 12  | 2448   | 0.553               | 0.539            | 0.596               | 0.526            |
| State bigrams, K=6                | 36  | 2448   | 0.558               | 0.529            | 0.587               | 0.515            |
| State bigrams, K=12               | 144 | 2448   | 0.605               | 0.551            | 0.714               | 0.541            |
| Gaze heatmap, 5x5                 | 25  | 2448   | 0.573               | 0.548            | 0.919               | 0.570            |
| Gaze heatmap, 10x10               | 100 | 2448   | 0.629               | **0.589**        | 0.933               | **0.590**        |
| Maze identity (reference)         | 6   | 2448   | 0.710               | 0.710            | 0.710               | 0.710            |

Bold: best CV accuracy per model, maze-identity reference excluded.
