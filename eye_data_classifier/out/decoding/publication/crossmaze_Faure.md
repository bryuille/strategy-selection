Faure -- across-maze decoding (unit-H features, scope: publication). Balanced accuracy, chance = 0.500

| Feature set                       | d   | Trials | Logistic (L2) train | Logistic (L2) CV | Random forest train | Random forest CV |
|-----------------------------------|-----|--------|---------------------|------------------|---------------------|------------------|
| Codebook occupancy (ms), K=6      | 6   | 660    | 0.593               | 0.569            | 0.936               | 0.531            |
| Codebook occupancy (ms), K=12     | 12  | 660    | 0.600               | 0.551            | 0.957               | 0.570            |
| Codebook occupancy (binary), K=6  | 6   | 660    | 0.574               | 0.562            | 0.588               | 0.518            |
| Codebook occupancy (binary), K=12 | 12  | 660    | 0.599               | 0.568            | 0.672               | 0.583            |
| State bigrams, K=6                | 36  | 660    | 0.614               | **0.580**        | 0.671               | 0.585            |
| State bigrams, K=12               | 144 | 660    | 0.684               | 0.571            | 0.766               | 0.578            |
| Gaze heatmap, 5x5                 | 25  | 660    | 0.616               | 0.568            | 0.913               | 0.576            |
| Gaze heatmap, 10x10               | 100 | 660    | 0.689               | 0.578            | 0.948               | **0.595**        |
| Maze identity (reference)         | 6   | 660    | 0.888               | 0.888            | 0.888               | 0.888            |

Bold: best CV accuracy per model, maze-identity reference excluded.
