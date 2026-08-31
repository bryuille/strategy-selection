Nielsen -- per-maze decoding (degrees features, scope: publication). Balanced accuracy, chance = 0.500

| Feature set                       | d   | Trials | Mazes | Logistic (L2) train | Logistic (L2) CV | Random forest train | Random forest CV |
|-----------------------------------|-----|--------|-------|---------------------|------------------|---------------------|------------------|
| Codebook occupancy (ms), K=6      | 6   | 608    | 4     | 0.638               | 0.547            | 0.937               | 0.521            |
| Codebook occupancy (ms), K=12     | 12  | 608    | 4     | 0.685               | 0.519            | 0.952               | 0.514            |
| Codebook occupancy (binary), K=6  | 6   | 608    | 4     | 0.614               | 0.544            | 0.663               | 0.567            |
| Codebook occupancy (binary), K=12 | 12  | 608    | 4     | 0.673               | 0.529            | 0.752               | 0.545            |
| State bigrams, K=6                | 36  | 608    | 4     | 0.703               | 0.534            | 0.729               | **0.572**        |
| State bigrams, K=12               | 144 | 608    | 4     | 0.846               | 0.540            | 0.790               | 0.566            |
| Gaze heatmap, 5x5                 | 25  | 608    | 4     | 0.731               | **0.595**        | 0.887               | 0.559            |
| Gaze heatmap, 10x10               | 100 | 608    | 4     | 0.846               | 0.528            | 0.941               | 0.534            |

Bold: best CV accuracy per model.
