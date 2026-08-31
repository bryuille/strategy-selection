Nielsen -- per-maze CV balanced accuracy by maze (scope: publication). Cells: Logistic (L2) / Random forest

| Feature set                       | Maze 1            | Maze 2            | Maze 3            | Maze 4                | Maze 5  | Maze 6  |
|-----------------------------------|-------------------|-------------------|-------------------|-----------------------|---------|---------|
| Codebook occupancy (ms), K=6      | 0.581 / 0.486     | 0.444 / 0.469     | 0.556 / 0.532     | 0.608 / 0.584         | -- / -- | -- / -- |
| Codebook occupancy (ms), K=12     | 0.437 / 0.477     | 0.552 / 0.511     | 0.513 / 0.535     | 0.556 / 0.526         | -- / -- | -- / -- |
| Codebook occupancy (binary), K=6  | 0.525 / 0.510     | 0.491 / 0.530     | 0.509 / 0.531     | **0.642** / **0.679** | -- / -- | -- / -- |
| Codebook occupancy (binary), K=12 | 0.469 / 0.427     | 0.525 / **0.534** | 0.541 / 0.642     | 0.567 / 0.554         | -- / -- | -- / -- |
| State bigrams, K=6                | 0.534 / 0.539     | 0.379 / 0.468     | 0.595 / **0.649** | 0.619 / 0.621         | -- / -- | -- / -- |
| State bigrams, K=12               | 0.561 / 0.570     | 0.441 / 0.496     | **0.674** / 0.599 | 0.488 / 0.596         | -- / -- | -- / -- |
| Gaze heatmap, 5x5                 | **0.626** / 0.478 | **0.560** / 0.492 | 0.638 / 0.607     | 0.562 / 0.636         | -- / -- | -- / -- |
| Gaze heatmap, 10x10               | 0.594 / **0.588** | 0.480 / 0.503     | 0.470 / 0.521     | 0.579 / 0.533         | -- / -- | -- / -- |

Bold: each model's best CV accuracy in each maze. '--': fewer than 5 trials of one strategy in that maze.
