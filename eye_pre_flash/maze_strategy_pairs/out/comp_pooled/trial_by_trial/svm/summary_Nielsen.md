# Pooled p by maze and K - Nielsen / trial_by_trial / svm

Codebook fitted per (monkey, maze), pooled across every in-scope
session; the estimator pools every in-scope session's trials of one
maze into a single estimate -- no per-session averaging. Each cell
is `p (n)`, where `n` is the number of sessions pooled (constant
across K for a given maze); a blank cell means the codebook itself
could not be fit at that K.

## occupancy / full

| maze | K=7 | K=9 | best K | best p |
|---|---|---|---|---|
| 4 | 0.0020 (10) | 0.0010 (10) | 9 | 0.0010 |

## occupancy / mean_removed

| maze | K=7 | K=9 | best K | best p |
|---|---|---|---|---|
| 4 | 0.0020 (10) | 0.0040 (10) | 7 | 0.0020 |

## occupancy / no_origin

| maze | K=7 | K=9 | best K | best p |
|---|---|---|---|---|
| 4 | 0.0020 (10) | 0.0010 (10) | 9 | 0.0010 |

## occupancy_bin / full

| maze | K=7 | K=9 | best K | best p |
|---|---|---|---|---|
| 4 | 0.0020 (10) | 0.0010 (10) | 9 | 0.0010 |

## occupancy_bin / mean_removed

| maze | K=7 | K=9 | best K | best p |
|---|---|---|---|---|
| 4 | 0.0020 (10) | 0.0010 (10) | 9 | 0.0010 |

## occupancy_bin / no_origin

| maze | K=7 | K=9 | best K | best p |
|---|---|---|---|---|
| 4 | 0.0020 (10) | 0.0010 (10) | 9 | 0.0010 |
