# Nielsen: belief tables

q_M: model M's belief over the unchosen exits given that M's rule chose the chosen exit. q_M uncond: M's probability of choosing each exit, renormalised over the unchosen exits (Figure Five). Observed: share of first counterfactual looks. See counterfactual/README.md and counterfactual/tables.py.

## 1. All mazes, by relation to the chosen exit

Mean probability on each unchosen exit by its relation to the chosen one (rows sum to 1), over the labelled trials with a counterfactual look.

**All labelled trials** (n 608)

| | sibling | horizontal | diagonal |
|---|---|---|---|
| q_H | 0.45 | 0.18 | 0.37 |
| q_H uncond | 0.43 | 0.19 | 0.38 |
| q_S | 0.36 | 0.25 | 0.39 |
| q_S uncond | 0.37 | 0.25 | 0.38 |
| observed | 0.40 | 0.25 | 0.35 |

**H-labelled, correct** (n 276)

| | sibling | horizontal | diagonal |
|---|---|---|---|
| q_H | 0.65 | 0.11 | 0.24 |
| q_H uncond | 0.62 | 0.11 | 0.26 |
| q_S | 0.30 | 0.24 | 0.46 |
| q_S uncond | 0.32 | 0.25 | 0.44 |
| observed | 0.42 | 0.30 | 0.28 |

**H-labelled, error** (n 46)

| | sibling | horizontal | diagonal |
|---|---|---|---|
| q_H | 0.66 | 0.16 | 0.18 |
| q_H uncond | 0.63 | 0.18 | 0.20 |
| q_S | 0.41 | 0.34 | 0.25 |
| q_S uncond | 0.43 | 0.31 | 0.26 |
| observed | 0.39 | 0.17 | 0.43 |

**S-labelled, correct** (n 195)

| | sibling | horizontal | diagonal |
|---|---|---|---|
| q_H | 0.23 | 0.20 | 0.57 |
| q_H uncond | 0.21 | 0.20 | 0.59 |
| q_S | 0.44 | 0.19 | 0.37 |
| q_S uncond | 0.44 | 0.19 | 0.37 |
| observed | 0.37 | 0.19 | 0.44 |

**S-labelled, error** (n 91)

| | sibling | horizontal | diagonal |
|---|---|---|---|
| q_H | 0.19 | 0.37 | 0.44 |
| q_H uncond | 0.19 | 0.39 | 0.42 |
| q_S | 0.36 | 0.32 | 0.32 |
| q_S uncond | 0.36 | 0.35 | 0.29 |
| observed | 0.42 | 0.24 | 0.34 |

## 2. Per maze

Each model stores one row per chosen exit: its probability that the monkey looks next at each of the other three exits (the row sums to 1). The row is built from the path that ends at the chosen exit (draws kept when the model's rule picks it) and applied to every trial that chose it, error trials included. Under each chosen exit: the H and S rows, then the observed share of first counterfactual looks (correct and error trials).

### Maze 1: h1..h6 = 5, 7, 11, 10, 9, 5

| chosen | | LU | LD | RU | RD |
|---|---|---|---|---|---|
| LU | q_H | – | 0.81 | 0.12 | 0.07 |
|  | q_S | – | 0.10 | 0.62 | 0.27 |
| | **observed** (n 19) | – | 0.26 | 0.11 | 0.63 |
| | observed, H-labelled (n 19) | – | 0.26 | 0.11 | 0.63 |
| | observed, S-labelled (n 0) | – |  |  |  |
| LD | q_H | 0.70 | – | 0.28 | 0.02 |
|  | q_S | 0.01 | – | 0.98 | 0.00 |
| | **observed** (n 25) | 0.40 | – | 0.04 | 0.56 |
| | observed, H-labelled (n 24) | 0.42 | – | 0.04 | 0.54 |
| | observed, S-labelled (n 1) | 0.00 | – | 0.00 | 1.00 |
| RU | q_H | 0.15 | 0.20 | – | 0.65 |
|  | q_S | 0.37 | 0.63 | – | 0.00 |
| | **observed** (n 20) | 0.20 | 0.05 | – | 0.75 |
| | observed, H-labelled (n 20) | 0.20 | 0.05 | – | 0.75 |
| | observed, S-labelled (n 0) |  |  | – |  |
| RD | q_H | 0.17 | 0.03 | 0.80 | – |
|  | q_S | 0.91 | 0.01 | 0.08 | – |
| | **observed** (n 25) | 0.28 | 0.48 | 0.24 | – |
| | observed, H-labelled (n 23) | 0.30 | 0.52 | 0.17 | – |
| | observed, S-labelled (n 2) | 0.00 | 0.00 | 1.00 | – |

### Maze 2: h1..h6 = 5, 9, 11, 10, 7, 5

| chosen | | LU | LD | RU | RD |
|---|---|---|---|---|---|
| LU | q_H | – | 0.94 | 0.06 | 0.01 |
|  | q_S | – | 0.63 | 0.37 | 0.00 |
| | **observed** (n 18) | – | 0.33 | 0.44 | 0.22 |
| | observed, H-labelled (n 15) | – | 0.33 | 0.47 | 0.20 |
| | observed, S-labelled (n 3) | – | 0.33 | 0.33 | 0.33 |
| LD | q_H | 0.92 | – | 0.08 | 0.00 |
|  | q_S | 0.98 | – | 0.01 | 0.00 |
| | **observed** (n 22) | 0.41 | – | 0.05 | 0.55 |
| | observed, H-labelled (n 22) | 0.41 | – | 0.05 | 0.55 |
| | observed, S-labelled (n 0) |  | – |  |  |
| RU | q_H | 0.07 | 0.02 | – | 0.91 |
|  | q_S | 0.62 | 0.10 | – | 0.27 |
| | **observed** (n 30) | 0.30 | 0.40 | – | 0.30 |
| | observed, H-labelled (n 27) | 0.30 | 0.44 | – | 0.26 |
| | observed, S-labelled (n 3) | 0.33 | 0.00 | – | 0.67 |
| RD | q_H | 0.04 | 0.01 | 0.94 | – |
|  | q_S | 0.08 | 0.01 | 0.91 | – |
| | **observed** (n 35) | 0.37 | 0.26 | 0.37 | – |
| | observed, H-labelled (n 31) | 0.35 | 0.29 | 0.35 | – |
| | observed, S-labelled (n 4) | 0.50 | 0.00 | 0.50 | – |

### Maze 3: h1..h6 = 5, 7, 11, 7.5, 9, 5

| chosen | | LU | LD | RU | RD |
|---|---|---|---|---|---|
| LU | q_H | – | 0.41 | 0.37 | 0.22 |
|  | q_S | – | 0.10 | 0.62 | 0.27 |
| | **observed** (n 23) | – | 0.39 | 0.09 | 0.52 |
| | observed, H-labelled (n 21) | – | 0.38 | 0.10 | 0.52 |
| | observed, S-labelled (n 2) | – | 0.50 | 0.00 | 0.50 |
| LD | q_H | 0.27 | – | 0.72 | 0.01 |
|  | q_S | 0.01 | – | 0.98 | 0.00 |
| | **observed** (n 19) | 0.58 | – | 0.05 | 0.37 |
| | observed, H-labelled (n 19) | 0.58 | – | 0.05 | 0.37 |
| | observed, S-labelled (n 0) |  | – |  |  |
| RU | q_H | 0.32 | 0.47 | – | 0.21 |
|  | q_S | 0.37 | 0.63 | – | 0.00 |
| | **observed** (n 35) | 0.23 | 0.09 | – | 0.69 |
| | observed, H-labelled (n 33) | 0.24 | 0.09 | – | 0.67 |
| | observed, S-labelled (n 2) | 0.00 | 0.00 | – | 1.00 |
| RD | q_H | 0.63 | 0.02 | 0.35 | – |
|  | q_S | 0.91 | 0.01 | 0.08 | – |
| | **observed** (n 28) | 0.68 | 0.11 | 0.21 | – |
| | observed, H-labelled (n 23) | 0.70 | 0.04 | 0.26 | – |
| | observed, S-labelled (n 5) | 0.60 | 0.40 | 0.00 | – |

### Maze 4: h1..h6 = 5, 9, 11, 7.5, 7, 5

| chosen | | LU | LD | RU | RD |
|---|---|---|---|---|---|
| LU | q_H | – | 0.71 | 0.28 | 0.01 |
|  | q_S | – | 0.63 | 0.37 | 0.00 |
| | **observed** (n 16) | – | 0.56 | 0.25 | 0.19 |
| | observed, H-labelled (n 10) | – | 0.50 | 0.20 | 0.30 |
| | observed, S-labelled (n 6) | – | 0.67 | 0.33 | 0.00 |
| LD | q_H | 0.65 | – | 0.35 | 0.00 |
|  | q_S | 0.98 | – | 0.01 | 0.00 |
| | **observed** (n 7) | 0.57 | – | 0.00 | 0.43 |
| | observed, H-labelled (n 3) | 0.33 | – | 0.00 | 0.67 |
| | observed, S-labelled (n 4) | 0.75 | – | 0.00 | 0.25 |
| RU | q_H | 0.34 | 0.07 | – | 0.59 |
|  | q_S | 0.62 | 0.10 | – | 0.27 |
| | **observed** (n 39) | 0.08 | 0.18 | – | 0.74 |
| | observed, H-labelled (n 16) | 0.06 | 0.25 | – | 0.69 |
| | observed, S-labelled (n 23) | 0.09 | 0.13 | – | 0.78 |
| RD | q_H | 0.27 | 0.03 | 0.70 | – |
|  | q_S | 0.08 | 0.01 | 0.91 | – |
| | **observed** (n 24) | 0.58 | 0.25 | 0.17 | – |
| | observed, H-labelled (n 14) | 0.71 | 0.14 | 0.14 | – |
| | observed, S-labelled (n 10) | 0.40 | 0.40 | 0.20 | – |

### Maze 5: h1..h6 = 5, 7, 11, 5, 9, 5  (h1 = h4: the first interval cannot tell the sides apart)

| chosen | | LU | LD | RU | RD |
|---|---|---|---|---|---|
| LU | q_H | – | 0.07 | 0.58 | 0.35 |
|  | q_S | – | 0.10 | 0.62 | 0.28 |
| | **observed** (n 15) | – | 0.60 | 0.07 | 0.33 |
| | observed, H-labelled (n 1) | – | 0.00 | 0.00 | 1.00 |
| | observed, S-labelled (n 14) | – | 0.64 | 0.07 | 0.29 |
| LD | q_H | 0.04 | – | 0.96 | 0.00 |
|  | q_S | 0.01 | – | 0.98 | 0.00 |
| | **observed** (n 13) | 0.31 | – | 0.08 | 0.62 |
| | observed, H-labelled (n 0) |  | – |  |  |
| | observed, S-labelled (n 13) | 0.31 | – | 0.08 | 0.62 |
| RU | q_H | 0.40 | 0.59 | – | 0.02 |
|  | q_S | 0.37 | 0.63 | – | 0.00 |
| | **observed** (n 40) | 0.17 | 0.17 | – | 0.65 |
| | observed, H-labelled (n 1) | 0.00 | 0.00 | – | 1.00 |
| | observed, S-labelled (n 39) | 0.18 | 0.18 | – | 0.64 |
| RD | q_H | 0.96 | 0.01 | 0.03 | – |
|  | q_S | 0.91 | 0.01 | 0.08 | – |
| | **observed** (n 46) | 0.93 | 0.00 | 0.07 | – |
| | observed, H-labelled (n 0) |  |  |  | – |
| | observed, S-labelled (n 46) | 0.93 | 0.00 | 0.07 | – |

### Maze 6: h1..h6 = 5, 9, 11, 5, 7, 5  (h1 = h4: the first interval cannot tell the sides apart)

| chosen | | LU | LD | RU | RD |
|---|---|---|---|---|---|
| LU | q_H | – | 0.21 | 0.77 | 0.02 |
|  | q_S | – | 0.63 | 0.37 | 0.00 |
| | **observed** (n 28) | – | 0.21 | 0.46 | 0.32 |
| | observed, H-labelled (n 0) | – |  |  |  |
| | observed, S-labelled (n 28) | – | 0.21 | 0.46 | 0.32 |
| LD | q_H | 0.17 | – | 0.83 | 0.00 |
|  | q_S | 0.98 | – | 0.01 | 0.00 |
| | **observed** (n 32) | 0.16 | – | 0.47 | 0.38 |
| | observed, H-labelled (n 0) |  | – |  |  |
| | observed, S-labelled (n 32) | 0.16 | – | 0.47 | 0.38 |
| RU | q_H | 0.78 | 0.14 | – | 0.08 |
|  | q_S | 0.62 | 0.10 | – | 0.27 |
| | **observed** (n 15) | 0.00 | 0.20 | – | 0.80 |
| | observed, H-labelled (n 0) |  |  | – |  |
| | observed, S-labelled (n 15) | 0.00 | 0.20 | – | 0.80 |
| RD | q_H | 0.80 | 0.08 | 0.12 | – |
|  | q_S | 0.08 | 0.01 | 0.91 | – |
| | **observed** (n 34) | 0.59 | 0.15 | 0.26 | – |
| | observed, H-labelled (n 0) |  |  |  | – |
| | observed, S-labelled (n 34) | 0.59 | 0.15 | 0.26 | – |
