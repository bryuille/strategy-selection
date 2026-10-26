# Faure: belief tables

q_M: model M's belief over the unchosen exits given that M's rule chose the chosen exit. q_M uncond: M's probability of choosing each exit, renormalised over the unchosen exits (Figure Five). Observed: share of first counterfactual looks. See counterfactual/README.md and counterfactual/tables.py.

## 1. All mazes, by relation to the chosen exit

Mean probability on each unchosen exit by its relation to the chosen one (rows sum to 1), over the labelled trials with a counterfactual look.

**All labelled trials** (n 224)

| | sibling | horizontal | diagonal |
|---|---|---|---|
| q_H | 0.39 | 0.13 | 0.47 |
| q_H uncond | 0.41 | 0.13 | 0.46 |
| q_S | 0.43 | 0.15 | 0.42 |
| q_S uncond | 0.42 | 0.15 | 0.43 |
| observed | 0.74 | 0.12 | 0.14 |

**H-labelled, correct** (n 75)

| | sibling | horizontal | diagonal |
|---|---|---|---|
| q_H | 0.57 | 0.04 | 0.38 |
| q_H uncond | 0.58 | 0.04 | 0.37 |
| q_S | 0.28 | 0.09 | 0.63 |
| q_S uncond | 0.29 | 0.09 | 0.62 |
| observed | 0.85 | 0.08 | 0.07 |

**H-labelled, error** (n 14)

| | sibling | horizontal | diagonal |
|---|---|---|---|
| q_H | 0.53 | 0.21 | 0.26 |
| q_H uncond | 0.53 | 0.22 | 0.25 |
| q_S | 0.20 | 0.42 | 0.37 |
| q_S uncond | 0.21 | 0.38 | 0.42 |
| observed | 0.50 | 0.29 | 0.21 |

**S-labelled, correct** (n 105)

| | sibling | horizontal | diagonal |
|---|---|---|---|
| q_H | 0.30 | 0.12 | 0.58 |
| q_H uncond | 0.31 | 0.11 | 0.58 |
| q_S | 0.58 | 0.11 | 0.31 |
| q_S uncond | 0.57 | 0.10 | 0.33 |
| observed | 0.82 | 0.10 | 0.09 |

**S-labelled, error** (n 30)

| | sibling | horizontal | diagonal |
|---|---|---|---|
| q_H | 0.21 | 0.36 | 0.43 |
| q_H uncond | 0.24 | 0.38 | 0.38 |
| q_S | 0.34 | 0.33 | 0.32 |
| q_S uncond | 0.34 | 0.35 | 0.31 |
| observed | 0.30 | 0.20 | 0.50 |

## 2. Per maze

Each model stores one row per chosen exit: its probability that the monkey looks next at each of the other three exits (the row sums to 1). The row is built from the path that ends at the chosen exit (draws kept when the model's rule picks it) and applied to every trial that chose it, error trials included. Under each chosen exit: the H and S rows, then the observed share of first counterfactual looks (correct and error trials).

### Maze 1: h1..h6 = 5, 7, 11, 10, 9, 5

| chosen | | LU | LD | RU | RD |
|---|---|---|---|---|---|
| LU | q_H | – | 0.78 | 0.13 | 0.09 |
|  | q_S | – | 0.12 | 0.60 | 0.28 |
| | **observed** (n 4) | – | 0.75 | 0.25 | 0.00 |
| | observed, H-labelled (n 4) | – | 0.75 | 0.25 | 0.00 |
| | observed, S-labelled (n 0) | – |  |  |  |
| LD | q_H | 0.66 | – | 0.32 | 0.01 |
|  | q_S | 0.02 | – | 0.98 | 0.00 |
| | **observed** (n 15) | 1.00 | – | 0.00 | 0.00 |
| | observed, H-labelled (n 15) | 1.00 | – | 0.00 | 0.00 |
| | observed, S-labelled (n 0) |  | – |  |  |
| RU | q_H | 0.15 | 0.21 | – | 0.64 |
|  | q_S | 0.38 | 0.62 | – | 0.00 |
| | **observed** (n 1) | 0.00 | 1.00 | – | 0.00 |
| | observed, H-labelled (n 1) | 0.00 | 1.00 | – | 0.00 |
| | observed, S-labelled (n 0) |  |  | – |  |
| RD | q_H | 0.18 | 0.02 | 0.80 | – |
|  | q_S | 0.88 | 0.02 | 0.10 | – |
| | **observed** (n 12) | 0.08 | 0.17 | 0.75 | – |
| | observed, H-labelled (n 11) | 0.00 | 0.18 | 0.82 | – |
| | observed, S-labelled (n 1) | 1.00 | 0.00 | 0.00 | – |

### Maze 2: h1..h6 = 5, 9, 11, 10, 7, 5

| chosen | | LU | LD | RU | RD |
|---|---|---|---|---|---|
| LU | q_H | – | 0.92 | 0.08 | 0.01 |
|  | q_S | – | 0.62 | 0.38 | 0.00 |
| | **observed** (n 4) | – | 0.75 | 0.25 | 0.00 |
| | observed, H-labelled (n 3) | – | 0.67 | 0.33 | 0.00 |
| | observed, S-labelled (n 1) | – | 1.00 | 0.00 | 0.00 |
| LD | q_H | 0.89 | – | 0.10 | 0.00 |
|  | q_S | 0.98 | – | 0.02 | 0.00 |
| | **observed** (n 9) | 1.00 | – | 0.00 | 0.00 |
| | observed, H-labelled (n 6) | 1.00 | – | 0.00 | 0.00 |
| | observed, S-labelled (n 3) | 1.00 | – | 0.00 | 0.00 |
| RU | q_H | 0.08 | 0.02 | – | 0.90 |
|  | q_S | 0.60 | 0.12 | – | 0.28 |
| | **observed** (n 8) | 0.50 | 0.25 | – | 0.25 |
| | observed, H-labelled (n 5) | 0.40 | 0.40 | – | 0.20 |
| | observed, S-labelled (n 3) | 0.67 | 0.00 | – | 0.33 |
| RD | q_H | 0.05 | 0.01 | 0.93 | – |
|  | q_S | 0.10 | 0.02 | 0.88 | – |
| | **observed** (n 10) | 0.20 | 0.30 | 0.50 | – |
| | observed, H-labelled (n 5) | 0.00 | 0.20 | 0.80 | – |
| | observed, S-labelled (n 5) | 0.40 | 0.40 | 0.20 | – |

### Maze 3: h1..h6 = 5, 7, 11, 7.5, 9, 5

| chosen | | LU | LD | RU | RD |
|---|---|---|---|---|---|
| LU | q_H | – | 0.41 | 0.36 | 0.23 |
|  | q_S | – | 0.12 | 0.60 | 0.28 |
| | **observed** (n 1) | – | 1.00 | 0.00 | 0.00 |
| | observed, H-labelled (n 1) | – | 1.00 | 0.00 | 0.00 |
| | observed, S-labelled (n 0) | – |  |  |  |
| LD | q_H | 0.28 | – | 0.72 | 0.01 |
|  | q_S | 0.02 | – | 0.98 | 0.00 |
| | **observed** (n 10) | 1.00 | – | 0.00 | 0.00 |
| | observed, H-labelled (n 10) | 1.00 | – | 0.00 | 0.00 |
| | observed, S-labelled (n 0) |  | – |  |  |
| RU | q_H | 0.32 | 0.46 | – | 0.23 |
|  | q_S | 0.38 | 0.62 | – | 0.00 |
| | **observed** (n 2) | 1.00 | 0.00 | – | 0.00 |
| | observed, H-labelled (n 2) | 1.00 | 0.00 | – | 0.00 |
| | observed, S-labelled (n 0) |  |  | – |  |
| RD | q_H | 0.60 | 0.02 | 0.38 | – |
|  | q_S | 0.88 | 0.02 | 0.10 | – |
| | **observed** (n 10) | 0.10 | 0.10 | 0.80 | – |
| | observed, H-labelled (n 9) | 0.11 | 0.00 | 0.89 | – |
| | observed, S-labelled (n 1) | 0.00 | 1.00 | 0.00 | – |

### Maze 4: h1..h6 = 5, 9, 11, 7.5, 7, 5

| chosen | | LU | LD | RU | RD |
|---|---|---|---|---|---|
| LU | q_H | – | 0.68 | 0.30 | 0.02 |
|  | q_S | – | 0.62 | 0.38 | 0.00 |
| | **observed** (n 4) | – | 0.25 | 0.50 | 0.25 |
| | observed, H-labelled (n 1) | – | 0.00 | 0.00 | 1.00 |
| | observed, S-labelled (n 3) | – | 0.33 | 0.67 | 0.00 |
| LD | q_H | 0.62 | – | 0.38 | 0.00 |
|  | q_S | 0.98 | – | 0.02 | 0.00 |
| | **observed** (n 19) | 0.95 | – | 0.05 | 0.00 |
| | observed, H-labelled (n 2) | 1.00 | – | 0.00 | 0.00 |
| | observed, S-labelled (n 17) | 0.94 | – | 0.06 | 0.00 |
| RU | q_H | 0.34 | 0.08 | – | 0.58 |
|  | q_S | 0.60 | 0.12 | – | 0.28 |
| | **observed** (n 2) | 0.00 | 0.50 | – | 0.50 |
| | observed, H-labelled (n 1) | 0.00 | 1.00 | – | 0.00 |
| | observed, S-labelled (n 1) | 0.00 | 0.00 | – | 1.00 |
| RD | q_H | 0.27 | 0.04 | 0.69 | – |
|  | q_S | 0.10 | 0.02 | 0.88 | – |
| | **observed** (n 5) | 0.00 | 0.20 | 0.80 | – |
| | observed, H-labelled (n 0) |  |  |  | – |
| | observed, S-labelled (n 5) | 0.00 | 0.20 | 0.80 | – |

### Maze 5: h1..h6 = 5, 7, 11, 5, 9, 5  (h1 = h4: the first interval cannot tell the sides apart)

| chosen | | LU | LD | RU | RD |
|---|---|---|---|---|---|
| LU | q_H | – | 0.08 | 0.56 | 0.36 |
|  | q_S | – | 0.12 | 0.60 | 0.28 |
| | **observed** (n 12) | – | 0.83 | 0.08 | 0.08 |
| | observed, H-labelled (n 3) | – | 1.00 | 0.00 | 0.00 |
| | observed, S-labelled (n 9) | – | 0.78 | 0.11 | 0.11 |
| LD | q_H | 0.05 | – | 0.95 | 0.00 |
|  | q_S | 0.02 | – | 0.98 | 0.00 |
| | **observed** (n 15) | 1.00 | – | 0.00 | 0.00 |
| | observed, H-labelled (n 2) | 1.00 | – | 0.00 | 0.00 |
| | observed, S-labelled (n 13) | 1.00 | – | 0.00 | 0.00 |
| RU | q_H | 0.40 | 0.58 | – | 0.02 |
|  | q_S | 0.38 | 0.62 | – | 0.00 |
| | **observed** (n 13) | 0.23 | 0.62 | – | 0.15 |
| | observed, H-labelled (n 2) | 0.00 | 1.00 | – | 0.00 |
| | observed, S-labelled (n 11) | 0.27 | 0.55 | – | 0.18 |
| RD | q_H | 0.94 | 0.02 | 0.04 | – |
|  | q_S | 0.88 | 0.02 | 0.10 | – |
| | **observed** (n 17) | 0.35 | 0.06 | 0.59 | – |
| | observed, H-labelled (n 2) | 0.00 | 0.50 | 0.50 | – |
| | observed, S-labelled (n 15) | 0.40 | 0.00 | 0.60 | – |

### Maze 6: h1..h6 = 5, 9, 11, 5, 7, 5  (h1 = h4: the first interval cannot tell the sides apart)

| chosen | | LU | LD | RU | RD |
|---|---|---|---|---|---|
| LU | q_H | – | 0.22 | 0.75 | 0.03 |
|  | q_S | – | 0.62 | 0.38 | 0.00 |
| | **observed** (n 4) | – | 1.00 | 0.00 | 0.00 |
| | observed, H-labelled (n 0) | – |  |  |  |
| | observed, S-labelled (n 4) | – | 1.00 | 0.00 | 0.00 |
| LD | q_H | 0.18 | – | 0.82 | 0.00 |
|  | q_S | 0.98 | – | 0.02 | 0.00 |
| | **observed** (n 20) | 0.95 | – | 0.05 | 0.00 |
| | observed, H-labelled (n 4) | 1.00 | – | 0.00 | 0.00 |
| | observed, S-labelled (n 16) | 0.94 | – | 0.06 | 0.00 |
| RU | q_H | 0.74 | 0.17 | – | 0.09 |
|  | q_S | 0.60 | 0.12 | – | 0.28 |
| | **observed** (n 9) | 0.11 | 0.67 | – | 0.22 |
| | observed, H-labelled (n 0) |  |  | – |  |
| | observed, S-labelled (n 9) | 0.11 | 0.67 | – | 0.22 |
| RD | q_H | 0.76 | 0.10 | 0.14 | – |
|  | q_S | 0.10 | 0.02 | 0.88 | – |
| | **observed** (n 18) | 0.00 | 0.17 | 0.83 | – |
| | observed, H-labelled (n 0) |  |  |  | – |
| | observed, S-labelled (n 18) | 0.00 | 0.17 | 0.83 | – |
