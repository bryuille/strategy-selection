# Maze-to-maze similarity CV

All three plots are 6×6 matrices of Pearson *r* between maze types 1–6. Cell `(i, j)` is maze *i* vs maze *j*. It is **not** trial *i* vs trial *j*.

The split-half procedure is the same idea everywhere: within a session, split each maze’s trials in half, average each half into one feature vector, correlate held-out halves, repeat, then average the session matrices.

---

## 1. Heatmap similarities

Code: `eye_data_plotting/pre_flash_similarities_heatmap.py` (linear `count/trial`) and `eye_data_plotting/pre_flash_similarities_heatmap_log.py` (`log1p(count/trial)`).

**What is correlated.** Two flattened unit-H gaze maps. Each map is `count / n_trials` (linear) or `log1p(count / n_trials)` (log) over spatial bins. Pearson *r* is over bins that had gaze in that split, not over trials or time.

**Trials.** Pre-fixation samples from `fix_start − 1466 ms` through `fix_start − 300 ms`. Gaze is warped to unit H (`to_maze`) so exits sit at `(±1, ±1)`. Keep QC-passed fixed-geometry trials (`path_type != -99`, photodiode OK, `trial_fade == 0`). A maze is used in a session only if it has at least 4 trials.

**Per session (20 random splits):**

1. For each maze type, randomly cut that maze’s trials in half (`n // 2` vs the rest).
2. Average the trials in each half into one heatmap: sum counts, divide by the number of trials in that half, then optionally `log1p`. Maze 1 now has maps `A1` and `B1`, maze 2 has `A2` and `B2`, and so on.
3. For every maze pair `(i, j)`:

   \[
   r_{ij} = \mathrm{mean}\big(\mathrm{corr}(A_i, B_j),\; \mathrm{corr}(B_i, A_j)\big)
   \]

   `corr` is Pearson *r* of the two flattened maps, using only bins with gaze in that split.

4. Average the 20 split matrices into one 6×6 for the session.

**Across sessions.** Average those session matrices. Diagonal entries are split-half reliability of the same maze (independent trial halves). Off-diagonal entries are similarity of two maze types, estimated without using the same trials twice.

---

## 2. Occupancy and transition

Code:

- `eye_data_plotting.pre_flash_similarities_occupancy.py`
- `eye_data_plotting.pre_flash_similarities_occupancy_bin.py`
- `eye_data_plotting.pre_flash_similarities_transition.py`

Shared CV: `eye_data_plotting/similarities_common.py` (seconds occupancy currently inlines the same steps).

The CV is the heatmap procedure with a different per-trial vector. Pearson *r* is over the entries of that vector, not over pixels.

**Trials.** Same pre-fixation window relative to `fix_start` (1533 ms in `plot_io.py`). Valid-path trials (`path_type != -99`) with a finite feature vector. A maze is used in a session if it has at least 2 trials.

**Per-trial feature (the only difference between the plots):**

| Plot | Vector for each trial |
|---|---|
| Occupancy | Seconds spent in each of the *K* codebook states |
| Occupancy (binary) | Length-*K* 0/1 vector: 1 if that state was visited, else 0 |
| Transition | Bigram + trigram proportions of collapsed state runs (dwell time ignored). Path `k1 → k5 → k2` contributes bigrams `(k1, k5)`, `(k5, k2)` and trigram `(k1, k5, k2)` |

**Per session (20 random splits):**

1. For each maze type, randomly cut that maze’s trials in half.
2. Average the feature vectors in each half. Maze *i* has means `A_i` and `B_i`.
3. For every maze pair `(i, j)`:

   \[
   r_{ij} = \mathrm{mean}\big(\mathrm{corr}(A_i, B_j),\; \mathrm{corr}(B_i, A_j)\big)
   \]

4. Average the 20 split matrices into one 6×6 for the session.

**Across sessions.** Average those session matrices, same as heatmaps.
