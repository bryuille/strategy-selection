# Choosing the codebook size K

Three scripts in `eye_data_plotting/` bear on K. They are not alternatives of
equal standing — read them in this order:

| Script | Measures | Can it select K? |
| ------ | -------- | ---------------- |
| `pre_flash_centroid_stability_better` | held-out coverage × centroid reproducibility | **yes — this is the one to read.** Interior maximum; read the argmax |
| `pre_flash_centroid_stability` | centroid reproducibility alone (RMS drift, maze units) | only as a diagnostic — trivially satisfied at small K |
| `pre_flash_mse_error` | held-out quantization error | no. Falls monotonically in K |

All three pool valid in-maze fixation `(x, y)` samples across every eye session
for a monkey (`codebook_xy_pool`), in maze-normalized units, capped at
`--max-samples` (default 80000) — exactly the pool the production codebook is
fit on. All three fit with the production call
(`fit_code_xy_kmeans`, k-means++, `n_init=10`).

Each fit sees **half** the pool, so every number here is a conservative read of
the production codebook, which is fit on all of it. Halving the sample lowers
coverage and raises drift at every K; the argmax is the robust part, not the
score height.

```bash
uv run python -m eye_data_plotting.pre_flash_centroid_stability_better --monkey Faure
uv run python -m eye_data_plotting.pre_flash_centroid_stability_better --monkey Nielsen --n-repeats 50
uv run python -m eye_data_plotting.pre_flash_centroid_stability --monkey Faure
uv run python -m eye_data_plotting.pre_flash_mse_error --monkey Faure
```

Shared flags: `--ks`, `--n-repeats` (default 30), `--max-samples`, `--seed`.
`_better` adds `--tolerance` and `--exponent`. Default sweep is
`2,3,4,5,6,7,8,9,10,12,14,16,20`. Cost is `n_repeats × |ks| × 2` k-means fits on
half the pool each.

Output: `eye_data_plotting/out/<script name>/<monkey>/<script name>.png`.

---

## 1. Why reconstruction error cannot pick K

`pre_flash_mse_error` measures held-out mean squared distance from a gaze sample
to its nearest centroid. For k-means that quantity **falls monotonically in K** —
a finer codebook always reconstructs held-out points better, all the way to one
centroid per sample — so it has no minimum. Read it for the absolute
reconstruction scale and the train/test gap, nothing more.

The question with an interior optimum is not "how well does the codebook
reconstruct?" but **"are the centroids the same objects when re-estimated from a
different sample?"** Real attractors are properties of the behaviour, so they
reappear in any sufficiently large sample. Surplus centroids have nothing to lock
onto, so they migrate to wherever that particular sample happens to be lumpy —
sample-dependent centres are the fingerprint of too large a K.

## 2. Split-half reproducibility, and why it is not enough alone

`pre_flash_centroid_stability` asks that question directly. Per repeat, per K:

1. **Split A/B.** Shuffle the pool and cut it into two **disjoint** halves.
   Disjoint matters: overlapping subsamples share points, which correlates the
   two fits and inflates stability.
2. **Fit both halves**, on the **same** k-means seed, so the intended difference
   between the two fits is the sample, not the initialization. `n_init=10` keeps
   the remaining local-minimum risk small.
3. **Correspondence.** Build the K×K matrix of squared distances between A's
   centroids and B's, and solve for the one-to-one matching that minimizes the
   total, exactly, with the Hungarian algorithm
   (`scipy.optimize.linear_sum_assignment`).

   This step is required, not cosmetic: k-means state labels are arbitrary, so
   A's `k1` and B's `k1` are unrelated and matching by index would measure
   nothing. Greedy nearest-neighbour matching is *not* used either — it can
   double-book one centroid and then overstate drift when two centroids swap
   places.
4. **Score.** Mean squared distance between corresponding centroids, reported as
   its root: **RMS drift in maze units**, so it reads on the scale of the maze
   itself (exits at `(±1, ±1)`, `ASSIGN_RADIUS = 1.0`). The **worst matched
   pair** is plotted alongside the mean, because a codebook can look fine on
   average while one centroid is pure sample noise — usually the first thing to
   go as K passes the true value. At Faure K=12 the mean drift is 0.36 but the
   worst centroid moves 1.0 maze units, past the assignment radius, i.e. that
   state's location is not a property of the behaviour at all.

Drift stays low and flat while every centroid has a real landmark to sit on, and
climbs once K exceeds the number of landmarks there are. It also spikes at
*under*-parameterized K: too few centroids to cover the real modes have no
unique grouping either, so they flip between samples — Faure's K=3 jumps to RMS
0.50 while K=4,5,6 sit at 0.02.

**But drift alone cannot choose K, because K=2 is trivially its minimum.** Two
centroids have almost nothing to disagree about. Every split-half stability
measure is biased toward the small end, so its argmin is not an estimate of K.
Reading this plot means picking out a *band* — the longest contiguous run of Ks
holding RMS drift under `DRIFT_STABLE = 0.10` maze units — and a contiguity rule
is a weaker instrument than an argmax.

The mean over matched pairs is unweighted, so a centroid holding 2% of the
fixations counts as much as one holding 30%. That is the point: an unneeded
centroid parks in a sparse tail and wanders, and weighting by mass would hide
exactly the overfitting this is meant to detect.

## 3. The criterion that does pick K

`pre_flash_centroid_stability_better` scores reproducibility **and** explanatory
power together, on a common 0–1 scale, and takes the product. The two failure
modes sit at opposite ends of K:

```
small K -> reproducible but uninformative
large K -> informative but not reproducible
```

so neither end can win and the optimum is an interior maximum.

**`coverage`** — explanatory power, held out. Fraction of the held-out half's
variance that the *other* half's codebook explains:

```
coverage = 1 - E[ min_j |b - c_A,j|^2 ] / E[ |b - mean(B)|^2 ]
```

The numerator is the quantization error of A's centroids applied to B — the same
quantity `pre_flash_mse_error` plots — and the denominator is B's total variance,
i.e. what a single centroid at the mean would leave unexplained. So this is the
clustering R², computed across the split so it is never in-sample. It rises
steeply while centroids are still picking up real landmarks, then saturates:
exactly the penalty K=2 deserves, and the reason MSE alone cannot select K —
nothing about this term ever turns around.

**`reproducibility`** — spatial stability, made dimensionless:

```
reproducibility = 1 - RMS drift / mean centroid spacing      (clipped at 0)
```

Same Hungarian matching as §2. Dividing by the centroids' own mean
nearest-neighbour spacing removes the shrinking-scale confound: as K grows the
centroids pack closer together, so raw drift in maze units would fall for free.
1 is a codebook that reproduces itself exactly; 0 means centroids move as far as
their own spacing and no longer have identities that survive a resample.

**`score = coverage × reproducibility`** — read the argmax. The equal weighting
is a convention, not a derivation, but a mild one: the two curves turn over
sharply in opposite directions, so the argmax is set by where they cross rather
than by the exact exchange rate. `--exponent e` reweights
(`score = coverage × reproducibility ** e`) if you want to see how little it
matters.

**`k_eff`** (printed, not plotted) — how many matched pairs moved less than
`--tolerance` × that centroid's own nearest-neighbour spacing (default 0.5:
move less than halfway to your nearest sibling and the two fits are
unambiguously describing the same landmark). Unlike the RMS, this says *how
many* states are trustworthy rather than how bad the average is. It is a
diagnostic only, because it saturates rather than turning over — a K=12 codebook
with 8 solid centroids and 4 wandering ones scores `k_eff = 8`, which flatters a
codebook whose remaining third is junk.

## Rejected variants

- **Raw drift normalized by centroid spacing, in `pre_flash_centroid_stability`
  itself.** Defensible, but not load-bearing there: over this sweep spacing falls
  by under a factor of two (1.80 → 0.95) while drift varies by more than an order
  of magnitude, and both versions select the same band for both monkeys. The
  `spacing` column is printed beside drift so that stays checkable. `_better`
  does normalize, because there the term has to be commensurable with coverage.
- **A structure-free null baseline.** An earlier version divided drift by the same
  statistic on uniform points over the pool's bounding box, to calibrate how much
  stability a given K buys from geometry alone. Removed: uniform data over a
  *square* has its own degeneracies, and k-means flips between equal-cost tilings
  whenever the optimal tiling comes in more than one orientation — verified
  directly, the split axis at K=2 alternates between vertical and horizontal, and
  the K=6 solution appears in two 90°-apart rotations. So null drift is spiky in
  K for reasons of square-tiling combinatorics (low at K=4,8,9, high at
  K=2,3,6,12,14,20) that have nothing to do with gaze. A ratio against it is not
  interpretable.
