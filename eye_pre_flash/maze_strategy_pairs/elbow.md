# Reading the elbow-selected-K panels

`out/comp_pooled/` sweeps K over 13 values and reports `best_k` = the K
minimising `p`. That is selection on the outcome statistic: a `best_p` of
0.0196 is a minimum over 13 correlated tests, and as a single number it means
much less than it reads as. This analysis replaces that selection with one the
H/S labels never enter, and fixes K **before** the estimator runs.

Everything downstream of the selection is `comp_pooled`'s estimator unchanged,
so **[`CAVEATS.md`](CAVEATS.md) still applies in full** — the two-diagonal
check, the pooling artifact, the `mean_removed` arithmetic, all of it. What
follows is only what is different here.

---

## How K is chosen

For each (monkey, maze), pooling that maze's trials across the in-scope
sessions:

1. Split the trials **80/20 at the trial level**, 20 times with different seeds.
2. Fit the k-means on the train side's gaze samples, at every K from 2 to 20.
3. Score the held-out side and average the 20 curves.
4. Take the **kneedle** knee of that mean curve. That K is fixed from here on.

The grid is uniform on purpose. The legacy sweep grid
(2,3,4,5,6,8,10,12,14,16,20,25,30) has spacing 1,1,1,1,2,2,2,2,2,4,5,5, and
every elbow rule reads the curve's *shape* — a second difference on an uneven
grid is the wrong quantity, and kneedle's normalised x axis gets stretched
exactly where the elbow is expected to sit.

### What the held-out score is, and why it is not plain SSE

`occ_ms` depends on the codebook only through a **categorical** outcome.
`assign_fixation_states` reduces each fixation to the mean of its samples, and
`assign_states_inference` maps that centroid to the nearest prototype within
`ASSIGN_RADIUS`, or to `-1` — and `-1` is dropped from `occ_ms` entirely. Once a
centroid is already the nearest-in-radius pick, moving it *closer* changes
`occ_ms` not at all.

A plain sum of squared distances is therefore dominated by the one effect the
feature is blind to:

| effect of raising K | changes `occ_ms`? | changes plain EV? |
|---|---|---|
| a centroid switches which prototype wins | **yes** | slightly |
| a centroid crosses into radius (was `-1`) | **yes** | slightly |
| an already-assigned centroid moves nearer its prototype | **no** | **yes, dominant** |

Two corrections follow. The score is computed on **fixation centroids**, not
raw samples — the objects assignment actually consumes. (This part is small:
measured on one Faure session, within-fixation variance is 0.15% of total
raw-sample variance, so it rarely moves K. Taken because it is free.) And the
**radius gate is mirrored**: an unassigned centroid contributes its full
deviation from the test mean, explained by nothing, rather than a merely large
distance. `assign_states_inference` itself does the gating, so the metric and
the pipeline cannot drift apart.

`EV` uses the **test-set** mean as its denominator, which makes that denominator
the exact K = 1 case of the numerator. So `EV(1) = 0` identically and every
marginal gain is measured against a fixed origin.

---

## Read `selected_k.csv` before any p-value

The EV curve is a continuous proxy for a categorical decision, and the elbow
may not exist at all. Three columns say whether the K you are about to trust is
a measurement or an artifact.

**`n_distinct_per_split` and `modal_share`.** Kneedle is run on each of the 20
individual split curves as well as on their mean. If those 20 picks are spread
across many values and no single K holds a majority, "the elbow" is not
identified and the headline must say so. A `modal_share` near 1.0 with two or
three distinct values is a real knee; a share near 0.2 across eight values is
noise with a number attached.

**`k_marginal_gain` and `k_max_curvature`.** The other two rules, reported but
never selecting. Agreement with `k` is reassurance. Disagreement is a finding
about the curve's shape, not a reason to pick whichever you prefer — the
kneedle pick is pre-registered and stays.

**`stability_at_k`.** Mean pairwise adjusted Rand index between the assignments
produced by each split's own codebook, over the full centroid pool. This is the
categorical counterpart to EV: it asks whether the partition the features rest
on is determined by the data at that K, or is an artifact of which trials
landed in the fit. It falls as K rises by construction. What matters is whether
it is still respectable at the selected K — an EV knee sitting where stability
has already collapsed is a knee in a quantity `occ_ms` does not use.

Note this replaces the more obvious "which prototype wins changes between K and
K+1" diagnostic, which cannot be computed: prototype *indices* are not
comparable across two different k-means fits, so "the same prototype" has no
meaning between K and K+1.

**`at_grid_edge`.** A knee at K = 2 or K = 20 is a grid failure to report, not
a selection. It means the curve never bent inside the range searched.

---

## Caveats specific to this analysis

**1. Fixing K does not make this one test.** 4 mazes × 2 features × 3 variants
× 2 estimator methods = **48 cells**. Fixing K removes one degree of freedom;
it does not remove the other four. The pre-committed primary is:

> **`occupancy` / `full` / `trial_by_trial`**, with **Holm correction over the
> four mazes** — four tests, not 48.

`occupancy` over `occupancy_bin` because dwell time uses strictly more
information than visited/not. `full` because the other two variants are derived
transforms of it. `trial_by_trial` because, per `CAVEATS.md` caveat 0,
`block_means` cells scale with group size and so are not comparable between
panels, while `trial_by_trial` carries no such dependence.

**Every other cell is exploratory.** They are computed and reported because
they are cheap and because agreement across variants is informative (see
`CAVEATS.md` caveat 5), but a `p ≤ 0.05` outside the primary cell is a
hypothesis, not a result.

**2. The trial-level split inflates held-out EV, and biases K upward.** The real
dependence structure in this data is **session** — calibration, posture,
tracker drift — not trial. A trial-level split therefore leaves near-duplicate
samples on both sides, so an over-fitted codebook still "generalises" to the
same sessions. A leave-one-session-out split would measure whether prototypes
transfer across days and would generally select a *lower* K.

The trial split was chosen deliberately, because the question being asked is
"what resolution describes this maze's gaze," not "do prototypes transfer
across days." The bias direction is documented, not corrected. Any K reported
here should be read as an **upper bound** on the resolution the data supports.

**3. `p` here is not comparable to `out/comp_pooled/`'s.** Two things differ at
once, and they are not separated. The codebook here is fitted on the **in-scope
sessions only** (10 for `svm/top_ten`); `comp_pooled_features` fits label-blind
over every session that clears QC, up to 27 for Faure. So any difference in `p`
between the two trees mixes "the K choice mattered" with "the codebook's fit
pool mattered." Do not read one against the other cell-for-cell.

The scope restriction is deliberate: it makes the selected K size the same
population it is applied to. But it is a confound with respect to the older
numbers, and the comparison is simply not available.

**4. The EV curve may only flatten, never turn over.** Held-out EV for k-means
on 2-D positions has no interior maximum in practice — with these sample sizes
a codebook at K = 20 in two dimensions is nowhere near over-fitting, so the
curve climbs and gradually flattens. Kneedle finds the sharpest bend of a
flattening curve, which is a defensible and pre-registered choice, but it is
not the same thing as finding an optimum. If `n_distinct_per_split` is wide,
that is this problem showing up.

*A one-session smoke test on maze 5 had EV still climbing at K = 10 with no
visible plateau.* Whether the full ten-session curves flatten inside the 2–20
grid is the first thing to check in `ev_curves.png`, before reading any K.

**5. `no_origin` and `mean_removed` are maze-scoped.** Each maze has its own
codebook, so the dropped origin dimension and the subtracted grand mean are
recomputed per maze from that maze's own centres and its own labelled rows. No
wider fit shares these dimensions, and a narrower one would reintroduce the
per-session averaging `comp_pooled` exists to avoid.

**6. Selection and estimation use the same trials.** The 80/20 split protects
the *choice of K* from the outcome, which is what it was for. It does not hold
out data from the estimator: once K is fixed, the codebook is refitted on all of
that maze's in-scope trials and every one of them enters the permutation test.
This is correct — the selection is the thing that needed protecting — but it
means the panels are not an out-of-sample validation of the H/S effect itself.

---

## Verifying the criterion

`uv run python -m eye_pre_flash.maze_strategy_pairs.elbow_checks` — synthetic
data, no cache needed, and the only part of this analysis that runs on a laptop.
It pins: `EV(1) = 0` exactly at the test mean; EV non-decreasing up to a known
blob count; an unreachable codebook scoring `EV = 0` rather than a large finite
value; the gate being inert at a wide radius; **distance shrinkage moving EV
while leaving the assignment unchanged** (the trap the metric exists to bound);
kneedle recovering a planted knee, refusing a flat curve, and ignoring a
split-noise dip; the split being a trial-level partition with both sides
non-empty; and the per-trial seed being identity-based and stable across
processes.
