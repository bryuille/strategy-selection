# Reading these panels

Each figure is one (monkey, maze, feature, variant, label source), with K = 6
and K = 12 side by side. Within one maze, trials are labelled H (hierarchical)
or S (sequential) by a decoder applied to neural population activity. Maze
geometry is held fixed, so a difference cannot be a visual confound.

**The estimator.** Pool every trial of that maze from every in-scope session.
Cut them into four disjoint groups of equal size `m = min(n_H, n_S) // 2` —
two from H, two from S — average each group into one vector, and take the
correlation between those two means. Average over 100 fresh random groupings.
Output lives under `out/<source>/`.

* **Diagonal** (`HH`, `SS`) — how similar the two group means of that strategy
  are.
* **Off-diagonal** (`HS`, `SH`) — how similar an H group mean and an S group
  mean are. These are two draws of one quantity, not two directions of an
  axis, and they agree to about three decimals.
* **`n = …`** under each diagonal cell — how many trials of that strategy
  the groups were drawn from, pooled over every session in scope. **No trial
  is excluded**, so this is identical across both K and all three variants.
  Each cell is one correlation of two means of `m = min(n_H, n_S) // 2`
  trials, averaged over the rounds; `m` is what sets the precision and lives
  in `results.csv`.
* **Δ** = `0.5·(r_HH + r_SS) − 0.5·(r_HS + r_SH)`.
* **z** = `(Δ − null_mean) / null_sd`, the null being 1000 reshuffles of the
  H/S labels **within each session**.
* **p** — two-sided fraction of shuffles at least as extreme. A leading `~`
  means no shuffle beat the observed gap, so `p` is at the floor
  `1/(n_perm+1)`: "at least this significant", not a measurement. `z` says by
  how much.

**Colour is a within-figure read only.** The scale spans each figure's own
min-to-max, so the diagonal-vs-off-diagonal gap stays visible. A noise panel
therefore gets the same visual contrast as a strong one. Judge strength by
`z`, never by how colourful the panel looks. `--vmax 1.0` forces a fixed scale
if panels need comparing by eye.

---

## Caveats

**0a. Zero-variance trials stay in the pool.**
A trial with no variance — no gaze in the window, or under `no_origin` one
that never looked away from centre — still enters its group's mean. It is
counted in `n_degenerate` and in `n`, and it is not dropped.

Excluding such trials would be worse on two counts. It would make the trial
set depend on K and variant — `no_origin` strands ~15% of trials, at different
rates per K — so no two panels would rest on the same data. And the stranded
trials are not a random slice: centre-only trials are about 1.4x more common
in S than in H (Faure 12.6% vs 8.8%, Nielsen 18.9% vs 12.0%), so dropping them
removes a strategy-correlated subset of the data.

**0. Raw cells rise with group size. Compare `z`, not `r`.**
Averaging `m` trials suppresses their independent noise *before* the
correlation is taken, so the diagonal climbs toward 1 as `m` grows. Real `m`
runs from 5 (Faure/dendro maze 6) to 175 (Nielsen/svm maze 4), so a high
diagonal may mean nothing but a bigger pool. `z` stays comparable, because
each panel's null is built at that panel's own `m`.

**1. Pooling mixes within-session and between-session trial pairs.** Two trials
from the same day are more similar than two trials from different days —
calibration, arousal, posture. Sessions differ in H/S balance, so the
within-strategy cells carry more same-session pairs than the cross-strategy
cell does, which lifts the diagonal on its own.

*This one is controlled.* The null shuffles labels **within** each session, so
every session keeps its H and S counts and the shuffled pool carries the
same-session enrichment in exactly the same proportion. The null reproduces the
artifact, so `z` and `p` are calibrated against it. **Raw Δ is not.** Read `z`.

**2. Unequal H/S noise inflates Δ, and the null cannot see it.** If one
strategy's trials are intrinsically noisier, `r_HH ≠ r_SS`, and the
cross-strategy term sits near their *geometric* mean while Δ subtracts it from
the *arithmetic* mean. That is positive even when both strategies share one
identical underlying gaze pattern. Shuffling makes both pseudo-groups equally
noisy, so the null sits at ~0 while the data does not.

**Look at the two diagonal cells before believing a Δ.** The failure looks like
`r_HH = 0.31`, `r_SS = 0.37`, `r_HS = 0.33` — where the off-diagonal is *above*
one of the diagonals and Δ is positive only because the other diagonal is high.
That is not "H and S differ"; it is "S trials are more stereotyped than H
trials". This pattern is common in these panels, so check it every time.

**3. Intrasession drift is not controlled.** If the animal switched strategy
partway through a session, H and S occupy different parts of session time, and
eye-tracker drift alone will make within-strategy pairs more similar. The
shuffle is blind to trial order, so a Δ driven entirely by this can still clear
the null.

**4. Δ is not comparable across features, variants, K, or group size.**
Different transforms and dimensionalities live on different scales, and the
correlation of group means rises with `m` (caveat 0). `z` is the cross-panel
number, being normalised by each panel's own chance variability.

**5. `mean_removed` pushes correlations negative by arithmetic, not biology.**
Subtracting the pooled grand-mean profile leaves residuals that sum to ~zero,
and vectors summing to zero have negative mean pairwise correlation. Absolute
`r` is uninterpretable under this variant; only the diagonal-vs-off-diagonal
gap survives. Read `no_origin` — a structural rather than fitted way of
stripping the same common profile — alongside it. Agreement means the finding
is robust; if only the fitted one shows an effect, distrust it.

**6. Rounds are not independent.** Each trial is redrawn into many groupings.
That is fine for a permutation test, which never assumes independence, but do
not compute a parametric standard error from the round count.

**7. The SVM source's maze-1 S and maze-6 H cells are near-empty by
construction.** That classifier is trained on mazes 1 vs 6, so those cells are
artifacts of its own training axis rather than measurements. They clear the
10-trial rule on pooled counts and so get panels; treat `svm` mazes 1 and 6
with more suspicion than the rest.

**8. Monte Carlo noise at the default 100 rounds is about ±0.001 on Δ.**
Measured by re-running with relabelled groups. Δ is printed to three decimals,
so the last digit is not stable. `z` is less affected, because the null is
computed with the same number of rounds and most of the noise cancels. Raise
`--n-rounds` if a Δ near zero matters.

**9. Scope and multiplicity.** One maze and one monkey per figure — a result
says nothing about the others. The two K panels are the same trials at two
codebook resolutions, not independent replications. `dendro/publication` and
`svm/top_ten` are different session pools *and* a different label source;
agreement between them is informative, but neither is a rerun of the other.
And every panel is its own test, so `p ≤ 0.05` on one of many carries the usual
multiple-comparisons caveat.

---

## Verifying the estimator

`uv run python -m eye_pre_flash.maze_strategy_pairs.checks` — synthetic data,
no cache needed. It pins: the four groups are disjoint and exactly `m` (so no
trial is ever correlated with itself); independent trials give a flat 2×2;
relabelling H↔S moves Δ only by sampling noise, which shrinks as
`1/sqrt(n_rounds)` rather than sitting at a fixed offset; the shuffle preserves
each session's counts; a null case gives `|z| < 3` even with session offsets
present; a planted H/S difference is detected; the diagonal rises with `m`;
zero-variance trials stay in the pool; and the coverage rule admits exactly
`n_H ≥ 10 and n_S ≥ 10`.
