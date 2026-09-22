"""Choosing K per maze from held-out data, without ever consulting a label.

`comp_pooled_build` sweeps K and reports the K that minimises `p`. That selects
the codebook size *on the outcome statistic*, so the reported `p` is a minimum
over 13 correlated tests and means much less than it looks like it does. This
module replaces that with a criterion the H/S labels never enter: split a maze's
trials into train and test, fit the codebook on train, score how well it
reconstructs held-out data, and take the elbow. K is then fixed before
`estimator.estimate` is called even once.

The pool is built here rather than read from a cache because no cache has it.
`comp_pooled_features` pools per-trial samples into a local dict, fits the
k-means, and keeps only the centres and a sample count -- the raw samples are
gone, and the pooling loop is inlined rather than factored out. This module
re-expresses that loop with two deliberate differences, both forced by what the
elbow needs:

* **Scope-restricted.** `comp_pooled_features` pools label-blind over every
  session that clears QC (up to 27 for Faure). Here the pool is restricted to
  the in-scope sessions the estimator actually analyses, so the K that gets
  selected sizes the same population it is applied to.
* **Trial provenance retained**, so the split can happen at the trial level.
  Samples within one trial are heavily dependent; splitting them individually
  would leak train into test.

## What the criterion scores, and why it is not plain SSE

`occ_ms` depends on the codebook only through a **categorical** outcome.
`assign_fixation_states` reduces each I-DT fixation to the mean of its valid
samples, and `assign_states_inference` maps that centroid to the nearest
prototype *within* `ASSIGN_RADIUS`, or to `-1` -- and a `-1` fixation is dropped
from `occ_ms` entirely. Once a centroid is already the nearest-in-radius pick,
moving it closer to its prototype changes `occ_ms` not at all.

A plain sum-of-squared-distances EV is therefore dominated by the one effect the
feature is blind to. Two corrections follow:

1. **Score fixation centroids, not raw samples.** The k-means fit is unchanged
   (still on samples, exactly as every other codebook in the repo), but the
   held-out score is computed on the centroids assignment actually consumes.
   This is a small correction: measured on one Faure session, within-fixation
   variance is 0.15% of total raw-sample variance, so the two curves pick the
   same K almost always. Taken because it is free and marginally more correct.
2. **Mirror the radius gate.** A held-out centroid with no prototype in range
   contributes its full deviation from the test mean -- explained by nothing --
   rather than a merely large distance. `assign_states_inference` itself does
   the gating, so the metric and the pipeline cannot drift apart.

Because distance is a continuous proxy for a categorical decision even after
those corrections, the EV curve ships alongside a **stability** diagnostic: how
reproducible the partition itself is at each K. See `stability_curve`.

Nothing here is cached. The pool is consumed only by this module, and a format
written once and read once is not worth designing.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from data.attractor import (
    ASSIGN_RADIUS,
    MAZE_SCREEN_LIM,
    assign_states_inference,
)
from data.builder import trial_qc_ok
from data.config import PRE_FIX_END_MS, PRE_FIX_START_MS
from data.loader import load_attractor_eye_data, load_eye_behavioral_data
from data.occupancy import behavioral_lookup
from eye_pre_flash.classifier.features import (
    N_MAZES,
    POOL_MAX,
    POOL_PER_TRIAL,
    SEED,
    _event_lookup,
    _trial_arrays,
    _trial_window,
    clip_fixation_spans,
)

DEFAULT_START_MS = PRE_FIX_START_MS
DEFAULT_END_MS = PRE_FIX_END_MS

# Uniform by design. The legacy sweep grid (2,3,4,5,6,8,10,12,14,16,20,25,30)
# has spacing 1,1,1,1,2,2,2,2,2,4,5,5, and every elbow rule reads the curve's
# shape: a second difference on an uneven grid is simply the wrong quantity,
# and kneedle's normalised x axis gets stretched exactly where the elbow is
# expected to sit. 20 is the top because the tail above it is flat and only the
# head carries the decision.
KS = tuple(range(2, 21))

N_SPLITS = 20
TEST_FRACTION = 0.2

# Below this many labelled-population trials a maze cannot be split at all:
# the test side would be one or two trials and the EV curve pure noise.
MIN_TRIALS_FOR_SPLIT = 20

# An EV span narrower than this over the whole grid is a flat curve, and a
# flat curve has no knee to find. Reported as "no knee" rather than argmaxed.
EV_EPS = 1e-4

# First K whose marginal EV gain falls below this is the threshold rule's pick.
# One percentage point of held-out explained variance. A threshold, not a knee,
# and named as such everywhere it is reported.
MARGINAL_GAIN = 0.01


@dataclass
class TrialPool:
    """One trial's contribution, split by what each part is for.

    `samples` feeds the k-means (subsampled, screen-clipped, matching
    `comp_pooled_features`' pool exactly). `centroids` is what the held-out
    score is computed on -- one point per clipped I-DT fixation, built with the
    same mask `assign_fixation_states` uses, which deliberately does *not*
    apply the screen-limit clip.
    """

    session: str
    trial_id: int
    samples: np.ndarray
    centroids: np.ndarray


def _trial_seed(session, trial_id):
    """A stable per-trial seed.

    Not `hash()`: Python randomises string hashing per process unless
    PYTHONHASHSEED is pinned, so a `hash`-seeded pool would differ between runs.
    CRC32 is stable, cheap, and good enough to decorrelate 50-sample draws.
    """
    return zlib.crc32(f"{session}:{int(trial_id)}".encode())


def maze_pools(
    monkey,
    *,
    keep_sessions,
    mazes,
    start_ms=DEFAULT_START_MS,
    end_ms=DEFAULT_END_MS,
):
    """``{maze: [TrialPool, ...]}`` for the in-scope sessions only.

    Steps 1-2 of the feature pipeline (clip to the window, warp to unit H) are
    already baked into the attractor cache. This adds the per-fixation centroid
    that `comp_pooled_features` computes and then throws away.

    Each trial is subsampled with its **own** RNG, seeded from (session,
    trial_id). `comp_pooled_features` shares one RNG across the whole loop,
    which makes trial *i*'s draw depend on how many trials preceded it -- so a
    scope-restricted pool would not be a subset of the full-pool draws, and
    would not reproduce if the session set changed. Per-trial seeding makes the
    pool invariant to iteration order and to scope.
    """
    attractor = load_attractor_eye_data(monkey)
    behavioral = load_eye_behavioral_data(monkey)
    beh = behavioral_lookup(behavioral)
    sessions = np.asarray(attractor["session"]).astype(str)
    trials = np.asarray(attractor["trial_indices_all"]).astype(int)
    events = _event_lookup(monkey, set(sessions.tolist()))

    keep = set(str(s) for s in keep_sessions)
    wanted = set(int(m) for m in mazes)
    pools = {m: [] for m in sorted(wanted)}

    for i in range(sessions.size):
        session, trial_id = sessions[i], int(trials[i])
        if session not in keep:
            continue
        beh_i = beh.get((session, trial_id))
        if not trial_qc_ok(behavioral, beh_i):
            continue
        maze = int(behavioral["geo_type"][beh_i])
        if not 1 <= maze <= N_MAZES or maze not in wanted:
            continue
        bounds = _trial_window(behavioral, beh_i, start_ms, end_ms)
        if bounds is None:
            continue
        lo, hi = bounds

        t_ms, valid, x, y = _trial_arrays(attractor, i)
        if t_ms.size < 2:
            continue

        in_window = np.isfinite(t_ms) & (t_ms >= lo) & (t_ms <= hi)
        finite = np.isfinite(x) & np.isfinite(y)

        # The k-means pool: screen-clipped, exactly `comp_pooled_features`.
        keep_u = in_window & valid & finite
        keep_u &= (np.abs(x) <= MAZE_SCREEN_LIM) & (np.abs(y) <= MAZE_SCREEN_LIM)
        if keep_u.sum() < 2:
            continue
        idx = np.flatnonzero(keep_u)
        if idx.size > POOL_PER_TRIAL:
            rng = np.random.default_rng(_trial_seed(session, trial_id))
            idx = rng.choice(idx, POOL_PER_TRIAL, replace=False)
        samples = np.stack([x[idx], y[idx]], axis=1).astype(np.float32)

        # The score targets: one centroid per clipped fixation, built on the
        # same `valid_win` mask `assign_fixation_states` is handed -- no
        # screen-limit clip, because the assignment step does not apply one.
        valid_win = valid & in_window & finite
        centroids = []
        for _name, o0, o1 in clip_fixation_spans(
            events.get((session, trial_id), ()), lo, hi
        ):
            hit = valid_win & (t_ms >= o0) & (t_ms <= o1)
            if not hit.any():
                continue
            centroids.append([x[hit].mean(), y[hit].mean()])
        if not centroids:
            continue

        pools[maze].append(
            TrialPool(
                session=session,
                trial_id=trial_id,
                samples=samples,
                centroids=np.asarray(centroids, dtype=np.float32),
            )
        )

    for maze, trial_pools in pools.items():
        n_pool = sum(t.samples.shape[0] for t in trial_pools)
        # `comp_pooled_features` caps the pool at POOL_MAX and subsamples above
        # it. Measured counts are ~29-55k for ten sessions, so the cap is not
        # expected to bind; it is deliberately not applied here, because a
        # random cap would differ between the train side and the full fit and
        # make the swept curve describe a different pool than the final
        # codebook. Warn rather than assert, so a larger session set degrades
        # loudly instead of failing.
        if n_pool > POOL_MAX:
            print(
                f"  WARNING maze {maze}: pool is {n_pool} samples, above "
                f"POOL_MAX={POOL_MAX}. `comp_pooled_features` would subsample "
                f"here; this module does not, so the selected K sizes a larger "
                f"pool than the final codebook is fitted on."
            )

    return pools


def fit_codebook(samples, k, *, seed=SEED):
    """K-means over pooled samples. `None` when there is less data than K."""
    xy = np.asarray(samples, dtype=np.float32)
    if xy.shape[0] < k:
        return None
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    km.fit(xy)
    centers = np.clip(km.cluster_centers_, -MAZE_SCREEN_LIM, MAZE_SCREEN_LIM)
    return centers.astype(np.float32)


def gated_ev(centroids, codebook, *, radius=ASSIGN_RADIUS):
    """``(ev, assigned_fraction, sse_per_n)`` for held-out centroids.

    `TSS` is the deviation from the **test** mean, which makes it the exact
    K = 1 case of `SSE` -- one prototype, placed at the test centroid -- so
    `EV(1) == 0` identically and `EV` reads as a real R-squared on the held-out
    sample. The train mean would add a term constant in K: harmless for a
    min-max-normalised kneedle, but it rescales every marginal gain by an
    arbitrary split-dependent factor and so corrupts the threshold rule.

    An unassigned centroid (nothing within `radius`) contributes its full
    deviation from the test mean rather than its distance to the nearest
    out-of-range prototype. That is what `occ_ms` does with it: drops it. The
    gate comes from `assign_states_inference` itself so the two cannot diverge.
    """
    c = np.asarray(centroids, dtype=np.float32)
    mean = c.mean(axis=0)
    dev2 = ((c - mean) ** 2).sum(axis=1)
    tss = float(dev2.sum())

    state = assign_states_inference(c, codebook, radius=radius)
    hit = state >= 0
    sse_vec = dev2.copy()
    if hit.any():
        picked = np.asarray(codebook, dtype=np.float32)[state[hit]]
        sse_vec[hit] = ((c[hit] - picked) ** 2).sum(axis=1)
    sse = float(sse_vec.sum())

    ev = 0.0 if tss <= 0 else 1.0 - sse / tss
    return ev, float(hit.mean()), sse / max(c.shape[0], 1)


def split_trials(n, rng, *, test_fraction=TEST_FRACTION):
    """``(train_idx, test_idx)`` at the trial level, disjoint and complete."""
    order = rng.permutation(n)
    n_test = max(1, int(round(n * test_fraction)))
    return order[n_test:], order[:n_test]


def ev_curve(trial_pools, ks=KS, *, n_splits=N_SPLITS, seed=0, with_stability=True):
    """Held-out EV and partition stability against K, in one pass.

    Both curves come from the *same* codebooks. That is not just an
    optimisation: computing them separately would fit every (split, K) twice
    with identical seeds, doubling the only expensive part of the selection,
    and would let the two diagnostics drift apart under a later edit.

    `ev` is the continuous score. `stability` is its categorical counterpart:
    each split's codebook is used to assign the **full** centroid pool, and the
    mean pairwise adjusted Rand index between those label vectors says whether
    the partition `occ_ms` actually rests on is determined by the data at that
    K, or is an artifact of which trials landed in the fit. Unassigned
    centroids keep `-1` and form their own label -- correct, not a fudge, since
    "dropped from `occ_ms`" is a real category, and a K that drops a
    reproducible set of fixations is genuinely more stable than one that drops
    a different set every time.

    Stability replaces the "which prototype wins changes between K and K+1"
    diagnostic the obvious framing suggests, which cannot be computed at all:
    prototype *indices* are not comparable across two different k-means fits,
    so "the same prototype" has no meaning between K and K+1.

    The per-split EV matrix comes back too, because running the elbow rule on
    each split separately is what says whether the knee is identified rather
    than an artifact of one lucky partition.

    K is the outer loop so only `n_splits` label vectors are held at a time.
    """
    n = len(trial_pools)
    if n < MIN_TRIALS_FOR_SPLIT:
        return None, f"{n} trials; need >= {MIN_TRIALS_FOR_SPLIT} to split"

    ks = tuple(int(k) for k in ks)
    ev = np.full((n_splits, len(ks)), np.nan)
    frac = np.full((n_splits, len(ks)), np.nan)
    sse_n = np.full((n_splits, len(ks)), np.nan)
    stability = np.full(len(ks), np.nan)

    # The splits are drawn once and reused at every K, so a difference between
    # two columns is the codebook size and nothing else.
    splits = [split_trials(n, np.random.default_rng((seed, r))) for r in range(n_splits)]
    train_pools = [
        np.concatenate([trial_pools[i].samples for i in tr]) for tr, _te in splits
    ]
    test_centroids = [
        np.concatenate([trial_pools[i].centroids for i in te]) for _tr, te in splits
    ]
    all_c = np.concatenate([t.centroids for t in trial_pools]) if with_stability else None

    for j, k in enumerate(ks):
        labels = []
        for r in range(n_splits):
            codebook = fit_codebook(train_pools[r], k, seed=seed + r)
            if codebook is None:
                continue
            ev[r, j], frac[r, j], sse_n[r, j] = gated_ev(test_centroids[r], codebook)
            if with_stability:
                labels.append(assign_states_inference(all_c, codebook))
        if with_stability and len(labels) >= 2:
            stability[j] = float(
                np.mean([
                    adjusted_rand_score(labels[a], labels[b])
                    for a in range(len(labels))
                    for b in range(a + 1, len(labels))
                ])
            )

    if not np.isfinite(ev).any():
        return None, "no split produced a fittable codebook"

    return {
        "ks": np.asarray(ks, dtype=int),
        "ev": np.nanmean(ev, axis=0),
        "ev_sd": np.nanstd(ev, axis=0),
        "ev_per_split": ev,
        "assigned_fraction": np.nanmean(frac, axis=0),
        "sse_per_n": np.nanmean(sse_n, axis=0),
        "stability": stability if with_stability else None,
    }, ""


def kneedle(ks, ev):
    """The knee, or `None` on a curve too flat to have one.

    On a uniform grid this needs no `kneed` dependency and no S parameter. Once
    both axes are min-max normalised, the curve runs from (0,0) to (1,1), so
    the chord is the 45-degree line and perpendicular distance to it is
    `(y - x) / sqrt(2)` -- maximised wherever `y - x` is.

    EV is made non-decreasing first. A dip from split noise would otherwise
    move the normalisation floor and can put the argmax on the recovery rather
    than the bend. `np.argmax` returns the first maximum, so ties break to the
    smaller K, which is the conservative direction.
    """
    ks = np.asarray(ks, dtype=int)
    y = np.maximum.accumulate(np.asarray(ev, dtype=float))
    span = float(y.max() - y.min())
    if not np.isfinite(span) or span < EV_EPS:
        return None
    x = (ks - ks[0]) / (ks[-1] - ks[0])
    yn = (y - y.min()) / span
    return int(ks[int(np.argmax(yn - x))])


def marginal_gain_k(ks, ev, *, threshold=MARGINAL_GAIN):
    """First K whose gain over K-1 falls below `threshold`.

    A threshold, not a knee, and reported under that name. `threshold` is in
    absolute units of held-out explained variance, so 0.01 means "one
    percentage point of the held-out dispersion". Absolute rather than relative
    because the test-mean denominator already puts every maze's curve on a
    comparable 0-to-1 scale.
    """
    ks = np.asarray(ks, dtype=int)
    ev = np.asarray(ev, dtype=float)
    gains = np.diff(ev)
    below = np.flatnonzero(gains < threshold)
    if below.size == 0:
        return None
    return int(ks[below[0] + 1])


def max_curvature_k(ks, ev):
    """K at the sharpest bend, by discrete second difference.

    Valid only because the grid is uniform -- `ev[i-1] - 2 ev[i] + ev[i+1]` is
    a curvature estimate at equal spacing and nothing else otherwise. An
    increasing concave curve has negative second differences, so the sharpest
    bend is the most negative one.
    """
    ks = np.asarray(ks, dtype=int)
    ev = np.asarray(ev, dtype=float)
    if ev.size < 3:
        return None
    d2 = ev[:-2] - 2.0 * ev[1:-1] + ev[2:]
    if not np.isfinite(d2).any():
        return None
    return int(ks[1 + int(np.nanargmin(d2))])


def select_k(curve):
    """All three rules on one EV curve, plus the identifiability diagnostics.

    `k` is the kneedle pick and is the only one that selects; the other two are
    reported so a reader can see whether the choice is rule-dependent.
    `k_per_split` runs kneedle on each split's own curve -- if its mode is
    spread across several values, "the elbow" is not a measurement, and the
    headline has to say so.
    """
    ks, ev = curve["ks"], curve["ev"]
    knee = kneedle(ks, ev)

    per_split = [
        kneedle(ks, row) for row in curve["ev_per_split"] if np.isfinite(row).all()
    ]
    per_split = [k for k in per_split if k is not None]
    if per_split:
        values, counts = np.unique(np.asarray(per_split), return_counts=True)
        modal_k = int(values[int(np.argmax(counts))])
        modal_share = float(counts.max() / counts.sum())
        spread = int(values.size)
    else:
        modal_k, modal_share, spread = None, float("nan"), 0

    at_edge = knee is not None and knee in (int(ks[0]), int(ks[-1]))
    return {
        "k": knee,
        "k_marginal_gain": marginal_gain_k(ks, ev),
        "k_max_curvature": max_curvature_k(ks, ev),
        "k_modal_per_split": modal_k,
        "modal_share": modal_share,
        "n_distinct_per_split": spread,
        "at_grid_edge": at_edge,
        "ev_at_k": (
            float(ev[int(np.flatnonzero(ks == knee)[0])]) if knee is not None else float("nan")
        ),
    }
