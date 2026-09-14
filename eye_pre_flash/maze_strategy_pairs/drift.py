"""Intrasession-drift controls for the H/S 2x2.

The label-shuffle permutation null in `pair.permutation_null` does not control
for drift, and the reason is structural rather than incidental. Shuffling H/S
labels without regard to trial order **destroys any real time-clustering of
the labels before the null is even built**, so the null distribution is
assembled almost entirely from time-balanced pseudo-groups. A delta driven
entirely by "H and S occupy different parts of the session, and the gaze
signal drifts slowly across it" can still land far outside that null and look
significant. The permutation test answers *"is this delta unusual for any
random 2-way split of this size?"*, not *"is this delta explained by H/S's
actual temporal positions?"*

The asymmetry it misses is specific: both halves of the H cell are random
subsets of H's own trials, so they span the same session time by construction.
H-vs-S does not, if the animal switched strategy partway through. Eye-tracker
calibration drift then inflates the diagonal for reasons that have nothing to
do with strategy.

Three escalating checks:

1. `label_time_association` -- is the confound even plausible in this session?
2. `delta_time` -- run the *same* estimator on a split defined by time alone.
3. `detrend_pools` -- regress trial index out and recompute.

**A rejected alternative, recorded so it is not re-proposed.** Stratifying
`pair.build_split_plan` to sample each half evenly across session time was
considered and rejected. Two random halves of one cell's own trials are
already temporally indistinguishable in expectation, so stratifying the split
reduces the split-to-split variance of the *diagonal* but cannot fix a
between-cell temporal mismatch: if H's whole trial pool occupies different
session time than S's, there is no "late half of H" to sample. The confound
lives at the H-vs-S comparison level and has to be addressed there.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import stats

from eye_pre_flash.maze_strategy_pairs.cells import STRATEGIES
from eye_pre_flash.maze_strategy_pairs.pair import (
    H_CELL,
    N_SPLITS,
    R_CLIP,
    S_CELL,
    build_plans,
    pair_delta,
    pair_matrix_from_pools,
    pair_table,
)

# `early_late` detects monotonic drift. `odd_even` is temporally interleaved,
# so it detects cyclic structure -- and, more usefully, it measures **the
# estimator's own noise floor on real data** at this m and d with two groups
# that are genuinely exchangeable. It should sit at ~0; if it does not, the
# floor is not where the permutation null says it is.
TIME_SCHEMES = ("early_late", "odd_even")

# Appended to the pool's RNG key so the two schemes draw independent splits.
# Without this, a session whose median and parity splits happen to give the
# same group sizes would get a byte-identical split plan for both, correlating
# the two controls through their draws rather than leaving them independent.
SCHEME_SEED_TAG = {"early_late": 31, "odd_even": 37}


def label_time_association(pool):
    """Is H/S associated with time-in-session for this (session, maze)?

    Two views of the same question, reported per session so the drift concern
    is auditable rather than assumed either way:

    * Mann-Whitney U on the two groups' trial indices -- distribution-free, no
      assumption that drift is linear.
    * Point-biserial r of the label against trial index -- a signed effect
      size, so the direction (H early or S early) is visible.

    If H and S are well interleaved in time in most sessions, `delta_time` and
    `detrend_pools` are confirmatory. If strategy shifts systematically within
    a block, they are load-bearing.
    """
    t = np.asarray(pool.trial_index, dtype=float)
    y = np.asarray(pool.y, dtype=int)
    idx_h, idx_s = t[y == H_CELL], t[y == S_CELL]
    out = dict(
        session=pool.session,
        n_H=int(idx_h.size),
        n_S=int(idx_s.size),
        t_median_H=float(np.median(idx_h)) if idx_h.size else np.nan,
        t_median_S=float(np.median(idx_s)) if idx_s.size else np.nan,
        mw_U=np.nan,
        mw_p=np.nan,
        r_pb=np.nan,
        r_pb_p=np.nan,
    )
    if idx_h.size and idx_s.size:
        u, p = stats.mannwhitneyu(idx_h, idx_s, alternative="two-sided")
        out["mw_U"], out["mw_p"] = float(u), float(p)
    # Point-biserial needs both classes present and trial index non-constant.
    if np.unique(y).size == 2 and np.unique(t).size > 1:
        r_pb, p_pb = stats.pointbiserialr(y, t)
        out["r_pb"], out["r_pb_p"] = float(r_pb), float(p_pb)
    return out


def association_summary(pools):
    """Aggregate `label_time_association` over sessions, for the figure footer."""
    rows = [label_time_association(p) for p in pools]
    r_pb = np.array([r["r_pb"] for r in rows], dtype=float)
    p_mw = np.array([r["mw_p"] for r in rows], dtype=float)
    finite = np.isfinite(r_pb)
    return rows, dict(
        r_pb_median_abs=float(np.median(np.abs(r_pb[finite]))) if finite.any() else np.nan,
        r_pb_max_abs=float(np.max(np.abs(r_pb[finite]))) if finite.any() else np.nan,
        n_sessions_time_assoc=int(np.sum(np.isfinite(p_mw) & (p_mw < 0.05))),
        n_sessions_tested=int(len(rows)),
    )


def time_labels(pool, scheme):
    """A two-group split of this pool's trials from **time alone**.

    Nothing about strategy enters. `early_late` is a median split on trial
    index; `odd_even` is trial index parity.
    """
    t = np.asarray(pool.trial_index, dtype=float)
    if scheme == "early_late":
        return (t >= np.median(t)).astype(int)
    if scheme == "odd_even":
        return (np.asarray(pool.trial_index, dtype=int) % 2).astype(int)
    raise ValueError(f"unknown scheme {scheme!r}; choose from {TIME_SCHEMES}")


def surrogate_pools(pools, scheme):
    """`pools` relabelled by time, **holding `m` at its H/S value**.

    This is the detail that decides whether the control means anything. A
    median or parity split yields two near-equal groups, so the surrogate's
    own ``min(n_0, n_1) // 2`` would typically *exceed* the H/S `m`, which the
    minority strategy caps. A larger `m` raises every correlation, and
    `delta_time` would come out inflated for a reason having nothing to do
    with drift. Forcing the H/S `m` is the same match-the-counts discipline the
    main estimator applies between cells, applied here between the analysis and
    its control.

    For `early_late` the forced size always fits: with ``n = n_H + n_S`` and
    ``m = min(n_H, n_S) // 2``, the smaller median group has
    ``floor(n / 2) >= min(n_H, n_S) >= 2 * m`` trials. For `odd_even` the
    groups can in principle skew, so a session that cannot supply ``2 * m`` in
    both parity groups is dropped and counted.

    Returns ``(surrogates, dropped)``.
    """
    surrogates, dropped = [], {}
    for pool in pools:
        y_t = time_labels(pool, scheme)
        counts = {s: int(np.sum(y_t == s)) for s in STRATEGIES}
        if min(counts.values()) < 2 * pool.m:
            dropped[pool.session] = (
                f"{scheme}: smaller group has {min(counts.values())} trials, "
                f"needs 2*m = {2 * pool.m}"
            )
            continue
        surrogates.append(
            replace(
                pool,
                y=y_t,
                counts=counts,
                rng_seed=tuple(pool.rng_seed) + (SCHEME_SEED_TAG[scheme],),
            )
        )
    return surrogates, dropped


def delta_time(
    pools,
    scheme,
    *,
    maze,
    variant="full",
    n_splits=N_SPLITS,
    r_clip=R_CLIP,
    **kw,
):
    """Delta for a time-defined split, through the identical estimator.

    Same `pair_matrix_from_pools`, same equal-`m` rule, same Fisher-z
    aggregation, same `pair_delta` -- only the grouping differs. That is the
    whole point: a number produced by a different code path would not be
    comparable to `delta_obs`.

    How to read it:

    * ``delta_time ~= 0`` -- drift is not a meaningful force at these half
      sizes and windows; the observed H/S delta stands on its own.
    * ``delta_time`` comparable to the observed H/S delta -- **the observed
      result cannot be read as strategy-specific** without `detrend_pools`.

    Returns ``(delta, info)``.
    """
    surrogates, dropped = surrogate_pools(pools, scheme)
    info = dict(
        scheme=scheme,
        n_sessions=len(surrogates),
        n_sessions_dropped=len(dropped),
        dropped=dropped,
    )
    if not surrogates:
        return np.nan, info

    plans = build_plans(surrogates, n_splits=n_splits)
    res = pair_matrix_from_pools(
        surrogates, plans, maze=maze, variant=variant, r_clip=r_clip, **kw
    )
    r_00, r_11, r_01, n_sess = pair_table(res, r_clip=r_clip)
    info.update(r_early_early=r_00, r_late_late=r_11, r_cross=r_01, n_used=n_sess)
    return pair_delta(r_00, r_11, r_01), info


def detrend_pools(pools, *, order=1):
    """Regress trial index out of each pool's features, within (session, maze).

    Least squares of every feature dimension on ``[1, t, t^2, ...]`` up to
    `order`, keeping the residuals. Run this and recompute both the H/S delta
    and `delta_time` on the same residuals; the decisive pattern is **the H/S
    delta surviving while `delta_time` collapses toward zero**, which says the
    effect is strategy-specific rather than a drift artifact.

    Two caveats that matter when reading a detrended panel:

    * The design includes an intercept, so this also removes each session's
      own mean profile -- it is partly a per-session `mean_removed` and
      interacts with the variant axis rather than being orthogonal to it.
    * It consumes ``order + 1`` degrees of freedom per dimension, which is not
      negligible at d = 5 (occupancy / `no_origin` / k = 6).

    The scope is (session, maze), matching where the confound lives: the
    question is whether H and S occupy different time within *this maze's*
    trials.
    """
    out = []
    for pool in pools:
        t = np.asarray(pool.trial_index, dtype=float)
        # Centre and scale t so the Vandermonde stays well conditioned at
        # order > 1; the fit is unchanged, only the basis.
        span = t.max() - t.min()
        tn = (t - t.mean()) / (span if span > 0 else 1.0)
        design = np.vander(tn, order + 1, increasing=True)
        coef, *_ = np.linalg.lstsq(design, pool.X, rcond=None)
        out.append(replace(pool, X=pool.X - design @ coef))
    return out
