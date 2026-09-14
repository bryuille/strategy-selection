"""Invariant checks for the 2x2 estimator. ``python -m ...maze_strategy_pairs.checks``

There is no output regression to run against: the rewrite changed the half-size
rule, the NaN policy, the symmetrisation, the averaging rule and the RNG
consumption order, so no number is bit-comparable to a previous run. These
checks test the *properties* the analysis depends on instead, on synthetic data
that needs no feature cache.

The load-bearing one is `check_relabel_exchangeable`: the claim this whole
package makes is that the diagonal exceeds the off-diagonal for reasons of
biology rather than of procedure, and that only holds if the estimator has no
built-in preference for the diagonal over the off-diagonal, or for H over S.
"""

from __future__ import annotations

import sys

import numpy as np

from eye_pre_flash.maze_strategy_pairs import drift
from eye_pre_flash.maze_strategy_pairs.matrix import pairwise_pearson_matrix
from eye_pre_flash.maze_strategy_pairs.pair import (
    H_CELL,
    S_CELL,
    SessionPool,
    SplitPlan,
    amgm_floor,
    build_split_plan,
    cross_r,
    matrix_delta,
    pearson_batch_2x2,
    session_pair_matrix,
    split_deltas,
)

PASS, FAIL = "  PASS", "  FAIL"
_failures = []


def report(name, ok, detail=""):
    print(f"{PASS if ok else FAIL}  {name}{'  — ' + detail if detail else ''}")
    if not ok:
        _failures.append(name)
    return ok


def make_pool(n_h, n_s, d, *, rng, profile_h=None, profile_s=None, noise_s=1.0,
              time_ordered=False, drift_vec=None, seed=(0, 1)):
    """A synthetic `SessionPool`: two cells, optional distinct mean profiles."""
    if profile_h is None:
        profile_h = np.zeros(d)
    if profile_s is None:
        profile_s = profile_h
    X = np.empty((n_h + n_s, d))
    X[:n_h] = profile_h + rng.normal(size=(n_h, d))
    X[n_h:] = profile_s + noise_s * rng.normal(size=(n_s, d))
    y = np.concatenate([np.zeros(n_h, int), np.ones(n_s, int)])

    if time_ordered:
        # H occupies the early part of the session, S the late part.
        trial_index = np.arange(1, n_h + n_s + 1)
    else:
        # Interleaved: labels carry no time information.
        trial_index = np.arange(1, n_h + n_s + 1)
        order = rng.permutation(n_h + n_s)
        X, y = X[order], y[order]

    if drift_vec is not None:
        span = trial_index.max() - trial_index.min()
        t = (trial_index - trial_index.mean()) / (span if span else 1)
        X = X + np.outer(t, drift_vec)

    return SessionPool(
        session="synthetic",
        X=X,
        y=y,
        trial_index=trial_index,
        counts={H_CELL: int(n_h), S_CELL: int(n_s)},
        m=min(n_h, n_s) // 2,
        rng_seed=seed,
    )


def symmetric_plan(pool, n_splits):
    """A plan whose two cells share one draw. Requires ``n_H == n_S``.

    Only meaningful when the counts match, and it exists so
    `check_relabel_exchangeable` can be an *exact* algebraic identity rather
    than a Monte-Carlo comparison: with one shared draw, relabelling H<->S
    swaps which rows the same weights are applied to and nothing else.
    """
    base = build_split_plan(pool, n_splits=n_splits)
    return SplitPlan(
        m=base.m,
        n_splits=base.n_splits,
        n_cell=base.n_cell,
        idx={s: base.idx[H_CELL] for s in (H_CELL, S_CELL)},
        W={s: base.W[H_CELL] for s in (H_CELL, S_CELL)},
    )


def check_relabel_exchangeable():
    """Relabelling H<->S must reverse both axes and leave delta untouched.

    With a shared split draw, ``a_i`` and ``b_j`` are the same weights applied
    to cell *i* / *j*'s rows, so swapping the two cells' rows gives
    ``R_swap[i, j] = R_orig[1-i, 1-j]`` exactly -- the diagonal entries trade
    places and so do the two cross draws. Delta averages both diagonals and
    both cross draws, so it must be *identical*, not merely close.

    A failure here means the estimator treats one cell, or the diagonal,
    differently from the other by construction -- which would make the headline
    comparison meaningless.
    """
    rng = np.random.default_rng(7)
    pool = make_pool(30, 30, 10, rng=rng, profile_h=np.arange(10) * 0.4)
    plan = symmetric_plan(pool, 64)

    orig = session_pair_matrix(pool, pool.y, plan).r
    swapped = session_pair_matrix(pool, 1 - pool.y, plan).r

    rotated = np.allclose(swapped, orig[::-1, ::-1], atol=1e-12)
    same_delta = abs(matrix_delta(orig) - matrix_delta(swapped)) < 1e-12
    return report(
        "relabel H<->S reverses both axes and preserves delta",
        rotated and same_delta,
        f"max|swap - rot(orig)| = {np.max(np.abs(swapped - orig[::-1, ::-1])):.2e}, "
        f"|Δd| = {abs(matrix_delta(orig) - matrix_delta(swapped)):.2e}",
    )


def check_zero_signal_equal_n():
    """No signal, equal counts -> delta indistinguishable from zero."""
    rng = np.random.default_rng(11)
    pool = make_pool(40, 40, 12, rng=rng)
    plan = build_split_plan(pool, n_splits=400)
    sm = session_pair_matrix(pool, pool.y, plan, keep_splits=True)
    delta = matrix_delta(sm.r)
    per_split = split_deltas([sm])
    sem = np.nanstd(per_split, ddof=1) / np.sqrt(np.isfinite(per_split).sum())
    return report(
        "zero signal, equal n -> delta ~ 0",
        abs(delta) < 4 * sem,
        f"Δ = {delta:+.4f}, 4*sem = {4 * sem:.4f}",
    )


def check_zero_signal_unequal_n():
    """No signal, 45 vs 12 trials -> delta still ~0.

    This is the check that would have caught the old unequal-half-size
    handicap: under the previous rule `r_HH` averaged 22-trial halves while
    `r_HS` correlated a 22-trial half against a 6-trial one, so delta came out
    positive on pure noise.
    """
    rng = np.random.default_rng(13)
    pool = make_pool(45, 12, 12, rng=rng)
    plan = build_split_plan(pool, n_splits=400)
    sm = session_pair_matrix(pool, pool.y, plan, keep_splits=True)
    delta = matrix_delta(sm.r)
    per_split = split_deltas([sm])
    sem = np.nanstd(per_split, ddof=1) / np.sqrt(np.isfinite(per_split).sum())
    return report(
        "zero signal, unequal n (45 vs 12) -> delta ~ 0",
        abs(delta) < 5 * sem,
        f"Δ = {delta:+.4f}, 5*sem = {5 * sem:.4f}, m = {pool.m}",
    )


def check_amgm_floor_is_measured():
    """Unequal within-cell variance -> delta > 0 with NO true difference.

    Both cells share one mean profile (rho_true = 1), so any positive delta is
    the arithmetic-vs-geometric mean gap, not a strategy effect. This turns the
    `amgm_floor` caveat into a measured property: the observed delta should sit
    near the floor computed from the two diagonals.
    """
    rng = np.random.default_rng(17)
    profile = np.sin(np.arange(12) * 0.7) * 2.0
    pool = make_pool(40, 40, 12, rng=rng, profile_h=profile, profile_s=profile, noise_s=3.0)
    plan = build_split_plan(pool, n_splits=400)
    sm = session_pair_matrix(pool, pool.y, plan, keep_splits=True)
    r_hh = sm.r[H_CELL, H_CELL]
    r_ss = sm.r[S_CELL, S_CELL]
    delta = matrix_delta(sm.r)
    floor, beyond, imbalance = amgm_floor(r_hh, r_ss, cross_r(sm.r))
    return report(
        "unequal cell variance -> delta > 0 from the AM-GM floor alone",
        delta > 0 and np.isfinite(floor) and abs(beyond) < max(0.05, 0.5 * delta),
        f"r_HH={r_hh:.3f} r_SS={r_ss:.3f} Δ={delta:.4f} "
        f"floor={floor:.4f} beyond={beyond:+.4f} imbalance={imbalance:+.3f}",
    )


def check_cross_draws_exchangeable():
    """R[H,S] and R[S,H] must have the same mean -- they are one quantity.

    If these diverged systematically, reporting them separately would be
    reporting a directional effect, which the split construction cannot
    support (which half is "a" is an arbitrary per-cell coin flip).
    """
    rng = np.random.default_rng(19)
    pool = make_pool(40, 40, 12, rng=rng, profile_h=np.arange(12) * 0.3)
    plan = build_split_plan(pool, n_splits=800)
    sm = session_pair_matrix(pool, pool.y, plan, keep_splits=True)
    ab = sm.r[H_CELL, S_CELL]
    ba = sm.r[S_CELL, H_CELL]
    z = sm.z_splits[np.isfinite(sm.z_splits).all(axis=(1, 2))]
    sem = np.std(z[:, H_CELL, S_CELL] - z[:, S_CELL, H_CELL], ddof=1) / np.sqrt(len(z))
    gap_z = abs(np.mean(z[:, H_CELL, S_CELL] - z[:, S_CELL, H_CELL]))
    return report(
        "the two cross draws are exchangeable (equal means)",
        gap_z < 4 * sem,
        f"r[H,S]={ab:+.4f} r[S,H]={ba:+.4f} |Δz|={gap_z:.4f} 4*sem={4 * sem:.4f}",
    )


def check_equal_counts():
    """Every half-mean averages exactly m trials, and the halves are disjoint."""
    rng = np.random.default_rng(23)
    pool = make_pool(45, 12, 8, rng=rng)
    plan = build_split_plan(pool, n_splits=50)
    ok = True
    detail = []
    for s in (H_CELL, S_CELL):
        for half, w in enumerate(plan.W[s]):
            ok &= np.allclose(w.sum(axis=1), 1.0)
            ok &= np.all((w > 0).sum(axis=1) == plan.m)
        idx_a, idx_b = plan.idx[s]
        overlap = max(
            len(set(a.tolist()) & set(b.tolist())) for a, b in zip(idx_a, idx_b)
        )
        ok &= overlap == 0
        detail.append(f"cell{s}: m={plan.m}, max overlap={overlap}")
    return report("equal counts and disjoint halves", ok, "; ".join(detail))


def check_plan_invariant_under_shuffle():
    """The split plan must not depend on the labels.

    This is the assumption that lets one plan serve the observed value and all
    1000 permutations -- and hence that makes null variation purely label
    variation.
    """
    rng = np.random.default_rng(29)
    pool = make_pool(30, 18, 8, rng=rng)
    a = build_split_plan(pool, n_splits=32)
    shuffled = SessionPool(
        session=pool.session, X=pool.X, y=rng.permutation(pool.y),
        trial_index=pool.trial_index, counts=pool.counts, m=pool.m,
        rng_seed=pool.rng_seed,
    )
    b = build_split_plan(shuffled, n_splits=32)
    ok = all(
        np.array_equal(a.idx[s][h], b.idx[s][h])
        for s in (H_CELL, S_CELL)
        for h in (0, 1)
    )
    return report("split plan is invariant under relabelling", ok)


def check_matmul_equals_gather():
    """The weight-matmul half-mean must equal the fancy-index gather."""
    rng = np.random.default_rng(31)
    pool = make_pool(40, 26, 14, rng=rng)
    plan = build_split_plan(pool, n_splits=40)
    worst = 0.0
    for s in (H_CELL, S_CELL):
        rows = pool.X[pool.y == s]
        for half in (0, 1):
            by_matmul = plan.W[s][half] @ rows
            by_gather = rows[plan.idx[s][half]].mean(axis=1)
            worst = max(worst, float(np.max(np.abs(by_matmul - by_gather))))
    return report(
        "weight matmul == fancy-index gather", worst < 1e-12, f"max|diff| = {worst:.2e}"
    )


def check_batched_equals_scalar_pearson():
    """`pearson_batch_2x2` must match `matrix.pairwise_pearson_matrix` exactly,
    including the zero-variance -> NaN convention."""
    rng = np.random.default_rng(37)
    A = rng.normal(size=(6, 2, 9))
    B = rng.normal(size=(6, 2, 9))
    A[2, 0, :] = 4.0  # zero variance -> NaN row
    batched = pearson_batch_2x2(A, B)
    worst, nan_ok = 0.0, True
    for t in range(A.shape[0]):
        ref = pairwise_pearson_matrix(A[t], B[t])
        nan_ok &= np.array_equal(np.isnan(ref), np.isnan(batched[t]))
        both = np.isfinite(ref) & np.isfinite(batched[t])
        if both.any():
            worst = max(worst, float(np.max(np.abs(ref[both] - batched[t][both]))))
    return report(
        "batched Pearson == scalar reference (incl. NaN convention)",
        worst < 1e-12 and nan_ok,
        f"max|diff| = {worst:.2e}, NaN pattern match = {nan_ok}",
    )


def check_all_or_nothing_splits():
    """A degenerate half-mean must drop the whole split, not one entry.

    Constant features make every half-mean zero-variance, so every split is
    dropped and the session must come back all-NaN rather than partly
    populated.
    """
    rng = np.random.default_rng(41)
    pool = make_pool(20, 20, 6, rng=rng)
    pool = SessionPool(
        session=pool.session, X=np.ones_like(pool.X), y=pool.y,
        trial_index=pool.trial_index, counts=pool.counts, m=pool.m,
        rng_seed=pool.rng_seed,
    )
    plan = build_split_plan(pool, n_splits=16)
    sm = session_pair_matrix(pool, pool.y, plan)
    ok = sm.n_splits_used == 0 and sm.n_splits_dropped == 16 and np.isnan(sm.r).all()
    return report(
        "degenerate splits are dropped whole",
        ok,
        f"used={sm.n_splits_used} dropped={sm.n_splits_dropped}",
    )


def check_drift_surrogates():
    """`delta_time` must fire on injected drift and stay ~0 without it.

    Interleaved labels with no drift: `odd_even` measures the estimator's own
    floor on real-shaped data, so it should sit at ~0. Time-ordered labels plus
    a linear drift: `early_late` should be clearly positive, and detrending
    should collapse it.
    """
    rng = np.random.default_rng(43)
    d = 12
    clean = make_pool(40, 40, d, rng=rng)
    delta_oe, _ = drift.delta_time(
        [clean], "odd_even", maze=4, n_splits=200
    )

    drifted = make_pool(
        40, 40, d, rng=rng, time_ordered=True, drift_vec=np.arange(d) * 3.0
    )
    delta_el, _ = drift.delta_time([drifted], "early_late", maze=4, n_splits=200)
    detrended = drift.detrend_pools([drifted], order=1)
    delta_el_dt, _ = drift.delta_time(detrended, "early_late", maze=4, n_splits=200)

    ok = (
        abs(delta_oe) < 0.15
        and delta_el > 0.2
        and abs(delta_el_dt) < 0.5 * delta_el
    )
    return report(
        "delta_time: ~0 when interleaved, fires on drift, collapses on detrend",
        ok,
        f"odd/even={delta_oe:+.4f}  early/late={delta_el:+.4f}  "
        f"detrended={delta_el_dt:+.4f}",
    )


def check_label_time_association():
    """The association test must separate time-ordered from interleaved labels."""
    rng = np.random.default_rng(47)
    interleaved = make_pool(40, 40, 8, rng=rng)
    ordered = make_pool(40, 40, 8, rng=rng, time_ordered=True)
    a = drift.label_time_association(interleaved)
    b = drift.label_time_association(ordered)
    ok = abs(a["r_pb"]) < 0.3 and abs(b["r_pb"]) > 0.8 and b["mw_p"] < 1e-6
    return report(
        "label/time association separates ordered from interleaved",
        ok,
        f"interleaved r_pb={a['r_pb']:+.3f}, ordered r_pb={b['r_pb']:+.3f} "
        f"(p={b['mw_p']:.1e})",
    )


CHECKS = (
    check_relabel_exchangeable,
    check_zero_signal_equal_n,
    check_zero_signal_unequal_n,
    check_amgm_floor_is_measured,
    check_cross_draws_exchangeable,
    check_equal_counts,
    check_plan_invariant_under_shuffle,
    check_matmul_equals_gather,
    check_batched_equals_scalar_pearson,
    check_all_or_nothing_splits,
    check_drift_surrogates,
    check_label_time_association,
)


def main():
    print(f"Running {len(CHECKS)} invariant checks\n")
    for fn in CHECKS:
        fn()
    print()
    if _failures:
        print(f"{len(_failures)} FAILED: {', '.join(_failures)}")
        return 1
    print(f"All {len(CHECKS)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
