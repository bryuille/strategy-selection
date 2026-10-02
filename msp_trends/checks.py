"""Synthetic checks for msp_trends; no cache needed.

    uv run python -m msp_trends.checks
"""

from __future__ import annotations

import numpy as np

from msp import estimator as msp_est
from msp_trends import estimator, features
from msp import features as msp_features
from msp_trends.config import CODEBOOKS


def _synthetic(rng, n_h, n_s, p_h, p_s, n_sessions=4):
    y = np.r_[np.zeros(n_h, int), np.ones(n_s, int)]
    p = np.where(y[:, None] == 0, p_h, p_s)
    X = (rng.random((y.size, len(p_h))) < p).astype(float)
    sess = rng.integers(0, n_sessions, y.size).astype(str)
    return X, y, sess


def check_matches_msp():
    rng = np.random.default_rng(1)
    X, y, sess = _synthetic(rng, 80, 60, [0.8, 0.9, 0.2], [0.6, 0.95, 0.3])
    run, _ = estimator.run_maze(X, y, sess, seed=7, n_rounds=20, n_perm=50)
    ref, _ = msp_est.estimate(X, y, sess, seed=7, n_rounds=20, n_perm=50)
    a, b = run.result, ref
    assert np.array_equal(a.q, b.q), (a.q, b.q)
    for f in ("delta", "z", "p", "null_mean", "null_sd", "n_degenerate"):
        assert getattr(a, f) == getattr(b, f), (f, getattr(a, f), getattr(b, f))
    print("ok  run_maze == msp.estimator.estimate")

    t = features.balanced_mean_removed
    run, _ = estimator.run_maze(X, y, sess, seed=7, n_rounds=20, n_perm=50, transform=t)
    ref, _ = msp_est.estimate(X, y, sess, seed=7, n_rounds=20, n_perm=50, transform=t)
    assert np.array_equal(run.result.q, ref.q)
    for f in ("delta", "z", "p", "null_mean", "null_sd", "n_degenerate"):
        assert getattr(run.result, f) == getattr(ref, f), f
    print("ok  run_maze == msp.estimator.estimate with a refit-per-shuffle transform")


def check_pooled_is_weighted_mean():
    rng = np.random.default_rng(2)
    runs = []
    for maze, (nh, ns) in zip((2, 3, 4), ((60, 40), (100, 15), (20, 120))):
        X, y, sess = _synthetic(rng, nh, ns, [0.8, 0.9, 0.2], [0.6, 0.95, 0.3])
        runs.append(estimator.run_maze(X, y, sess, seed=maze, n_rounds=20, n_perm=60)[0])
    w = estimator.weights_for(runs)
    pooled = estimator.pooled(runs)
    assert np.isclose(pooled.delta, sum(wi * r.result.delta for wi, r in zip(w, runs)))
    assert np.isclose(w.sum(), 1.0)
    assert pooled.n_H == sum(r.result.n_H for r in runs)
    print(f"ok  pooled Δ = weighted mean of maze Δ (w = {np.round(w, 3)})")


def check_power_and_calibration():
    rng = np.random.default_rng(3)
    planted, null = [], []
    for maze in (2, 3, 4, 5):
        X, y, s = _synthetic(rng, 150, 150, [0.8, 0.85, 0.2], [0.55, 0.95, 0.45])
        planted.append(estimator.run_maze(X, y, s, seed=maze, n_rounds=30, n_perm=200)[0])
        X, y, s = _synthetic(rng, 150, 150, [0.7, 0.9, 0.3], [0.7, 0.9, 0.3])
        null.append(estimator.run_maze(X, y, s, seed=maze, n_rounds=30, n_perm=200)[0])
    zp, zn = estimator.pooled(planted).z, estimator.pooled(null).z
    assert zp > 4, zp
    assert abs(zn) < 3, zn
    print(f"ok  planted pooled z = {zp:+.1f}, no-effect pooled z = {zn:+.2f}")


def check_tiny_strategy():
    """<= 5 trials of one strategy: undefined null draws are dropped and z stays
    finite; if the observed score itself is undefined, z and p are NaN (never
    a spurious floor p)."""
    rng = np.random.default_rng(5)
    X, y, sess = _synthetic(rng, 90, 4, [0.7, 0.8, 0.1], [0.2, 0.4, 0.0], n_sessions=2)
    X[y == 1] = [[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 1]]
    run, reason = estimator.run_maze(X, y, sess, seed=1, n_rounds=50, n_perm=300)
    assert run is not None, reason
    assert np.isfinite(run.result.z) and np.isfinite(run.result.p)
    assert estimator.low_n(run.result)
    print(f"ok  n_S = 4: z = {run.result.z:+.2f}, {run.n_null_undefined} undefined draws dropped")

    X[y == 1] = [[1, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]  # every split has a constant half
    run, _ = estimator.run_maze(X, y, sess, seed=1, n_rounds=50, n_perm=100)
    assert not np.isfinite(run.result.delta)
    assert np.isnan(run.result.z) and np.isnan(run.result.p) and not run.result.p_at_floor
    print("ok  undefined observed score -> z, p NaN (no floor p)")


def check_balanced_centre():
    """Residual means are w_other·(μ_X − μ_other), with w ∝ sqrt(n)."""
    rng = np.random.default_rng(6)
    X, y, _ = _synthetic(rng, 400, 25, [0.8, 0.9, 0.2], [0.5, 0.95, 0.4])
    R = features.balanced_mean_removed(X, y)
    mu_h, mu_s = X[y == 0].mean(0), X[y == 1].mean(0)
    w_h, w_s = np.sqrt(400) / (np.sqrt(400) + np.sqrt(25)), np.sqrt(25) / (np.sqrt(400) + np.sqrt(25))
    assert np.allclose(R[y == 0].mean(0), w_s * (mu_h - mu_s))
    assert np.allclose(R[y == 1].mean(0), w_h * (mu_s - mu_h))
    print(f"ok  balanced centre: w_H = {w_h:.2f}, w_S = {w_s:.2f} (grand mean: 0.94 / 0.06)")


def check_balanced_diagonals():
    """10:1 imbalance, planted effect: the grand mean leaves the diagonals far
    apart; sqrt(n) centring brings them together."""
    rng = np.random.default_rng(7)
    gaps = {"mean_removed": [], "mean_removed_balanced": []}
    for rep in range(6):
        X, y, sess = _synthetic(rng, 500, 50, [0.75, 0.9, 0.25], [0.6, 0.93, 0.4])
        for variant in gaps:
            run, _ = estimator.run_maze(
                features.apply_variant(X, variant), y, sess, seed=rep,
                n_rounds=100, n_perm=2, transform=features.variant_transform(variant),
            )
            q = run.result.q
            gaps[variant].append(abs(q[0, 0] - q[1, 1]))
    g_grand, g_bal = np.mean(gaps["mean_removed"]), np.mean(gaps["mean_removed_balanced"])
    assert g_bal < 0.5 * g_grand, (g_grand, g_bal)
    print(f"ok  |r_HH − r_SS| at 500:50 — grand mean {g_grand:.2f}, sqrt(n) {g_bal:.2f}")


def check_two_state_degenerate():
    """Why three states: Pearson of two 2-vectors is always ±1."""
    rng = np.random.default_rng(4)
    A, B = rng.random((500, 2)), rng.random((500, 2))
    r = msp_est._rowwise_pearson(A, B)
    assert np.allclose(np.abs(r[np.isfinite(r)]), 1.0)
    print("ok  2-state block-mean correlation is ±1 (hence K = 3)")


def check_merge():
    data = {
        "occ_bin": np.array([[1, 0, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]], float),
        "occ_ms": np.array([[100, 0, 50, 0, 0], [0, 20, 30, 50, 0], [0, 0, 0, 0, 0]], float),
        "fix_state": np.array([0, 1, 2, 3, 4, -1]),
    }
    cb = CODEBOOKS["balls"]  # left, origin, right
    b = features.merge_block(data, "binary", cb)
    assert b.shape == (3, 3)
    assert np.array_equal(b[0], [1, 1, 0]) and np.array_equal(b[1], [1, 0, 1])
    d = features.merge_block(data, "dwell", cb)
    assert np.allclose(d[0], [50 / 150, 100 / 150, 0]) and np.allclose(d[1], [0.5, 0, 0.5])
    assert np.array_equal(d[2], [0, 0, 0])
    assert list(features.merged_fix_state(data, cb)) == [1, 0, 0, 2, 2, -1]
    assert np.array_equal(features.merge_block(data, "binary", CODEBOOKS["halves"]), b)

    q = CODEBOOKS["quads"]  # LU, LD, origin, RU, RD
    bq = features.merge_block(data, "binary", q)
    assert np.array_equal(bq[0], [0, 1, 1, 0, 0]) and np.array_equal(bq[1], [1, 1, 0, 1, 0])
    assert list(features.merged_fix_state(data, q)) == [2, 0, 1, 3, 4, -1]
    print("ok  5-state → codebook merges (balls / halves 3, quads 5; binary, dwell, fixations)")


def check_region_rule():
    """Origin ball first; inside the rounded square the r1 exit balls trace,
    by quadrant (x >= 0 right, y >= 0 up); outside, unassigned."""
    O, LU, LD, RU, RD = 0, 1, 2, 3, 4
    pts = np.array([
        [0.1, 0.1],    # origin ball (r 0.5)
        [-1.0, 1.0],   # LU exit
        [-0.6, -0.2],  # LD, mid arm
        [1.9, 0.0],    # RU (y = 0 ties up), near right edge
        [0.4, -1.95],  # RD, bottom edge between exits
        [0.0, 1.0],    # stem top: x = 0 ties right
        [1.75, 1.75],  # corner beyond the exit ball (dist 1.06 from (1, 1))
        [2.1, 0.0],    # past the right edge
        [-1.7, -1.7],  # LD corner inside (dist 0.99)
    ])
    got = features.region_states(pts, radius=0.5)
    assert list(got) == [O, LU, LD, RU, RD, RU, -1, -1, LD], got
    assert features.inside_maze(np.array([[2.0, 0.0], [0.0, -2.0]])).all()
    print("ok  region rule: origin ball, rounded-square perimeter, quadrant ties")


def check_stem_rule():
    """Stem strip taken after the origin ball and before the quadrant split."""
    O, LU, RU, STEM = 0, 1, 3, 5
    pts = np.array([
        [0.1, 0.3],    # origin ball wins over the stem
        [-0.2, 0.8],   # stem (would be LU)
        [0.2, 1.2],    # stem top (would be RU)
        [0.0, 1.3],    # above STEM_TOP -> RU
        [-0.3, 0.8],   # outside the strip -> LU
        [0.1, -0.8],   # below the origin: no stem there -> RD
    ])
    got = features.region_states(pts, radius=0.5, stem=True)
    assert list(got) == [O, STEM, STEM, RU, LU, 4], got
    assert list(features.region_states(pts, radius=0.5)) == [O, LU, RU, RU, LU, 4]
    m = {r: features.merge_columns(np.eye(6), "max", CODEBOOKS[r]) for r in ("halvesstem", "quadsstem")}
    assert np.array_equal(m["halvesstem"][5], [0, 0, 1, 0])  # stem column
    assert np.array_equal(m["quadsstem"][5], [0, 0, 0, 1, 0, 0])
    print("ok  stem rule: origin first, strip |x| <= 0.25 up to y 1.25, then quadrants")


def check_region_patch_scoped():
    original, k = msp_features.assign_states, msp_features.K
    with features._rule_context(features.REGION_RULE):
        assert msp_features.assign_states is features.region_states and msp_features.K == 5
    with features._rule_context(features.STEM_RULE):
        assert msp_features.K == 6
        assert msp_features.assign_states(np.array([[0.0, 0.8]]), None, 0.5)[0] == 5
    assert msp_features.assign_states is original and msp_features.K == k
    print("ok  region / stem rules swapped into msp only inside the context")


def main():
    check_merge()
    check_region_rule()
    check_stem_rule()
    check_region_patch_scoped()
    check_two_state_degenerate()
    check_matches_msp()
    check_pooled_is_weighted_mean()
    check_power_and_calibration()
    check_tiny_strategy()
    check_balanced_centre()
    check_balanced_diagonals()
    print("all checks passed")


if __name__ == "__main__":
    main()
