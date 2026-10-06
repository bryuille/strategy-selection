"""Synthetic checks, no data: estimator calibration and codebook sanity.

    uv run python -m msp_revised.checks
"""

from __future__ import annotations

import numpy as np

from msp_revised import codebooks, estimator, features


def check_null_calibration():
    rng = np.random.default_rng(0)
    X = rng.random((80, 5))
    y = np.r_[np.zeros(40, int), np.ones(40, int)]
    sess = np.where(np.arange(80) % 2 == 0, "a", "b")
    run, reason = estimator.run(X, y, sess, n_perm=200, n_rounds=30)
    assert run is not None, reason
    assert abs(run.result.z) < 3, run.result.z
    assert 0.0 < run.result.p <= 1.0
    print(f"null data: z={run.result.z:+.2f} p={run.result.p:.3f} (expect |z| < 3)")


def check_signal_detected():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(60, int), np.ones(60, int)]
    X = rng.random((120, 5)) * 0.3
    X[y == 1, 0] += 0.5  # S looks at state 0 more
    sess = np.where(np.arange(120) % 2 == 0, "a", "b")  # both strategies in each session
    run, _ = estimator.run(X, y, sess, n_perm=200, n_rounds=30)
    assert run.result.z > 3, run.result.z
    print(f"planted signal: z={run.result.z:+.2f}")


def check_pooled_matches_single():
    rng = np.random.default_rng(2)
    X = rng.random((50, 4))
    y = np.r_[np.zeros(25, int), np.ones(25, int)]
    sess = np.array(["a"] * 50)
    run, _ = estimator.run(X, y, sess, n_perm=100, n_rounds=20)
    pooled = estimator.pooled([run])
    assert np.isclose(pooled.delta, run.result.delta) and np.isclose(pooled.z, run.result.z)
    print("pooled over one maze equals that maze")


def check_two_dims_refused():
    X = np.random.default_rng(3).random((40, 2))
    y = np.r_[np.zeros(20, int), np.ones(20, int)]
    run, reason = estimator.run(X, y, np.array(["a"] * 40), n_perm=10, n_rounds=5)
    assert run is None and "±1" in reason
    print("d = 2 refused")


def check_codebooks():
    rng = np.random.default_rng(4)
    n = 2000
    xy_u = rng.normal(0, 1.2, (n, 2))
    h = np.tile([5, 9, 11, 7.5, 7, 5], (n, 1)).astype(float)
    # degree positions consistent with the warp for the right/up quadrant only
    # (sanity, not exactness): scale by typical arm lengths
    xy_d = xy_u * np.array([6.0, 8.0])
    fix = dict(xy_u=xy_u, xy_d=xy_d, h=h)
    for key, cb in codebooks.REGISTRY.items():
        st = cb.assign(fix)
        assert st.min() >= -1 and st.max() < cb.k, key
        used = np.unique(st[st >= 0]).size
        assert used >= 2, f"{key}: only {used} state(s) used on random gaze"
    print(f"{len(codebooks.REGISTRY)} codebooks assign within range")


def check_feature_kinds():
    data = dict(
        session=np.array(["a", "a", "a"]),
        fix_row=np.array([0, 0, 1, 1, 1]),
        fix_xy_u=np.array([[0, 0], [-1, 1], [1, 1], [0, 0], [-1, -1]], float),
        fix_xy_d=np.zeros((5, 2)),
        fix_onset_ms=np.array([10, 300, 50, 400, 900], float),
        fix_offset_ms=np.array([200, 500, 250, 600, 1100], float),
        fix_valid_ms=np.array([100, 200, 100, 100, 100], float),
        h=np.tile([5, 9, 11, 10, 7, 5], (3, 1)).astype(float),
        pupil_mean_raw=np.array([1.0, 2.0, np.nan]),
        pupil_change_raw=np.array([0.1, -0.1, 0.0]),
    )
    cb = codebooks.get("balls_u0.5")
    B = features.gaze_block(data, cb, "binary")
    assert B.shape == (3, 5) and B[0].tolist() == [1, 1, 0, 0, 0] and B[2].sum() == 0
    D = features.gaze_block(data, cb, "dwell")
    assert np.isclose(D[0, 1], 2 / 3) and D[2].sum() == 0
    F = features.gaze_block(data, cb, "first")
    L = features.gaze_block(data, cb, "last")
    assert F[0].argmax() == 0 and L[0].argmax() == 1 and F[1].argmax() == 3 and L[1].argmax() == 2
    P = features.pupil_block(data)
    assert P.shape == (3, 2) and P[2, 0] == 0.0
    Xc = features.apply_variant(np.arange(12.0).reshape(6, 2), "mean_removed")
    assert np.allclose(Xc.mean(axis=0), 0)
    print("feature kinds: binary / dwell / first / last / pupil / mean_removed behave")


def main():
    check_null_calibration()
    check_signal_detected()
    check_pooled_matches_single()
    check_two_dims_refused()
    check_codebooks()
    check_feature_kinds()
    print("all checks passed")


if __name__ == "__main__":
    main()
