"""Synthetic checks, no data.

    uv run python -m classifier.checks
"""

from __future__ import annotations

import numpy as np

from classifier import model as M


def _meta(n, rng, n_sessions=2):
    maze = rng.integers(1, 7, n)
    session = np.asarray(list("abcdefghij"))[rng.integers(0, n_sessions, n)]
    return dict(maze=maze, session=session)


def check_within_maze_auc_ignores_maze():
    rng = np.random.default_rng(0)
    n = 600
    meta = _meta(n, rng)
    y = (meta["maze"] >= 4).astype(int)  # label is maze
    y[rng.random(n) < 0.15] ^= 1  # a few exceptions so every maze has both
    p = meta["maze"] / 6.0  # a "classifier" that only reads maze
    auc, per = M.within_maze_auc(p, y, meta["maze"])
    assert abs(auc - 0.5) < 1e-9, auc  # constant within maze -> all ties -> 0.5
    assert len(per) == 6
    print("within-maze AUC gives 0.5 to a maze-only predictor")


def check_null_calibrated_and_signal_found():
    rng = np.random.default_rng(1)
    n = 400
    meta = _meta(n, rng)
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 8))
    ev = M.evaluate(X, y, meta, "logreg", n_perm=30, n_repeats=1)
    assert abs(ev.z) < 3, ev.z
    X[:, 0] += 1.0 * y
    ev2 = M.evaluate(X, y, meta, "logreg", n_perm=30, n_repeats=1)
    assert ev2.auc > 0.7 and ev2.z > 3, (ev2.auc, ev2.z)
    print(f"null: AUC {ev.auc:.2f} z {ev.z:+.1f}; planted: AUC {ev2.auc:.2f} z {ev2.z:+.1f}")


def check_session_holdout_scales():
    """Two sessions: one entry per session. Ten: leave-one-out over ten, and
    a planted signal survives the held-out fit."""
    rng = np.random.default_rng(2)
    n = 800
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 6))
    X[:, 1] += 1.0 * y
    for k in (2, 10):
        meta = _meta(n, rng, n_sessions=k)
        per, pooled = M.session_holdout(X, y, meta, "logreg")
        assert len(per) == k, (k, per)
        assert pooled > 0.7, (k, pooled)
    print(f"leave-one-session-out: {k} sessions, pooled AUC {pooled:.2f}")


def check_regions_partition():
    """Every fixation lands in exactly one region; dwell shares sum to 1 on
    trials with any fixation time; a trial at the RU exit lights only RU."""
    from classifier import features as F
    rng = np.random.default_rng(3)
    n_fix = 500
    h = np.tile([7.0, 7.0, 7.0, 7.0, 7.0, 7.0], (n_fix, 1))
    xy = rng.uniform(-15, 15, (n_fix, 2))
    idx = F.region_index(xy, h)
    assert idx.min() >= 0 and idx.max() < len(F.REGION_NAMES)
    d = dict(session=np.array(["a"] * 3), fix_row=np.array([0, 0, 1, 1, 1]),
             fix_xy_d=np.array([[7.0, 7.0], [7.1, 6.9], [-1.0, 1.0], [-1.0, -1.0], [1.0, 1.0]]),
             fix_valid_ms=np.array([100.0, 50.0, 10.0, 20.0, 30.0]), h=np.tile(h[0], (3, 1)))
    X, names = F.regions_matrix(d)
    dwell = X[:, :len(F.REGION_NAMES)]
    assert np.allclose(dwell[:2].sum(1), 1.0) and dwell[2].sum() == 0.0
    assert dwell[0, names.index("dwell: exit RU")] == 1.0
    assert X[0, names.index("binary: exit RU")] == 1.0 and X[0, names.index("binary: origin LU")] == 0.0
    assert np.allclose(dwell[1, :4], [10 / 60, 20 / 60, 30 / 60, 0.0])
    print("regions: one region per fixation, dwell shares sum to 1")
    # exits: the four exit-ball columns of the same matrix, nothing else
    Xe, enames = F.regions_matrix(d, exits=True)
    assert enames == [f"{k}: {n}" for k in ("dwell", "binary") for n in F.REGION_NAMES[4:8]], enames
    nr = len(F.REGION_NAMES)
    assert np.array_equal(Xe[:, :4], X[:, 4:8]) and np.array_equal(Xe[:, 4:], X[:, nr + 4:nr + 8])
    assert np.allclose(Xe[:, :4].sum(1), [1.0, 0.0, 0.0])  # exit shares sum to the time spent at any exit
    print("exits: the exit-ball columns of regions, shares not forced to sum to 1")
    # regions_plus: elsewhere splits at FAR_DEG, shares still sum to 1, pupil change appended
    d["fix_row"] = np.array([0, 0, 1, 1, 1, 2])
    d["fix_xy_d"] = np.array([[7.0, 7.0], [0.0, 8.0], [0.0, 12.0], [-1.0, -1.0], [1.0, 1.0], [0.0, -12.0]])
    d["fix_valid_ms"] = np.array([100.0, 50.0, 10.0, 20.0, 30.0, 40.0])
    d["pupil_change_raw"] = np.array([1.0, 2.0, np.nan])
    X, names = F.regions_matrix(d, plus=True)
    k = len(F.REGION_PLUS_NAMES)
    near, far = names.index(f"dwell: {F.REGION_PLUS_NAMES[8]}"), names.index(f"dwell: {F.REGION_PLUS_NAMES[9]}")
    assert X.shape[1] == 2 * k + 1 and names[-1] == "pupil change z"
    assert np.allclose(X[:, :k].sum(1), 1.0)
    assert np.isclose(X[0, near], 50 / 150) and np.isclose(X[1, far], 10 / 60) and X[2, far] == 1.0
    assert np.isclose(X[0, -1], -1.0) and np.isclose(X[1, -1], 1.0) and np.isnan(X[2, -1])
    print("regions_plus: near / far split keeps shares summing to 1; pupil change z within session")


def check_lor_partition():
    """Origin ball wins; inside the perimeter x < 0 is left, x >= 0 right; the
    perimeter is the exit quadrilateral grown by LOR_PERIMETER_DEG."""
    from classifier import features as F
    from classifier.config import LOR_ORIGIN_DEG, LOR_PERIMETER_DEG
    r, p = LOR_ORIGIN_DEG, LOR_PERIMETER_DEG
    h = np.array([5.0, 7.0, 11.0, 10.0, 9.0, 4.0])  # LU (-5, 7), LD (-5, -11), RU (10, 9), RD (10, -4)
    pts = np.array([[0.0, 0.0], [0.0, r - 0.1], [0.0, r + 0.1],  # origin, origin, right (x >= 0)
                    [-5.0, 7.0], [-5.0, 7.0 + p - 0.1], [-5.0, 7.0 + p + 0.1],  # left, left, outside
                    [10.0, -4.0], [10.0 + p - 0.1, -4.0], [10.0 + p + 0.1, -4.0],  # right, right, outside
                    [-5.0 - p - 0.1, 0.0], [-5.0 - p + 0.1, 0.0],  # outside, left
                    [-5.0 - p / np.sqrt(2) + 0.1, 7.0 + p / np.sqrt(2) + 0.1],  # outside the LU corner arc
                    [-5.0 - p / np.sqrt(2) + 0.1, 7.0 + p / np.sqrt(2) - 0.1],  # inside it
                    [2.0, -4.5]])  # inside the quad, right
    got = F.lor_index(pts, np.tile(h, (len(pts), 1)))
    want = [0, 0, 2, 1, 1, -1, 2, 2, -1, -1, 1, -1, 1, 2]
    assert got.tolist() == want, (got.tolist(), want)
    d = dict(session=np.array(["a", "a"]), fix_row=np.array([0, 0, 0, 1]),
             fix_xy_d=np.array([[0.0, 0.0], [-5.0, 7.0], [20.0, 20.0], [10.0, 9.0]]),
             fix_valid_ms=np.array([100.0, 100.0, 200.0, 50.0]), h=np.tile(h, (2, 1)))
    X, names = F.regions_matrix(d, lor=True)
    assert names == ["dwell: origin", "dwell: left", "dwell: right", "binary: origin", "binary: left", "binary: right"]
    assert np.allclose(X[0], [0.25, 0.25, 0.0, 1, 1, 0]) and np.allclose(X[1], [0, 0, 1.0, 0, 0, 1]), X
    print("lor: origin ball, then left / right inside the exit-hugging perimeter; outside unassigned")


def check_balls_partition():
    """Origin ball + four exit balls in degrees, nothing else; the exit columns
    equal the `regions` exit columns (same balls at the same landmarks)."""
    from classifier import features as F
    from classifier.config import BALLS_ORIGIN_DEG, EXIT_BALL_DEG
    r, e = BALLS_ORIGIN_DEG, EXIT_BALL_DEG
    h = np.array([5.0, 7.0, 11.0, 10.0, 9.0, 4.0])  # LU (-5, 7), LD (-5, -11), RU (10, 9), RD (10, -4)
    pts = np.array([[0.0, 0.0], [0.0, r - 0.1], [0.0, r + 0.1], [-5.0, 7.0], [-5.0, 7.0 + e + 0.1],
                    [-5.0, -11.0], [10.0, 9.0], [10.0 + e - 0.1, -4.0], [-2.5, 2.5]])  # last: |xy| 3.5 > r
    got = F.balls_index(pts, np.tile(h, (len(pts), 1)))
    assert got.tolist() == [0, 0, -1, 1, -1, 2, 3, 4, -1], got.tolist()
    rng = np.random.default_rng(4)
    xy = rng.uniform(-15, 15, (3000, 2))
    H = np.tile(h, (len(xy), 1))
    b, g = F.balls_index(xy, H), F.region_index(xy, H)
    assert np.array_equal(b >= 1, (g >= 4) & (g <= 7)) and np.array_equal(b[b >= 1], g[b >= 1] - 3)
    assert F.BALLS_NAMES == ("origin", "exit LU", "exit LD", "exit RU", "exit RD")
    print("balls: origin ball + the four exit balls of regions, in degrees; outside unassigned")


def check_capsules():
    """Origin ball wins; each capsule is the stadium of radius CAPSULE_DEG about
    the segment origin -> exit (parallel tangents); the rest is elsewhere, so
    dwell shares sum to 1. Shrinking the origin ball hands the ring to the capsule."""
    from classifier import features as F
    from classifier.config import CAPSULE_DEG, CAPSULE_ORIGIN_DEG, CAPSULE_ORIGIN_TIGHT_DEG
    h = np.array([5.0, 7.0, 11.0, 10.0, 9.0, 4.0])  # LU (-5, 7)
    e = np.array([-5.0, 7.0])
    u = e / np.linalg.norm(e)
    n = np.array([-u[1], u[0]])
    mid = 0.5 * e
    pts = np.array([[0.0, 0.0], 2.2 * u, e, mid + (CAPSULE_DEG - 0.05) * n, mid + (CAPSULE_DEG + 0.2) * n,
                    [20.0, 20.0]])
    got = F.capsule_index(pts, np.tile(h, (len(pts), 1)))
    assert got.tolist() == [0, 0, 1, 1, 5, 5], got.tolist()
    tight = F.capsule_index(pts, np.tile(h, (len(pts), 1)), origin_deg=CAPSULE_ORIGIN_TIGHT_DEG)
    assert tight[1] == 1 and tight[0] == 0, tight.tolist()  # 2.2° is inside the 3° ball, outside the 1.5° ball
    d = dict(session=np.array(["a"]), fix_row=np.zeros(4, int),
             fix_xy_d=np.array([[0.0, 0.0], e, [20.0, 20.0], mid]),
             fix_valid_ms=np.full(4, 25.0), h=h[None, :])
    X, names = F.regions_matrix(d, ("dwell",), capsule_origin=CAPSULE_ORIGIN_DEG)
    assert names == [f"dwell: {n}" for n in F.CAPSULE_NAMES]
    assert np.isclose(X.sum(), 1.0) and np.isclose(X[0, 0], 0.25) and np.isclose(X[0, 1], 0.5)
    assert np.isclose(X[0, names.index("dwell: elsewhere")], 0.25)
    print("capsules: stadiums with parallel tangents, origin ball carved out, elsewhere fills the rest")


def check_visit_count():
    """A visit is an entry into a region: LU, LU, elsewhere, LU (stored out of
    time order) is two LU visits and one elsewhere visit."""
    from classifier.features import regions_matrix
    xy = np.array([[-6.0, 5.0], [20.0, 20.0], [-6.2, 5.1], [-6.0, 5.0]])
    d = dict(session=np.array(["a"]), fix_row=np.zeros(4, int), fix_xy_d=xy,
             fix_valid_ms=np.full(4, 100.0), h=np.array([[6.0, 5, 11, 10, 9, 5]]),
             fix_onset_ms=np.array([0.0, 400, 200, 600]))
    X, names = regions_matrix(d, ("count",))
    got = dict(zip(names, X[0]))
    assert got["count: exit LU"] == 2 and got["count: elsewhere"] == 1 and sum(got.values()) == 3, got
    print("visit count: consecutive fixations in one region are one visit, in time order")


def check_svm_scope():
    from classifier.config import LABEL_SCOPES, SESSIONS, SVM_SESSIONS
    for m, ss in SVM_SESSIONS.items():
        assert len(ss) == 10 and set(SESSIONS[m]) <= set(ss), (m, ss)
    assert LABEL_SCOPES["svm"]["mazes"] == (2, 3, 4, 5)
    print("svm scope: " + "; ".join(f"{m}: {', '.join(ss)}" for m, ss in SVM_SESSIONS.items()))


def main():
    check_within_maze_auc_ignores_maze()
    check_null_calibrated_and_signal_found()
    check_session_holdout_scales()
    check_regions_partition()
    check_lor_partition()
    check_balls_partition()
    check_capsules()
    check_visit_count()
    check_svm_scope()
    print("all checks passed")


if __name__ == "__main__":
    main()
