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
    check_svm_scope()
    print("all checks passed")


if __name__ == "__main__":
    main()
