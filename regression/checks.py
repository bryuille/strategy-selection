"""Synthetic invariants for the model layer. No cache, seconds.

    python -m regression.checks
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from regression.config import K, STATE_NAMES
from regression.model import (
    analyse,
    build_design,
    fit_logit,
    select_mazes,
    select_states,
)

N_MAZES = 5
N_PER_MAZE = 80
ORIGIN, LU, LD, RU, RD = range(K)


def synth(rng, *, beta_occ_lu=0.0, beta_dur_ld=0.0, geometry=True):
    """Mazes whose geometry sets both the gaze profile and the S base rate.

    Within a maze the label depends on gaze only through the planted betas, so
    with both at 0 any gaze effect a model reports is geometry leaking.
    Origin and LD get revisits (visit counts vary); the exits are almost always
    a single visit, so visit count cannot be estimated there.
    """
    mazes, occ, visits, dwell, dur, base = [], [], [], [], [], []
    for c in range(N_MAZES):
        geo = c / (N_MAZES - 1) if geometry else 0.5
        p_visit = np.array([0.6, 0.8 - 0.5 * geo, 0.7 - 0.4 * geo, 0.2 + 0.5 * geo, 0.15 + 0.4 * geo])
        lam = np.array([1.0, 0.03, 0.8, 0.03, 0.03])
        n = N_PER_MAZE
        o = (rng.random((n, K)) < p_visit).astype(float)
        v = np.where(o > 0, 1 + rng.poisson(lam, (n, K)), 0).astype(float)
        d = np.where(o > 0, np.exp(rng.normal(5.3, 0.4, (n, K))), np.nan)
        mazes += [c + 1] * n
        occ.append(o), visits.append(v), dur.append(d)
        dwell.append(np.where(o > 0, d * v, 0.0))
        base.append(np.full(n, -1.5 + 3.0 * geo) + beta_occ_lu * (o[:, LU] - p_visit[LU]))
    meas = {
        "occ": np.vstack(occ), "visits": np.vstack(visits),
        "dwell_ms": np.vstack(dwell), "dur_ms": np.vstack(dur),
    }
    # planted duration effect: per SD of log mean duration among visited LD trials, 0 if unvisited
    seen = meas["occ"][:, LD] > 0
    lw = np.log(np.where(seen, meas["dur_ms"][:, LD], 1.0))
    z_ld = np.where(seen, (lw - lw[seen].mean()) / lw[seen].std(), 0.0)
    eta = np.concatenate(base) + beta_dur_ld * z_ld
    y = (rng.random(eta.size) < expit(eta)).astype(float)
    return meas, np.asarray(mazes), y


def _row(rows, predictor, state):
    return next(r for r in rows if r["predictor"] == predictor and r["state"] == state)


def check_matches_sklearn():
    from sklearn.linear_model import LogisticRegression

    meas, mazes, y = synth(np.random.default_rng(3), beta_occ_lu=0.8)
    d = build_design(meas, mazes, sorted(set(mazes)), list(range(K)))
    ours = fit_logit(d.X, y).beta
    sk = LogisticRegression(C=np.inf, fit_intercept=False, max_iter=10000, tol=1e-10)
    err = float(np.max(np.abs(ours - sk.fit(d.X, y).coef_.ravel())))
    print(f"  IRLS vs sklearn max |diff| = {err:.2e}")
    assert err < 1e-4, err


def check_pure_mazes_dropped():
    mazes = np.asarray([1] * 20 + [2] * 20 + [3] * 20)
    y = np.r_[np.zeros(20), np.r_[np.zeros(4), np.ones(16)], np.r_[np.zeros(10), np.ones(10)]]
    kept, _ = select_mazes(mazes, y, min_per_label=5)
    assert kept == [3], kept
    print("  pure / thin mazes dropped: ok")


def check_constant_state_dropped():
    occ = np.ones((100, K))
    occ[:, 1] = np.r_[np.ones(50), np.zeros(50)]
    occ[:, 2] = np.r_[np.ones(2), np.zeros(98)]
    kept, _ = select_states(occ)
    assert kept == [1], kept
    print("  near-constant states dropped: ok")


def check_offsets_remove_geometry(n_seeds=200):
    """Null within mazes: with per-maze offsets the gaze coefficient holds its
    size; with one global intercept it does not (geometry predicts the label
    across mazes)."""
    rej_off = rej_glob = 0
    for seed in range(n_seeds):
        meas, mazes, y = synth(np.random.default_rng(seed))
        for ids, name in ((mazes, "off"), (np.ones_like(mazes), "glob")):
            rows, _ = analyse(meas, ids, y, sorted(set(ids)), list(range(K)))
            p = _row(rows, "occupied", "LU")["p"]
            if name == "off":
                rej_off += p < 0.05
            else:
                rej_glob += p < 0.05
    size_off, size_glob = rej_off / n_seeds, rej_glob / n_seeds
    print(f"  null rejection of occupied[LU] at 0.05: per-maze offsets {size_off:.3f}, global intercept {size_glob:.3f}")
    assert size_off < 0.10, size_off
    assert size_glob > 0.5, size_glob


def check_planted_effects(n_seeds=40):
    """Planted occupied[LU] = 1.5 and duration[LD] = 1.0 per SD: recovered, and
    the 95% interval covers the truth about 95% of the time."""
    cover, est = [], []
    for seed in range(n_seeds):
        meas, mazes, y = synth(np.random.default_rng(100 + seed), beta_occ_lu=1.5, beta_dur_ld=1.0)
        rows, _ = analyse(meas, mazes, y, sorted(set(mazes)), list(range(K)))
        a, b = _row(rows, "occupied", "LU"), _row(rows, "duration", "LD")
        est.append((a["beta"], b["beta"]))
        cover.append((a["ci_lo"] < 1.5 < a["ci_hi"], b["ci_lo"] < 1.0 < b["ci_hi"]))
    est, cover = np.mean(est, axis=0), np.mean(cover, axis=0)
    print(f"  planted occupied[LU]=1.5: mean est {est[0]:+.2f}, CI coverage {cover[0]:.2f}")
    print(f"  planted duration[LD]=1.0: mean est {est[1]:+.2f}, CI coverage {cover[1]:.2f}")
    assert abs(est[0] - 1.5) < 0.25 and abs(est[1] - 1.0) < 0.25
    assert (cover > 0.8).all(), cover


def check_offsets_are_maze_log_odds():
    """No gaze effect: the centred offset is each maze's own S log-odds."""
    meas, mazes, y = synth(np.random.default_rng(5))
    names = sorted(set(mazes))
    rows, _ = analyse(meas, mazes, y, names, list(range(K)))
    off = {r["maze"]: r["beta"] for r in rows if r["kind"] == "offset"}
    err = []
    for m in names:
        p = y[mazes == m].mean()
        err.append(abs(off[m] - np.log(p / (1 - p))))
    print(f"  offset vs raw maze log-odds: max |diff| = {max(err):.2f}")
    assert max(err) < 0.35, err


def check_redundant_predictors_reported():
    """Exits with single visits: visits is dropped (no variation) with its
    reason; mean duration is still fitted there. Where visits vary, both fit."""
    meas, mazes, y = synth(np.random.default_rng(6))
    rows, _ = analyse(meas, mazes, y, sorted(set(mazes)), list(range(K)))
    vis = _row(rows, "visits", "LU")["status"]
    dur = _row(rows, "duration", "LU")["status"]
    assert vis.startswith("not fit: only"), vis
    assert dur == "fit", dur
    assert _row(rows, "visits", "origin")["status"] == "fit"
    assert _row(rows, "duration", "origin")["status"] == "fit"
    print(f"  LU visits -> '{vis}'; LU duration -> '{dur}'")


def check_separation_dropped():
    """Labels fully determined by LU occupancy: occupied[LU] runs off to
    infinity. The fit must converge by dropping a column, with the reason
    reported, and leave the other landmarks' columns fitted."""
    meas, mazes, y = synth(np.random.default_rng(10))
    y = (meas["occ"][:, LU] > 0).astype(float)
    rows, summary = analyse(meas, mazes, y, sorted(set(mazes)), list(range(K)))
    st = _row(rows, "occupied", "LU")["status"]
    assert summary["converged"], summary
    assert st.startswith("not fit: separates the labels"), st
    assert _row(rows, "occupied", "origin")["status"] == "fit"
    print(f"  separating column dropped, fit converged: LU occupied -> '{st}'")


def check_window_term():
    """Variable window: a `window` column is fitted (kind 'window'), not plotted
    as a gaze bar, and does not disturb the gaze rows."""
    meas, mazes, y = synth(np.random.default_rng(8))
    meas["window_ms"] = np.random.default_rng(9).uniform(800, 2500, y.size)
    rows, _ = analyse(meas, mazes, y, sorted(set(mazes)), list(range(K)))
    assert sum(r["kind"] == "window" for r in rows) == 1
    assert not any(r["kind"] == "gaze" and r["predictor"] == "" for r in rows)
    print("  window term fitted, kept out of the gaze rows: ok")


def check_figure_renders(tmp="/tmp/regression_check.png"):
    import tempfile
    from pathlib import Path

    from regression.figures import coefficient_figure

    meas, mazes, y = synth(np.random.default_rng(7), beta_occ_lu=1.0)
    rows, _ = analyse(meas, mazes, y, sorted(set(mazes)), list(range(K)))
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "c.png"
        coefficient_figure({"synthetic": rows}, out)
        assert out.stat().st_size > 10_000
    print("  figure renders: ok")


def main():
    for check in (
        check_matches_sklearn,
        check_pure_mazes_dropped,
        check_constant_state_dropped,
        check_planted_effects,
        check_offsets_remove_geometry,
        check_offsets_are_maze_log_odds,
        check_redundant_predictors_reported,
        check_separation_dropped,
        check_window_term,
        check_figure_renders,
    ):
        print(check.__name__)
        check()
    print("all checks passed")


if __name__ == "__main__":
    main()
