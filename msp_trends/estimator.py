"""Per-maze msp 2x2 that keeps its null draws, and the pooled-over-mazes estimate.

`run_maze` is `msp.estimator.estimate` computed from msp's own pieces, with
the same seeding, so for the same integer seed it returns the identical
`Result` (`checks.py` pins this). It additionally keeps every shuffle's Δ and
cells so the pooled estimate can reuse them instead of running a second null.

Pooled: each maze keeps its own 2x2 (block means never mix mazes). The pooled
Δ is the trial-count-weighted mean of the per-maze Δs, and null draw ``i`` is
the same weighted mean of the per-maze null draws ``i``. Mazes shuffle
independently and each within its own sessions, so the joint draw is a
within-(session, maze) label shuffle of the whole pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from msp import estimator as msp_est
from msp.estimator import H, N_PERM, N_ROUNDS, NULL_SEED_TAG, S, Result
from msp_trends.config import LOW_N, MIN_TRIALS


@dataclass
class MazeRun:
    result: Result
    draws: np.ndarray  # (n_perm,) null Δ
    cells: np.ndarray  # (n_perm, 2, 2) null cells
    session_ids: np.ndarray = field(repr=False)

    @property
    def n_null_undefined(self):
        return int((~np.isfinite(self.draws)).sum())


def maze_seed(seed, maze):
    return int(seed) * 100 + int(maze)


def _summarise(d_obs, draws):
    """``(centre, sd, z, p, at_floor)`` over the defined null draws.

    A draw is undefined when, with a strategy of only a few trials, some
    shuffled half has a constant block mean in every round (Pearson 0/0).
    msp's estimator lets one such draw turn z into NaN and p into a spurious
    floor; here those draws are dropped (counted as ``n_null_undefined``).
    Identical to msp whenever every draw is defined.
    """
    draws = draws[np.isfinite(draws)]
    if draws.size < 2 or not np.isfinite(d_obs):
        return np.nan, np.nan, np.nan, np.nan, False
    centre = float(draws.mean())
    sd = float(draws.std(ddof=1))
    z = (d_obs - centre) / sd if sd > 0 else np.nan
    n_extreme = int(np.sum(np.abs(draws - centre) >= abs(d_obs - centre)))
    p = (1 + n_extreme) / (draws.size + 1)
    return centre, sd, float(z), float(p), n_extreme == 0


def run_maze(X, y, session_ids, *, seed=0, n_rounds=N_ROUNDS, n_perm=N_PERM,
             min_trials=MIN_TRIALS, transform=None):
    """``(MazeRun, "")`` or ``(None, reason)``; mirrors `msp.estimator.estimate`.

    `transform` is ``f(X, labels) -> X'``, refit under every shuffle exactly as
    in msp: the observed score uses ``f(X, y)``, each null draw
    ``f(X, y_shuffled)``. Use it only for label-dependent preprocessing.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    session_ids = np.asarray(session_ids)

    n_h, n_s, m_h, m_s, reason = msp_est.coverage(y, min_trials=min_trials)
    if reason:
        return None, reason

    if transform is None:
        features = lambda labels: X
    else:
        features = lambda labels: np.asarray(transform(X, labels), dtype=float)

    blocks = msp_est.session_blocks(session_ids)
    rng = np.random.default_rng(seed)
    q_obs = msp_est.quadrant_block_means(features(y), y, rng, n_rounds=n_rounds)
    d_obs = msp_est.delta(q_obs)

    rng_null = np.random.default_rng((seed, NULL_SEED_TAG))
    draws = np.empty(n_perm)
    cells = np.empty((n_perm, 2, 2))
    for i in range(n_perm):
        y_perm = msp_est.permute_within_session(y, blocks, rng_null)
        cells[i] = msp_est.quadrant_block_means(
            features(y_perm), y_perm, rng_null, n_rounds=n_rounds
        )
        draws[i] = msp_est.delta(cells[i])

    centre, sd, z, p, at_floor = _summarise(d_obs, draws)
    result = Result(
        q=q_obs, delta=float(d_obs), z=z, p=p, p_at_floor=at_floor,
        null_mean=centre, null_sd=sd,
        q_null_mean=np.nanmean(cells, axis=0), q_null_sd=np.nanstd(cells, axis=0, ddof=1),
        n_H=n_h, n_S=n_s, m_H=m_h, m_S=m_s, d=X.shape[1],
        n_sessions=len(blocks), n_rounds=n_rounds, n_perm=n_perm,
        n_degenerate=int(msp_est.undefined_trials(features(y)).sum()),
        method=msp_est.BLOCK_MEANS,
    )
    return MazeRun(result, draws, cells, session_ids), ""


def low_n(result):
    """True when the smaller strategy has <= `LOW_N` trials (score is marked)."""
    return result is not None and min(result.n_H, result.n_S) <= LOW_N


def weights_for(runs):
    """Trial-count weights, one per run, summing to 1."""
    w = np.array([r.result.n_H + r.result.n_S for r in runs], dtype=float)
    return w / w.sum()


def pooled(runs):
    """Weighted pooled `Result` over the per-maze `runs` (all same n_perm)."""
    if not runs:
        return None
    n_perm = {r.result.n_perm for r in runs}
    if len(n_perm) != 1:
        raise ValueError(f"mazes ran with different n_perm: {sorted(n_perm)}")
    w = weights_for(runs)
    q = sum(wi * r.result.q for wi, r in zip(w, runs))
    d_obs = float(sum(wi * r.result.delta for wi, r in zip(w, runs)))
    draws = sum(wi * r.draws for wi, r in zip(w, runs))
    cells = sum(wi * r.cells for wi, r in zip(w, runs))
    centre, sd, z, p, at_floor = _summarise(d_obs, draws)
    sessions = set()
    for r in runs:
        sessions.update(np.unique(r.session_ids).tolist())
    rs = [r.result for r in runs]
    return Result(
        q=q, delta=d_obs, z=z, p=p, p_at_floor=at_floor,
        null_mean=centre, null_sd=sd,
        q_null_mean=np.nanmean(cells, axis=0), q_null_sd=np.nanstd(cells, axis=0, ddof=1),
        n_H=sum(r.n_H for r in rs), n_S=sum(r.n_S for r in rs),
        m_H=sum(r.m_H for r in rs), m_S=sum(r.m_S for r in rs),
        d=rs[0].d, n_sessions=len(sessions), n_rounds=rs[0].n_rounds,
        n_perm=rs[0].n_perm, n_degenerate=sum(r.n_degenerate for r in rs),
        method=msp_est.BLOCK_MEANS,
    )


__all__ = ["H", "S", "MazeRun", "low_n", "maze_seed", "pooled", "run_maze", "weights_for"]
