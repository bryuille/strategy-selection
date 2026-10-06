"""The split-half 2x2 of group-mean correlations, its within-session shuffle
null, and the pooled-over-mazes estimate. Copied from `msp.estimator` and
`msp_trends.estimator` (same seeding: identical `Result` for the same seed).

For one (monkey, maze, codebook, kind, variant): pool every trial of the
maze across the in-scope sessions; per round cut each strategy into two
disjoint halves of ``n_strategy // 2``; average each half into one
d-vector; correlate the means. ``Δ = ½(r_HH + r_SS) − ½(r_HS + r_SH)`` is
tested against label shuffles **within session** (within (session, maze)
when mazes are pooled first). Read `z`, not raw `r`: cells rise with group
size.

Pooled: trial-count-weighted mean of the per-maze Δs, null draw i the same
mean of the per-maze draws i (each maze keeps its own 2x2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from msp_revised.config import LOW_N, MIN_TRIALS

H, S = 0, 1
NULL_SEED_TAG = 1
N_ROUNDS = 100
N_PERM = 1000


def undefined_trials(X):
    """Rows with no feature variance (all-zero or constant). Counted, kept."""
    X = np.asarray(X, dtype=float)
    return ~(np.isfinite(X).all(axis=1) & (X.std(axis=1) > 0))


def coverage(y, *, min_trials=MIN_TRIALS):
    n_h = int(np.sum(y == H))
    n_s = int(np.sum(y == S))
    if min(n_h, n_s) < min_trials:
        return n_h, n_s, 0, 0, f"n_H={n_h}, n_S={n_s}; need >= {min_trials} of each"
    return n_h, n_s, n_h // 2, n_s // 2, ""


def _group_indicators(idx, m, n, n_rounds, rng):
    order = np.argsort(rng.random((n_rounds, idx.size)), axis=1)
    picks = idx[order[:, : 2 * m]]
    rows = np.arange(n_rounds)[:, None]
    u_a = np.zeros((n_rounds, n))
    u_b = np.zeros((n_rounds, n))
    u_a[rows, picks[:, :m]] = 1.0
    u_b[rows, picks[:, m:]] = 1.0
    return u_a, u_b


def _round_groups(y, n_rounds, rng):
    n = len(y)
    sizes = {H: int(np.sum(y == H)) // 2, S: int(np.sum(y == S)) // 2}
    a, b = {}, {}
    for strategy in (H, S):
        a[strategy], b[strategy] = _group_indicators(
            np.flatnonzero(y == strategy), sizes[strategy], n, n_rounds, rng
        )
    return a, b, sizes


def _rowwise_pearson(A, B):
    Ac = A - A.mean(axis=1, keepdims=True)
    Bc = B - B.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(Ac, axis=1) * np.linalg.norm(Bc, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denom > 0, (Ac * Bc).sum(axis=1) / denom, np.nan)
    return np.clip(r, -1.0, 1.0)


def quadrant_block_means(X, y, rng, *, n_rounds=N_ROUNDS):
    """The 2x2 from correlating group means: average first, correlate after."""
    a, b, sizes = _round_groups(y, n_rounds, rng)
    means_a = {s: (a[s] @ X) / sizes[s] for s in (H, S)}
    means_b = {s: (b[s] @ X) / sizes[s] for s in (H, S)}
    q = np.empty((2, 2))
    for i in (H, S):
        for j in (H, S):
            q[i, j] = np.nanmean(_rowwise_pearson(means_a[i], means_b[j]))
    return q


def delta(q):
    return 0.5 * (q[H, H] + q[S, S]) - 0.5 * (q[H, S] + q[S, H])


def session_blocks(session_ids):
    return [np.flatnonzero(session_ids == s) for s in np.unique(session_ids)]


def permute_within_session(y, blocks, rng):
    """Shuffle labels inside each block, keeping its H/S counts. Within-session
    (or within-(session, maze)) shuffling keeps same-day and same-maze
    enrichment in the null, so z is calibrated against it."""
    out = np.asarray(y).copy()
    for block in blocks:
        out[block] = rng.permutation(out[block])
    return out


@dataclass
class Result:
    q: np.ndarray
    delta: float
    z: float
    p: float
    p_at_floor: bool
    null_mean: float
    null_sd: float
    q_null_mean: np.ndarray
    q_null_sd: np.ndarray
    n_H: int
    n_S: int
    m_H: int
    m_S: int
    d: int
    n_sessions: int
    n_rounds: int
    n_perm: int
    n_degenerate: int


@dataclass
class MazeRun:
    result: Result
    draws: np.ndarray
    cells: np.ndarray
    session_ids: np.ndarray = field(repr=False)

    @property
    def n_null_undefined(self):
        return int((~np.isfinite(self.draws)).sum())


def maze_seed(seed, maze):
    return int(seed) * 100 + int(maze)


def _summarise(d_obs, draws):
    """``(centre, sd, z, p, at_floor)`` over the defined null draws."""
    draws = draws[np.isfinite(draws)]
    if draws.size < 2 or not np.isfinite(d_obs):
        return np.nan, np.nan, np.nan, np.nan, False
    centre = float(draws.mean())
    sd = float(draws.std(ddof=1))
    z = (d_obs - centre) / sd if sd > 0 else np.nan
    n_extreme = int(np.sum(np.abs(draws - centre) >= abs(d_obs - centre)))
    p = (1 + n_extreme) / (draws.size + 1)
    return centre, sd, float(z), float(p), n_extreme == 0


def run(X, y, session_ids, *, seed=0, n_rounds=N_ROUNDS, n_perm=N_PERM,
        min_trials=MIN_TRIALS):
    """``(MazeRun, "")`` or ``(None, reason)``. `session_ids` are the shuffle
    blocks: session names for one maze, ``"<session>|maze<M>"`` for a pool."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    session_ids = np.asarray(session_ids)

    n_h, n_s, m_h, m_s, reason = coverage(y, min_trials=min_trials)
    if reason:
        return None, reason
    if X.shape[1] < 3:
        return None, f"d={X.shape[1]}; a 2-vector correlation is always ±1"

    blocks = session_blocks(session_ids)
    rng = np.random.default_rng(seed)
    q_obs = quadrant_block_means(X, y, rng, n_rounds=n_rounds)
    d_obs = delta(q_obs)

    rng_null = np.random.default_rng((seed, NULL_SEED_TAG))
    draws = np.empty(n_perm)
    cells = np.empty((n_perm, 2, 2))
    for i in range(n_perm):
        y_perm = permute_within_session(y, blocks, rng_null)
        cells[i] = quadrant_block_means(X, y_perm, rng_null, n_rounds=n_rounds)
        draws[i] = delta(cells[i])

    centre, sd, z, p, at_floor = _summarise(d_obs, draws)
    result = Result(
        q=q_obs, delta=float(d_obs), z=z, p=p, p_at_floor=at_floor,
        null_mean=centre, null_sd=sd,
        q_null_mean=np.nanmean(cells, axis=0), q_null_sd=np.nanstd(cells, axis=0, ddof=1),
        n_H=n_h, n_S=n_s, m_H=m_h, m_S=m_s, d=X.shape[1],
        n_sessions=len(blocks), n_rounds=n_rounds, n_perm=n_perm,
        n_degenerate=int(undefined_trials(X).sum()),
    )
    return MazeRun(result, draws, cells, session_ids), ""


def low_n(result):
    return result is not None and min(result.n_H, result.n_S) <= LOW_N


def weights_for(runs):
    w = np.array([r.result.n_H + r.result.n_S for r in runs], dtype=float)
    return w / w.sum()


def pooled(runs):
    """Weighted pooled `Result` over per-maze `runs` (all the same n_perm)."""
    runs = [r for r in runs if np.isfinite(r.result.delta)]
    if not runs:
        return None
    if len({r.result.n_perm for r in runs}) != 1:
        raise ValueError("mazes ran with different n_perm")
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
        q=q, delta=d_obs, z=z, p=p, p_at_floor=at_floor, null_mean=centre, null_sd=sd,
        q_null_mean=np.nanmean(cells, axis=0), q_null_sd=np.nanstd(cells, axis=0, ddof=1),
        n_H=sum(r.n_H for r in rs), n_S=sum(r.n_S for r in rs),
        m_H=sum(r.m_H for r in rs), m_S=sum(r.m_S for r in rs),
        d=rs[0].d, n_sessions=len(sessions), n_rounds=rs[0].n_rounds,
        n_perm=rs[0].n_perm, n_degenerate=sum(r.n_degenerate for r in rs),
    )
