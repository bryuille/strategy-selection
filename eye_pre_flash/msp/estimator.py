"""Pooled H-vs-S gaze similarity of group means, and its label-shuffle null.

The whole method lives in this file. For one (monkey, maze, feature, K,
variant) it pools every trial from every in-scope session, then asks: is the
mean gaze of two groups that share a decoded strategy more similar than the
mean gaze of two groups that do not?

A pairing is scored by averaging each group into one d-dimensional vector and
taking the single correlation between those two means. Averaging `m` trials
suppresses their independent noise before the correlation, so the cells sit
high and rise with group size. Compare `z` across panels, never the raw cells.

Per round, the pooled trials are cut into four disjoint groups of exactly
`m` -- two from H, two from S -- and each of the four cross-group pairings is
scored. The groups are disjoint, so a trial is never compared with itself.
100 rounds of fresh random groups are averaged.

`msp.md` records what this does and does not control for. The short
version: read `z`, not `delta`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

H, S = 0, 1
STRATEGY_TAG = {H: "H", S: "S"}

MIN_TRIALS = 10  # per strategy, pooled across sessions
NULL_SEED_TAG = 1  # keeps the null's rng stream disjoint from the observed one
N_ROUNDS = 100
N_PERM = 1000

BLOCK_MEANS = "block_means"
METHODS = (BLOCK_MEANS,)


def undefined_trials(X):
    """Mask of trials with no feature variance.

    A row that is all-zero (no gaze in the window) or constant. These are
    **counted, not removed**: they still enter their group's mean, and
    dropping them would make the trial set depend on K and on the variant.
    """
    X = np.asarray(X, dtype=float)
    return ~(np.isfinite(X).all(axis=1) & (X.std(axis=1) > 0))


def coverage(y, *, min_trials=MIN_TRIALS):
    """``(n_H, n_S, m, reason)``. `reason` is empty when the maze qualifies.

    `m` is the group size: with four disjoint groups of equal size, the
    minority strategy caps it at ``min(n_H, n_S) // 2``. Equal groups keep the
    four cells on one scale, because a correlation between group means rises
    with the number of trials that were averaged.
    """
    n_h = int(np.sum(y == H))
    n_s = int(np.sum(y == S))
    if min(n_h, n_s) < min_trials:
        return n_h, n_s, 0, f"n_H={n_h}, n_S={n_s}; need >= {min_trials} of each"
    return n_h, n_s, min(n_h, n_s) // 2, ""


def _group_indicators(idx, m, n, n_rounds, rng):
    """Two ``(n_rounds, n)`` 0/1 matrices: one strategy's "a" and "b" groups.

    Each row is one round's random partition of `idx` into two disjoint
    groups of exactly `m`. ``argsort`` of uniform noise is the vectorised
    equivalent of one `rng.permutation` per round.
    """
    order = np.argsort(rng.random((n_rounds, idx.size)), axis=1)
    picks = idx[order[:, : 2 * m]]
    rows = np.arange(n_rounds)[:, None]
    u_a = np.zeros((n_rounds, n))
    u_b = np.zeros((n_rounds, n))
    u_a[rows, picks[:, :m]] = 1.0
    u_b[rows, picks[:, m:]] = 1.0
    return u_a, u_b


def _round_groups(y, m, n_rounds, rng):
    """``(a, b)``: per-strategy ``(n_rounds, n)`` indicator matrices.

    One round's partition of each strategy into two disjoint groups of `m`.
    """
    n = len(y)
    a, b = {}, {}
    for strategy in (H, S):
        a[strategy], b[strategy] = _group_indicators(
            np.flatnonzero(y == strategy), m, n, n_rounds, rng
        )
    return a, b


def _rowwise_pearson(A, B):
    """Pearson between ``A[r]`` and ``B[r]`` for every row `r`."""
    Ac = A - A.mean(axis=1, keepdims=True)
    Bc = B - B.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(Ac, axis=1) * np.linalg.norm(Bc, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denom > 0, (Ac * Bc).sum(axis=1) / denom, np.nan)
    return np.clip(r, -1.0, 1.0)


def quadrant_block_means(X, y, m, rng, *, n_rounds=N_ROUNDS):
    """The 2x2 from correlating **group means**: average first, correlate after.

    Each group's `m` trials collapse to one d-dimensional mean and the pairing
    scores one correlation between two such means.

    Averaging `m` trials suppresses their independent noise, so these
    correlations sit high and rise with `m`. That makes raw cells incomparable
    across panels with different group sizes. Read `z`, and see `msp.md`.
    """
    a, b = _round_groups(y, m, n_rounds, rng)
    means_a = {s: (a[s] @ X) / m for s in (H, S)}  # (n_rounds, d)
    means_b = {s: (b[s] @ X) / m for s in (H, S)}

    q = np.empty((2, 2))
    for i in (H, S):
        for j in (H, S):
            q[i, j] = np.nanmean(_rowwise_pearson(means_a[i], means_b[j]))
    return q


def delta(q):
    """``0.5*(r_HH + r_SS) - 0.5*(r_HS + r_SH)``: the within-minus-across gap.

    Both off-diagonal cells enter, because ``q[H, S]`` and ``q[S, H]`` are two
    draws of one quantity -- which group of a strategy is called "a" is an
    arbitrary coin flip per round -- rather than two directions of an axis.
    """
    return 0.5 * (q[H, H] + q[S, S]) - 0.5 * (q[H, S] + q[S, H])


def session_blocks(session_ids):
    """Row indices of each session, for the within-session shuffle."""
    return [np.flatnonzero(session_ids == s) for s in np.unique(session_ids)]


def permute_within_session(y, blocks, rng):
    """Shuffle H/S labels inside each session, leaving its counts untouched.

    Shuffling **within** session rather than across the pool is load-bearing.
    Trials recorded on one day are more similar to each other than to trials
    from another day, for reasons that have nothing to do with strategy. If
    sessions differ in H/S balance, the pooled within-strategy cells carry
    more same-session pairs than the cross-strategy cell does, which lifts
    the diagonal on its own. Holding each session's H and S counts fixed
    makes the shuffled data carry that same-session enrichment in exactly the
    same proportion, so the null reproduces the artifact and `z` and `p` are
    calibrated against it. A shuffle across the whole pool would destroy it
    in the null while leaving it in the data. See `msp.md`.
    """
    out = np.asarray(y).copy()
    for block in blocks:
        out[block] = rng.permutation(out[block])
    return out


@dataclass
class Result:
    q: np.ndarray  # (2, 2), rows/cols ordered H, S
    delta: float
    z: float
    p: float
    p_at_floor: bool
    null_mean: float
    null_sd: float
    n_H: int
    n_S: int
    m: int
    d: int
    n_sessions: int
    n_rounds: int
    n_perm: int
    n_degenerate: int
    method: str


def estimate(
    X,
    y,
    session_ids,
    *,
    method=BLOCK_MEANS,
    n_rounds=N_ROUNDS,
    n_perm=N_PERM,
    seed=0,
    min_trials=MIN_TRIALS,
):
    """The 2x2, delta, and the within-session label-shuffle null for one panel.

    Returns ``(Result, "")`` or ``(None, reason)``. `X` is the pooled
    ``(n, d)`` trial matrix for one maze, `y` its H/S labels, `session_ids` a
    parallel array naming each trial's recording session. `method` is
    `block_means`, the only scoring method.

    **Every labelled trial is kept**, including trials whose features have no
    variance. Those still enter their group's mean; dropping them would make
    the trial set depend on K and on the variant. `n_degenerate` records how
    many there were. Centre-only trials are more common in S than in H, so
    dropping them would remove a strategy-correlated slice of the data.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    session_ids = np.asarray(session_ids)

    n_degenerate = int(undefined_trials(X).sum())

    n_h, n_s, m, reason = coverage(y, min_trials=min_trials)
    if reason:
        return None, reason

    score = lambda labels, rng: quadrant_block_means(
        X, labels, m, rng, n_rounds=n_rounds
    )

    blocks = session_blocks(session_ids)

    rng = np.random.default_rng(seed)
    q_obs = score(y, rng)
    d_obs = delta(q_obs)

    # A session with only one strategy present is invariant under the shuffle.
    # It still contributes its trials to every draw, so it is kept -- it just
    # cannot contribute label variation, which is the honest behaviour.
    rng_null = np.random.default_rng((seed, NULL_SEED_TAG))
    draws = np.empty(n_perm)
    for i in range(n_perm):
        draws[i] = delta(score(permute_within_session(y, blocks, rng_null), rng_null))

    centre = float(draws.mean())
    sd = float(draws.std(ddof=1))
    z = (d_obs - centre) / sd if sd > 0 else np.nan
    n_extreme = int(np.sum(np.abs(draws - centre) >= abs(d_obs - centre)))
    p = (1 + n_extreme) / (n_perm + 1)

    return (
        Result(
            q=q_obs,
            delta=float(d_obs),
            z=float(z),
            p=float(p),
            p_at_floor=n_extreme == 0,
            null_mean=centre,
            null_sd=sd,
            n_H=n_h,
            n_S=n_s,
            m=m,
            d=X.shape[1],
            n_sessions=len(blocks),
            n_rounds=n_rounds,
            n_perm=n_perm,
            n_degenerate=n_degenerate,
            method=method,
        ),
        "",
    )
