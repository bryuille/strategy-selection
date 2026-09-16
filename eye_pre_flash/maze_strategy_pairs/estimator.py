"""Pooled trial-by-trial H-vs-S gaze similarity, and its label-shuffle null.

The whole method lives in this file. For one (monkey, maze, feature, K,
variant) it pools every trial from every in-scope session, then asks: are two
trials that share a decoded strategy more similar to each other than two
trials that do not?

The unit of comparison is a **pair of individual trials**, not a pair of
group means. That is the one design choice everything else follows from: a
mean pairwise trial correlation means the same thing whatever the group size,
so there is no need to equalise counts to keep the four cells comparable, and
no group-size-dependent inflation to correct for afterwards.

Per round, the pooled trials are cut into four disjoint groups of exactly
`m` -- two from H, two from S -- and each of the four cross-group pairings is
scored. The groups are disjoint, so a trial is never compared with itself.
100 rounds of fresh random groups are averaged.

Two ways to score a pairing, swept side by side as `METHODS`:

``trial_by_trial``  the mean correlation over the pairing's `m x m` individual
                    trial pairs. Correlate first, average after.
``block_means``     average each group into one d-dimensional vector, then
                    take the single correlation between those two means.
                    Average first, correlate after.

They are **not** two estimates of one number. A correlation between two
`m`-trial means is pulled up by each group's internal consistency, roughly as
Spearman-Brown predicts, so `block_means` runs far higher and its scale moves
with `m`; `trial_by_trial` does not depend on group size. Compare `z` between
them, never the raw cells, and expect `block_means` to look much stronger for
that reason alone.

`CAVEATS.md` records what this does and does not control for. The short
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

TRIAL_BY_TRIAL = "trial_by_trial"
BLOCK_MEANS = "block_means"
METHODS = (TRIAL_BY_TRIAL, BLOCK_MEANS)


def trial_correlation(X):
    """``C[i, j]`` = Pearson r between trials `i` and `j`, over the d features.

    Rows are centred and scaled to unit norm, so the whole matrix is one
    ``Xc @ Xc.T`` rather than ``n^2`` calls to `np.corrcoef`. That matters:
    `C` is computed once per panel and then re-read by 100 rounds x 1000
    permutations, all of which only ever select submatrices of it.

    A zero-variance trial (no gaze in the window, or -- under `no_origin` --
    one that never looked away from centre) has no defined correlation with
    anything, so its row and column come back NaN. Those are *skipped pair by
    pair* in `quadrant_means`, not dropped from the pool; `undefined_trials`
    counts them.
    """
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(Xc, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        Xc = np.where(norm > 0, Xc / norm, np.nan)
    # Float error puts |r| a few ulp past 1, which is harmless here but would
    # look like a bug in a printed cell. Clip rather than explain it later.
    return np.clip(Xc @ Xc.T, -1.0, 1.0)


def undefined_trials(X):
    """Mask of trials whose correlation with anything is undefined.

    A row with no variance -- all-zero (no gaze in the window) or constant --
    gives Pearson a zero denominator. These are **counted, not removed**: see
    `quadrant_means` for why excluding them would be the more dangerous
    choice.
    """
    X = np.asarray(X, dtype=float)
    return ~(np.isfinite(X).all(axis=1) & (X.std(axis=1) > 0))


def coverage(y, *, min_trials=MIN_TRIALS):
    """``(n_H, n_S, m, reason)``. `reason` is empty when the maze qualifies.

    `m` is the group size: with four disjoint groups of equal size, the
    minority strategy caps it at ``min(n_H, n_S) // 2``. Equal groups are not
    needed to keep the four cells unbiased -- a mean trial-pair correlation
    does not depend on how many pairs it averages -- but they give all four
    cells the same number of pairs and so the same sampling variance, which
    is what makes the 2x2 readable as four comparable numbers.
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

    Shared by both estimators, so the two differ only in how a pairing is
    scored and not in which trials land in which group.
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

    The estimator this package used before the rewrite, kept as a comparison
    arm. Each group's `m` trials collapse to one d-dimensional mean and the
    pairing scores one correlation between two such means.

    Read `CAVEATS.md` before comparing its cells to `quadrant_means`: averaging
    `m` trials suppresses their independent noise, so these correlations sit
    much higher and rise with `m`, which makes them incomparable across panels
    with different group sizes. That group-size dependence is the whole reason
    the trial-by-trial arm exists.
    """
    a, b = _round_groups(y, m, n_rounds, rng)
    means_a = {s: (a[s] @ X) / m for s in (H, S)}  # (n_rounds, d)
    means_b = {s: (b[s] @ X) / m for s in (H, S)}

    q = np.empty((2, 2))
    for i in (H, S):
        for j in (H, S):
            q[i, j] = np.nanmean(_rowwise_pearson(means_a[i], means_b[j]))
    return q


def quadrant_means(C, y, m, rng, *, n_rounds=N_ROUNDS, defined=None):
    """The 2x2 of mean trial-pair similarity, averaged over `n_rounds` rounds.

    Row is the strategy of the "a" group, column that of the "b" group, so
    ``q[H, S]`` is H's first group against S's second. The diagonal is
    within-strategy similarity, the off-diagonal cross-strategy.

    **No trial is excluded.** A trial whose features have no variance -- one
    with no gaze in the window, or, under `no_origin`, one that never looked
    away from centre -- has an undefined correlation with everything, so its
    pairs are *skipped* rather than its trial being dropped from the pool.
    That distinction matters: those trials are not a random subset. Centre-only
    trials run ~1.4x more common in S than in H, so excluding them would remove
    a strategy-correlated slice of the data and could manufacture or mask the
    very difference being measured. Keeping them in the pool keeps `n`, the
    group draws and the coverage rule identical across every K, variant and
    estimator arm.

    So a cell is the mean over its **defined** pairs, which can be fewer than
    ``m * m``. `defined` is the boolean matrix of which pairs those are;
    `estimate` passes ``isfinite(C)``.

    Mechanically this is exactly ``np.nanmean`` over each round's submatrix
    -- verified equal to a literal fancy-indexed `np.nanmean` to 6e-17 given
    the same partitions. It is written as ``sum / count`` because NaN
    propagates through BLAS and would poison the whole product, so undefined
    pairs contribute 0 to both and drop out. ``sum(C[A, B])`` is
    ``1_A @ C @ 1_B``, which lets all `n_rounds` go through one matrix
    multiplication; measured about 5x faster than the loop at these sizes,
    over 100 rounds x 1000 permutations per panel.
    """
    if defined is None:
        defined = np.isfinite(C)
    # NaN would poison the matmul, so undefined pairs contribute 0 to the sum
    # and 0 to the count, which is exactly "skip this pair".
    C_filled = np.where(defined, C, 0.0)
    D = defined.astype(float)

    a, b = _round_groups(y, m, n_rounds, rng)

    q = np.empty((2, 2))
    for i in (H, S):
        ac = a[i] @ C_filled  # (n_rounds, n)
        ad = a[i] @ D
        for j in (H, S):
            total = (ac * b[j]).sum(axis=1)
            count = (ad * b[j]).sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                per_round = np.where(count > 0, total / count, np.nan)
            # A round in which one group is entirely undefined contributes
            # nothing rather than dragging the cell to NaN.
            q[i, j] = np.nan if np.isnan(per_round).all() else np.nanmean(per_round)
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
    in the null while leaving it in the data. See `CAVEATS.md`.
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
    method=TRIAL_BY_TRIAL,
    n_rounds=N_ROUNDS,
    n_perm=N_PERM,
    seed=0,
    min_trials=MIN_TRIALS,
):
    """The 2x2, delta, and the within-session label-shuffle null for one panel.

    Returns ``(Result, "")`` or ``(None, reason)``. `X` is the pooled
    ``(n, d)`` trial matrix for one maze, `y` its H/S labels, `session_ids` a
    parallel array naming each trial's recording session. `method` selects
    which of `METHODS` scores a pairing.

    **Every labelled trial is kept.** Trials whose features have no variance
    have no defined correlation, so their *pairs* are skipped inside
    `quadrant_means` while the trials themselves stay in the pool, in `n_H` /
    `n_S`, and in the coverage decision. `n_degenerate` records how many there
    were. This keeps the trial set identical across both K, all three variants
    and both estimator arms, which per-panel exclusion did not: `no_origin`
    strands ~15% of trials (those that never looked away from centre) and
    at different rates per K, so excluding them made every panel a different
    dataset. It is also the safer choice on the merits -- centre-only trials
    are ~1.4x more common in S than in H, so dropping them removes a
    strategy-correlated slice of the data.
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

    # `trial_by_trial` reads the same (n, n) correlation matrix for every
    # round and every permutation, so it is built once. `block_means` has to
    # re-average the raw trials each round and cannot reuse anything.
    if method == TRIAL_BY_TRIAL:
        C = trial_correlation(X)
        finite = np.isfinite(C)
        score = lambda labels, rng: quadrant_means(
            C, labels, m, rng, n_rounds=n_rounds, defined=finite
        )
    else:
        score = lambda labels, rng: quadrant_block_means(
            X, labels, m, rng, n_rounds=n_rounds
        )

    blocks = session_blocks(session_ids)

    rng = np.random.default_rng(seed)
    q_obs = score(y, rng)
    d_obs = delta(q_obs)

    # A session with only one strategy present is invariant under the shuffle.
    # It still contributes its pairs to every draw, so it is kept -- it just
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
