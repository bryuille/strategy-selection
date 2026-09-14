"""The 2x2 (strategy x strategy) split-half similarity for a single maze.

For one maze, correlate half-means of trials grouped by decoded strategy
(H = hierarchical, S = sequential). The diagonal is each strategy's own
split-half reliability; the off-diagonal is the cross-strategy correlation.
Maze geometry is identical on both sides, so ``r(H,S)`` falling below that
maze's own reliability is strategy-dependent gaze sampling and not a visual
confound.

The whole design problem is that **the diagonal is structurally advantaged**,
and a 2x2 where the diagonal wins for procedural reasons says nothing. Every
choice below exists to strip an advantage that is not the strategy effect.
`METHODS.md` is the full write-up; the load-bearing points:

1. **One half size, both cells, no exceptions.** ``m = min(n_H, n_S) // 2``,
   so all four half-means in every split average exactly `m` trials. The
   earlier rule (`matrix.session_half_sizes`) let a cell below ``MIN_STABLE``
   fall back to its own ``n // 2``, which meant ``r_HH`` could average 22-trial
   halves while ``r_HS`` correlated a 22-trial half against a 5-trial half.
   Pearson between two noisy means is dragged down by the noise in *both*, so
   the off-diagonal lost trivially. Sessions with ``m < min_half`` are dropped
   rather than estimated badly.

2. **All-or-nothing splits and sessions.** A split contributes all four
   entries or none; a session likewise. The earlier code accumulated each
   entry against its own denominator, so one degenerate H half-mean dropped
   ``r_HH`` and ``r_HS`` but kept ``r_SS`` -- and ``0.5 * (r_HH + r_SS)`` then
   mixed two different split sets.

3. **The off-diagonal is not symmetrised.** ``R[H,S]`` and ``R[S,H]`` are
   reported separately. See `session_pair_matrix` for why that is a
   transparency measure and **not** a directional finding.

4. **Fisher-z aggregation, raw reported alongside.** See `fisher_mean`.

What the permutation null does and does not buy, since this is the single
easiest thing to over-read: the within-(session, maze) label shuffle preserves
`n_H` and `n_S` exactly, so **it calibrates everything that depends on the
counts and nothing that depends on the within-cell variances or on trial
timing.** For the variance gap see `amgm_floor`; for timing see `drift.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb, log10, sqrt

import numpy as np

from eye_pre_flash.classifier.labels import labels_for_rows
from eye_pre_flash.maze_strategy_pairs.cells import STRATEGIES, STRATEGY_TAG
from eye_pre_flash.maze_strategy_pairs.matrix import MIN_TRIALS, session_seed

N_CELLS = 2
H_CELL, S_CELL = 0, 1  # matrix row/col 0 = hierarchical, 1 = sequential

# Deliberately package-local, NOT `plotting.similarities.common.N_SPLITS`.
# That constant is 20 and has four external consumers (similarities/
# occupancy_bin, similarities/transition, similarities/heatmap, classifier/
# pairwise); `similarity.md` documents "20 random splits" for them. Raising it
# there would silently move three figure families and a classifier analysis.
N_SPLITS = 200

# Two-sided clip before `arctanh`. Mandatory, not cosmetic: the
# `cov / (a_sd * b_sd)` division reaches |r| = 1 + 2e-16, which `arctanh`
# turns into NaN, and at d = 5 (occupancy/no_origin/k=6) exact |r| = 1 is
# genuinely reachable. |z| <= 7.25 here, against ~2.65 for the largest
# plausible real r, so the clip only ever engages on near-degeneracy.
# Two-sided because `mean_removed` makes negative r routine and a one-sided
# clip would bias delta.
R_CLIP = 1.0 - 1e-6

# Below this the session is dropped. A 2-trial half-mean at d = 5 is noise,
# and sessions are averaged unweighted, so it would count as much as an
# m = 20 session. This is a session-inclusion rule, not an exception to the
# equal-count rule: H and S always use the same m.
MIN_HALF = 5

NULL_SEED_TAG = 11


def cell_labels(maze):
    """``["4H", "4S"]`` for maze 4."""
    return [f"{maze}{STRATEGY_TAG[s]}" for s in STRATEGIES]


def half_size(counts, *, n_half=None, min_half=MIN_HALF):
    """The one half size for both cells, or 0 if this session cannot supply it.

    ``min(n_H, n_S) // 2`` unless `n_half` forces a size (used by `drift.py`
    to hold `m` fixed across a surrogate split, and by ``--n-half`` for a
    like-for-like check against an older run). Returns 0 -- meaning "drop this
    session" -- when the size is below `min_half`, or when a forced size is
    larger than either cell can supply two disjoint halves of.
    """
    m = int(n_half) if n_half is not None else min(counts[s] for s in STRATEGIES) // 2
    if m < min_half:
        return 0
    if any(counts[s] < 2 * m for s in STRATEGIES):
        return 0
    return m


@dataclass(frozen=True)
class SessionPool:
    """One session's maze-restricted trials, everything the CV needs.

    Held separately from the matrix computation because the permutation null
    reuses it: shuffling labels within (session, maze) permutes `y` but leaves
    `counts` -- and therefore `m` -- invariant, so the pool and its `SplitPlan`
    are built once and only `y` is resampled.
    """

    session: str
    X: np.ndarray  # (n_maze_trials, d)
    y: np.ndarray  # (n_maze_trials,) int, 0 = H, 1 = S
    trial_index: np.ndarray  # (n_maze_trials,) int, 1-based trial id = time order
    counts: dict  # {0: n_H, 1: n_S}
    m: int  # the ONE half size, both cells
    rng_seed: tuple  # for np.random.default_rng


@dataclass(frozen=True)
class SplitPlan:
    """The `n_splits` fixed split draws for one pool.

    Built once per pool and reused by the observed run and every permutation.
    That is sound because a within-(session, maze) label shuffle preserves both
    cell counts exactly, so `m` and the index space are shuffle-invariant; and
    it is better than reseeding per permutation because it makes "null
    variation is label variation, not split noise" a structural property
    rather than a consequence of matching seeds, and it gives common random
    numbers between the observed value and the null.

    `W` holds row-stochastic weight matrices so a half-mean is one BLAS call
    (``W_a @ rows``) instead of a ``(n_splits, m, d)`` fancy-index gather.
    `idx` is kept for auditing and for the gather-equivalence check.
    """

    m: int
    n_splits: int
    n_cell: dict  # {0: n_H, 1: n_S} -- what `y` must match
    idx: dict  # {cell: (idx_a, idx_b)}, each (n_splits, m) int
    W: dict  # {cell: (W_a, W_b)}, each (n_splits, n_cell) float


@dataclass
class SessionMatrix:
    """One session's 2x2, in both averaging conventions, plus its audit trail."""

    r: np.ndarray  # (2,2) tanh(mean z) -- the reported matrix
    z: np.ndarray  # (2,2) Fisher-z mean over kept splits
    r_raw: np.ndarray  # (2,2) arithmetic mean over kept splits
    z_splits: np.ndarray | None  # (n_splits,2,2), NaN at dropped splits
    n_splits_used: int
    n_splits_dropped: int
    n_nan_by_entry: np.ndarray  # (2,2) int
    n_clipped_by_entry: np.ndarray  # (2,2) int


@dataclass
class PairResult:
    """Session-averaged 2x2, and everything needed to rebuild it."""

    maze: int
    variant: str
    mean: np.ndarray  # (2, 2) Fisher-z session average, back-transformed
    mean_raw: np.ndarray  # (2, 2) arithmetic session average of r_raw
    session_mats: list  # [SessionMatrix], aligned with session_names
    session_names: list
    cell_counts: list  # [{0|1 -> n trials}], aligned
    half_sizes: list  # [int m], aligned
    session_deltas: list  # per-session delta, aligned
    census: np.ndarray  # (2,) trials pooled over the sessions kept
    n_sessions: int
    n_sessions_in_scope: int
    n_sessions_dropped_nan: int
    min_trials: int
    min_half: int
    skipped_cells: tuple = ()
    d: int = 0
    extras: dict = field(default_factory=dict)


def build_pools(
    X,
    sessions,
    trials,
    mazes,
    *,
    maze,
    keep_sessions,
    label_lookup,
    min_trials=MIN_TRIALS,
    min_half=MIN_HALF,
    n_half=None,
    seed=0,
):
    """One `SessionPool` per session that can supply this maze's two cells.

    Sessions are iterated in sorted order. That is not cosmetic: iterating a
    `set` of session names would put the null's strata order at the mercy of
    Python's per-process string hash randomisation, so every p-value would
    vary between runs despite ``--seed``.

    A session is dropped unless both cells clear `min_trials` **and** the
    resulting `m` clears `min_half`. Dropped sessions are returned in the
    second element so the caller can report why coverage is what it is.
    """
    sessions = np.asarray(sessions).astype(str)
    trials = np.asarray(trials, dtype=int)
    mazes = np.asarray(mazes, dtype=int)
    y_full = labels_for_rows(sessions, trials, label_lookup)
    present_sessions = set(sessions.tolist())
    keep = [s for s in sorted(set(keep_sessions)) if s in present_sessions]

    pools, dropped = [], {}
    for session in keep:
        row_mask = (sessions == session) & np.isfinite(y_full) & (mazes == maze)
        if not row_mask.any():
            dropped[session] = "no labelled trial in this maze"
            continue
        ys = y_full[row_mask].astype(int)
        counts = {s: int(np.sum(ys == s)) for s in STRATEGIES}
        thin = [s for s in STRATEGIES if counts[s] < min_trials]
        if thin:
            dropped[session] = (
                f"cell {STRATEGY_TAG[thin[0]]} has {counts[thin[0]]} < {min_trials} trials"
            )
            continue

        m = half_size(counts, n_half=n_half, min_half=min_half)
        if m == 0:
            dropped[session] = (
                f"m = {min(counts.values()) // 2} < min_half {min_half} "
                f"(n_H={counts[H_CELL]}, n_S={counts[S_CELL]})"
            )
            continue

        pools.append(
            SessionPool(
                session=session,
                X=X[row_mask],
                y=ys,
                trial_index=trials[row_mask],
                counts=counts,
                m=m,
                # The maze is in the seed key, not just the session: sweeping
                # `--maze 4 5 6` off a session-only key would reuse the same
                # split draws across mazes and correlate their results.
                rng_seed=session_seed(seed, f"{session}|m{maze}"),
            )
        )
    return pools, dropped


def build_split_plan(pool, *, n_splits=N_SPLITS):
    """The `n_splits` disjoint equal-size half pairs for each of the two cells.

    ``argsort(rng.random((n_splits, n_s)))`` is the vectorised equivalent of
    `n_splits` independent ``rng.permutation(n_s)`` calls: each row is a
    uniform permutation, so slicing ``[:m]`` and ``[m:2m]`` gives two disjoint
    uniform random subsets of size `m`.

    Do **not** swap in `argpartition`. The 2m smallest keys would still be a
    uniform random subset, but their order *within* that block is a
    deterministic function of the key values, so splitting the block into two
    halves would not be a uniform random split.
    """
    rng = np.random.default_rng(pool.rng_seed)
    m = pool.m
    rows = np.arange(n_splits)[:, None]
    idx, W = {}, {}
    for s in STRATEGIES:  # fixed order 0, 1 -- the draw must be deterministic
        n_s = pool.counts[s]
        order = np.argsort(rng.random((n_splits, n_s)), axis=1)
        idx_a, idx_b = order[:, :m], order[:, m : 2 * m]
        w_a = np.zeros((n_splits, n_s))
        w_b = np.zeros((n_splits, n_s))
        w_a[rows, idx_a] = 1.0 / m
        w_b[rows, idx_b] = 1.0 / m
        idx[s], W[s] = (idx_a, idx_b), (w_a, w_b)
    return SplitPlan(
        m=m, n_splits=n_splits, n_cell=dict(pool.counts), idx=idx, W=W
    )


def pearson_batch_2x2(A, B):
    """``R[t, i, j] = pearson(A[t, i, :], B[t, j, :])``, batched over splits.

    Identical algebra and identical zero-variance -> NaN convention to
    `matrix.pairwise_pearson_matrix`, which stays in the tree as the scalar
    reference the checks compare against: same ``d < 2`` guard, same
    population std (``ddof=0``), and ``np.outer(a_std, b_std)`` expressed as
    the batched outer product.
    """
    d = A.shape[-1]
    if d < 2:
        return np.full(A.shape[:-1] + (B.shape[-2],), np.nan)
    Ac = A - A.mean(axis=-1, keepdims=True)
    Bc = B - B.mean(axis=-1, keepdims=True)
    a_sd = Ac.std(axis=-1)
    b_sd = Bc.std(axis=-1)
    cov = Ac @ np.swapaxes(Bc, -1, -2) / d
    denom = a_sd[..., :, None] * b_sd[..., None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        R = cov / denom
    zero = (a_sd[..., :, None] == 0) | (b_sd[..., None, :] == 0)
    return np.where(zero, np.nan, R)


def _nan_session_matrix(n_splits, n_dropped, n_nan_by_entry, keep_splits):
    nan2 = np.full((N_CELLS, N_CELLS), np.nan)
    return SessionMatrix(
        r=nan2.copy(),
        z=nan2.copy(),
        r_raw=nan2.copy(),
        z_splits=(
            np.full((n_splits, N_CELLS, N_CELLS), np.nan) if keep_splits else None
        ),
        n_splits_used=0,
        n_splits_dropped=n_dropped,
        n_nan_by_entry=n_nan_by_entry,
        n_clipped_by_entry=np.zeros((N_CELLS, N_CELLS), dtype=int),
    )


def session_pair_matrix(pool, y, plan, *, r_clip=R_CLIP, keep_splits=False):
    """One session's 2x2, from the plan's fixed splits.

    **The off-diagonal is not symmetrised.** The four entries are four
    distinct quantities::

        R[H,H] = pearson(a_H, b_H)      R[H,S] = pearson(a_H, b_S)
        R[S,S] = pearson(a_S, b_S)      R[S,H] = pearson(a_S, b_H)

    (Symmetrisation was always a no-op on the diagonal, since Pearson is
    symmetric; only the off-diagonal changes.)

    ``R[H,S]`` and ``R[S,H]`` are **two exchangeable draws of the same
    population quantity, not two directions of a meaningful axis.** Within a
    split, which half of a cell is "a" and which is "b" is an arbitrary random
    assignment, drawn independently per cell. So a gap between them is a
    statement about how noisy the estimate is at this `m`, and **must not be
    read as evidence of real asymmetry** between the strategies. Reporting
    them separately instead of pre-averaging buys exactly one thing: the
    spread is visible rather than hidden inside a mean.

    A split contributes all four entries or none (`keep` below). Without that,
    a zero-variance half-mean in cell *i* NaNs row *i* of `R`, so `r_ii` and
    `r_i,1-i` would be averaged over fewer splits than `r_1-i,1-i`, and
    ``0.5 * (r_HH + r_SS)`` would mix two different split sets.

    `y` is passed separately from `pool` so the permutation null and the drift
    surrogates can hand in a substituted label vector against an otherwise
    identical pool and plan.
    """
    counts_y = {s: int(np.sum(y == s)) for s in STRATEGIES}
    if counts_y != plan.n_cell:
        raise ValueError(
            f"label vector changes the cell counts ({counts_y} vs plan "
            f"{plan.n_cell}); the fixed split plan is only valid under a "
            "count-preserving relabelling"
        )

    n_splits, d = plan.n_splits, pool.X.shape[1]
    A = np.empty((n_splits, N_CELLS, d))
    B = np.empty((n_splits, N_CELLS, d))
    for s in STRATEGIES:
        rows_s = pool.X[y == s]
        w_a, w_b = plan.W[s]
        A[:, s, :] = w_a @ rows_s
        B[:, s, :] = w_b @ rows_s

    R = pearson_batch_2x2(A, B)
    n_nan_by_entry = np.isnan(R).sum(axis=0).astype(int)
    keep = np.isfinite(R).all(axis=(1, 2))
    n_dropped = int((~keep).sum())
    if not keep.any():
        return _nan_session_matrix(n_splits, n_dropped, n_nan_by_entry, keep_splits)

    R_keep = R[keep]
    n_clipped = (np.abs(R_keep) > r_clip).sum(axis=0).astype(int)
    Z = np.arctanh(np.clip(R_keep, -r_clip, r_clip))
    z_mean = Z.mean(axis=0)

    z_splits = None
    if keep_splits:
        z_splits = np.full((n_splits, N_CELLS, N_CELLS), np.nan)
        z_splits[keep] = Z

    return SessionMatrix(
        r=np.tanh(z_mean),
        z=z_mean,
        r_raw=R_keep.mean(axis=0),
        z_splits=z_splits,
        n_splits_used=int(keep.sum()),
        n_splits_dropped=n_dropped,
        n_nan_by_entry=n_nan_by_entry,
        n_clipped_by_entry=n_clipped,
    )


def fisher_mean(values, *, r_clip=R_CLIP):
    """``tanh(mean(arctanh(r)))`` over a 1-D set of correlations.

    Used to average splits, sessions, and the two cross draws. Fisher-z is the
    right aggregation rule for correlations, but it is not neutral here and the
    direction matters: expanding about the raw mean with split variance
    ``s^2`` gives ``tanh(mean(arctanh r)) - mean(r) ~= s^2 * r / (1 - r^2)``,
    which is odd in r and grows steeply in |r|. The diagonal sits higher than
    the off-diagonal, so **it gets the larger upward push and Fisher-z inflates
    delta -- in the direction of the hypothesis**, by roughly 4-10% in
    plausible regimes. `z` and `p` largely cancel it because the null uses the
    same rule, but not exactly. That is why `r_raw` / `delta_obs_raw` are
    carried everywhere alongside: if the two ever disagree qualitatively, the
    raw one is the one to believe, because it has no mechanism favouring the
    diagonal.
    """
    v = np.asarray(values, dtype=float)
    if not np.isfinite(v).all():
        return np.nan
    return float(np.tanh(np.mean(np.arctanh(np.clip(v, -r_clip, r_clip)))))


def cross_r(mat, *, r_clip=R_CLIP):
    """The single off-diagonal that enters delta: Fisher-z mean of both draws."""
    return fisher_mean([mat[H_CELL, S_CELL], mat[S_CELL, H_CELL]], r_clip=r_clip)


def cross_gap(mat):
    """``|R[H,S] - R[S,H]||`` -- how far apart the two exchangeable draws landed.

    A QC number, not a finding. See `session_pair_matrix`.
    """
    a, b = mat[H_CELL, S_CELL], mat[S_CELL, H_CELL]
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    return float(abs(a - b))


def pair_delta(r_hh, r_ss, r_hs):
    """``0.5 * (r_HH + r_SS) - r_HS``, NaN-propagating.

    Formed in **r-space** -- back-transform each cell, then difference. A
    z-space delta would manufacture the exact artifact this module is built to
    avoid: `arctanh` blows up near 1, so ``0.5 * (z_HH + z_SS) - z_HS`` would
    report how close the diagonal sits to the ceiling rather than the size of
    the gap (r_diag 0.90 -> 0.95 moves the r-space delta by 0.05 and the
    z-space delta by 0.36).
    """
    if not (np.isfinite(r_hh) and np.isfinite(r_ss) and np.isfinite(r_hs)):
        return np.nan
    return 0.5 * (r_hh + r_ss) - r_hs


def matrix_delta(mat, *, r_clip=R_CLIP):
    return pair_delta(mat[H_CELL, H_CELL], mat[S_CELL, S_CELL], cross_r(mat, r_clip=r_clip))


def amgm_floor(r_hh, r_ss, r_hs):
    """The part of delta that unequal H/S reliability explains on its own.

    Under classical test theory, with rho_H and rho_S the reliabilities of an
    m-trial mean and rho_true the correlation between the two strategies' true
    profiles::

        r_HH ~= rho_H     r_SS ~= rho_S     r_HS ~= sqrt(rho_H * rho_S) * rho_true

    The off-diagonal is attenuated by the **geometric** mean of the two
    reliabilities, while delta subtracts it from the **arithmetic** mean. So
    even at ``rho_true = 1`` -- the two strategies sharing one identical gaze
    profile, the very null this analysis argues against --

        delta = 0.5 * (rho_H + rho_S) - sqrt(rho_H * rho_S)
              = 0.5 * (sqrt(rho_H) - sqrt(rho_S))^2  >=  0

    r_HH = 0.8 against r_SS = 0.4 puts that floor at 0.034; 0.8 against 0.1
    puts it at 0.167. Equal `m` removes the *count* source of
    ``rho_H != rho_S`` but not the *within-cell variance* source, and H and S
    need not be equally variable trial to trial.

    **The permutation null cannot see this.** Shuffling labels makes both
    pseudo-cells draws from one mixture, so ``rho_H ~= rho_S`` in the null and
    the floor is ~0 there while positive in the data. It is not absorbed by z
    or p, which is why it is reported.

    Returns ``(floor, beyond, imbalance)``: the floor, the part of delta that
    survives it (``sqrt(r_HH * r_SS) - r_HS``, still a raw difference of
    correlations within one panel), and ``r_HH - r_SS``. The first two are NaN
    when either diagonal is negative -- routine under `mean_removed`, where
    `imbalance` is the number to read instead.
    """
    if not (np.isfinite(r_hh) and np.isfinite(r_ss)):
        return np.nan, np.nan, np.nan
    imbalance = float(r_hh - r_ss)
    if r_hh < 0 or r_ss < 0 or not np.isfinite(r_hs):
        return np.nan, np.nan, imbalance
    gm = sqrt(r_hh * r_ss)
    return float(0.5 * (r_hh + r_ss) - gm), float(gm - r_hs), imbalance


def pair_table(result, *, r_clip=R_CLIP):
    """``(r_HH, r_SS, r_HS, n_sessions)`` from a `PairResult`.

    All four entries come from one session set by construction:
    `pair_matrix_from_pools` admits a session only when its whole 2x2 is
    finite, and averages with a plain mean rather than `nanmean`, so no NaN can
    reach the average and no entry can be built from different recording days
    than its neighbours.
    """
    if not result.n_sessions:
        return np.nan, np.nan, np.nan, 0
    mat = result.mean
    return (
        float(mat[H_CELL, H_CELL]),
        float(mat[S_CELL, S_CELL]),
        cross_r(mat, r_clip=r_clip),
        int(result.n_sessions),
    )


def split_deltas(session_mats):
    """``(n_splits,)`` per-split, session-averaged delta. NaN where any session
    dropped that split index.

    Two caveats, both real:

    * ``mean_i(delta_i) != delta_headline``. The headline averages first and
      differences after a nonlinear `tanh`, so the mean of these is not it.
      These exist to measure *spread*, not to restate the estimate.
    * Split index `i` is an arbitrary pairing across sessions -- nothing links
      session 1's split `i` to session 2's. That is harmless for the
      distribution (the draws are i.i.d. and independent across sessions, so
      any pairing has the same law), but the resulting SD is **not** a
      confidence interval on delta across sessions or animals. It holds the
      data fixed and varies only the split draw.
    """
    usable = [sm for sm in session_mats if sm.z_splits is not None]
    if not usable:
        return np.array([])
    Z = np.stack([sm.z_splits for sm in usable])  # (n_sessions, T, 2, 2)
    full = np.isfinite(Z).all(axis=(2, 3)).all(axis=0)  # (T,)
    out = np.full(Z.shape[1], np.nan)
    if not full.any():
        return out
    Zm = Z[:, full].mean(axis=0)  # (n_full, 2, 2)
    cross = np.tanh(0.5 * (Zm[:, H_CELL, S_CELL] + Zm[:, S_CELL, H_CELL]))
    diag = 0.5 * (np.tanh(Zm[:, H_CELL, H_CELL]) + np.tanh(Zm[:, S_CELL, S_CELL]))
    out[full] = diag - cross
    return out


def pair_matrix_from_pools(
    pools,
    plans,
    *,
    maze,
    variant="full",
    ys=None,
    r_clip=R_CLIP,
    min_trials=MIN_TRIALS,
    min_half=MIN_HALF,
    n_sessions_in_scope=0,
    skipped_cells=(),
    keep_splits=False,
):
    """Session-averaged 2x2 over `pools`, optionally against substituted labels.

    `ys` is a list of label vectors aligned with `pools` -- the hook the
    permutation null and the `drift` surrogates both use to push different
    groupings through a provably identical estimator.

    Sessions are averaged **unweighted**, in z-space. Weighting by trial count
    is deliberately not done: a session with more trials gets cleaner
    half-means and hence a higher diagonal, so n-weighting would amplify
    exactly the count bias the equal-`m` rule exists to remove. A pooled
    z-mean would do the same thing by the back door, weighting each session by
    its surviving-split count.
    """
    session_mats, session_names = [], []
    cell_counts, half_used, deltas = [], [], []
    census = np.zeros(N_CELLS, dtype=int)
    n_dropped_nan = 0

    for i, pool in enumerate(pools):
        y = pool.y if ys is None else ys[i]
        sm = session_pair_matrix(
            pool, y, plans[i], r_clip=r_clip, keep_splits=keep_splits
        )
        if not np.isfinite(sm.z).all():
            n_dropped_nan += 1
            continue
        session_mats.append(sm)
        session_names.append(pool.session)
        counts = {s: int(np.sum(y == s)) for s in STRATEGIES}
        for s in STRATEGIES:
            census[s] += counts[s]
        cell_counts.append(counts)
        half_used.append(pool.m)
        deltas.append(matrix_delta(sm.r, r_clip=r_clip))

    if not session_mats:
        nan2 = np.full((N_CELLS, N_CELLS), np.nan)
        return PairResult(
            maze=maze, variant=variant, mean=nan2.copy(), mean_raw=nan2.copy(),
            session_mats=[], session_names=[], cell_counts=[], half_sizes=[],
            session_deltas=[], census=census, n_sessions=0,
            n_sessions_in_scope=n_sessions_in_scope,
            n_sessions_dropped_nan=n_dropped_nan,
            min_trials=min_trials, min_half=min_half,
            skipped_cells=tuple(skipped_cells),
        )

    # Plain mean, never nanmean: every admitted session is fully finite, so
    # the four entries provably share one session set.
    mean_z = np.stack([sm.z for sm in session_mats]).mean(axis=0)
    mean_raw = np.stack([sm.r_raw for sm in session_mats]).mean(axis=0)

    return PairResult(
        maze=maze, variant=variant, mean=np.tanh(mean_z), mean_raw=mean_raw,
        session_mats=session_mats, session_names=session_names,
        cell_counts=cell_counts, half_sizes=half_used, session_deltas=deltas,
        census=census, n_sessions=len(session_mats),
        n_sessions_in_scope=n_sessions_in_scope,
        n_sessions_dropped_nan=n_dropped_nan,
        min_trials=min_trials, min_half=min_half,
        skipped_cells=tuple(skipped_cells),
    )


def build_plans(pools, *, n_splits=N_SPLITS):
    return [build_split_plan(pool, n_splits=n_splits) for pool in pools]


def permutation_null(
    pools,
    *,
    maze,
    variant="full",
    n_splits=N_SPLITS,
    n_perm=1000,
    seed=0,
    r_clip=R_CLIP,
    plans=None,
    **kw,
):
    """Within-(session, maze) label-shuffle null for `pair_delta`.

    Shuffles H/S labels among one maze's trials within each session and
    recomputes the session-averaged delta `n_perm` times. The shuffle is
    confined to a (session, maze) stratum, so it preserves every cell's trial
    count exactly -- and therefore `m`, the split plan and the admitted session
    set. The shuffled matrices carry a byte-identical estimator, which is what
    makes this a test of the labels rather than of the coverage, and why
    `null_mean` rather than zero is delta's reference: at d as low as 5 the
    estimator's own floor is measured, not assumed.

    A permutation null rather than a bootstrap CI: the question is not how much
    delta varies between recording days but whether it is distinguishable from
    its own noise floor at this dimensionality and these counts.

    **What it does not cover.** The shuffle is blind to trial order, so it
    destroys any real time-clustering of the labels before the null is built.
    A delta driven entirely by "H and S occupy different parts of the session,
    and the signal drifts" can still land far outside this null. It answers
    "is this delta unusual for any random 2-way split of this size?", not "is
    it explained by H/S's actual temporal positions?" -- that is `drift.py`.

    Returns ``(row, observed, draws)``.
    """
    if plans is None:
        plans = build_plans(pools, n_splits=n_splits)

    observed = pair_matrix_from_pools(
        pools, plans, maze=maze, variant=variant, r_clip=r_clip,
        keep_splits=True, **kw
    )
    r_hh, r_ss, r_hs, n_both = pair_table(observed, r_clip=r_clip)
    obs = pair_delta(r_hh, r_ss, r_hs)

    raw = observed.mean_raw
    obs_raw = (
        pair_delta(
            raw[H_CELL, H_CELL],
            raw[S_CELL, S_CELL],
            0.5 * (raw[H_CELL, S_CELL] + raw[S_CELL, H_CELL]),
        )
        if observed.n_sessions
        else np.nan
    )

    # A session whose maze trials are all one label is invariant under every
    # permutation, so it would dilute the null toward the observed and make the
    # test conservative. `build_pools` already excludes those, but count them
    # so the report states the exclusion rather than leaving it implicit.
    strata, unexchangeable, log10_distinct = set(), 0, 0.0
    for i, pool in enumerate(pools):
        if np.unique(pool.y).size >= 2:
            strata.add(i)
            log10_distinct += log10(
                comb(pool.y.size, int(np.sum(pool.y == H_CELL)))
            )
        else:
            unexchangeable += 1

    rng_perm = np.random.default_rng((seed, NULL_SEED_TAG))
    draws = np.full(n_perm, np.nan)
    for p in range(n_perm):
        ys = [
            rng_perm.permutation(pool.y) if i in strata else pool.y
            for i, pool in enumerate(pools)
        ]
        res = pair_matrix_from_pools(
            pools, plans, maze=maze, variant=variant, ys=ys, r_clip=r_clip, **kw
        )
        draws[p] = pair_delta(*pair_table(res, r_clip=r_clip)[:3])

    floor, beyond, imbalance = amgm_floor(r_hh, r_ss, r_hs)
    per_split = split_deltas(observed.session_mats)
    finite_split = per_split[np.isfinite(per_split)]
    clip_total = (
        np.stack([sm.n_clipped_by_entry for sm in observed.session_mats]).sum(axis=0)
        if observed.n_sessions
        else np.zeros((N_CELLS, N_CELLS), dtype=int)
    )

    half_sizes = observed.half_sizes or [0]
    null = draws[np.isfinite(draws)]
    row = dict(
        r_HH=r_hh, r_SS=r_ss, r_HS=r_hs,
        r_HS_ab=float(observed.mean[H_CELL, S_CELL]) if observed.n_sessions else np.nan,
        r_HS_ba=float(observed.mean[S_CELL, H_CELL]) if observed.n_sessions else np.nan,
        cross_gap=cross_gap(observed.mean) if observed.n_sessions else np.nan,
        delta_obs=obs,
        delta_obs_raw=obs_raw,
        delta_fisher_minus_raw=(
            obs - obs_raw if np.isfinite(obs) and np.isfinite(obs_raw) else np.nan
        ),
        delta_ab=pair_delta(r_hh, r_ss, observed.mean[H_CELL, S_CELL])
        if observed.n_sessions else np.nan,
        delta_ba=pair_delta(r_hh, r_ss, observed.mean[S_CELL, H_CELL])
        if observed.n_sessions else np.nan,
        delta_amgm_floor=floor,
        delta_beyond_floor=beyond,
        diag_imbalance=imbalance,
        n_sessions_both=n_both,
        n_sessions_in_scope=observed.n_sessions_in_scope,
        n_sessions_dropped_nan=observed.n_sessions_dropped_nan,
        n_strata=len(strata),
        n_sessions_unexchangeable=unexchangeable,
        n_perm_distinct_log10=round(log10_distinct, 3),
        n_splits=n_splits,
        n_splits_dropped_total=int(
            sum(sm.n_splits_dropped for sm in observed.session_mats)
        ),
        n_clipped_HH=int(clip_total[H_CELL, H_CELL]),
        n_clipped_SS=int(clip_total[S_CELL, S_CELL]),
        n_clipped_HS=int(clip_total[H_CELL, S_CELL] + clip_total[S_CELL, H_CELL]),
        half_size_min=int(min(half_sizes)),
        half_size_median=int(np.median(half_sizes)),
        half_size_max=int(max(half_sizes)),
        delta_split_sd=float(np.std(finite_split, ddof=1)) if finite_split.size > 1 else np.nan,
        delta_split_p025=float(np.percentile(finite_split, 2.5)) if finite_split.size else np.nan,
        delta_split_p975=float(np.percentile(finite_split, 97.5)) if finite_split.size else np.nan,
        n_splits_full=int(finite_split.size),
        null_mean=np.nan, null_sd=np.nan, null_p025=np.nan, null_p975=np.nan,
        p_two_sided=np.nan, p_one_sided_greater=np.nan, z_perm=np.nan,
        n_perm=int(null.size),
    )
    for entry, name in ((H_CELL, "r_HH"), (S_CELL, "r_SS")):
        vals = [sm.r[entry, entry] for sm in observed.session_mats]
        row[f"{name}_session_sd"] = (
            float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        )
    row["delta_split_sem"] = (
        row["delta_split_sd"] / sqrt(row["n_splits_full"])
        if np.isfinite(row["delta_split_sd"]) and row["n_splits_full"]
        else np.nan
    )

    if np.isfinite(obs) and null.size:
        null_mean = float(np.mean(null))
        null_sd = float(np.std(null, ddof=1)) if null.size > 1 else np.nan
        row.update(
            null_mean=null_mean,
            null_sd=null_sd,
            null_p025=float(np.percentile(null, 2.5)),
            null_p975=float(np.percentile(null, 97.5)),
            # +1 Laplace smoothing, so a permutation p is never exactly 0, and
            # centred on the null mean rather than on zero.
            p_two_sided=float(
                (np.sum(np.abs(null - null_mean) >= abs(obs - null_mean)) + 1)
                / (null.size + 1)
            ),
            # Directional: the hypothesis is that cross-strategy similarity
            # falls *below* the reliability ceiling, i.e. delta > 0.
            p_one_sided_greater=float((np.sum(null >= obs) + 1) / (null.size + 1)),
            z_perm=(
                float((obs - null_mean) / null_sd)
                if np.isfinite(null_sd) and null_sd > 0
                else np.nan
            ),
        )
        # The "is n_splits enough?" acceptance criterion: residual split-draw
        # noise in the headline delta, relative to the null's own width.
        if np.isfinite(row["delta_split_sem"]) and np.isfinite(null_sd) and null_sd > 0:
            row["split_sem_over_null_sd"] = row["delta_split_sem"] / null_sd
        else:
            row["split_sem_over_null_sd"] = np.nan
    else:
        row["split_sem_over_null_sd"] = np.nan

    return row, observed, draws
