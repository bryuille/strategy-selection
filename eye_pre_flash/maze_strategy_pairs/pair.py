"""The 2x2 (strategy x strategy) split-half similarity for a single maze.

`eye_pre_flash.maze_strategy_pairs` builds a 12x12 (maze x decoded strategy) matrix and
argues that its decisive read is the same-maze, H-vs-S comparison: geometry is
identical on both sides, so ``r(H,S)`` falling below that maze's own split-half
reliability is strategy-dependent gaze sampling and not a visual confound.
This module promotes that read to the whole analysis -- two cells, one maze --
and in doing so fixes the estimator handicap that read carried inside the
12x12.

The handicap is `corr.matrix.session_half_sizes`: it picks **one** shared half
size per session, the minimum over every cell that clears ``min_stable``, so
maze 4's comfortable trial counts get split into halves sized by maze 1's
near-empty sequential cell. Measured from corr's own
`results_raw_k6.csv` on 2026-09-10, Nielsen ``Nov_1_g0`` has 45 maze-4
hierarchical and 36 sequential trials but corr splits it into halves of 5;
restricted to those two cells the shared half is 18. Same estimator, 3.6x the
trials per half. The consequence is that the numbers here will not equal
corr's maze-4 cells -- every ``r`` rises -- and that is a precision gain, not
a finding. `--n-half` forces corr's size back for a like-for-like check, and
``half_corr_rule`` in `results_raw` records what corr's unrestricted rule
would have chosen on every session.

Everything else is corr's estimator unchanged, and deliberately so: the
generic pieces are imported from `corr.matrix` rather than reimplemented
(`session_half_sizes`, `pairwise_pearson_matrix`, `session_seed`,
`MIN_TRIALS`, `MIN_STABLE`), so the two packages cannot drift apart in the
half-size rule, the zero-variance convention or the seeding. Only the three
places corr hardcodes ``cells.N_CELLS = 12`` are re-expressed here for two
cells.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import comb, log10

import numpy as np

from eye_pre_flash.classifier.labels import labels_for_rows
from eye_pre_flash.maze_strategy_pairs.cells import MAZES, STRATEGIES, STRATEGY_TAG
from eye_pre_flash.maze_strategy_pairs.cells import cell_index as corr_cell_index
from eye_pre_flash.maze_strategy_pairs.matrix import (
    MIN_STABLE,
    MIN_TRIALS,
    pairwise_pearson_matrix,
    session_half_sizes,
    session_seed,
)
from eye_pre_flash.plotting.similarities.common import N_SPLITS
from eye_pre_flash.plotting.similarities.heatmap_eq import equal_halves

N_CELLS = 2
H_CELL, S_CELL = 0, 1  # matrix row/col 0 = hierarchical, 1 = sequential
DEFAULT_MAZE = 4
DEFAULT_MONKEY = "Nielsen"

# How the two cells' half size is chosen. `stable_shared` is corr's rule
# verbatim (`corr.matrix.session_half_sizes`): the shared size is the minimum
# over cells clearing `min_stable`, and a cell that misses `min_stable` falls
# back to its own n // 2. With twelve cells that fallback is a corner case;
# with two it lands directly on the statistic, because r(H,H) computed from
# two large halves is less noisy than r(H,S) computed from a large half
# against a small one, and delta is exactly their difference. `min_pair`
# instead takes ``min(n_H, n_S) // 2`` whenever both cells clear `min_trials`,
# so the halves are always balanced and the thin flag becomes purely a
# warning. The two rules coincide whenever both cells clear `min_stable`,
# which is every Nielsen maze-4 session in every scope -- they differ only
# where one strategy is genuinely thin, e.g. Faure maze 4.
HALF_RULES = ("stable_shared", "min_pair")
NULL_SEED_TAG = 11  # tags 1, 2, 3 are taken by corr.examples and corr.matrix


def cell_labels(maze):
    """``["4H", "4S"]`` for maze 4."""
    return [f"{maze}{STRATEGY_TAG[s]}" for s in STRATEGIES]


def skip_pair_cells(source_skip_cells, maze):
    """corr's 12-cell skip indices, translated to this maze's 0/1 cells.

    `corr.labels.SOURCE_SKIP_CELLS` blanks 1S and 6H under the SVM source --
    the anchor-minority cells that are near-empty by the SVM's own
    construction, skipped rather than marked thin. Translated here so
    ``--source svm --maze 1`` reports "no pair exists" instead of silently
    correlating one cell against nothing.
    """
    out = []
    for strategy in STRATEGIES:
        if corr_cell_index(maze, strategy) in tuple(source_skip_cells):
            out.append(strategy)
    return tuple(out)


@dataclass
class SessionPool:
    """One session's maze-restricted trials, everything the CV needs.

    Held separately from the matrix computation because the permutation null
    reuses it: shuffling labels within (session, maze) permutes `y` but leaves
    `counts`, `half_sizes` and `stable` invariant -- each cell keeps its trial
    count exactly -- so the pool is built once and only `y` is resampled.
    That is what makes the null a test of the labels rather than of the
    coverage, and it is also why the null is cheap here.
    """

    session: str
    X: np.ndarray  # (n_maze_trials, d)
    y: np.ndarray  # (n_maze_trials,) int, 0 = H, 1 = S
    counts: dict  # {0: n_H, 1: n_S}
    half_sizes: dict  # {0: half_H, 1: half_S}
    stable: dict  # {0: bool, 1: bool}
    half_corr_rule: dict  # {0, 1} -> what corr's unrestricted 12-cell rule gives
    rng_seed: tuple  # for np.random.default_rng

    @property
    def equal_n(self):
        return self.half_sizes[H_CELL] == self.half_sizes[S_CELL]


@dataclass
class PairResult:
    """Session-averaged 2x2, and everything needed to rebuild it."""

    maze: int
    variant: str
    mean: np.ndarray  # (2, 2)
    session_mats: list  # each (2, 2), aligned with session_names
    session_names: list
    presence: np.ndarray  # (n_sessions, 2) bool
    stable: list  # [{0|1 -> bool}], aligned
    cell_counts: list  # [{0|1 -> n trials}], aligned
    half_sizes: list  # [{0|1 -> half size}], aligned
    half_corr_rule: list  # [{0|1 -> corr's 12-cell half size}], aligned
    session_deltas: list  # per-session 0.5*(r_HH + r_SS) - r_HS, aligned
    census: np.ndarray  # (2,) trials pooled over the sessions kept
    used: np.ndarray  # (2,) sessions the cell appeared in
    n_sessions: int
    n_sessions_in_scope: int
    n_degenerate: int
    min_trials: int
    min_stable: int
    half_rule: str
    skipped_cells: tuple = ()


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
    min_stable=MIN_STABLE,
    n_half=None,
    half_rule="stable_shared",
    seed=0,
):
    """One `SessionPool` per session that has both of this maze's cells.

    Sessions are iterated in sorted order. That is not cosmetic: `corr`'s
    `permutation_null` iterates ``set(keep_sessions)``, and Python randomises
    string hashing per process, so its strata order -- and therefore every
    p-value it writes -- varies between runs despite ``--seed``. Sorting here
    makes this package's null reproducible.

    A session is dropped unless **both** cells clear `min_trials`. With two
    cells a single cell yields no correlation at all, so there is nothing to
    mark thin and nothing to average.
    """
    if half_rule not in HALF_RULES:
        raise ValueError(f"unknown half_rule {half_rule!r}; choose from {HALF_RULES}")

    sessions = np.asarray(sessions).astype(str)
    trials = np.asarray(trials, dtype=int)
    mazes = np.asarray(mazes, dtype=int)
    y_full = labels_for_rows(sessions, trials, label_lookup)
    present_sessions = set(sessions.tolist())
    keep = [s for s in sorted(set(keep_sessions)) if s in present_sessions]

    pools = []
    for session in keep:
        labelled = (sessions == session) & np.isfinite(y_full)
        row_mask = labelled & (mazes == maze)
        if not row_mask.any():
            continue
        ys = y_full[row_mask].astype(int)
        counts = {s: int(np.sum(ys == s)) for s in STRATEGIES}
        if any(counts[s] < min_trials for s in STRATEGIES):
            continue

        cells = {s: X[row_mask][ys == s] for s in STRATEGIES}
        half_sizes, stable = session_half_sizes(
            cells, min_stable=min_stable, n_half=n_half
        )
        if half_rule == "min_pair" and n_half is None:
            shared = min(counts[s] for s in STRATEGIES) // 2
            half_sizes = {s: shared for s in STRATEGIES}

        # What corr's unrestricted 12-cell rule would have chosen for this
        # session, so the half-size difference is auditable from the CSV
        # alone and can never be mistaken for a result.
        corr_cells = {}
        for m in MAZES:
            for s in STRATEGIES:
                rows = X[labelled][
                    (mazes[labelled] == m) & (y_full[labelled].astype(int) == s)
                ]
                if len(rows) >= min_trials:
                    corr_cells[corr_cell_index(m, s)] = rows
        corr_halves, _ = session_half_sizes(
            corr_cells, min_stable=min_stable, n_half=None
        )
        half_corr_rule = {
            s: corr_halves.get(corr_cell_index(maze, s), 0) for s in STRATEGIES
        }

        pools.append(
            SessionPool(
                session=session,
                X=X[row_mask],
                y=ys,
                counts=counts,
                half_sizes=half_sizes,
                stable=stable,
                half_corr_rule=half_corr_rule,
                # The maze is in the seed key, not just the session: sweeping
                # `--maze 4 5 6` off a session-only key would reuse the same
                # split permutations across mazes and correlate their results
                # through the draws.
                rng_seed=session_seed(seed, f"{session}|m{maze}"),
            )
        )
    return pools


def session_pair_matrix(pool, y, rng, *, n_splits=N_SPLITS):
    """One session's 2x2 CV Pearson r, and its degenerate-split count.

    `corr.matrix.session_cv_matrix`'s body with ``n = 2``: `equal_halves`
    subsamples each cell to its half size, the two half-means go through
    `pairwise_pearson_matrix`, and the result is symmetrised as
    ``0.5 * (R_ab + R_ab.T)`` -- so the diagonal is a cell's own split-half
    reliability (its two halves are disjoint) and the off-diagonal is the
    cross-strategy correlation at the same half size.

    `y` is passed separately from `pool` so the permutation null can hand in a
    shuffled label vector against an otherwise identical pool.
    """
    acc = np.zeros((N_CELLS, N_CELLS))
    counts = np.zeros((N_CELLS, N_CELLS))
    n_degenerate = 0

    rows_by_cell = {s: pool.X[y == s] for s in STRATEGIES}
    present = sorted(
        s
        for s in STRATEGIES
        if pool.half_sizes.get(s, 0) >= 1
        and len(rows_by_cell[s]) >= 2 * pool.half_sizes[s]
    )
    if len(present) < N_CELLS:
        return np.full((N_CELLS, N_CELLS), np.nan), 0

    d = pool.X.shape[1]
    for _ in range(n_splits):
        mean_a = np.empty((N_CELLS, d))
        mean_b = np.empty((N_CELLS, d))
        for s in present:
            rows = rows_by_cell[s]
            half = pool.half_sizes[s]
            idx_a, idx_b = equal_halves(len(rows), half, rng)
            mean_a[s] = rows[idx_a].mean(axis=0)
            mean_b[s] = rows[idx_b].mean(axis=0)

        R_ab = pairwise_pearson_matrix(mean_a, mean_b)
        r_final = 0.5 * (R_ab + R_ab.T)
        n_degenerate += 2 * int(np.sum(np.isnan(R_ab)))

        finite = np.isfinite(r_final)
        acc += np.where(finite, r_final, 0.0)
        counts += finite

    out = np.full((N_CELLS, N_CELLS), np.nan)
    ok = counts > 0
    out[ok] = acc[ok] / counts[ok]
    return out, n_degenerate


def pair_delta(r_hh, r_ss, r_hs):
    """``0.5 * (r_HH + r_SS) - r_HS``, NaN-propagating.

    The reliability-referenced statistic: how far the cross-strategy
    correlation falls below the mean of the two strategies' own split-half
    ceilings, at the same half size on both sides.
    """
    if not (np.isfinite(r_hh) and np.isfinite(r_ss) and np.isfinite(r_hs)):
        return np.nan
    return 0.5 * (r_hh + r_ss) - r_hs


def matrix_delta(mat):
    return pair_delta(mat[H_CELL, H_CELL], mat[S_CELL, S_CELL], mat[H_CELL, S_CELL])


def pair_table(session_mats, presence):
    """``(r_HH, r_SS, r_HS, n_sessions)`` on the matched session set.

    `corr.cells.within_maze_table` collapsed to one maze, and it keeps that
    function's discipline for the same reason: all three numbers come from
    exactly the sessions where **both** cells qualified, so a cell that
    qualified on more recording days than its partner cannot pull its own
    average onto different days. Here that set is almost always every kept
    session -- `build_pools` already drops a session missing either cell --
    but the rule is enforced rather than assumed.
    """
    if presence.size == 0:
        return np.nan, np.nan, np.nan, 0
    both = np.flatnonzero(presence[:, H_CELL] & presence[:, S_CELL])
    if both.size == 0:
        return np.nan, np.nan, np.nan, 0
    mats = [session_mats[i] for i in both]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (
            float(np.nanmean([m[H_CELL, H_CELL] for m in mats])),
            float(np.nanmean([m[S_CELL, S_CELL] for m in mats])),
            float(np.nanmean([m[H_CELL, S_CELL] for m in mats])),
            int(both.size),
        )


def pair_matrix_from_pools(
    pools, *, maze, variant="full", n_splits=N_SPLITS, ys=None,
    half_rule="stable_shared", min_trials=MIN_TRIALS, min_stable=MIN_STABLE,
    n_sessions_in_scope=0, skipped_cells=(),
):
    """Session-averaged 2x2 over `pools`, optionally against substituted labels.

    `ys` is a list of label vectors aligned with `pools`, which is how the
    permutation null feeds shuffled labels through the identical estimator.
    Sessions are averaged unweighted, as `corr.matrix.label_similarity_matrix`
    does. Weighting by trial count is deliberately not done: a session with
    more trials gets cleaner half-means and hence a higher diagonal, so
    n-weighting would amplify exactly the count bias `equal_halves` exists to
    remove. Every per-session matrix reaches `results_raw`, so a weighted
    version stays reconstructable without a rerun.
    """
    session_mats, session_names, presence = [], [], []
    stable_flags, cell_counts, half_used, half_corr, deltas = [], [], [], [], []
    census = np.zeros(N_CELLS, dtype=int)
    used = np.zeros(N_CELLS, dtype=int)
    n_degenerate_total = 0

    for i, pool in enumerate(pools):
        y = pool.y if ys is None else ys[i]
        rng = np.random.default_rng(pool.rng_seed)
        mat, n_deg = session_pair_matrix(pool, y, rng, n_splits=n_splits)
        if not np.isfinite(mat).any():
            continue
        session_mats.append(mat)
        session_names.append(pool.session)
        n_degenerate_total += n_deg
        here = np.zeros(N_CELLS, dtype=bool)
        for s in STRATEGIES:
            n_s = int(np.sum(y == s))
            census[s] += n_s
            if n_s >= min_trials:
                used[s] += 1
                here[s] = True
        presence.append(here)
        stable_flags.append(dict(pool.stable))
        cell_counts.append({s: int(np.sum(y == s)) for s in STRATEGIES})
        half_used.append(dict(pool.half_sizes))
        half_corr.append(dict(pool.half_corr_rule))
        deltas.append(matrix_delta(mat))

    if not session_mats:
        return PairResult(
            maze=maze, variant=variant,
            mean=np.full((N_CELLS, N_CELLS), np.nan),
            session_mats=[], session_names=[],
            presence=np.zeros((0, N_CELLS), dtype=bool),
            stable=[], cell_counts=[], half_sizes=[], half_corr_rule=[],
            session_deltas=[], census=census, used=used, n_sessions=0,
            n_sessions_in_scope=n_sessions_in_scope, n_degenerate=n_degenerate_total,
            min_trials=min_trials, min_stable=min_stable, half_rule=half_rule,
            skipped_cells=tuple(skipped_cells),
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(np.stack(session_mats), axis=0)

    return PairResult(
        maze=maze, variant=variant, mean=mean, session_mats=session_mats,
        session_names=session_names, presence=np.stack(presence),
        stable=stable_flags, cell_counts=cell_counts, half_sizes=half_used,
        half_corr_rule=half_corr, session_deltas=deltas, census=census, used=used,
        n_sessions=len(session_mats), n_sessions_in_scope=n_sessions_in_scope,
        n_degenerate=n_degenerate_total, min_trials=min_trials,
        min_stable=min_stable, half_rule=half_rule,
        skipped_cells=tuple(skipped_cells),
    )


def thin_mask(result):
    """``(2,)`` bool: cell was thin in most sessions it appeared in.

    `corr.figures.cell_thin_mask`'s rule at two cells, kept here rather than
    in `figures` because the report needs it too.
    """
    stable_count = np.zeros(N_CELLS)
    seen_count = np.zeros(N_CELLS)
    for stable in result.stable:
        for s, is_stable in stable.items():
            seen_count[s] += 1
            stable_count[s] += int(is_stable)
    thin = np.ones(N_CELLS, dtype=bool)
    have = seen_count > 0
    thin[have] = (stable_count[have] / seen_count[have]) < 0.5
    return thin


def permutation_null(pools, *, maze, variant="full", n_splits=N_SPLITS, n_perm=1000,
                     seed=0, **kw):
    """Within-(session, maze) label-shuffle null for `pair_delta`.

    Shuffles H/S labels among **one maze's** trials within each session and
    recomputes the session-averaged delta `n_perm` times. Because the shuffle
    is confined to a (session, maze) stratum it preserves every cell's trial
    count exactly, and therefore both half sizes, both thin flags and the
    presence matrix as well -- the shuffled matrices carry the identical
    estimator, down to an unequal half size where one strategy is thin. That
    invariance is what makes this a test of the labels rather than of the
    coverage, and it is why `null_mean` rather than zero is delta's reference:
    at d as low as 6 the estimator's own floor is measured, not assumed.

    A permutation null rather than a bootstrap CI, for corr's reason
    (`corr.md` section 9): the question is not how much delta varies between
    recording days but whether it is distinguishable from its own noise floor
    at this dimensionality and these counts.

    Returns ``(row, observed, draws)``.
    """
    observed = pair_matrix_from_pools(
        pools, maze=maze, variant=variant, n_splits=n_splits, **kw
    )
    r_hh, r_ss, r_hs, n_both = pair_table(observed.session_mats, observed.presence)
    obs = pair_delta(r_hh, r_ss, r_hs)

    # A session whose maze trials are all one label is invariant under every
    # permutation: its 2x2 never changes, so it dilutes the null toward the
    # observed and makes the test conservative. `build_pools` already excludes
    # those (both cells must clear min_trials), but count them so the report
    # can say the exclusion happened rather than leaving it implicit.
    strata, unexchangeable, log10_distinct = [], 0, 0.0
    for i, pool in enumerate(pools):
        if np.unique(pool.y).size >= 2:
            strata.append(i)
            n = pool.y.size
            log10_distinct += log10(comb(n, int(np.sum(pool.y == H_CELL))))
        else:
            unexchangeable += 1

    rng_perm = np.random.default_rng((seed, NULL_SEED_TAG))
    draws = np.full(n_perm, np.nan)
    for p in range(n_perm):
        ys = []
        for i, pool in enumerate(pools):
            ys.append(rng_perm.permutation(pool.y) if i in strata else pool.y)
        res = pair_matrix_from_pools(
            pools, maze=maze, variant=variant, n_splits=n_splits, ys=ys, **kw
        )
        draws[p] = pair_delta(*pair_table(res.session_mats, res.presence)[:3])

    null = draws[np.isfinite(draws)]
    row = dict(
        r_HH=r_hh, r_SS=r_ss, r_HS=r_hs, delta_obs=obs,
        n_sessions_both=n_both, n_strata=len(strata),
        n_sessions_unexchangeable=unexchangeable,
        n_perm_distinct_log10=round(log10_distinct, 3),
        null_mean=np.nan, null_sd=np.nan, null_p025=np.nan, null_p975=np.nan,
        p_two_sided=np.nan, p_one_sided_greater=np.nan, z_perm=np.nan,
        n_perm=int(null.size),
    )
    if np.isfinite(obs) and null.size:
        null_mean = float(np.mean(null))
        null_sd = float(np.std(null, ddof=1)) if null.size > 1 else np.nan
        row.update(
            null_mean=null_mean,
            null_sd=null_sd,
            null_p025=float(np.percentile(null, 2.5)),
            null_p975=float(np.percentile(null, 97.5)),
            # +1 Laplace smoothing, so a permutation p is never exactly 0.
            # Centred on the null mean, not on zero -- corr's convention, so
            # the two packages' p-values mean the same thing.
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
    return row, observed, draws
