"""The split-half equal-halves CV engine, and a label-shuffle permutation null.

Per session, per split: draw an equal-size half from every qualifying (maze,
strategy) cell, mean each half into one d-dimensional vector, then
``r_ij = mean(pearson(A_i, B_j), pearson(B_i, A_j))`` -- independent trial
halves throughout, so the diagonal is a cell's own split-half reliability.
Average the 20 splits per session, then average sessions.

Three deliberate divergences from the deleted
`eye_pre_flash.plotting.similarities.heatmap_labels` (which did the same thing
for 102,400-dim spatial gaze maps):

1. **No active-bin mask.** That mask restricted the Pearson to bins any cell
   touched, necessary at 102,400 dims. At d <= 144 a mask can only shrink an
   already-short vector, so this uses `common.pearson` unmasked over every
   dimension.
2. **Every cell is computed.** `MIN_TRIALS` is the arithmetic floor (a cell
   must split into two halves of >= 2); below `MIN_STABLE` a cell is still
   computed, at its own smaller half size, and the caller marks it thin rather
   than dropping it.
3. **The half size is set by the stable cells only**, not by the thinnest cell
   in the session (see `session_half_sizes`) -- a session with one 2-trial cell
   no longer drags every other cell in it down to 1-trial halves.
"""

from __future__ import annotations

import warnings
import zlib
from dataclasses import dataclass

import numpy as np

from eye_pre_flash.corr import cells as cellmod
from eye_pre_flash.classifier.labels import labels_for_rows
from eye_pre_flash.plotting.similarities.common import N_SPLITS
from eye_pre_flash.plotting.similarities.heatmap_eq import equal_halves

MIN_TRIALS = 4  # hard floor: a cell must split into two halves of >= 2
MIN_STABLE = 8  # below this a cell is computed but marked thin


def session_seed(seed, session):
    """Deterministic per-session sub-seed. Not Python's `hash()` -- that is
    randomised per interpreter run for strings, which would make a session's
    matrix depend on process start rather than only on `seed`."""
    return (seed, zlib.crc32(str(session).encode()) & 0xFFFFFFFF)


def session_cells(X, mazes, y, *, min_trials=MIN_TRIALS, skip_cells=()):
    """``cell index -> (n_i, d) rows`` for one session's trials.

    `skip_cells` excludes cell indices entirely (used for the SVM source's
    anchor-minority cells, 1S and 6H -- see `eye_pre_flash.corr.labels`),
    rather than merely marking them thin.
    """
    mazes = np.asarray(mazes, dtype=int)
    y = np.asarray(y, dtype=int)
    out = {}
    for maze in cellmod.MAZES:
        for strategy in cellmod.STRATEGIES:
            idx = cellmod.cell_index(maze, strategy)
            if idx in skip_cells:
                continue
            rows = X[(mazes == maze) & (y == strategy)]
            if len(rows) < min_trials:
                continue
            out[idx] = rows
    return out


def session_half_sizes(cells, *, min_stable=MIN_STABLE, n_half=None):
    """One shared half size for the session's stable cells, per-cell sizes for thin ones.

    Returns ``(half_sizes, stable)``, both ``cell index -> value`` dicts.
    Setting the shared size from the stable cells (rather than the thinnest
    cell in the session, as the deleted `heatmap_labels` did) keeps stable
    cells mutually comparable and count-balanced without one thin cell forcing
    every cell in the session down to 1-trial halves -- unacceptable now that
    every cell is computed rather than dropped. Unbalanced halves are not
    merely noisier, they are biased in the direction being tested: the
    majority-strategy cell would get cleaner halves and hence a higher
    diagonal, so confining that bias to cells marked thin matters.

    ``n_half``, if given, fixes the shared size directly instead of deriving
    it from the data; cells too small to reach it fall back to their own
    ``n // 2`` and are marked thin, same as the data-derived case.
    """
    if n_half is not None:
        shared = int(n_half)
    else:
        stable_candidates = [
            len(rows) // 2 for rows in cells.values() if len(rows) >= min_stable
        ]
        shared = min(stable_candidates) if stable_candidates else None

    half_sizes, stable = {}, {}
    for idx, rows in cells.items():
        n = len(rows)
        if shared is not None and n >= max(min_stable, 2 * shared):
            half_sizes[idx] = shared
            stable[idx] = True
        else:
            half_sizes[idx] = n // 2
            stable[idx] = False
    return half_sizes, stable


def pairwise_pearson_matrix(A, B):
    """``R[i, j] = pearson(A[i], B[j])`` for every row pair, vectorised.

    Equivalent to calling `common.pearson` on every row pair (same
    zero-variance -> NaN convention), but as two matrix multiplications
    instead of ``len(A) * len(B)`` individual `np.corrcoef` calls -- the
    difference between milliseconds and minutes once this runs inside a
    1000-resample permutation null over 20+ sessions.
    """
    d = A.shape[1]
    if d < 2:
        return np.full((A.shape[0], B.shape[0]), np.nan)
    Ac = A - A.mean(axis=1, keepdims=True)
    Bc = B - B.mean(axis=1, keepdims=True)
    a_std = Ac.std(axis=1)
    b_std = Bc.std(axis=1)
    cov = Ac @ Bc.T / d
    with np.errstate(divide="ignore", invalid="ignore"):
        R = cov / np.outer(a_std, b_std)
    zero = (a_std == 0)[:, None] | (b_std == 0)[None, :]
    return np.where(zero, np.nan, R)


def session_cv_matrix(cells, rng, *, half_sizes, n_splits=N_SPLITS):
    """12x12 split-half CV Pearson r for one session, per-cell half sizes.

    Returns ``(matrix, n_degenerate)`` -- `n_degenerate` counts split-halves
    where both cells had data but the Pearson came back NaN anyway (a
    zero-variance half-mean, e.g. an `occ_bin` cell whose every trial visited
    every state), which would otherwise render identically to "no coverage".
    """
    n = cellmod.N_CELLS
    acc = np.zeros((n, n))
    counts = np.zeros((n, n))
    n_degenerate = 0

    present = sorted(
        idx
        for idx, rows in cells.items()
        if half_sizes.get(idx, 0) >= 1 and len(rows) >= 2 * half_sizes[idx]
    )
    if len(present) < 2:
        return np.full((n, n), np.nan), 0
    d = next(iter(cells.values())).shape[1]
    grid = np.ix_(present, present)

    for _ in range(n_splits):
        mean_a = np.empty((len(present), d))
        mean_b = np.empty((len(present), d))
        for pos, idx in enumerate(present):
            rows = cells[idx]
            half = half_sizes[idx]
            idx_a, idx_b = equal_halves(len(rows), half, rng)
            mean_a[pos] = rows[idx_a].mean(axis=0)
            mean_b[pos] = rows[idx_b].mean(axis=0)

        # r_final[i, j] = mean(pearson(a_i, b_j), pearson(b_i, a_j)); the
        # second term is pearson(a_j, b_i) with i, j swapped, i.e. R_ab.T.
        R_ab = pairwise_pearson_matrix(mean_a, mean_b)
        r_final = 0.5 * (R_ab + R_ab.T)
        n_degenerate += 2 * int(np.sum(np.isnan(R_ab)))

        finite = np.isfinite(r_final)
        acc[grid] += np.where(finite, r_final, 0.0)
        counts[grid] += finite

    out = np.full((n, n), np.nan)
    ok = counts > 0
    out[ok] = acc[ok] / counts[ok]
    return out, n_degenerate


@dataclass
class LabelSimilarityResult:
    mean: np.ndarray
    session_mats: list
    session_names: list
    presence: np.ndarray  # (n_sessions, N_CELLS) bool
    stable: list  # list of {cell index -> bool}, aligned with session_names
    cell_counts: list  # list of {cell index -> n trials}, aligned with session_names
    half_sizes: list  # list of {cell index -> half size}, aligned with session_names
    census: np.ndarray  # (N_CELLS,) trials pooled across ALL kept sessions
    used: np.ndarray  # (N_CELLS,) sessions the cell appeared in
    n_sessions: int
    n_degenerate: int
    min_trials: int
    min_stable: int


def label_similarity_matrix(
    X,
    sessions,
    trials,
    mazes,
    *,
    keep_sessions,
    label_lookup,
    n_splits=N_SPLITS,
    min_trials=MIN_TRIALS,
    min_stable=MIN_STABLE,
    skip_cells=(),
    n_half=None,
    seed=0,
):
    """Session-averaged 12x12 CV r, and everything needed to rebuild it.

    The CV is deliberately feature-agnostic: `X` is just an (n, d) block, the
    aggregate is always the half-mean, and the similarity is always Pearson.
    There is no per-feature branch anywhere below this point and there should
    not be one -- `occupancy`, `occupancy_bin` and `bigram` are only
    comparable to each other because the identical estimator is applied to
    all three, and a feature-specific aggregate (binarising `occ_bin`'s
    half-mean, say) would buy nothing and cost that comparability.

    ``X`` is the (n, d) feature block (one row per trial, any of `occ_ms`,
    `occ_bin`, `bigram`, already through a `eye_pre_flash.corr.variants`
    transform); ``sessions``, ``trials``, ``mazes`` are the aligned per-trial
    arrays from `eye_pre_flash.classifier.features.load_features`.
    ``label_lookup`` is ``(session, trial_id) -> 0.0/1.0`` (dendro or SVM).
    """
    sessions = np.asarray(sessions).astype(str)
    trials = np.asarray(trials, dtype=int)
    mazes = np.asarray(mazes, dtype=int)
    y_full = labels_for_rows(sessions, trials, label_lookup)
    keep = [s for s in sorted(set(keep_sessions)) if s in set(sessions.tolist())]

    n_cells = cellmod.N_CELLS
    census = np.zeros(n_cells, dtype=int)
    used = np.zeros(n_cells, dtype=int)
    session_mats, session_names, presence, stable_flags, cell_counts, half_used = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    n_degenerate_total = 0

    for session in keep:
        row_mask = (sessions == session) & np.isfinite(y_full)
        if not row_mask.any():
            continue
        Xs = X[row_mask]
        mz = mazes[row_mask]
        ys = y_full[row_mask].astype(int)

        for maze in cellmod.MAZES:
            for strategy in cellmod.STRATEGIES:
                idx = cellmod.cell_index(maze, strategy)
                census[idx] += int(np.sum((mz == maze) & (ys == strategy)))

        cells = session_cells(Xs, mz, ys, min_trials=min_trials, skip_cells=skip_cells)
        if len(cells) < 2:
            continue
        half_sizes, stable = session_half_sizes(
            cells, min_stable=min_stable, n_half=n_half
        )
        rng = np.random.default_rng(session_seed(seed, session))
        mat, n_deg = session_cv_matrix(
            cells, rng, half_sizes=half_sizes, n_splits=n_splits
        )
        if not np.isfinite(mat).any():
            continue

        session_mats.append(mat)
        session_names.append(session)
        n_degenerate_total += n_deg
        here = np.zeros(n_cells, dtype=bool)
        for idx in cells:
            used[idx] += 1
            here[idx] = True
        presence.append(here)
        stable_flags.append(stable)
        cell_counts.append({idx: len(rows) for idx, rows in cells.items()})
        half_used.append(half_sizes)

    if not session_mats:
        return LabelSimilarityResult(
            mean=np.full((n_cells, n_cells), np.nan),
            session_mats=[],
            session_names=[],
            presence=np.zeros((0, n_cells), dtype=bool),
            stable=[],
            cell_counts=[],
            half_sizes=[],
            census=census,
            used=used,
            n_sessions=0,
            n_degenerate=n_degenerate_total,
            min_trials=min_trials,
            min_stable=min_stable,
        )

    with warnings.catch_warnings():
        # Cells absent from every session are an all-NaN column, not an error.
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(np.stack(session_mats), axis=0)

    return LabelSimilarityResult(
        mean=mean,
        session_mats=session_mats,
        session_names=session_names,
        presence=np.stack(presence),
        stable=stable_flags,
        cell_counts=cell_counts,
        half_sizes=half_used,
        census=census,
        used=used,
        n_sessions=len(session_mats),
        n_degenerate=n_degenerate_total,
        min_trials=min_trials,
        min_stable=min_stable,
    )


def permutation_null(
    X,
    sessions,
    trials,
    mazes,
    *,
    keep_sessions,
    label_lookup,
    n_perm=1000,
    seed=0,
    **kw,
):
    """Within-(session, maze) label-shuffle null for the within-maze delta.

    Shuffles strategy labels among trials sharing a (session, maze), which
    preserves every cell's trial count exactly, then recomputes the full
    `label_similarity_matrix` and the within-maze delta `n_perm` times. This
    measures the delta's noise floor empirically at the actual dimensionality
    and session coverage, rather than assuming a bootstrap's between-session
    resampling is the right error model at d as low as 6.

    Returns ``(rows, observed)`` -- `rows` is a list of per-maze dicts ready
    for `nulls_k<K>.csv`, `observed` is the un-shuffled `LabelSimilarityResult`.
    """
    from eye_pre_flash.corr.cells import within_maze_delta, within_maze_table

    sessions = np.asarray(sessions).astype(str)
    trials = np.asarray(trials, dtype=int)
    mazes = np.asarray(mazes, dtype=int)
    keep = set(keep_sessions)
    y_obs = labels_for_rows(sessions, trials, label_lookup)
    keep_mask = np.isin(sessions, list(keep)) & np.isfinite(y_obs)

    observed = label_similarity_matrix(
        X, sessions, trials, mazes, keep_sessions=keep_sessions,
        label_lookup=label_lookup, seed=seed, **kw,
    )
    obs_delta = within_maze_delta(
        within_maze_table(observed.session_mats, observed.presence)
    )
    obs_table = within_maze_table(observed.session_mats, observed.presence)

    strata = []
    for session in keep:
        for maze in cellmod.MAZES:
            idx = np.flatnonzero(
                keep_mask & (sessions == session) & (mazes == maze)
            )
            if idx.size >= 2:
                strata.append(idx)

    rng_perm = np.random.default_rng((seed, 3))  # tag 3: permutation null
    null_deltas = {m: [] for m in cellmod.MAZES}
    trial_ids_by_row = trials  # aligned with sessions/mazes/y_obs

    for _ in range(n_perm):
        y_shuffled = y_obs.copy()
        for idx in strata:
            y_shuffled[idx] = rng_perm.permutation(y_shuffled[idx])
        perm_lookup = {
            (sessions[i], int(trial_ids_by_row[i])): float(y_shuffled[i])
            for i in np.flatnonzero(keep_mask)
        }
        result = label_similarity_matrix(
            X, sessions, trials, mazes, keep_sessions=keep_sessions,
            label_lookup=perm_lookup, seed=seed, **kw,
        )
        delta = within_maze_delta(within_maze_table(result.session_mats, result.presence))
        for m in cellmod.MAZES:
            null_deltas[m].append(delta[m])

    rows = []
    for maze in cellmod.MAZES:
        obs = obs_delta[maze]
        null = np.asarray(null_deltas[maze], dtype=float)
        null = null[np.isfinite(null)]
        n_sessions_maze = obs_table[maze][3]
        if not np.isfinite(obs) or null.size == 0:
            rows.append(
                dict(
                    maze=maze, delta_obs=obs, null_mean=np.nan, null_sd=np.nan,
                    p_two_sided=np.nan, null_p025=np.nan, null_p975=np.nan,
                    n_perm=int(null.size), n_sessions=n_sessions_maze,
                )
            )
            continue
        null_mean = float(np.mean(null))
        null_sd = float(np.std(null, ddof=1)) if null.size > 1 else np.nan
        # +1 smoothing: a permutation test can never report p = 0.
        p = float(
            (np.sum(np.abs(null - null_mean) >= abs(obs - null_mean)) + 1)
            / (null.size + 1)
        )
        rows.append(
            dict(
                maze=maze, delta_obs=obs, null_mean=null_mean, null_sd=null_sd,
                p_two_sided=p,
                null_p025=float(np.percentile(null, 2.5)),
                null_p975=float(np.percentile(null, 97.5)),
                n_perm=int(null.size), n_sessions=n_sessions_maze,
            )
        )
    return rows, observed
