"""Cumulative-pool variant of `pair`: one 2x2 per maze, built by concatenating
every kept session's trials before the split-half CV, instead of computing one
2x2 per session and averaging them.

This is a deliberate side-by-side comparison against the canonical
per-session-then-average estimate, not a replacement for it: pooling erases
the session boundary that `pair.py` uses to keep sessions with more trials
from dominating the estimate and to keep cross-session drift out of a
within-day comparison. `build_pooled.py` writes these under `out/pooled/`
precisely so they are never confused with the canonical figures.

It inherits the whole rewritten estimator -- equal `m` with no exceptions, the
unsymmetrised off-diagonal, all-or-nothing splits, Fisher-z aggregation -- by
sharing `pair.SessionPool`, `pair.build_split_plan` and
`pair.session_pair_matrix`. What it cannot inherit is `drift.py`: once
sessions are concatenated, `trial_index` no longer orders anything, so the
time-based controls are not meaningful here and are not computed.
"""

from __future__ import annotations

import numpy as np

from eye_pre_flash.classifier.labels import labels_for_rows
from eye_pre_flash.maze_strategy_pairs.cells import STRATEGIES
from eye_pre_flash.maze_strategy_pairs.matrix import MIN_TRIALS, session_seed
from eye_pre_flash.maze_strategy_pairs.pair import (
    H_CELL,
    MIN_HALF,
    N_SPLITS,
    R_CLIP,
    S_CELL,
    SessionPool,
    amgm_floor,
    build_split_plan,
    cross_gap,
    cross_r,
    half_size,
    matrix_delta,
    pair_delta,
    session_pair_matrix,
    split_deltas,
)

POOLED_NULL_SEED_TAG = 12  # 11 is pair.py's


def build_pooled_pool(
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
    """Every kept session's maze trials concatenated into one `SessionPool`.

    Same qualifying rule as `pair.build_pools` -- both cells clear
    `min_trials`, and `m` clears `min_half` -- but applied once to the pooled
    counts rather than per session, so a session contributing a single trial to
    an otherwise well-populated pool is not dropped; only a (maze, label)
    combination that is thin in total is. Returns ``None`` when that happens.
    """
    sessions = np.asarray(sessions).astype(str)
    trials = np.asarray(trials, dtype=int)
    mazes = np.asarray(mazes, dtype=int)
    y_full = labels_for_rows(sessions, trials, label_lookup)
    keep = [s for s in sorted(set(keep_sessions)) if s in set(sessions.tolist())]

    row_mask = np.isin(sessions, keep) & np.isfinite(y_full) & (mazes == maze)
    if not row_mask.any():
        return None
    ys = y_full[row_mask].astype(int)
    counts = {s: int(np.sum(ys == s)) for s in STRATEGIES}
    if any(counts[s] < min_trials for s in STRATEGIES):
        return None
    m = half_size(counts, n_half=n_half, min_half=min_half)
    if m == 0:
        return None

    return SessionPool(
        session="__pooled__",
        X=X[row_mask],
        y=ys,
        # Concatenated across sessions, so this does NOT order time globally.
        # Carried only to satisfy the dataclass; `drift.py` must not be run on
        # a pooled pool.
        trial_index=trials[row_mask],
        counts=counts,
        m=m,
        rng_seed=session_seed(seed, f"pooled|m{maze}"),
    )


def pooled_permutation_null(
    pool, *, n_splits=N_SPLITS, n_perm=1000, seed=0, r_clip=R_CLIP
):
    """Label-shuffle null for the pooled delta.

    Shuffles H/S labels across *every* pooled trial at once -- there is no
    session boundary left to shuffle within, since pooling already erased it.
    The split plan is built once and reused by the observed value and every
    permutation: the shuffle preserves both cell counts, so the draws are
    valid throughout, and each permutation then differs only in which trials
    carry which label, not in split noise.
    """
    plan = build_split_plan(pool, n_splits=n_splits)
    obs_mat = session_pair_matrix(
        pool, pool.y, plan, r_clip=r_clip, keep_splits=True
    )
    obs = matrix_delta(obs_mat.r, r_clip=r_clip)

    rng_perm = np.random.default_rng((seed, POOLED_NULL_SEED_TAG))
    draws = np.full(n_perm, np.nan)
    for p in range(n_perm):
        sm = session_pair_matrix(
            pool, rng_perm.permutation(pool.y), plan, r_clip=r_clip
        )
        draws[p] = matrix_delta(sm.r, r_clip=r_clip)

    r_hh = float(obs_mat.r[H_CELL, H_CELL])
    r_ss = float(obs_mat.r[S_CELL, S_CELL])
    r_hs = cross_r(obs_mat.r, r_clip=r_clip)
    floor, beyond, imbalance = amgm_floor(r_hh, r_ss, r_hs)
    raw = obs_mat.r_raw
    obs_raw = pair_delta(
        raw[H_CELL, H_CELL],
        raw[S_CELL, S_CELL],
        0.5 * (raw[H_CELL, S_CELL] + raw[S_CELL, H_CELL]),
    )
    per_split = split_deltas([obs_mat])
    finite_split = per_split[np.isfinite(per_split)]

    null = draws[np.isfinite(draws)]
    row = dict(
        r_HH=r_hh, r_SS=r_ss, r_HS=r_hs,
        r_HS_ab=float(obs_mat.r[H_CELL, S_CELL]),
        r_HS_ba=float(obs_mat.r[S_CELL, H_CELL]),
        cross_gap=cross_gap(obs_mat.r),
        delta_obs=obs,
        delta_obs_raw=obs_raw,
        delta_amgm_floor=floor,
        delta_beyond_floor=beyond,
        diag_imbalance=imbalance,
        n_trials_H=pool.counts[H_CELL],
        n_trials_S=pool.counts[S_CELL],
        half_size_median=pool.m,
        n_splits=n_splits,
        n_splits_used=obs_mat.n_splits_used,
        n_splits_dropped_total=obs_mat.n_splits_dropped,
        n_clipped_HH=int(obs_mat.n_clipped_by_entry[H_CELL, H_CELL]),
        n_clipped_SS=int(obs_mat.n_clipped_by_entry[S_CELL, S_CELL]),
        n_clipped_HS=int(
            obs_mat.n_clipped_by_entry[H_CELL, S_CELL]
            + obs_mat.n_clipped_by_entry[S_CELL, H_CELL]
        ),
        delta_split_sd=(
            float(np.std(finite_split, ddof=1)) if finite_split.size > 1 else np.nan
        ),
        n_splits_full=int(finite_split.size),
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
            # +1 Laplace smoothing, centred on the null mean -- same
            # convention as `pair.permutation_null`, so the two packages'
            # p-values mean the same thing.
            p_two_sided=float(
                (np.sum(np.abs(null - null_mean) >= abs(obs - null_mean)) + 1)
                / (null.size + 1)
            ),
            p_one_sided_greater=float((np.sum(null >= obs) + 1) / (null.size + 1)),
            z_perm=(
                float((obs - null_mean) / null_sd)
                if np.isfinite(null_sd) and null_sd > 0
                else np.nan
            ),
        )
    return row, obs_mat, draws
