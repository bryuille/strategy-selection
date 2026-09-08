"""The 12x12 (maze x strategy) cell algebra.

Ported from the deleted `eye_pre_flash.plotting.similarities.heatmap_labels`
(recoverable with ``git show HEAD:eye_pre_flash/plotting/similarities/heatmap_labels.py``).
Cells are ``(maze, strategy)``: mazes 1-6 hierarchical, then mazes 1-6
sequential, so the matrix reads as two 6x6 quadrants along each axis.

Unlike the deleted version, this package computes and plots *every* cell
regardless of trial count -- thin cells are marked, not dropped -- so the
exhaustive-subset "comparable core" restriction (`common_cell_set`,
`pair_session_sets`, `unreliable_mask`, `matched_matrix`) is deliberately not
carried over into the plotting path. Every per-session correlation still
lands in `results_raw.csv`, so that restriction remains reconstructable later
without re-running anything -- it is dropped from here, not lost.
"""

from __future__ import annotations

import numpy as np

MAZES = tuple(range(1, 7))
STRATEGIES = (0, 1)
STRATEGY_TAG = {0: "H", 1: "S"}
N_CELLS = len(MAZES) * len(STRATEGIES)


def cell_index(maze, strategy):
    """Row/column of the (maze, strategy) cell in the 12x12 matrix."""
    return strategy * len(MAZES) + (maze - 1)


def cell_maze_strategy(index):
    """Inverse of `cell_index`: ``(maze, strategy)`` for a 0-11 cell index."""
    strategy, maze0 = divmod(index, len(MAZES))
    return maze0 + 1, strategy


def cell_labels():
    return [
        f"{maze}{STRATEGY_TAG[strategy]}" for strategy in STRATEGIES for maze in MAZES
    ]


def within_maze_table(session_mats, presence):
    """``maze -> (r_HH, r_SS, r_HS, n sessions)`` on matched session sets.

    A cell can qualify in more sessions than its partner, and sessions differ
    in overall correlation level, so averaging `r(H,H)` over four sessions
    against `r(H,S)` over three would compare them on different recording
    days. Each maze's three numbers come from exactly the sessions where
    **both** of its cells qualified (``presence``, one boolean row per
    session, `N_CELLS` columns).
    """
    rows = {}
    for maze in MAZES:
        h, s = cell_index(maze, 0), cell_index(maze, 1)
        if presence.size == 0:
            rows[maze] = (np.nan, np.nan, np.nan, 0)
            continue
        both = np.flatnonzero(presence[:, h] & presence[:, s])
        if both.size == 0:
            rows[maze] = (np.nan, np.nan, np.nan, 0)
            continue
        mats = [session_mats[k] for k in both]
        rows[maze] = (
            float(np.nanmean([m[h, h] for m in mats])),
            float(np.nanmean([m[s, s] for m in mats])),
            float(np.nanmean([m[h, s] for m in mats])),
            int(both.size),
        )
    return rows


def within_maze_delta(rows):
    """``maze -> delta`` = ``0.5*(r_HH + r_SS) - r_HS`` from `within_maze_table`."""
    out = {}
    for maze, (r_hh, r_ss, r_hs, n) in rows.items():
        if n == 0 or not np.isfinite(r_hs):
            out[maze] = np.nan
            continue
        out[maze] = 0.5 * (r_hh + r_ss) - r_hs
    return out
