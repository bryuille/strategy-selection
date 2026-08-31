"""Cross-validated binary occupancy similarities, averaged across sessions.

Each trial is a K-dimensional 0/1 vector: 1 if that codebook state was
visited from ``fix_start − 1466 ms`` through ``fix_start``, else 0
(k12 by default). Within a session, maze visit profiles are estimated on independent
trial splits and Pearson-correlated (split-half CV). Those 6×6 matrices
are then averaged across sessions.

Usage:
    uv run python -m eye_data_plotting.pre_flash_similarities_occupancy_bin --monkey Faure
    uv run python -m eye_data_plotting.pre_flash_similarities_occupancy_bin --monkey Nielsen
    uv run python -m eye_data_plotting.pre_flash_similarities_occupancy_bin --monkey Faure --k 12
"""

from __future__ import annotations

import argparse

from data.attractor import DEFAULT_K
from data.occupancy import attractor_occupancy_bin_rows
from data.loader import load_attractor_eye_data, load_eye_behavioral_data
from eye_data_plotting.plot_io import PRE_FIX_START_MS
from eye_data_plotting.similarities_common import (
    N_SPLITS,
    plot_cv_grid,
    session_averaged_cv,
)

VISUALIZER = "pre_flash_similarities_occupancy_bin"
PRE_FIX_END_MS = 0


def plot_similarity(monkey="Faure", *, k=DEFAULT_K, n_splits=N_SPLITS, seed=0):
    attractor = load_attractor_eye_data(monkey, k=k)
    behavioral = load_eye_behavioral_data(monkey)
    features, sessions, _, mazes = attractor_occupancy_bin_rows(
        attractor, behavioral, end_ms=PRE_FIX_END_MS
    )
    corr, counts, n_sessions = session_averaged_cv(
        features, sessions, mazes, n_splits=n_splits, seed=seed
    )
    codebook_k = int(attractor["codebook_k"])
    end_label = "fix_start" if PRE_FIX_END_MS == 0 else f"fix_start − {PRE_FIX_END_MS:g} ms"
    title = (
        f"{monkey} pre-fixation binary occupancy similarities (K={codebook_k}, CV)\n"
        f"split-half Pearson r of mean visit/no-visit | "
        f"avg over {n_sessions} sessions\n"
        f"fix_start − {PRE_FIX_START_MS:g} ms → {end_label} | {n_splits} splits | "
        f"n={','.join(str(n) for n in counts)}"
    )
    return plot_cv_grid(
        corr,
        counts,
        n_sessions,
        title=title,
        stem=f"similarities_k{codebook_k}",
        rel_dir=f"{VISUALIZER}/{monkey}",
        n_splits=n_splits,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monkey", default="Faure", choices=("Faure", "Nielsen"))
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    plot_similarity(
        monkey=args.monkey, k=args.k, n_splits=args.n_splits, seed=args.seed
    )


if __name__ == "__main__":
    main()
