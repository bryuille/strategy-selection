"""Cross-validated pre-fixation transition-order similarities.

Each trial is the bigram + trigram proportion vector of collapsed codebook
runs in [fix_start − 1466 ms, fix_start]. A path ``k1 → k5 → k2`` is the
bigrams (k1, k5), (k5, k2) and the trigram (k1, k5, k2). Dwell time is
ignored. Session split-half CV Pearson *r* is averaged across sessions.

Usage:
    uv run python -m eye_data_plotting.pre_flash_similarities_transition --monkey Faure
    uv run python -m eye_data_plotting.pre_flash_similarities_transition --monkey Nielsen
"""

from __future__ import annotations

import argparse

from data.attractor import DEFAULT_K
from data.occupancy import attractor_transition_rows
from data.loader import load_attractor_eye_data, load_eye_behavioral_data
from eye_data_plotting.plot_io import PRE_FIX_START_MS
from eye_data_plotting.similarities_common import (
    N_SPLITS,
    plot_cv_grid,
    session_averaged_cv,
)

VISUALIZER = "pre_flash_similarities_transition"
PRE_FIX_END_MS = 0


def plot_similarity(monkey="Faure", *, k=DEFAULT_K, n_splits=N_SPLITS, seed=0):
    attractor = load_attractor_eye_data(monkey, k=k)
    behavioral = load_eye_behavioral_data(monkey)
    features, sessions, _, mazes = attractor_transition_rows(
        attractor, behavioral, end_ms=PRE_FIX_END_MS
    )
    corr, counts, n_sessions = session_averaged_cv(
        features, sessions, mazes, n_splits=n_splits, seed=seed
    )
    codebook_k = int(attractor["codebook_k"])
    end_label = "fix_start" if PRE_FIX_END_MS == 0 else f"fix_start − {PRE_FIX_END_MS:g} ms"
    title = (
        f"{monkey} pre-fixation transition similarities (K={codebook_k}, CV)\n"
        f"split-half Pearson r of bigram+trigram run order | "
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
