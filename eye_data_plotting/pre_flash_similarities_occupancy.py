"""Cross-validated occupancy similarities, averaged across sessions.

Each trial is a K-dimensional vector of seconds spent in each codebook
state from ``fix_start − 1466 ms`` through ``fix_start`` (k12 by default).
Within a session, maze occupancy profiles are estimated on independent
trial splits and Pearson-correlated (split-half CV). Those 6×6 matrices
are then averaged across sessions.

Usage:
    uv run python -m eye_data_plotting.pre_flash_similarities_occupancy --monkey Faure
    uv run python -m eye_data_plotting.pre_flash_similarities_occupancy --monkey Nielsen
    uv run python -m eye_data_plotting.pre_flash_similarities_occupancy --monkey Faure --k 12
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from data.attractor import DEFAULT_K
from data.labeler import attractor_occupancy_rows
from data.loader import load_attractor_eye_data, load_eye_behavioral_data
from eye_data_plotting.plot_io import PRE_FIX_START_MS, save_figure
from eye_data_plotting.similarities_common import BLUE_YELLOW

VISUALIZER = "pre_flash_similarities_occupancy"
PRE_FIX_END_MS = 0
MAZES = range(1, 7)
N_SPLITS = 20


def _pearson(a, b):
    av = np.asarray(a, dtype=float).ravel()
    bv = np.asarray(b, dtype=float).ravel()
    if av.size < 2 or av.std() == 0 or bv.std() == 0:
        return np.nan
    return float(np.corrcoef(av, bv)[0, 1])


def _split_half_indices(n, rng):
    order = rng.permutation(n)
    mid = n // 2
    return order[:mid], order[mid:]


def _session_cv_matrix(features, mazes, rng, n_splits=N_SPLITS):
    """6×6 split-half CV Pearson *r* for one session."""
    by_maze = [features[mazes == maze] for maze in MAZES]
    acc = np.zeros((6, 6), dtype=float)
    counts = np.zeros((6, 6), dtype=float)

    for _ in range(n_splits):
        mean_a = [None] * 6
        mean_b = [None] * 6
        for i, rows in enumerate(by_maze):
            if len(rows) < 2:
                continue
            idx_a, idx_b = _split_half_indices(len(rows), rng)
            mean_a[i] = rows[idx_a].mean(axis=0)
            mean_b[i] = rows[idx_b].mean(axis=0)

        for i in range(6):
            for j in range(6):
                if mean_a[i] is None or mean_b[j] is None:
                    continue
                if mean_b[i] is None or mean_a[j] is None:
                    continue
                r = np.nanmean(
                    [
                        _pearson(mean_a[i], mean_b[j]),
                        _pearson(mean_b[i], mean_a[j]),
                    ]
                )
                if np.isfinite(r):
                    acc[i, j] += r
                    counts[i, j] += 1

    out = np.full((6, 6), np.nan)
    ok = counts > 0
    out[ok] = acc[ok] / counts[ok]
    return out


def maze_similarity_matrix(
    attractor, behavioral, *, n_splits=N_SPLITS, seed=0
):
    """Session-averaged 6×6 CV Pearson *r*, plus trial counts and n sessions."""
    features, sessions, _, mazes = attractor_occupancy_rows(
        attractor, behavioral, end_ms=PRE_FIX_END_MS
    )
    rng = np.random.default_rng(seed)

    session_mats = []
    for session in np.unique(sessions):
        mask = sessions == session
        mat = _session_cv_matrix(
            features[mask], mazes[mask], rng, n_splits=n_splits
        )
        if np.isfinite(mat).any():
            session_mats.append(mat)

    trial_counts = [int(np.sum(mazes == maze)) for maze in MAZES]
    if not session_mats:
        return np.full((6, 6), np.nan), trial_counts, 0

    corr = np.nanmean(np.stack(session_mats, axis=0), axis=0)
    return corr, trial_counts, len(session_mats)


def plot_similarity(
    monkey="Faure", *, k=DEFAULT_K, n_splits=N_SPLITS, seed=0
):
    attractor = load_attractor_eye_data(monkey, k=k)
    behavioral = load_eye_behavioral_data(monkey)
    corr, counts, n_sessions = maze_similarity_matrix(
        attractor, behavioral, n_splits=n_splits, seed=seed
    )
    codebook_k = int(np.asarray(attractor["codebook_k"]))

    finite = corr[np.isfinite(corr)]
    if finite.size:
        vmin = float(finite.min())
        vmax = float(finite.max())
        if vmin == vmax:
            vmax = vmin + 1e-12
    else:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(6.4, 5.6), layout="constrained")
    im = ax.imshow(
        corr,
        origin="upper",
        cmap=BLUE_YELLOW,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    ax.set_xticks(range(6), labels=range(1, 7))
    ax.set_yticks(range(6), labels=range(1, 7))
    ax.set_xlabel("Maze")
    ax.set_ylabel("Maze")
    end_label = "fix_start" if PRE_FIX_END_MS == 0 else f"fix_start − {PRE_FIX_END_MS:g} ms"
    ax.set_title(
        f"{monkey} pre-fixation occupancy similarities (K={codebook_k}, CV)\n"
        f"split-half Pearson r of mean seconds/state | "
        f"avg over {n_sessions} sessions\n"
        f"fix_start − {PRE_FIX_START_MS:g} ms → {end_label} | {n_splits} splits | "
        f"n={','.join(str(n) for n in counts)}",
        fontsize=10,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("correlation coefficient")

    for i in range(6):
        for j in range(6):
            val = corr[i, j]
            if not np.isfinite(val):
                ax.text(j, i, "—", ha="center", va="center", color="white")
                continue
            r, g, b, _ = im.cmap(im.norm(val))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="black" if lum > 0.55 else "white",
                fontsize=9,
            )

    save_figure(
        fig,
        f"similarities_k{codebook_k}",
        rel_dir=f"{VISUALIZER}/{monkey}",
    )
    plt.close(fig)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monkey", default="Faure", choices=("Faure", "Nielsen"))
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Attractor codebook size")
    parser.add_argument(
        "--n-splits",
        type=int,
        default=N_SPLITS,
        help="Repeated random split-halves per session",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    plot_similarity(
        monkey=args.monkey, k=args.k, n_splits=args.n_splits, seed=args.seed
    )


if __name__ == "__main__":
    main()
