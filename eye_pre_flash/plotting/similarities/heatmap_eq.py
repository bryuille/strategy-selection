"""Maze similarity matrices with trial counts equalised across mazes.

Same 6×6 split-half CV as `similarities.heatmap`, with one change: **every half
map averages the same number of trials.**

That change matters because the uncorrected matrices are biased by trial count.
Split-half CV averages ``n // 2`` trials into each half map, so a maze with more
trials gets a cleaner half map and a higher diagonal (its own split-half
reliability). Pearson *r* between two noisy means is dragged down by the noise
in **both**, so a maze measured more precisely also correlates more highly with
*every* other maze -- most of all with the other precisely-measured mazes. Both
animals run the most trials on mazes 4-6, which is enough on its own to make
those mazes look like a block.

Equalising removes the cause rather than correcting for it after the fact. In
each session the half size is ``min over included mazes of n // 2``, and every
maze's half maps are subsampled to exactly that many trials, so no maze enters a
correlation with a better estimate than any other. Pass ``--n-half`` to fix the
same size across sessions too; sessions still differ in how much data they have,
but that shifts every cell of a session's matrix together and so cannot create
block structure.

Everything else -- window, unit-H warp, QC, 20 random splits, session averaging
-- is unchanged from `similarities.heatmap`, so the figures are directly
comparable and any difference between them is the trial-count bias.

Usage:
    uv run python -m eye_pre_flash.plotting.similarities.heatmap_eq --monkey Faure
    uv run python -m eye_pre_flash.plotting.similarities.heatmap_eq --monkey Nielsen
    uv run python -m eye_pre_flash.plotting.similarities.heatmap_eq --monkey Faure --bin-w 0.142857
    uv run python -m eye_pre_flash.plotting.similarities.heatmap_eq --monkey Faure --n-half 20

Output: `eye_pre_flash/plotting/out/similarities/heatmap_eq/<monkey>/`.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from data.loader import load_eye_behavioral_data, load_eye_data
from eye_pre_flash.plotting.plot_io import save_figure
from eye_pre_flash.plotting.similarities.common import BLUE_YELLOW
from eye_pre_flash.plotting.similarities.heatmap_log import (
    COARSE_BIN,
    DEFAULT_BIN,
    MAZES,
    MIN_TRIALS,
    N_SPLITS,
    PRE_FIX_END_MS,
    PRE_FIX_START_MS,
    _collect_trials,
    _half_profile,
    _pearson,
    _trial_count_maps,
)

VISUALIZER = "similarities/heatmap_eq"


def equal_halves(n, n_half, rng):
    """Two disjoint index sets of exactly `n_half` trials drawn from `n`."""
    order = rng.permutation(n)
    return order[:n_half], order[n_half : 2 * n_half]


def session_half_size(maze_maps, min_trials=MIN_TRIALS):
    """The largest half size every included maze in this session can supply."""
    sizes = [
        len(m) // 2
        for m in maze_maps
        if m is not None and len(m) >= min_trials
    ]
    return min(sizes) if len(sizes) >= 2 else 0


def _session_cv_matrix(
    maze_maps, rng, *, n_half, n_splits=N_SPLITS, min_trials=MIN_TRIALS, use_log=False
):
    """6×6 split-half CV Pearson *r* for one session, all halves the same size."""
    acc = np.zeros((6, 6))
    counts = np.zeros((6, 6))

    for _ in range(n_splits):
        mean_a = [None] * 6
        mean_b = [None] * 6
        for i, maps in enumerate(maze_maps):
            if maps is None or len(maps) < max(min_trials, 2 * n_half):
                continue
            idx_a, idx_b = equal_halves(len(maps), n_half, rng)
            mean_a[i] = _half_profile(maps, idx_a, use_log=use_log)
            mean_b[i] = _half_profile(maps, idx_b, use_log=use_log)

        present = [m for m in mean_a + mean_b if m is not None]
        if len(present) < 2:
            continue
        active = np.any(np.stack(present, axis=0) > 0, axis=0)

        for i in range(6):
            for j in range(6):
                if any(
                    m is None
                    for m in (mean_a[i], mean_b[j], mean_b[i], mean_a[j])
                ):
                    continue
                r = np.nanmean(
                    [
                        _pearson(mean_a[i], mean_b[j], active),
                        _pearson(mean_b[i], mean_a[j], active),
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
    eye,
    behavioral,
    *,
    bin_w=DEFAULT_BIN,
    n_splits=N_SPLITS,
    min_trials=MIN_TRIALS,
    n_half=None,
    seed=0,
    use_log=False,
):
    """Session-averaged 6×6 CV *r* with equalised half sizes.

    `n_half` fixes the half size across every session; left as None, each
    session uses the largest size all of its own mazes can supply.
    """
    by_session = _collect_trials(eye, behavioral)
    rng = np.random.default_rng(seed)

    session_mats, half_sizes = [], []
    trial_counts = [0] * 6
    for maze_trials in by_session.values():
        for i, maze in enumerate(MAZES):
            trial_counts[i] += len(maze_trials[maze])
        maze_maps = [
            None
            if len(maze_trials[maze]) < min_trials
            else _trial_count_maps(maze_trials[maze], bin_w)
            for maze in MAZES
        ]
        size = n_half if n_half is not None else session_half_size(maze_maps, min_trials)
        if size < 1:
            continue
        mat = _session_cv_matrix(
            maze_maps,
            rng,
            n_half=size,
            n_splits=n_splits,
            min_trials=min_trials,
            use_log=use_log,
        )
        if np.isfinite(mat).any():
            session_mats.append(mat)
            half_sizes.append(size)

    if not session_mats:
        return np.full((6, 6), np.nan), trial_counts, 0, 0.0

    return (
        np.nanmean(np.stack(session_mats), axis=0),
        trial_counts,
        len(session_mats),
        float(np.mean(half_sizes)),
    )


def plot_similarity(
    monkey="Faure",
    *,
    bin_w=DEFAULT_BIN,
    n_splits=N_SPLITS,
    min_trials=MIN_TRIALS,
    n_half=None,
    seed=0,
    use_log=False,
    visualizer=VISUALIZER,
):
    eye = load_eye_data(monkey)
    behavioral = load_eye_behavioral_data(monkey)
    corr, counts, n_sessions, mean_half = maze_similarity_matrix(
        eye,
        behavioral,
        bin_w=bin_w,
        n_splits=n_splits,
        min_trials=min_trials,
        n_half=n_half,
        seed=seed,
        use_log=use_log,
    )

    finite = corr[np.isfinite(corr)]
    if finite.size:
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmin == vmax:
            vmax = vmin + 1e-12
    else:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(6.4, 5.6), layout="constrained")
    im = ax.imshow(
        corr, origin="upper", cmap=BLUE_YELLOW, vmin=vmin, vmax=vmax, aspect="equal"
    )
    ax.set_xticks(range(6), labels=range(1, 7))
    ax.set_yticks(range(6), labels=range(1, 7))
    ax.set_xlabel("Maze")
    ax.set_ylabel("Maze")
    rate_label = "log1p(count/trial)" if use_log else "count/trial"
    end_label = "fix_start" if PRE_FIX_END_MS == 0 else f"fix_start − {PRE_FIX_END_MS} ms"
    half_label = (
        f"{n_half} trials/half (fixed)"
        if n_half is not None
        else f"{mean_half:.1f} trials/half (per-session max)"
    )
    ax.set_title(
        f"{monkey} pre-fixation heatmap similarities (unit H, CV, equal n)\n"
        f"split-half Pearson r of {rate_label} | avg over {n_sessions} sessions | "
        f"{half_label}\n"
        f"fix_start − {PRE_FIX_START_MS} ms → {end_label} | {bin_w:g} unit H bins | "
        f"{n_splits} splits | n={','.join(str(n) for n in counts)}",
        fontsize=9,
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
                f"{val:.3f}",
                ha="center",
                va="center",
                color="black" if lum > 0.55 else "white",
                fontsize=8,
            )

    stem = f"similarities_bin{bin_w:g}" + ("" if n_half is None else f"_nhalf{n_half}")
    save_figure(fig, stem, rel_dir=f"{visualizer}/{monkey}")
    plt.close(fig)

    print(f"{monkey} bin {bin_w:g} — reliability (diagonal), equal n:")
    print("  " + "  ".join(f"m{m}={corr[i, i]:.3f}" for i, m in enumerate(MAZES)))
    return fig


def plot_all(monkey="Faure", **kwargs):
    plot_similarity(monkey=monkey, bin_w=DEFAULT_BIN, **kwargs)
    plot_similarity(monkey=monkey, bin_w=COARSE_BIN, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monkey", default="Faure", choices=("Faure", "Nielsen"))
    parser.add_argument(
        "--bin-w",
        type=float,
        default=None,
        help=f"Bin width in unit H (default: both {DEFAULT_BIN:g} and {COARSE_BIN:g})",
    )
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--min-trials", type=int, default=MIN_TRIALS)
    parser.add_argument(
        "--n-half",
        type=int,
        default=None,
        help="Fix the half size across sessions too (default: per-session maximum)",
    )
    parser.add_argument("--log", action="store_true", help="log1p(count/trial) maps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    kwargs = dict(
        n_splits=args.n_splits,
        min_trials=args.min_trials,
        n_half=args.n_half,
        use_log=args.log,
        seed=args.seed,
    )
    if args.bin_w is not None:
        plot_similarity(monkey=args.monkey, bin_w=args.bin_w, **kwargs)
    else:
        plot_all(monkey=args.monkey, **kwargs)


if __name__ == "__main__":
    main()
