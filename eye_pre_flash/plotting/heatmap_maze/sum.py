"""Pooled pre-fixation gaze heatmaps by strategy group.

Sums raw sample counts across hierarchical mazes and sequential mazes
separately. Screen degrees, no unit-H warp, no per-trial or L1
normalization.

    Faure  — hierarchical 1–3, sequential 4–6
    Nielsen — hierarchical 1–4, sequential 5–6

Usage:
    uv run python -m eye_pre_flash.plotting.heatmap_maze.sum --monkey Faure
    uv run python -m eye_pre_flash.plotting.heatmap_maze.sum --monkey Nielsen
    uv run python -m eye_pre_flash.plotting.heatmap_maze.sum --monkey Faure --bin-deg 1
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from data.labeler import MAZE_GROUPS
from data.loader import load_eye_behavioral_data, load_eye_data
from eye_pre_flash.plotting.heatmap_maze.core import (
    DEFAULT_BIN_DEG,
    HALF_DEG,
    PRE_FIX_END_MS,
    PRE_FIX_START_MS,
    PRE_FIX_WINDOW_MS,
    _accumulate_xy,
    _collect_maze_samples,
    _map_size,
)
from eye_pre_flash.plotting.plot_io import save_figure, window_label

VISUALIZER = "heatmap_maze/sum"

COARSE_BIN_DEG = 1.0


def _sum_group(eye, behavioral, mazes, *, bin_deg=DEFAULT_BIN_DEG):
    eyemap = np.zeros((_map_size(bin_deg), _map_size(bin_deg)), dtype=np.int64)
    n_trials = 0
    for maze in mazes:
        samples, _, _ = _collect_maze_samples(eye, behavioral, maze)
        for x, y in samples:
            _accumulate_xy(eyemap, x, y, bin_deg=bin_deg)
        n_trials += len(samples)
    return eyemap, n_trials


def plot_group_heatmap(
    mazes,
    label,
    monkey="Faure",
    eye_data=None,
    behavioral=None,
    *,
    bin_deg=DEFAULT_BIN_DEG,
):
    eye = eye_data if eye_data is not None else load_eye_data(monkey)
    behavioral = (
        behavioral if behavioral is not None else load_eye_behavioral_data(monkey)
    )
    eyemap, n_trials = _sum_group(eye, behavioral, mazes, bin_deg=bin_deg)
    if n_trials == 0:
        raise ValueError(f"No trials for monkey={monkey!r}, {label} mazes {mazes}")

    display = eyemap.astype(float)
    vmax = float(display.max()) if display.max() > 0 else 1.0
    maze_label = ",".join(str(m) for m in mazes)

    fig, ax = plt.subplots(figsize=(6.5, 6.0), layout="constrained")
    im = ax.imshow(
        display,
        origin="lower",
        extent=(-HALF_DEG, HALF_DEG, -HALF_DEG, HALF_DEG),
        aspect="equal",
        cmap="magma",
        vmin=0,
        vmax=vmax,
    )
    ax.set_xlim(-HALF_DEG, HALF_DEG)
    ax.set_ylim(-HALF_DEG, HALF_DEG)
    ax.set_xlabel("eye_x (deg)")
    ax.set_ylabel("eye_y (deg)")
    ax.set_title(
        f"{monkey} {label} mazes {maze_label} pre-fixation gaze (sum)\n"
        f"{window_label()}\n"
        f"±{HALF_DEG:g}°, {bin_deg:g}° bins, n={n_trials} (all sessions)",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, shrink=0.85, label="count")
    save_figure(
        fig,
        f"bin{bin_deg:g}_{label}",
        rel_dir=f"{VISUALIZER}/{monkey}",
    )
    plt.close(fig)
    return fig


def plot_all(monkey="Faure", *, bin_deg=DEFAULT_BIN_DEG):
    eye = load_eye_data(monkey)
    behavioral = load_eye_behavioral_data(monkey)
    hier, seq = MAZE_GROUPS[monkey]
    plot_group_heatmap(
        hier,
        "hierarchical",
        monkey=monkey,
        eye_data=eye,
        behavioral=behavioral,
        bin_deg=bin_deg,
    )
    plot_group_heatmap(
        seq,
        "sequential",
        monkey=monkey,
        eye_data=eye,
        behavioral=behavioral,
        bin_deg=bin_deg,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monkey", default="Faure", choices=("Faure", "Nielsen"))
    parser.add_argument(
        "--bin-deg",
        type=float,
        default=None,
        help=(
            f"Bin width in degrees (default: generate both "
            f"{DEFAULT_BIN_DEG:g} and {COARSE_BIN_DEG:g})"
        ),
    )
    args = parser.parse_args()
    bins = [args.bin_deg] if args.bin_deg is not None else [DEFAULT_BIN_DEG, COARSE_BIN_DEG]
    for bin_deg in bins:
        plot_all(monkey=args.monkey, bin_deg=bin_deg)


if __name__ == "__main__":
    main()
