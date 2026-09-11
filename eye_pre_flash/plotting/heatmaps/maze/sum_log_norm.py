"""Log-scale pooled pre-fixation gaze heatmaps by strategy group (unit H).

Same raw summed counts as ``heatmaps_sum_norm``, displayed as
``log1p(count)``. Normalized to unit H via ``data.attractor.to_maze``.

Usage:
    uv run python -m eye_pre_flash.plotting.heatmaps.maze.sum_log_norm --monkey Faure
    uv run python -m eye_pre_flash.plotting.heatmaps.maze.sum_log_norm --monkey Nielsen
    uv run python -m eye_pre_flash.plotting.heatmaps.maze.sum_log_norm --monkey Faure --bin-w 0.142857
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from data.loader import load_eye_behavioral_data, load_eye_data
from eye_pre_flash.plotting.heatmaps.maze.sum_norm import (
    COARSE_BIN,
    DEFAULT_BIN,
    DEG_TO_UNIT_H,
    FIXATION_STEM_DEG,
    HALF_UNIT,
    MAZE_GROUPS,
    PRE_FIX_END_MS,
    PRE_FIX_START_MS,
    PRE_FIX_WINDOW_MS,
    _sum_group,
)
from eye_pre_flash.plotting.plot_io import save_figure, window_label
from eye_pre_flash.plotting.heatmaps.paths import OUT_ROOT

VISUALIZER = "maze/sum_log_norm"


def plot_group_heatmap(
    mazes,
    label,
    monkey="Faure",
    eye_data=None,
    behavioral=None,
    *,
    bin_w=DEFAULT_BIN,
):
    eye = eye_data if eye_data is not None else load_eye_data(monkey)
    behavioral = (
        behavioral if behavioral is not None else load_eye_behavioral_data(monkey)
    )
    eyemap, n_trials = _sum_group(eye, behavioral, mazes, bin_w=bin_w)
    if n_trials == 0:
        raise ValueError(f"No trials for monkey={monkey!r}, {label} mazes {mazes}")

    display = np.log1p(eyemap.astype(float))
    vmax = float(display.max()) if display.max() > 0 else 1.0
    maze_label = ",".join(str(m) for m in mazes)

    fig, ax = plt.subplots(figsize=(6.5, 6.0), layout="constrained")
    im = ax.imshow(
        display,
        origin="lower",
        extent=(-HALF_UNIT, HALF_UNIT, -HALF_UNIT, HALF_UNIT),
        aspect="equal",
        cmap="magma",
        vmin=0,
        vmax=vmax,
    )
    ax.set_xlim(-HALF_UNIT, HALF_UNIT)
    ax.set_ylim(-HALF_UNIT, HALF_UNIT)
    ax.set_xlabel("eye_x (unit H)")
    ax.set_ylabel("eye_y (unit H)")
    ax.set_title(
        f"{monkey} {label} mazes {maze_label} pre-fixation gaze (sum, log, unit H)\n"
        f"{window_label()}\n"
        f"±{HALF_UNIT:.2f} unit H, {bin_w:g} bin_w, n={n_trials} (all sessions)",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, shrink=0.85, label="log1p(count)")
    save_figure(
        fig,
        f"bin{bin_w:g}_{label}",
        out_root=OUT_ROOT,
        rel_dir=f"{VISUALIZER}/{monkey}",
    )
    plt.close(fig)
    return fig


def plot_all(monkey="Faure", *, bin_w=DEFAULT_BIN):
    eye = load_eye_data(monkey)
    behavioral = load_eye_behavioral_data(monkey)
    hier, seq = MAZE_GROUPS[monkey]
    plot_group_heatmap(
        hier,
        "hierarchical",
        monkey=monkey,
        eye_data=eye,
        behavioral=behavioral,
        bin_w=bin_w,
    )
    plot_group_heatmap(
        seq,
        "sequential",
        monkey=monkey,
        eye_data=eye,
        behavioral=behavioral,
        bin_w=bin_w,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monkey", default="Faure", choices=("Faure", "Nielsen"))
    parser.add_argument(
        "--bin-w",
        type=float,
        default=None,
        help=(
            f"Bin width in unit H (default: generate both "
            f"{DEFAULT_BIN:g} and {COARSE_BIN:g})"
        ),
    )
    parser.add_argument(
        "--bin-deg",
        type=float,
        default=None,
        help=f"Bin width in degrees (converted via 1 / {FIXATION_STEM_DEG:g}°)",
    )
    args = parser.parse_args()
    if args.bin_w is not None:
        bins = [args.bin_w]
    elif args.bin_deg is not None:
        bins = [args.bin_deg * DEG_TO_UNIT_H]
    else:
        bins = [DEFAULT_BIN, COARSE_BIN]

    for bin_w in bins:
        plot_all(monkey=args.monkey, bin_w=bin_w)


if __name__ == "__main__":
    main()
