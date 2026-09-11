"""Log-scale difference of pooled unit-H pre-fixation gaze heatmaps.

Same raw summed counts as ``heatmaps_sum_log_norm``, displayed as
``log1p(count_hier) − log1p(count_seq)``.

    Faure   — (mazes 1–3) − (mazes 4–6)
    Nielsen — (mazes 1–4) − (mazes 5–6)

Usage:
    uv run python -m eye_pre_flash.plotting.heatmaps.maze.sum_log_norm_diff --monkey Faure
    uv run python -m eye_pre_flash.plotting.heatmaps.maze.sum_log_norm_diff --monkey Nielsen
    uv run python -m eye_pre_flash.plotting.heatmaps.maze.sum_log_norm_diff --monkey Faure --bin-w 0.142857
"""

from __future__ import annotations

import argparse

from data.loader import load_eye_behavioral_data, load_eye_data
from eye_pre_flash.plotting.heatmaps.maze.sum_norm import (
    COARSE_BIN,
    DEFAULT_BIN,
    DEG_TO_UNIT_H,
    FIXATION_STEM_DEG,
)
from eye_pre_flash.plotting.heatmaps.maze.sum_norm_diff import plot_diff as _plot_diff

VISUALIZER = "maze/sum_log_norm_diff"


def plot_diff(monkey="Faure", **kwargs):
    kwargs.setdefault("use_log", True)
    kwargs.setdefault("visualizer", VISUALIZER)
    return _plot_diff(monkey=monkey, **kwargs)


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

    eye = load_eye_data(args.monkey)
    behavioral = load_eye_behavioral_data(args.monkey)
    for bin_w in bins:
        plot_diff(
            monkey=args.monkey,
            eye_data=eye,
            behavioral=behavioral,
            bin_w=bin_w,
        )


if __name__ == "__main__":
    main()
