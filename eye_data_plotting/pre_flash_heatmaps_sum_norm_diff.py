"""Difference of pooled unit-H pre-fixation gaze heatmaps by strategy group.

Same raw summed counts as ``pre_flash_heatmaps_sum_norm``. Hierarchical
minus sequential, diverging around zero.

    Faure   — (mazes 1–3) − (mazes 4–6)
    Nielsen — (mazes 1–4) − (mazes 5–6)

Usage:
    uv run python -m eye_data_plotting.pre_flash_heatmaps_sum_norm_diff --monkey Faure
    uv run python -m eye_data_plotting.pre_flash_heatmaps_sum_norm_diff --monkey Nielsen
    uv run python -m eye_data_plotting.pre_flash_heatmaps_sum_norm_diff --monkey Faure --bin-w 0.142857
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from data.loader import load_eye_behavioral_data, load_eye_data
from eye_data_plotting.plot_io import save_figure
from eye_data_plotting.pre_flash_heatmaps_sum_norm import (
    COARSE_BIN,
    DEFAULT_BIN,
    DEG_TO_UNIT_H,
    FIXATION_STEM_DEG,
    HALF_UNIT,
    MAZE_GROUPS,
    PRE_FIX_END_MS,
    PRE_FIX_START_MS,
    _sum_group,
)

VISUALIZER = "pre_flash_heatmaps_sum_norm_diff"


def plot_diff(
    monkey="Faure",
    eye_data=None,
    behavioral=None,
    *,
    bin_w=DEFAULT_BIN,
    use_log=False,
    visualizer=VISUALIZER,
):
    eye = eye_data if eye_data is not None else load_eye_data(monkey)
    behavioral = (
        behavioral if behavioral is not None else load_eye_behavioral_data(monkey)
    )
    hier, seq = MAZE_GROUPS[monkey]
    hier_map, n_hier = _sum_group(eye, behavioral, hier, bin_w=bin_w)
    seq_map, n_seq = _sum_group(eye, behavioral, seq, bin_w=bin_w)
    if n_hier == 0 or n_seq == 0:
        raise ValueError(f"No trials for monkey={monkey!r} heatmap diff")

    hier_disp = hier_map.astype(float)
    seq_disp = seq_map.astype(float)
    if use_log:
        hier_disp = np.log1p(hier_disp)
        seq_disp = np.log1p(seq_disp)
    diff = hier_disp - seq_disp
    vmax = float(np.max(np.abs(diff))) if np.any(diff) else 1.0
    hier_label = ",".join(str(m) for m in hier)
    seq_label = ",".join(str(m) for m in seq)
    scale = "sum, log, unit H" if use_log else "sum, unit H"
    cbar = "log1p(count) (hier − seq)" if use_log else "count (hier − seq)"

    fig, ax = plt.subplots(figsize=(6.5, 6.0), layout="constrained")
    im = ax.imshow(
        diff,
        origin="lower",
        extent=(-HALF_UNIT, HALF_UNIT, -HALF_UNIT, HALF_UNIT),
        aspect="equal",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_xlim(-HALF_UNIT, HALF_UNIT)
    ax.set_ylim(-HALF_UNIT, HALF_UNIT)
    ax.set_xlabel("eye_x (unit H)")
    ax.set_ylabel("eye_y (unit H)")
    ax.set_title(
        f"{monkey} hierarchical − sequential pre-fixation gaze ({scale})\n"
        f"mazes {hier_label} − {seq_label}\n"
        f"fix_start − {PRE_FIX_START_MS} ms → fix_start − {PRE_FIX_END_MS} ms\n"
        f"±{HALF_UNIT:.2f} unit H, {bin_w:g} bin_w, n={n_hier} − {n_seq} (all sessions)",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, shrink=0.85, label=cbar)
    save_figure(
        fig,
        f"bin{bin_w:g}_diff",
        rel_dir=f"{visualizer}/{monkey}",
    )
    plt.close(fig)
    return fig


def plot_all(monkey="Faure", **kwargs):
    eye = load_eye_data(monkey)
    behavioral = load_eye_behavioral_data(monkey)
    plot_diff(monkey=monkey, eye_data=eye, behavioral=behavioral, **kwargs)


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
