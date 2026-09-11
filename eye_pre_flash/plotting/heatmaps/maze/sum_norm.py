"""Pooled pre-fixation gaze heatmaps by strategy group (unit H).

Sums raw sample counts across hierarchical mazes and sequential mazes
separately, normalized to the unit H coordinate frame via ``data.attractor.to_maze``.

    Faure   — hierarchical 1–3, sequential 4–6
    Nielsen — hierarchical 1–4, sequential 5–6

Usage:
    uv run python -m eye_pre_flash.plotting.heatmaps.maze.sum_norm --monkey Faure
    uv run python -m eye_pre_flash.plotting.heatmaps.maze.sum_norm --monkey Nielsen
    uv run python -m eye_pre_flash.plotting.heatmaps.maze.sum_norm --monkey Faure --bin-w 0.142857
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from data.attractor import to_maze
from data.loader import load_eye_behavioral_data, load_eye_data
from eye_pre_flash.plotting.heatmaps.maze.core import (
    DEFAULT_BIN_DEG,
    FIXATION_STEM_DEG,
    HALF_DEG,
    PRE_FIX_END_MS,
    PRE_FIX_START_MS,
    PRE_FIX_WINDOW_MS,
)
from eye_pre_flash.plotting.heatmaps.maze.sum import COARSE_BIN_DEG, MAZE_GROUPS
from eye_pre_flash.plotting.plot_io import save_figure, window_label
from eye_pre_flash.plotting.heatmaps.paths import OUT_ROOT

VISUALIZER = "maze/sum_norm"

DEG_TO_UNIT_H = 1.0 / FIXATION_STEM_DEG
HALF_UNIT = HALF_DEG * DEG_TO_UNIT_H
DEFAULT_BIN = DEFAULT_BIN_DEG * DEG_TO_UNIT_H
COARSE_BIN = COARSE_BIN_DEG * DEG_TO_UNIT_H


def _behavioral_lookup(behavioral):
    return {
        (str(session), int(trial_id)): i
        for session, trial_id, i in zip(
            behavioral["session"],
            behavioral["trial_indices_all"],
            range(len(behavioral["session"])),
        )
    }


def _fix_start_ms(behavioral, idx):
    return (
        behavioral["fix_start"][idx] - behavioral["geo_present"][idx]
    ) * 1000


def _map_size(bin_w):
    return int(2 * HALF_UNIT / bin_w)


def _accumulate_xy(eyemap, x, y, *, bin_w):
    size = _map_size(bin_w)
    xi = np.floor((x + HALF_UNIT) / bin_w).astype(np.int64)
    yi = np.floor((y + HALF_UNIT) / bin_w).astype(np.int64)
    ok = (xi >= 0) & (xi < size) & (yi >= 0) & (yi < size)
    np.add.at(eyemap, (yi[ok], xi[ok]), 1)


def _collect_maze_samples_norm(eye, behavioral, maze):
    """Return (samples, window) for one maze across all sessions in unit H."""
    lookup = _behavioral_lookup(behavioral)
    raw = []

    for eye_i in range(len(eye["session"])):
        session = str(eye["session"][eye_i])
        trial_id = int(eye["trial_indices_all"][eye_i])
        beh_i = lookup.get((session, trial_id))
        if beh_i is None:
            continue
        if behavioral["geo_type"][beh_i] != maze:
            continue
        if behavioral["path_type"][beh_i] == -99:
            continue
        if behavioral["photodiode_qc_bad"][beh_i]:
            continue
        if behavioral["trial_fade"][beh_i] != 0:
            continue

        t_ms = np.asarray(eye["time"][eye_i], dtype=float) * 1000
        x = np.asarray(eye["eye_x"][eye_i], dtype=float)
        y = np.asarray(eye["eye_y"][eye_i], dtype=float)
        fix_ms = _fix_start_ms(behavioral, beh_i)
        h = tuple(float(behavioral[f"h{i}"][beh_i]) for i in range(1, 7))
        raw.append({"t_ms": t_ms, "x": x, "y": y, "fix_ms": fix_ms, "h": h})

    if not raw:
        return [], (PRE_FIX_START_MS, PRE_FIX_END_MS)

    samples = []
    for r in raw:
        keep = (
            np.isfinite(r["t_ms"])
            & np.isfinite(r["x"])
            & np.isfinite(r["y"])
            & (r["t_ms"] >= r["fix_ms"] - PRE_FIX_START_MS)
            & (r["t_ms"] <= r["fix_ms"] - PRE_FIX_END_MS)
        )
        if keep.sum() < 2:
            continue
        x_n, y_n = to_maze(r["x"][keep], r["y"][keep], r["h"])
        samples.append((x_n, y_n))

    return samples, (PRE_FIX_START_MS, PRE_FIX_END_MS)


def _sum_group(eye, behavioral, mazes, *, bin_w=DEFAULT_BIN):
    eyemap = np.zeros((_map_size(bin_w), _map_size(bin_w)), dtype=np.int64)
    n_trials = 0
    for maze in mazes:
        samples, _ = _collect_maze_samples_norm(eye, behavioral, maze)
        for x, y in samples:
            _accumulate_xy(eyemap, x, y, bin_w=bin_w)
        n_trials += len(samples)
    return eyemap, n_trials


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

    display = eyemap.astype(float)
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
        f"{monkey} {label} mazes {maze_label} pre-fixation gaze (sum, unit H)\n"
        f"{window_label()}\n"
        f"±{HALF_UNIT:.2f} unit H, {bin_w:g} bin_w, n={n_trials} (all sessions)",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, shrink=0.85, label="count")
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
