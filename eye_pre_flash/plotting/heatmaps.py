"""Pre-fixation gaze heatmaps across all sessions, by maze.

Accumulates eye samples from fix_start − PRE_FIX_START_MS →
fix_start − PRE_FIX_END_MS for every QC-passed fixed-geometry trial for a
monkey, one heatmap per maze (geo_type).

Usage:
    uv run python -m eye_pre_flash.plotting.heatmaps --monkey Faure
    uv run python -m eye_pre_flash.plotting.heatmaps --monkey Faure --maze 1
    uv run python -m eye_pre_flash.plotting.heatmaps --monkey Nielsen --bin-deg 1
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from data.loader import load_eye_behavioral_data, load_eye_data
from eye_pre_flash.plotting.plot_io import PRE_FIX_END_MS, PRE_FIX_START_MS, save_figure

VISUALIZER = "heatmaps"
HALF_DEG = 20.0
DEFAULT_BIN_DEG = 0.125
# Pre-fixation window: fix_start − PRE_FIX_START_MS → fix_start − PRE_FIX_END_MS.
PRE_FIX_WINDOW_MS = PRE_FIX_START_MS
FIXATION_STEM_DEG = 7.0
TUNNEL_THICKNESS_DEG = 1.0
MAZE_EDGE_ALPHA = 0.5


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


def _map_size(bin_deg):
    return int(2 * HALF_DEG / bin_deg)


def _accumulate_xy(eyemap, x, y, *, bin_deg):
    map_size = _map_size(bin_deg)
    xi = np.floor((x + HALF_DEG) / bin_deg).astype(np.int64)
    yi = np.floor((y + HALF_DEG) / bin_deg).astype(np.int64)
    in_range = (xi >= 0) & (xi < map_size) & (yi >= 0) & (yi < map_size)
    np.add.at(eyemap, (yi[in_range], xi[in_range]), 1)


def _draw_h_maze(ax, h1, h2, h3, h4, h5, h6):
    """1°-thick H outline (single path, no overlapping edges); 50% opacity."""
    left_x, right_x = -h1, h4
    half = TUNNEL_THICKNESS_DEG / 2.0
    stem_top = FIXATION_STEM_DEG

    # Exterior outline of a thick H + fixation stem (counterclockwise).
    verts = [
        (left_x - half, -h3),
        (left_x + half, -h3),
        (left_x + half, -half),
        (right_x - half, -half),
        (right_x - half, -h6),
        (right_x + half, -h6),
        (right_x + half, h5),
        (right_x - half, h5),
        (right_x - half, half),
        (half, half),
        (half, stem_top),
        (-half, stem_top),
        (-half, half),
        (left_x + half, half),
        (left_x + half, h2),
        (left_x - half, h2),
        (left_x - half, -h3),
    ]
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    path = Path(verts, codes)
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor="white",
            linewidth=1.2,
            alpha=MAZE_EDGE_ALPHA,
            zorder=5,
            joinstyle="miter",
        )
    )


def _collect_maze_samples(eye, behavioral, maze):
    """Return (samples, window_ms, h_vals) for one maze across all sessions."""
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
        return [], (PRE_FIX_START_MS, PRE_FIX_END_MS), (5.0,) * 6

    samples = []
    h_vals = []
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
        samples.append((r["x"][keep], r["y"][keep]))
        h_vals.append(r["h"])

    h_arr = np.asarray(h_vals, dtype=float)
    h_med = tuple(float(np.median(h_arr[:, i])) for i in range(6))
    return samples, (PRE_FIX_START_MS, PRE_FIX_END_MS), h_med


def build_maze_heatmap(eye, behavioral, maze, *, bin_deg=DEFAULT_BIN_DEG):
    samples, window, h_vals = _collect_maze_samples(eye, behavioral, maze)
    eyemap = np.zeros((_map_size(bin_deg), _map_size(bin_deg)), dtype=np.int64)
    for x, y in samples:
        _accumulate_xy(eyemap, x, y, bin_deg=bin_deg)
    return eyemap, len(samples), window, h_vals


def plot_maze_heatmap(
    maze, monkey="Faure", eye_data=None, behavioral=None, *, bin_deg=DEFAULT_BIN_DEG
):
    eye = eye_data if eye_data is not None else load_eye_data(monkey)
    behavioral = (
        behavioral if behavioral is not None else load_eye_behavioral_data(monkey)
    )
    eyemap, n_trials, window, h_vals = build_maze_heatmap(
        eye, behavioral, maze, bin_deg=bin_deg
    )
    if n_trials == 0:
        raise ValueError(f"No trials for monkey={monkey!r}, maze={maze}")

    rate = eyemap.astype(float) / n_trials
    vmax = float(rate.max()) if rate.max() > 0 else 1.0

    fig, ax = plt.subplots(figsize=(6.5, 6.0), layout="constrained")
    im = ax.imshow(
        rate,
        origin="lower",
        extent=(-HALF_DEG, HALF_DEG, -HALF_DEG, HALF_DEG),
        aspect="equal",
        cmap="magma",
        vmin=0,
        vmax=vmax,
    )
    _draw_h_maze(ax, *h_vals)
    ax.set_xlim(-HALF_DEG, HALF_DEG)
    ax.set_ylim(-HALF_DEG, HALF_DEG)
    ax.set_xlabel("eye_x (deg)")
    ax.set_ylabel("eye_y (deg)")
    ax.set_title(
        f"{monkey} maze {maze} pre-fixation gaze\n"
        f"fix_start − {PRE_FIX_START_MS} ms → fix_start − {PRE_FIX_END_MS} ms\n"
        f"±{HALF_DEG:g}°, {bin_deg:g}° bins, n={n_trials} (all sessions)",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, shrink=0.85, label="count / trial")
    save_figure(
        fig,
        f"bin{bin_deg:g}_maze_{maze}",
        rel_dir=f"{VISUALIZER}/{monkey}",
    )
    plt.close(fig)
    return fig


def plot_all_maze_heatmaps(monkey="Faure", mazes=range(1, 7), *, bin_deg=DEFAULT_BIN_DEG):
    eye = load_eye_data(monkey)
    behavioral = load_eye_behavioral_data(monkey)
    for maze in mazes:
        plot_maze_heatmap(
            maze, monkey=monkey, eye_data=eye, behavioral=behavioral, bin_deg=bin_deg
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monkey", default="Faure", choices=("Faure", "Nielsen"))
    parser.add_argument(
        "--bin-deg",
        type=float,
        default=DEFAULT_BIN_DEG,
        help=f"Bin width in degrees (default: {DEFAULT_BIN_DEG})",
    )
    parser.add_argument(
        "--maze",
        type=int,
        default=None,
        choices=range(1, 7),
        metavar="1-6",
        help="Single maze (default: all mazes 1–6)",
    )
    args = parser.parse_args()

    if args.maze is None:
        plot_all_maze_heatmaps(monkey=args.monkey, bin_deg=args.bin_deg)
    else:
        plot_maze_heatmap(args.maze, monkey=args.monkey, bin_deg=args.bin_deg)


if __name__ == "__main__":
    main()
