"""Pre-fixation saccade plots aligned to fix_start.

Plots eye_x and eye_y from the shortest pre-fixation interval in the group
through fix_start. Time axes are relative to fix_start (0 = fix_start).

Usage:
    uv run python -m eye_pre_flash.plotting.saccades.traces --session june_24_g0 --maze 1
    uv run python -m eye_pre_flash.plotting.saccades.traces --session june_24_g0 --maze 1 --view trials
    uv run python -m eye_pre_flash.plotting.saccades.traces --session june_24_g0 --maze 1 --view trial --trial-id 141
    uv run python -m eye_pre_flash.plotting.saccades.traces --session june_24_g0 --maze 1 --mode 3d --save
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from data.loader import load_eye_behavioral_data, load_eye_data
from eye_pre_flash.plotting.plot_io import (
    PRE_FIX_WINDOW_MS,
    cue_rel_median,
    cue_rel_ms,
    draw_cue_marker,
    save_figure,
)

VISUALIZER = "saccades/traces"
FIXATION_STEM_DEG = 7.0
VIS_LIMIT = 30.0


def _out_dir(session, maze):
    return f"{VISUALIZER}/{session}/maze_{maze}"


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


def _exit_x_refs(h1, h4):
    return (-float(h1), float(h4))


def _exit_y_refs(h2, h3, h5, h6):
    return (float(h2), -float(h3), float(h5), -float(h6))


def _collect_maze_trials(eye, behavioral, maze, session):
    lookup = _behavioral_lookup(behavioral)
    raw = []

    for eye_i in range(len(eye["session"])):
        if str(eye["session"][eye_i]) != session:
            continue

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
        h = tuple(behavioral[f"h{i}"][beh_i] for i in range(1, 7))
        raw.append(
            {
                "trial_id": trial_id,
                "t_ms": t_ms,
                "x": x,
                "y": y,
                "fix_ms": fix_ms,
                "h": h,
                "cue_rel": cue_rel_ms(behavioral, beh_i),
            }
        )

    if not raw:
        return [], [], [], []

    # window_ms = int(round(min(r["fix_ms"] for r in raw)))
    window_ms = int(round(PRE_FIX_WINDOW_MS))
    trials = []
    x_refs = set()
    y_refs = set()
    h_vals = []

    for r in raw:
        keep = (
            np.isfinite(r["t_ms"])
            & np.isfinite(r["x"])
            & np.isfinite(r["y"])
            & (r["t_ms"] >= r["fix_ms"] - window_ms)
            & (r["t_ms"] <= r["fix_ms"])
        )
        if keep.sum() < 2:
            continue

        t_rel = r["t_ms"][keep] - r["fix_ms"]
        h = r["h"]
        trials.append(
            {
                "trial_id": r["trial_id"],
                "t_rel": t_rel,
                "x": r["x"][keep],
                "y": r["y"][keep],
                "h": h,
                "fix_ms": r["fix_ms"],
                "window_ms": window_ms,
                "cue_rel": r["cue_rel"],
            }
        )
        x_refs.update(_exit_x_refs(h[0], h[3]))
        y_refs.update(_exit_y_refs(h[1], h[2], h[4], h[5]))
        h_vals.append(h)

    return trials, sorted(x_refs), sorted(y_refs), h_vals


def _trial_by_id(trials, trial_id):
    for trial in trials:
        if trial["trial_id"] == trial_id:
            return trial
    ids = sorted(t["trial_id"] for t in trials)
    raise ValueError(
        f"Trial {trial_id} not found for this session/maze; "
        f"available: {ids[:10]}{'...' if len(ids) > 10 else ''}"
    )


def _median_geometry(h_vals):
    if not h_vals:
        return (5.0,) * 6
    arr = np.asarray(h_vals, dtype=float)
    return tuple(np.nanmedian(arr[:, i]) for i in range(6))


def _time_grid(trials):
    window_ms = trials[0]["window_ms"]
    return np.arange(-window_ms, 1, dtype=float)


def _time_window_label(trials):
    window_ms = trials[0]["window_ms"]
    return f"fix_start − {window_ms:.0f} ms → fix_start"


def _aligned_stack(trials, grid):
    xs, ys = [], []
    for trial in trials:
        xs.append(
            np.interp(
                grid,
                trial["t_rel"],
                trial["x"],
                left=np.nan,
                right=np.nan,
            )
        )
        ys.append(
            np.interp(
                grid,
                trial["t_rel"],
                trial["y"],
                left=np.nan,
                right=np.nan,
            )
        )
    return np.asarray(xs), np.asarray(ys)


def _trace_stats(trials):
    grid = _time_grid(trials)
    x_stack, y_stack = _aligned_stack(trials, grid)
    n_x = np.maximum(np.sum(np.isfinite(x_stack), axis=0), 1)
    n_y = np.maximum(np.sum(np.isfinite(y_stack), axis=0), 1)
    with np.errstate(invalid="ignore"):
        return {
            "t": grid,
            "t_min": float(grid[0]),
            "x_mean": np.nanmean(x_stack, axis=0),
            "x_sem": np.nanstd(x_stack, axis=0, ddof=1) / np.sqrt(n_x),
            "y_mean": np.nanmean(y_stack, axis=0),
            "y_sem": np.nanstd(y_stack, axis=0, ddof=1) / np.sqrt(n_y),
        }


def _style_2d_axes(ax_x, ax_y, x_refs, y_refs, t_min, cue_rel=None, cue_median=False):
    for xv in x_refs:
        ax_x.axhline(xv, color="k", linestyle=":", linewidth=1.0, alpha=0.75)
    for yv in y_refs:
        ax_y.axhline(yv, color="k", linestyle=":", linewidth=1.0, alpha=0.75)

    for ax in (ax_x, ax_y):
        ax.axvline(0, color="0.5", linewidth=0.8)
        ax.axvline(t_min, color="0.65", linewidth=0.6, linestyle="--")
        ax.set_xlim(t_min, 0)
        ax.set_xlabel("Time rel. fix_start (ms)")
        draw_cue_marker(ax, cue_rel, median=cue_median)

    ax_x.set_ylim(-VIS_LIMIT, VIS_LIMIT)
    ax_y.set_ylim(-VIS_LIMIT, VIS_LIMIT)
    ax_x.set_ylabel("eye_x (deg)")
    ax_y.set_ylabel("eye_y (deg)")


def _draw_h_maze_3d(ax, h1, h2, h3, h4, h5, h6, z0, z1=0.0):
    """Hollow maze at z0 with dotted vertical projection through time to z1."""
    left_x, right_x = -h1, h4
    base_kw = {"color": "k", "linewidth": 2.0}

    ax.plot([left_x, 0], [0, 0], [z0, z0], **base_kw)
    ax.plot([0, right_x], [0, 0], [z0, z0], **base_kw)
    ax.plot([left_x, left_x], [-h3, h2], [z0, z0], **base_kw)
    ax.plot([right_x, right_x], [-h6, h5], [z0, z0], **base_kw)
    ax.plot([0, 0], [0, FIXATION_STEM_DEG], [z0, z0], **base_kw)

    exit_kw = {
        "linestyle": "none",
        "marker": "o",
        "markersize": 7,
        "markerfacecolor": "w",
        "markeredgecolor": "k",
        "markeredgewidth": 1.5,
    }
    for x, y in (
        (left_x, h2),
        (left_x, -h3),
        (right_x, h5),
        (right_x, -h6),
    ):
        ax.plot([x], [y], [z0], **exit_kw)

    ax.plot(
        [0],
        [0],
        [z0],
        linestyle="none",
        marker="o",
        markersize=5,
        markerfacecolor="k",
        markeredgecolor="k",
    )

    proj_kw = {"color": "0.35", "linewidth": 1.0, "linestyle": ":"}
    vertices = (
        (left_x, 0),
        (0, 0),
        (right_x, 0),
        (left_x, h2),
        (left_x, -h3),
        (right_x, h5),
        (right_x, -h6),
        (0, FIXATION_STEM_DEG),
    )
    for x, y in vertices:
        ax.plot([x, x], [y, y], [z0, z1], **proj_kw)


def _style_3d_axes(ax, t_min):
    ax.set_xlabel("eye_x (deg)")
    ax.set_ylabel("eye_y (deg)")
    ax.set_zlabel("Time rel. fix_start (ms)")
    ax.set_xlim(-VIS_LIMIT, VIS_LIMIT)
    ax.set_ylim(-VIS_LIMIT, VIS_LIMIT)
    ax.set_zlim(t_min, 0)
    ax.view_init(elev=22, azim=-60)


def _finish_3d(fig, stem, session, maze, *, save):
    if save:
        path = save_figure(fig, stem, rel_dir=_out_dir(session, maze))
        plt.close(fig)
        return path
    plt.show()
    return fig


def plot_2d_saccades_average(
    maze, session, monkey="Faure", eye_data=None, behavioral=None
):
    """Mean eye_x / eye_y vs time across trials for one session and maze."""
    eye = eye_data if eye_data is not None else load_eye_data(monkey)
    behavioral = (
        behavioral if behavioral is not None else load_eye_behavioral_data(monkey)
    )
    trials, x_refs, y_refs, _ = _collect_maze_trials(eye, behavioral, maze, session)
    if not trials:
        raise ValueError(f"No trials for session={session!r}, maze={maze}")

    stats = _trace_stats(trials)
    fig, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")

    ax_x.plot(stats["t"], stats["x_mean"], color="C0", linewidth=2.0)
    ax_x.fill_between(
        stats["t"],
        stats["x_mean"] - stats["x_sem"],
        stats["x_mean"] + stats["x_sem"],
        color="C0",
        alpha=0.25,
    )
    ax_y.plot(stats["t"], stats["y_mean"], color="C1", linewidth=2.0)
    ax_y.fill_between(
        stats["t"],
        stats["y_mean"] - stats["y_sem"],
        stats["y_mean"] + stats["y_sem"],
        color="C1",
        alpha=0.25,
    )

    _style_2d_axes(
        ax_x, ax_y, x_refs, y_refs, stats["t_min"],
        cue_rel=cue_rel_median([t.get("cue_rel") for t in trials]),
        cue_median=True,
    )
    fig.suptitle(
        f"{session} maze {maze} pre-fixation saccades (2D average)\n"
        f"{_time_window_label(trials)}, n={len(trials)} trials"
    )
    save_figure(fig, "avg_2d", rel_dir=_out_dir(session, maze))
    plt.close(fig)
    return fig


def _save_2d_trial(maze, session, trial, trials):
    fig, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    ax_x.plot(trial["t_rel"], trial["x"], color="C0", linewidth=1.2)
    ax_y.plot(trial["t_rel"], trial["y"], color="C1", linewidth=1.2)
    _style_2d_axes(
        ax_x,
        ax_y,
        _exit_x_refs(trial["h"][0], trial["h"][3]),
        _exit_y_refs(trial["h"][1], trial["h"][2], trial["h"][4], trial["h"][5]),
        -trial["window_ms"],
        cue_rel=trial.get("cue_rel"),
    )
    fig.suptitle(
        f"{session} maze {maze} trial {trial['trial_id']} (2D)\n"
        f"{_time_window_label(trials)} ({trial['fix_ms']:.0f} ms pre-fixation)"
    )
    path = save_figure(
        fig,
        f"trial_{trial['trial_id']:03d}_2d",
        rel_dir=_out_dir(session, maze),
    )
    plt.close(fig)
    return path


def plot_2d_saccades_trials(
    maze, session, monkey="Faure", eye_data=None, behavioral=None
):
    """One 2D side-by-side plot per trial."""
    eye = eye_data if eye_data is not None else load_eye_data(monkey)
    behavioral = (
        behavioral if behavioral is not None else load_eye_behavioral_data(monkey)
    )
    trials, _, _, _ = _collect_maze_trials(eye, behavioral, maze, session)
    if not trials:
        raise ValueError(f"No trials for session={session!r}, maze={maze}")

    return [_save_2d_trial(maze, session, trial, trials) for trial in trials]


def plot_2d_saccade_trial(
    maze,
    session,
    trial_id,
    monkey="Faure",
    eye_data=None,
    behavioral=None,
):
    """One 2D side-by-side plot for a single trial."""
    eye = eye_data if eye_data is not None else load_eye_data(monkey)
    behavioral = (
        behavioral if behavioral is not None else load_eye_behavioral_data(monkey)
    )
    trials, _, _, _ = _collect_maze_trials(eye, behavioral, maze, session)
    trial = _trial_by_id(trials, trial_id)
    return _save_2d_trial(maze, session, trial, trials)


def plot_3d_saccades_average(
    maze,
    session,
    monkey="Faure",
    eye_data=None,
    behavioral=None,
    *,
    save=False,
):
    """Mean 3D gaze trace with maze projected upward through time."""
    eye = eye_data if eye_data is not None else load_eye_data(monkey)
    behavioral = (
        behavioral if behavioral is not None else load_eye_behavioral_data(monkey)
    )
    trials, _, _, h_vals = _collect_maze_trials(eye, behavioral, maze, session)
    if not trials:
        raise ValueError(f"No trials for session={session!r}, maze={maze}")

    stats = _trace_stats(trials)
    h1, h2, h3, h4, h5, h6 = _median_geometry(h_vals)
    maze_z = -trials[0]["window_ms"]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(stats["x_mean"], stats["y_mean"], stats["t"], color="C0", linewidth=2.0)
    _draw_h_maze_3d(ax, h1, h2, h3, h4, h5, h6, z0=maze_z)
    _style_3d_axes(ax, stats["t_min"])
    fig.suptitle(
        f"{session} maze {maze} pre-fixation saccades (3D average)\n"
        f"{_time_window_label(trials)}, n={len(trials)} trials"
    )
    return _finish_3d(fig, "avg_3d", session, maze, save=save)


def plot_3d_saccade_trial(
    maze,
    session,
    trial_id,
    monkey="Faure",
    eye_data=None,
    behavioral=None,
    *,
    save=False,
):
    """One 3D plot for a single trial."""
    eye = eye_data if eye_data is not None else load_eye_data(monkey)
    behavioral = (
        behavioral if behavioral is not None else load_eye_behavioral_data(monkey)
    )
    trials, _, _, _ = _collect_maze_trials(eye, behavioral, maze, session)
    trial = _trial_by_id(trials, trial_id)
    h1, h2, h3, h4, h5, h6 = trial["h"]
    maze_z = -trial["window_ms"]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(trial["x"], trial["y"], trial["t_rel"], color="C0", linewidth=1.2)
    _draw_h_maze_3d(ax, h1, h2, h3, h4, h5, h6, z0=maze_z)
    _style_3d_axes(ax, maze_z)
    fig.suptitle(
        f"{session} maze {maze} trial {trial_id} (3D)\n"
        f"{_time_window_label(trials)} ({trial['fix_ms']:.0f} ms pre-fixation)"
    )
    return _finish_3d(fig, f"trial_{trial_id:03d}_3d", session, maze, save=save)


def plot_saccades(
    maze,
    session,
    *,
    view="average",
    mode="2d",
    trial_id=None,
    save_3d=False,
    monkey="Faure",
    eye_data=None,
    behavioral=None,
):
    """Dispatch to the requested view and mode."""
    if view == "trial" and trial_id is None:
        raise ValueError("--trial-id is required when --view trial")

    if view == "average":
        if mode == "2d":
            return plot_2d_saccades_average(maze, session, monkey, eye_data, behavioral)
        return plot_3d_saccades_average(
            maze, session, monkey, eye_data, behavioral, save=save_3d
        )

    if view == "trials":
        if mode == "2d":
            return plot_2d_saccades_trials(maze, session, monkey, eye_data, behavioral)
        raise ValueError("3D batch trial plots are disabled; use --view trial")

    if mode == "2d":
        return plot_2d_saccade_trial(
            maze, session, trial_id, monkey, eye_data, behavioral
        )
    return plot_3d_saccade_trial(
        maze, session, trial_id, monkey, eye_data, behavioral, save=save_3d
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="june_24_g0")
    parser.add_argument("--mode", choices=("2d", "3d"), default="2d")
    parser.add_argument(
        "--view",
        choices=("average", "trials", "trial"),
        default="average",
        help="average: session mean; trials: all trials (2D only); trial: one trial",
    )
    parser.add_argument(
        "--trial-id",
        type=int,
        default=None,
        help="Trial ID (required for --view trial)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save 3D plots to disk (3D displays interactively by default)",
    )
    parser.add_argument(
        "--maze",
        type=int,
        required=True,
        choices=range(1, 7),
        metavar="1-6",
    )
    parser.add_argument("--monkey", default="Faure", choices=("Faure", "Nielsen"))
    args = parser.parse_args()

    plot_saccades(
        args.maze,
        args.session,
        view=args.view,
        mode=args.mode,
        trial_id=args.trial_id,
        save_3d=args.save,
        monkey=args.monkey,
    )


if __name__ == "__main__":
    main()
