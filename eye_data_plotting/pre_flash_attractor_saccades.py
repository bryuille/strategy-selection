"""Pre-fixation gaze plots with k-means codebook assignments and state IDs.

Shows raw gaze, prototype snap, maze-landmark codebook, and state
transition times aligned to fix_start.

Usage:
    uv run python -m eye_data_plotting.pre_flash_attractor_saccades --session june_24_g0 --maze 1
    uv run python -m eye_data_plotting.pre_flash_attractor_saccades --session june_24_g0 --maze 1 --view trials
    uv run python -m eye_data_plotting.pre_flash_attractor_saccades --session june_24_g0 --maze 1 --view trial --trial-id 141
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

from data.attractor import MAZE_SCREEN_LIM
from data.loader import load_eye_behavioral_data, load_eye_data, load_attractor_eye_data
from eye_data_plotting.plot_io import PRE_FIX_WINDOW_MS, save_figure
from eye_data_plotting.pre_flash_saccades import (
    _behavioral_lookup,
    _fix_start_ms,
    _median_geometry,
    _time_window_label,
    _trace_stats,
    _trial_by_id,
)

VISUALIZER = "pre_flash_attractor_saccades"
MAZE_LIM = MAZE_SCREEN_LIM


def _out_dir(session, maze, k):
    return f"{VISUALIZER}/{session}/k{int(k)}/maze_{maze}"


def _state_colors(k):
    cmap = plt.colormaps["tab20" if k > 10 else "tab10"]
    return [cmap(i % cmap.N) for i in range(k)]


def _attractor_lookup(attractor):
    return {
        (str(session), int(trial_id)): i
        for session, trial_id, i in zip(
            attractor["session"],
            attractor["trial_indices_all"],
            range(len(attractor["session"])),
        )
    }


def _runs_from_state(t_rel, state):
    n = state.size
    if n == 0:
        return []
    runs = []
    start = 0
    for i in range(1, n + 1):
        if i == n or state[i] != state[start]:
            sid = int(state[start])
            if sid >= 0:
                runs.append(
                    {
                        "state": sid,
                        "onset_rel": float(t_rel[start]),
                        "offset_rel": float(t_rel[min(i, n) - 1]),
                    }
                )
            start = i
    return runs


def _collect_maze_trials(eye, behavioral, attractor, maze, session):
    lookup = _behavioral_lookup(behavioral)
    att_lookup = _attractor_lookup(attractor)
    codebook_xy = np.asarray(attractor["codebook_xy"], dtype=float)
    codebook_name = (
        [str(n) for n in attractor["codebook_name"]]
        if "codebook_name" in attractor
        else [str(i) for i in range(len(codebook_xy))]
    )
    raw = []

    for eye_i in range(len(eye["session"])):
        if str(eye["session"][eye_i]) != session:
            continue

        trial_id = int(eye["trial_indices_all"][eye_i])
        beh_i = lookup.get((session, trial_id))
        att_i = att_lookup.get((session, trial_id))
        if beh_i is None or att_i is None:
            continue
        if behavioral["geo_type"][beh_i] != maze:
            continue
        if behavioral["path_type"][beh_i] == -99:
            continue
        if behavioral["photodiode_qc_bad"][beh_i]:
            continue
        if behavioral["trial_fade"][beh_i] != 0:
            continue

        t_ms = np.asarray(attractor["time"][att_i], dtype=float) * 1000
        x = np.asarray(attractor["eye_x"][att_i], dtype=float)
        y = np.asarray(attractor["eye_y"][att_i], dtype=float)
        recon_x = np.asarray(attractor["recon_x"][att_i], dtype=float)
        recon_y = np.asarray(attractor["recon_y"][att_i], dtype=float)
        state = np.asarray(attractor["state_id"][att_i], dtype=int)
        valid = np.asarray(attractor["valid"][att_i], dtype=bool)
        artifact = np.asarray(attractor["artifact_prob"][att_i], dtype=float)
        fix_ms = _fix_start_ms(behavioral, beh_i)
        h = tuple(behavioral[f"h{i}"][beh_i] for i in range(1, 7))
        n = min(
            t_ms.size,
            x.size,
            y.size,
            recon_x.size,
            recon_y.size,
            state.size,
            valid.size,
            artifact.size,
        )
        x, y = x[:n], y[:n]
        valid = valid[:n]
        state = np.where(valid, state[:n], -1)
        recon_x = np.where(state >= 0, recon_x[:n], np.nan)
        recon_y = np.where(state >= 0, recon_y[:n], np.nan)
        raw.append(
            {
                "trial_id": trial_id,
                "t_ms": t_ms[:n],
                "x": x,
                "y": y,
                "recon_x": recon_x,
                "recon_y": recon_y,
                "state": state,
                "valid": valid,
                "artifact": artifact[:n],
                "fix_ms": fix_ms,
                "h": h,
            }
        )

    if not raw:
        return [], codebook_xy, codebook_name, []

    window_ms = int(round(PRE_FIX_WINDOW_MS))
    trials = []
    h_vals = []

    for r in raw:
        keep = (
            np.isfinite(r["t_ms"])
            & (r["t_ms"] >= r["fix_ms"] - window_ms)
            & (r["t_ms"] <= r["fix_ms"])
        )
        if keep.sum() < 2:
            continue
        t_rel = r["t_ms"][keep] - r["fix_ms"]
        trials.append(
            {
                "trial_id": r["trial_id"],
                "t_rel": t_rel,
                "x": r["x"][keep],
                "y": r["y"][keep],
                "recon_x": r["recon_x"][keep],
                "recon_y": r["recon_y"][keep],
                "state": r["state"][keep],
                "valid": r["valid"][keep],
                "artifact": r["artifact"][keep],
                "events": _runs_from_state(t_rel, r["state"][keep]),
                "h": r["h"],
                "fix_ms": r["fix_ms"],
                "window_ms": window_ms,
            }
        )
        h_vals.append(r["h"])

    return trials, codebook_xy, codebook_name, h_vals


def _style_maze_axes(ax_x, ax_y, t_min):
    for v in (-1.0, 1.0):
        ax_x.axhline(v, color="k", linestyle=":", linewidth=1.0, alpha=0.75)
        ax_y.axhline(v, color="k", linestyle=":", linewidth=1.0, alpha=0.75)
    for ax in (ax_x, ax_y):
        ax.axvline(0, color="0.5", linewidth=0.8)
        ax.axvline(t_min, color="0.65", linewidth=0.6, linestyle="--")
        ax.set_xlim(t_min, 0)
        ax.set_xlabel("Time rel. fix_start (ms)")
    ax_x.set_ylim(-MAZE_LIM, MAZE_LIM)
    ax_y.set_ylim(-MAZE_LIM, MAZE_LIM)
    ax_x.set_ylabel("eye_x (maze)")
    ax_y.set_ylabel("eye_y (maze)")


def _draw_h_maze_2d(ax, h):
    kw = {"color": "k", "linewidth": 1.4, "alpha": 0.7, "zorder": 1}
    ax.plot([-1, 1], [0, 0], **kw)
    ax.plot([-1, -1], [-1, 1], **kw)
    ax.plot([1, 1], [-1, 1], **kw)
    ax.plot([0, 0], [0, 1.0], **kw)


def _draw_state_spans(ax_x, ax_y, events, colors, names=None):
    for event in events:
        color = colors[event["state"] % len(colors)]
        for ax in (ax_x, ax_y):
            ax.axvspan(
                event["onset_rel"],
                event["offset_rel"],
                color=color,
                alpha=0.12,
                linewidth=0,
                zorder=0,
            )
            ax.axvline(event["onset_rel"], color=color, linewidth=0.9, alpha=0.85, zorder=3)

        mid = 0.5 * (event["onset_rel"] + event["offset_rel"])
        trans = blended_transform_factory(ax_x.transData, ax_x.transAxes)
        sid = event["state"]
        label = names[sid] if names is not None and sid < len(names) else str(sid)
        ax_x.text(
            mid,
            0.98,
            label,
            transform=trans,
            ha="center",
            va="top",
            fontsize=7,
            color=color,
            clip_on=True,
            zorder=4,
        )


def _draw_codebook(ax, codebook_xy, h, colors, names=None):
    _draw_h_maze_2d(ax, h)
    for i, (px, py) in enumerate(codebook_xy):
        ax.scatter(
            px,
            py,
            marker="X",
            s=70,
            color=colors[i % len(colors)],
            edgecolors="k",
            linewidths=0.4,
            zorder=5,
        )
        label = names[i] if names is not None and i < len(names) else str(i)
        ax.annotate(label, (px, py), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("eye_x (maze)")
    ax.set_ylabel("eye_y (maze)")
    ax.set_xlim(-MAZE_LIM, MAZE_LIM)
    ax.set_ylim(-MAZE_LIM, MAZE_LIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("k-means codebook")


def _draw_recon_points(ax, recon_x, recon_y, valid):
    assigned = valid & np.isfinite(recon_x) & np.isfinite(recon_y)
    if assigned.any():
        ax.scatter(
            recon_x[assigned],
            recon_y[assigned],
            c="k",
            s=16,
            alpha=0.85,
            zorder=3,
        )


def _aligned_recon_stats(trials):
    stats = _trace_stats(trials)
    grid = stats["t"]
    rx, ry = [], []
    for trial in trials:
        rx.append(
            np.interp(grid, trial["t_rel"], trial["recon_x"], left=np.nan, right=np.nan)
        )
        ry.append(
            np.interp(grid, trial["t_rel"], trial["recon_y"], left=np.nan, right=np.nan)
        )
    rx = np.asarray(rx)
    ry = np.asarray(ry)
    with np.errstate(invalid="ignore"):
        stats["rx_mean"] = np.nanmean(rx, axis=0)
        stats["ry_mean"] = np.nanmean(ry, axis=0)
    return stats


def plot_2d_average(
    maze, session, monkey="Faure", eye_data=None, behavioral=None, attractor=None, k=None
):
    eye = eye_data if eye_data is not None else load_eye_data(monkey)
    behavioral = (
        behavioral if behavioral is not None else load_eye_behavioral_data(monkey)
    )
    attractor = (
        attractor if attractor is not None else load_attractor_eye_data(monkey, k=k)
    )
    trials, codebook_xy, codebook_name, h_vals = _collect_maze_trials(
        eye, behavioral, attractor, maze, session
    )
    if not trials:
        raise ValueError(f"No trials for session={session!r}, maze={maze}")

    stats_trials = []
    for trial in trials:
        masked = dict(trial)
        masked["x"] = np.where(trial["valid"], trial["x"], np.nan)
        masked["y"] = np.where(trial["valid"], trial["y"], np.nan)
        stats_trials.append(masked)
    stats = _aligned_recon_stats(stats_trials)
    colors = _state_colors(len(codebook_xy))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    ax_x, ax_y, ax_xy = axes

    ax_x.plot(stats["t"], stats["x_mean"], color="0.55", linewidth=1.6, label="raw")
    ax_x.plot(stats["t"], stats["rx_mean"], color="C0", linewidth=2.0, label="recon")
    ax_y.plot(stats["t"], stats["y_mean"], color="0.55", linewidth=1.6, label="raw")
    ax_y.plot(stats["t"], stats["ry_mean"], color="C1", linewidth=2.0, label="recon")
    ax_x.legend(fontsize=8, loc="lower left")
    ax_y.legend(fontsize=8, loc="lower left")
    _style_maze_axes(ax_x, ax_y, stats["t_min"])
    _draw_codebook(
        ax_xy, codebook_xy, _median_geometry(h_vals), colors, names=codebook_name
    )

    k = int(np.asarray(attractor["codebook_k"]))
    mae = float(np.asarray(attractor["valid_mae"]))
    fig.suptitle(
        f"{session} maze {maze} attractor pre-fixation gaze (2D average)\n"
        f"{_time_window_label(trials)}, n={len(trials)} trials, K={k}, "
        f"valid MAE={mae:.2f} maze units"
    )
    save_figure(fig, "avg_2d", rel_dir=_out_dir(session, maze, k))
    plt.close(fig)
    return fig


def _save_2d_trial(
    maze, session, trial, trials, codebook_xy, colors, k, mae, codebook_name=None
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    ax_x, ax_y, ax_xy = axes
    valid = trial["valid"]

    ax_x.plot(trial["t_rel"], trial["x"], color="0.75", linewidth=0.9, zorder=1)
    ax_y.plot(trial["t_rel"], trial["y"], color="0.75", linewidth=0.9, zorder=1)
    if valid.any():
        x_ok = np.where(valid, trial["x"], np.nan)
        y_ok = np.where(valid, trial["y"], np.nan)
        ax_x.plot(trial["t_rel"], x_ok, color="C0", linewidth=1.1, alpha=0.7, zorder=2)
        ax_y.plot(trial["t_rel"], y_ok, color="C1", linewidth=1.1, alpha=0.7, zorder=2)
    ax_x.plot(trial["t_rel"], trial["recon_x"], color="k", linewidth=1.6, zorder=3)
    ax_y.plot(trial["t_rel"], trial["recon_y"], color="k", linewidth=1.6, zorder=3)
    _draw_state_spans(ax_x, ax_y, trial["events"], colors, names=codebook_name)
    _style_maze_axes(ax_x, ax_y, -trial["window_ms"])

    _draw_codebook(ax_xy, codebook_xy, trial["h"], colors, names=codebook_name)
    ax_xy.plot(trial["x"], trial["y"], color="0.75", linewidth=0.8, alpha=0.45, zorder=2)
    if valid.any():
        ax_xy.plot(
            np.where(valid, trial["x"], np.nan),
            np.where(valid, trial["y"], np.nan),
            color="0.25",
            linewidth=1.0,
            alpha=0.8,
            zorder=2,
        )
    _draw_recon_points(ax_xy, trial["recon_x"], trial["recon_y"], valid)

    ax_x.legend(
        handles=[
            Line2D([0], [0], color="0.7", label="raw"),
            Line2D([0], [0], color="k", label="recon"),
            Patch(facecolor="0.7", alpha=0.4, label="state"),
        ],
        loc="lower left",
        fontsize=8,
        framealpha=0.9,
    )
    fig.suptitle(
        f"{session} maze {maze} trial {trial['trial_id']} (attractor)\n"
        f"{_time_window_label(trials)} ({trial['fix_ms']:.0f} ms pre-fixation), "
        f"K={k}, valid MAE={mae:.2f} maze units"
    )
    path = save_figure(
        fig,
        f"trial_{trial['trial_id']:03d}_2d",
        rel_dir=_out_dir(session, maze, k),
    )
    plt.close(fig)
    return path


def plot_2d_trials(
    maze, session, monkey="Faure", eye_data=None, behavioral=None, attractor=None, k=None
):
    eye = eye_data if eye_data is not None else load_eye_data(monkey)
    behavioral = (
        behavioral if behavioral is not None else load_eye_behavioral_data(monkey)
    )
    attractor = (
        attractor if attractor is not None else load_attractor_eye_data(monkey, k=k)
    )
    trials, codebook_xy, codebook_name, _ = _collect_maze_trials(
        eye, behavioral, attractor, maze, session
    )
    if not trials:
        raise ValueError(f"No trials for session={session!r}, maze={maze}")
    colors = _state_colors(len(codebook_xy))
    k = int(np.asarray(attractor["codebook_k"]))
    mae = float(np.asarray(attractor["valid_mae"]))
    return [
        _save_2d_trial(
            maze, session, trial, trials, codebook_xy, colors, k, mae, codebook_name
        )
        for trial in trials
    ]


def plot_2d_trial(
    maze,
    session,
    trial_id,
    monkey="Faure",
    eye_data=None,
    behavioral=None,
    attractor=None,
    k=None,
):
    eye = eye_data if eye_data is not None else load_eye_data(monkey)
    behavioral = (
        behavioral if behavioral is not None else load_eye_behavioral_data(monkey)
    )
    attractor = (
        attractor if attractor is not None else load_attractor_eye_data(monkey, k=k)
    )
    trials, codebook_xy, codebook_name, _ = _collect_maze_trials(
        eye, behavioral, attractor, maze, session
    )
    colors = _state_colors(len(codebook_xy))
    k = int(np.asarray(attractor["codebook_k"]))
    mae = float(np.asarray(attractor["valid_mae"]))
    trial = _trial_by_id(trials, trial_id)
    return _save_2d_trial(
        maze, session, trial, trials, codebook_xy, colors, k, mae, codebook_name
    )


def plot_attractor(
    maze,
    session,
    *,
    view="average",
    trial_id=None,
    monkey="Faure",
    eye_data=None,
    behavioral=None,
    attractor=None,
    k=None,
):
    if view == "trial" and trial_id is None:
        raise ValueError("--trial-id is required when --view trial")
    if view == "average":
        return plot_2d_average(
            maze, session, monkey, eye_data, behavioral, attractor, k=k
        )
    if view == "trials":
        return plot_2d_trials(
            maze, session, monkey, eye_data, behavioral, attractor, k=k
        )
    return plot_2d_trial(
        maze, session, trial_id, monkey, eye_data, behavioral, attractor, k=k
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="june_24_g0")
    parser.add_argument(
        "--view",
        choices=("average", "trials", "trial"),
        default="average",
        help="average: session mean; trials: all trials; trial: one trial",
    )
    parser.add_argument(
        "--trial-id",
        type=int,
        default=None,
        help="Trial ID (required for --view trial)",
    )
    parser.add_argument(
        "--maze",
        type=int,
        required=True,
        choices=range(1, 7),
        metavar="1-6",
    )
    parser.add_argument("--monkey", default="Faure", choices=("Faure", "Nielsen"))
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Attractor codebook size (default: loader DEFAULT_K)",
    )
    args = parser.parse_args()

    plot_attractor(
        args.maze,
        args.session,
        view=args.view,
        trial_id=args.trial_id,
        monkey=args.monkey,
        k=args.k,
    )


if __name__ == "__main__":
    main()
