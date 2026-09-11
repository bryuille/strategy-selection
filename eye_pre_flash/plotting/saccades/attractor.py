"""Pre-fixation gaze plots with k-means codebook assignments and state IDs.

Shows raw gaze, prototype snap, maze-landmark codebook, and state
transition times aligned to fix_start.

Usage:
    uv run python -m eye_pre_flash.plotting.saccades.attractor --session june_24_g0 --maze 1
    uv run python -m eye_pre_flash.plotting.saccades.attractor --session june_24_g0 --maze 1 --view trials
    uv run python -m eye_pre_flash.plotting.saccades.attractor --session june_24_g0 --maze 1 --view trial --trial-id 141
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

from data.attractor import MAZE_SCREEN_LIM
from data.attractor import assign_fixation_states
from data.loader import (
    load_attractor_eye_data,
    load_clean_eye_data,
    load_eye_behavioral_data,
    load_eye_data,
)
from data.occupancy import behavioral_lookup, fix_start_ms
from eye_pre_flash.classifier.features import (
    DEFAULT_K,
    clip_fixation_spans,
    load_features,
)
from eye_pre_flash.plotting.plot_io import (
    PRE_FIX_END_MS,
    PRE_FIX_START_MS,
    PRE_FIX_WINDOW_MS,
    cue_rel_median,
    cue_rel_ms,
    draw_cue_marker,
    save_figure,
)
from eye_pre_flash.plotting.saccades.paths import OUT_ROOT

MAZE_LIM = MAZE_SCREEN_LIM


def _out_dir(session, maze, k):
    return f"{session}/k{int(k)}/maze_{maze}"


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


def _time_window_label(trials):
    window_ms = trials[0]["window_ms"]
    return f"fix_start − {window_ms:.0f} ms → fix_start"


def _trace_stats(trials):
    grid = np.arange(-trials[0]["window_ms"], 1, dtype=float)
    xs, ys = [], []
    for trial in trials:
        xs.append(np.interp(grid, trial["t_rel"], trial["x"], left=np.nan, right=np.nan))
        ys.append(np.interp(grid, trial["t_rel"], trial["y"], left=np.nan, right=np.nan))
    x_stack, y_stack = np.asarray(xs), np.asarray(ys)
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


def _snap_mae(trials):
    """Mean |gaze - prototype| over assigned samples, in maze units."""
    err = [
        np.hypot(t["x"] - t["recon_x"], t["y"] - t["recon_y"])[
            np.isfinite(t["recon_x"])
        ]
        for t in trials
    ]
    if not err or not any(e.size for e in err):
        return float("nan")
    return float(np.nanmean(np.concatenate(err)))


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


def _collect_maze_trials(eye, behavioral, attractor, maze, session, monkey, k):
    lookup = behavioral_lookup(behavioral)
    att_lookup = _attractor_lookup(attractor)
    # The one codebook: fit by `classifier.features` on clipped, warped
    # in-window fixation samples. States are assigned here from I-DT spans
    # clipped to the same window, against that same book, so the figure
    # describes exactly the state space the feature blocks use.
    _feat = load_features(monkey, k=k)
    codebook_xy = np.asarray(_feat["codebook_xy"], dtype=float)
    _assign_radius = float(_feat["assign_radius"])
    _events = _event_rows(monkey)
    codebook_name = [f"k{i}" for i in range(len(codebook_xy))]
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
        valid = np.asarray(attractor["valid"][att_i], dtype=bool)
        fix_ms = fix_start_ms(behavioral, beh_i)
        h = tuple(behavioral[f"h{i}"][beh_i] for i in range(1, 7))
        n = min(t_ms.size, x.size, y.size, valid.size)
        x, y = x[:n], y[:n]
        valid = valid[:n]
        t_ms = t_ms[:n]
        lo, hi = fix_ms - PRE_FIX_WINDOW_MS, fix_ms
        in_win = np.isfinite(t_ms) & (t_ms >= lo) & (t_ms <= hi)
        valid_win = valid & in_win & np.isfinite(x) & np.isfinite(y)
        state = assign_fixation_states(
            np.stack([x, y], axis=1).astype(np.float32),
            t_ms,
            valid_win,
            clip_fixation_spans(_events.get((session, trial_id), ()), lo, hi),
            codebook_xy,
            radius=_assign_radius,
        )
        state = np.where(valid_win, state, -1).astype(int)
        # Reconstruction must come from the book that produced `state`. It used
        # to be read from the attractor cache, which snaps against the
        # whole-trial codebook -- and k-means renumbers prototypes between
        # fits, so index i means a different location in each book. That made a
        # sample labelled k10 plot at whatever the *old* book snapped it to,
        # i.e. one state id appearing at two coordinates, neither of them its
        # own. Indexing the windowed codebook directly makes the snap, the
        # label, the colour and the codebook panel all one book.
        recon_x = np.where(state >= 0, codebook_xy[np.clip(state, 0, None), 0], np.nan)
        recon_y = np.where(state >= 0, codebook_xy[np.clip(state, 0, None), 1], np.nan)
        raw.append(
            {
                "trial_id": trial_id,
                "t_ms": t_ms,
                "x": x,
                "y": y,
                "recon_x": recon_x,
                "recon_y": recon_y,
                "state": state,
                "valid": valid,
                "fix_ms": fix_ms,
                "h": h,
                "cue_rel": cue_rel_ms(behavioral, beh_i),
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
                "events": _runs_from_state(t_rel, r["state"][keep]),
                "h": r["h"],
                "fix_ms": r["fix_ms"],
                "window_ms": window_ms,
                "cue_rel": r["cue_rel"],
            }
        )
        h_vals.append(r["h"])

    return trials, codebook_xy, codebook_name, h_vals


def _event_rows(monkey):
    """``(session, trial) -> [(name, onset, offset), ...]`` from the clean events."""
    ev = load_clean_eye_data(monkey, start_ms=PRE_FIX_START_MS, end_ms=PRE_FIX_END_MS)
    names = np.asarray(ev["name"]).astype(str)
    sess = np.asarray(ev["session"]).astype(str)
    trials = np.asarray(ev["trial_indices_all"]).astype(int)
    onset = np.asarray(ev["onset"], dtype=float)
    offset = np.asarray(ev["offset"], dtype=float)
    out = {}
    for i in range(names.size):
        out.setdefault((sess[i], int(trials[i])), []).append(
            (names[i].lower(), onset[i], offset[i])
        )
    return out


def _style_maze_axes(ax_x, ax_y, t_min, cue_rel=None, cue_median=False):
    for v in (-1.0, 1.0):
        ax_x.axhline(v, color="k", linestyle=":", linewidth=1.0, alpha=0.75)
        ax_y.axhline(v, color="k", linestyle=":", linewidth=1.0, alpha=0.75)
    for ax in (ax_x, ax_y):
        ax.axvline(0, color="0.5", linewidth=0.8)
        ax.axvline(t_min, color="0.65", linewidth=0.6, linestyle="--")
        ax.set_xlim(t_min, 0)
        ax.set_xlabel("Time rel. fix_start (ms)")
        draw_cue_marker(ax, cue_rel, median=cue_median)
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
    k = DEFAULT_K if k is None else int(k)
    eye = eye_data if eye_data is not None else load_eye_data(monkey)
    behavioral = (
        behavioral if behavioral is not None else load_eye_behavioral_data(monkey)
    )
    attractor = (
        attractor if attractor is not None else load_attractor_eye_data(monkey)
    )
    trials, codebook_xy, codebook_name, h_vals = _collect_maze_trials(
        eye, behavioral, attractor, maze, session, monkey, k
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
    _style_maze_axes(
        ax_x, ax_y, stats["t_min"],
        cue_rel=cue_rel_median([t.get("cue_rel") for t in trials]),
        cue_median=True,
    )
    _draw_codebook(
        ax_xy, codebook_xy, _median_geometry(h_vals), colors, names=codebook_name
    )

    mae = _snap_mae(trials)
    fig.suptitle(
        f"{session} maze {maze} attractor pre-fixation gaze (2D average)\n"
        f"{_time_window_label(trials)}, n={len(trials)} trials, K={k}, "
        f"valid MAE={mae:.2f} maze units"
    )
    save_figure(fig, "avg_2d", out_root=OUT_ROOT, rel_dir=_out_dir(session, maze, k))
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
    _style_maze_axes(
        ax_x, ax_y, -trial["window_ms"], cue_rel=trial.get("cue_rel")
    )

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
        out_root=OUT_ROOT,
        rel_dir=_out_dir(session, maze, k),
    )
    plt.close(fig)
    return path


def plot_2d_trials(
    maze, session, monkey="Faure", eye_data=None, behavioral=None, attractor=None, k=None
):
    k = DEFAULT_K if k is None else int(k)
    eye = eye_data if eye_data is not None else load_eye_data(monkey)
    behavioral = (
        behavioral if behavioral is not None else load_eye_behavioral_data(monkey)
    )
    attractor = (
        attractor if attractor is not None else load_attractor_eye_data(monkey)
    )
    trials, codebook_xy, codebook_name, _ = _collect_maze_trials(
        eye, behavioral, attractor, maze, session, monkey, k
    )
    if not trials:
        raise ValueError(f"No trials for session={session!r}, maze={maze}")
    colors = _state_colors(len(codebook_xy))
    mae = _snap_mae(trials)
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
    k = DEFAULT_K if k is None else int(k)
    eye = eye_data if eye_data is not None else load_eye_data(monkey)
    behavioral = (
        behavioral if behavioral is not None else load_eye_behavioral_data(monkey)
    )
    attractor = (
        attractor if attractor is not None else load_attractor_eye_data(monkey)
    )
    trials, codebook_xy, codebook_name, _ = _collect_maze_trials(
        eye, behavioral, attractor, maze, session, monkey, k
    )
    colors = _state_colors(len(codebook_xy))
    mae = _snap_mae(trials)
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
        help="Codebook size (default: %(default)s)",
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
