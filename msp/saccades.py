"""Pre-fixation gaze plots under the fixed K=5 msp assignment.

Session/maze gaze panels (raw vs snapped gaze, state spans, codebook)
assigned against this package's fixed codebook and uniform balls
(``AssignmentSpec``). Depends only on ``data.*`` and ``msp.*``.

As in `msp.features`, each trial is re-detected and re-warped inside
``[geo_present, fix_start]`` with
`msp.features.redetect_trace`. The retired fixed-length window lives in
`msp.legacy.saccades`, which reuses this module's plotting through the
``collect`` / ``out_root`` hooks.

Outputs land under ``msp/out/saccades/<tag>/<Monkey>/<session>/`` as
``avg_2d_maze_<M>.png`` and ``trial_<id>_maze_<M>.png``. Default batch scope
is the four publication sessions.

Usage:
    uv run python -m msp.saccades --tag r0.5 --session june_24_g0 --maze 2
    uv run python -m msp.saccades --tag r0.5 --session june_24_g0 --maze 2 --view trials
    uv run python -m msp.saccades --all-analyses
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

from data.builder import trial_qc_ok
from data.config import PRE_FIX_START_MS
from data.loader import load_eye_behavioral_data, load_eye_data
from data.builder import behavioral_lookup, fix_start_ms
from msp import config as cfg
from msp import labels as labelmod
from msp.config import (
    ANALYSIS_TAGS,
    CODEBOOK_XY,
    K,
    MAZE_SCREEN_LIM,
    STATE_NAMES,
    AssignmentSpec,
    resolve_tag,
)
from msp.features import assign_trial, redetect_trace
from msp.figures import save_figure

MONKEYS = ("Faure", "Nielsen")
MAZES = (2, 3, 4, 5)
MAZE_LIM = MAZE_SCREEN_LIM

# Copied from plotting/plot_io (kept local so this module stays isolated).
CUE_FIELD = "fixation_cue_present"
CUE_COLOR = "#c44e52"


def cue_rel_ms(behavioral, beh_i, *, max_lag_ms=PRE_FIX_START_MS):
    if CUE_FIELD not in behavioral:
        return None
    cue = float(np.asarray(behavioral[CUE_FIELD])[beh_i])
    fix = float(np.asarray(behavioral["fix_start"])[beh_i])
    if not (np.isfinite(cue) and np.isfinite(fix)):
        return None
    rel = (cue - fix) * 1000.0
    if rel > 0.0 or rel < -float(max_lag_ms):
        return None
    return rel


def draw_cue_marker(ax, cue_rel, *, median=False, label=CUE_FIELD):
    if cue_rel is None:
        return False
    ax.axvline(
        cue_rel, color=CUE_COLOR, linewidth=1.2, linestyle="-.", zorder=4, alpha=0.9
    )
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        cue_rel,
        0.02,
        f"{label} (median)" if median else label,
        transform=trans,
        rotation=90,
        ha="right",
        va="bottom",
        fontsize=6,
        color=CUE_COLOR,
        zorder=5,
    )
    return True


def cue_rel_median(cue_rels):
    vals = [c for c in cue_rels if c is not None]
    return float(np.median(vals)) if vals else None


def _out_rel(tag, monkey, session):
    return cfg.saccades_rel_dir(tag, monkey, session)


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
    if trials[0]["window"] is not None:  # retired fixed-length window (msp.legacy)
        return f"fix_start − {trials[0]['window_ms']:.0f} ms → fix_start"
    med = float(np.median([t["window_ms"] for t in trials]))
    return f"maze onset → fix_start (median {med:.0f} ms)"


def _trace_stats(trials):
    grid = np.arange(-max(t["window_ms"] for t in trials), 1, dtype=float)
    xs, ys = [], []
    for trial in trials:
        xs.append(np.interp(grid, trial["t_rel"], trial["x"], left=np.nan, right=np.nan))
        ys.append(np.interp(grid, trial["t_rel"], trial["y"], left=np.nan, right=np.nan))
    x_stack, y_stack = np.asarray(xs), np.asarray(ys)
    # Variable-length windows: keep only the times at least half the
    # trials reach, so a few long trials cannot stretch the axis with
    # averages over a handful of traces. Fixed windows are unaffected.
    reach = np.array([-t["window_ms"] for t in trials], dtype=float)
    covered = (reach[:, None] <= grid[None, :]).mean(axis=0) >= 0.5
    grid = grid[covered]
    x_stack, y_stack = x_stack[:, covered], y_stack[:, covered]
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


def _snap_mae(trials):
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


def _raw_redetect(behavioral, eye, maze, session, assignment, *, monkey):
    """Default window: re-detect and re-warp each trial inside its window."""
    import pymovements as pm

    from data.builder import EYE_SAMPLING_RATE_HZ, pupil_for_trial, unpack_eye

    lookup = behavioral_lookup(behavioral)
    codebook_xy = np.asarray(CODEBOOK_XY, dtype=np.float32)
    experiment = pm.Experiment(sampling_rate=EYE_SAMPLING_RATE_HZ)
    pupils = {}
    raw = []
    sessions = np.asarray(eye["session"]).astype(str)
    for i in np.flatnonzero(sessions == session):
        t_s, x, y, _session, trial_id = unpack_eye(eye, i)
        beh_i = lookup.get((session, trial_id))
        if beh_i is None or not trial_qc_ok(behavioral, beh_i):
            continue
        if int(behavioral["geo_type"][beh_i]) != maze:
            continue
        fix_ms = float(fix_start_ms(behavioral, beh_i))
        if not np.isfinite(fix_ms) or fix_ms <= 0:
            continue
        lo, hi = assignment.window_bounds(fix_ms)
        h = tuple(float(behavioral[f"h{k}"][beh_i]) for k in range(1, 7))
        pupil = pupil_for_trial(pupils, monkey, session, trial_id)
        tr = redetect_trace(t_s, x, y, h, pupil, lo, hi, experiment=experiment)
        if tr is None:
            continue
        raw.append(_raw_entry(trial_id, tr["t_ms"], tr["x"], tr["y"], tr["valid"],
                              tr["valid_win"], tr["spans"], lo, hi, fix_ms, h,
                              cue_rel_ms(behavioral, beh_i), assignment, codebook_xy))
    return raw


def _raw_entry(trial_id, t_ms, x, y, valid, valid_win, spans, lo, hi, fix_ms, h,
               cue_rel, assignment, codebook_xy):
    state, _centroids, _fix_state = assign_trial(
        np.stack([x, y], axis=1).astype(np.float32), t_ms, valid_win, spans,
        codebook_xy, assignment,
    )
    state = np.where(valid_win, state, -1).astype(int)
    recon_x = np.where(state >= 0, codebook_xy[np.clip(state, 0, None), 0], np.nan)
    recon_y = np.where(state >= 0, codebook_xy[np.clip(state, 0, None), 1], np.nan)
    return {
        "trial_id": trial_id, "t_ms": t_ms, "x": x, "y": y,
        "recon_x": recon_x, "recon_y": recon_y, "state": state, "valid": valid,
        "fix_ms": fix_ms, "lo": lo, "hi": hi, "h": h, "cue_rel": cue_rel,
    }


def _collect_maze_trials(
    behavioral, source, maze, session, assignment: AssignmentSpec, *, monkey,
    collect=_raw_redetect,
):
    """Per-trial traces for one (session, maze) under `assignment`'s window.

    `collect` turns `source` (the pooled raw eye data by default) into raw
    per-trial entries (`_raw_entry`); `msp.legacy.saccades` swaps in its own.
    """
    raw = collect(behavioral, source, maze, session, assignment, monkey=monkey)
    codebook_xy = np.asarray(CODEBOOK_XY, dtype=np.float32)
    codebook_name = list(STATE_NAMES)
    if not raw:
        return [], codebook_xy, codebook_name, []

    trials = []
    h_vals = []
    for r in raw:
        keep = np.isfinite(r["t_ms"]) & (r["t_ms"] >= r["lo"]) & (r["t_ms"] <= r["hi"])
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
                "window_ms": int(round(r["fix_ms"] - r["lo"])),
                "window": assignment.window,
                "cue_rel": r["cue_rel"],
            }
        )
        h_vals.append(r["h"])
    return trials, codebook_xy, codebook_name, h_vals


def _monkey_for_session(session):
    from data.config import monkey_for_session

    return monkey_for_session(session)


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
            mid, 0.98, label, transform=trans, ha="center", va="top",
            fontsize=7, color=color, clip_on=True, zorder=4,
        )


def _draw_codebook(ax, codebook_xy, h, colors, names=None, *, assignment=None):
    from msp.figures import _draw_balls

    _draw_h_maze_2d(ax, h)
    if assignment is not None:
        _draw_balls(ax, codebook_xy, assignment.radius, MAZE_LIM, colours=None)
    else:
        for i, (px, py) in enumerate(codebook_xy):
            ax.scatter(
                px, py, marker="X", s=70, color=colors[i % len(colors)],
                edgecolors="k", linewidths=0.4, zorder=5,
            )
            label = names[i] if names is not None and i < len(names) else str(i)
            ax.annotate(label, (px, py), textcoords="offset points", xytext=(4, 4), fontsize=8)
    # Always label prototypes on top of the region outline.
    for i, (px, py) in enumerate(codebook_xy):
        label = names[i] if names is not None and i < len(names) else str(i)
        ax.annotate(label, (px, py), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("eye_x (maze)")
    ax.set_ylabel("eye_y (maze)")
    ax.set_xlim(-MAZE_LIM, MAZE_LIM)
    ax.set_ylim(-MAZE_LIM, MAZE_LIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("fixed codebook")


def _draw_recon_points(ax, recon_x, recon_y, valid):
    assigned = valid & np.isfinite(recon_x) & np.isfinite(recon_y)
    if assigned.any():
        ax.scatter(
            recon_x[assigned], recon_y[assigned], c="k", s=16, alpha=0.85, zorder=3
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


def _load_session_data(monkey):
    return load_eye_behavioral_data(monkey), load_eye_data(monkey)


def plot_2d_average(
    maze, session, *, monkey, assignment: AssignmentSpec, dpi=200,
    behavioral=None, source=None, collect=_raw_redetect, out_root=cfg.OUT_ROOT,
):
    if behavioral is None or source is None:
        behavioral, source = _load_session_data(monkey)
    trials, codebook_xy, codebook_name, h_vals = _collect_maze_trials(
        behavioral, source, maze, session, assignment, monkey=monkey, collect=collect
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
        ax_xy, codebook_xy, _median_geometry(h_vals), colors,
        names=codebook_name, assignment=assignment,
    )

    mae = _snap_mae(trials)
    fig.suptitle(
        f"{session} maze {maze} msp pre-fixation gaze (2D average)\n"
        f"{_time_window_label(trials)}, n={len(trials)} trials, "
        f"K={K} [{assignment.tag}: {assignment.label()}], "
        f"valid MAE={mae:.2f} maze units"
    )
    save_figure(
        fig, cfg.saccades_avg_stem(maze), out_root=out_root,
        rel_dir=_out_rel(assignment.tag, monkey, session), dpi=dpi,
    )
    plt.close(fig)
    return fig


def _save_2d_trial(
    maze, session, monkey, trial, trials, codebook_xy, colors, mae,
    codebook_name, assignment, dpi=200, out_root=cfg.OUT_ROOT,
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
    _style_maze_axes(ax_x, ax_y, -trial["window_ms"], cue_rel=trial.get("cue_rel"))

    _draw_codebook(
        ax_xy, codebook_xy, trial["h"], colors,
        names=codebook_name, assignment=assignment,
    )
    ax_xy.plot(trial["x"], trial["y"], color="0.75", linewidth=0.8, alpha=0.45, zorder=2)
    if valid.any():
        ax_xy.plot(
            np.where(valid, trial["x"], np.nan),
            np.where(valid, trial["y"], np.nan),
            color="0.25", linewidth=1.0, alpha=0.8, zorder=2,
        )
    _draw_recon_points(ax_xy, trial["recon_x"], trial["recon_y"], valid)

    ax_x.legend(
        handles=[
            Line2D([0], [0], color="0.7", label="raw"),
            Line2D([0], [0], color="k", label="recon"),
            Patch(facecolor="0.7", alpha=0.4, label="state"),
        ],
        loc="lower left", fontsize=8, framealpha=0.9,
    )
    fig.suptitle(
        f"{session} maze {maze} trial {trial['trial_id']} (msp)\n"
        f"{_time_window_label(trials)} ({trial['fix_ms']:.0f} ms pre-fixation), "
        f"K={K} [{assignment.tag}], valid MAE={mae:.2f} maze units"
    )
    path = save_figure(
        fig,
        cfg.saccades_trial_stem(trial["trial_id"], maze),
        out_root=out_root,
        rel_dir=_out_rel(assignment.tag, monkey, session),
        dpi=dpi,
    )
    plt.close(fig)
    return path


def plot_2d_trials(
    maze, session, *, monkey, assignment: AssignmentSpec, dpi=200,
    behavioral=None, source=None, collect=_raw_redetect, out_root=cfg.OUT_ROOT,
):
    if behavioral is None or source is None:
        behavioral, source = _load_session_data(monkey)
    trials, codebook_xy, codebook_name, _ = _collect_maze_trials(
        behavioral, source, maze, session, assignment, monkey=monkey, collect=collect
    )
    if not trials:
        raise ValueError(f"No trials for session={session!r}, maze={maze}")
    colors = _state_colors(len(codebook_xy))
    mae = _snap_mae(trials)
    return [
        _save_2d_trial(
            maze, session, monkey, trial, trials, codebook_xy, colors, mae,
            codebook_name, assignment, dpi=dpi, out_root=out_root,
        )
        for trial in trials
    ]


def plot_2d_trial(
    maze, session, trial_id, *, monkey, assignment: AssignmentSpec, dpi=200,
    behavioral=None, source=None, collect=_raw_redetect, out_root=cfg.OUT_ROOT,
):
    if behavioral is None or source is None:
        behavioral, source = _load_session_data(monkey)
    trials, codebook_xy, codebook_name, _ = _collect_maze_trials(
        behavioral, source, maze, session, assignment, monkey=monkey, collect=collect
    )
    colors = _state_colors(len(codebook_xy))
    mae = _snap_mae(trials)
    trial = _trial_by_id(trials, trial_id)
    return _save_2d_trial(
        maze, session, monkey, trial, trials, codebook_xy, colors, mae,
        codebook_name, assignment, dpi=dpi, out_root=out_root,
    )


def plot_session(
    maze,
    session,
    *,
    view="average",
    trial_id=None,
    monkey=None,
    assignment: AssignmentSpec,
    dpi=200,
    behavioral=None,
    source=None,
    collect=_raw_redetect,
    out_root=cfg.OUT_ROOT,
):
    monkey = monkey or _monkey_for_session(session)
    if view == "trial" and trial_id is None:
        raise ValueError("--trial-id is required when --view trial")
    kwargs = dict(
        monkey=monkey, assignment=assignment, dpi=dpi,
        behavioral=behavioral, source=source, collect=collect, out_root=out_root,
    )
    if view == "average":
        return plot_2d_average(maze, session, **kwargs)
    if view == "trials":
        return plot_2d_trials(maze, session, **kwargs)
    return plot_2d_trial(maze, session, trial_id, **kwargs)


def run_all_analyses(
    *,
    assignments=None,
    sessions=labelmod.PUBLICATION_SESSIONS,
    mazes=MAZES,
    views=("average", "trials"),
    dpi=200,
    load_source=load_eye_data,
    collect=_raw_redetect,
    out_root=cfg.OUT_ROOT,
):
    """Average and per-trial panels for every assignment × publication session × maze.

    `assignments` defaults to one per ANALYSIS_TAGS. Trial plots sit beside
    the averages as ``trial_<id>_maze_<M>.png``.
    """
    if assignments is None:
        assignments = [resolve_tag(tag) for tag in ANALYSIS_TAGS]
    n_ok, n_skip = 0, 0
    # Group publication sessions by monkey so each source is loaded once.
    by_monkey = {}
    for session in sessions:
        monkey = _monkey_for_session(session)
        by_monkey.setdefault(monkey, []).append(session)

    for monkey, monkey_sessions in by_monkey.items():
        print(f"\n=== saccades {monkey} ({len(monkey_sessions)} publication sessions) ===")
        behavioral = load_eye_behavioral_data(monkey)
        source = load_source(monkey)
        for assignment in assignments:
            tag = assignment.tag
            print(f"  tag={tag} ({assignment.label()})")
            for session in monkey_sessions:
                for maze in mazes:
                    for view in views:
                        try:
                            plot_session(
                                maze, session, view=view, monkey=monkey,
                                assignment=assignment, dpi=dpi,
                                behavioral=behavioral, source=source,
                                collect=collect, out_root=out_root,
                            )
                            n_ok += 1
                        except ValueError as exc:
                            n_skip += 1
                            print(
                                f"  skip {tag}/{monkey}/{session}/maze{maze}"
                                f" ({view}): {exc}"
                            )
    print(f"saccades: wrote {n_ok} panels, skipped {n_skip}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tag", default=None,
        help=f"analysis tag ({', '.join(ANALYSIS_TAGS)}); required unless --all-analyses",
    )
    parser.add_argument("--session", default=None)
    parser.add_argument("--monkey", default=None, choices=MONKEYS)
    parser.add_argument("--maze", type=int, default=None, choices=range(1, 7))
    parser.add_argument(
        "--view", choices=("average", "trials", "trial"), default="average"
    )
    parser.add_argument("--trial-id", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--all-analyses", action="store_true",
        help="avg_2d_maze_<M> and trial_<id>_maze_<M> for every "
             "ANALYSIS_TAGS × publication session × mazes 2-5",
    )
    args = parser.parse_args()

    if args.all_analyses:
        run_all_analyses(dpi=args.dpi)
        return

    if args.tag is None or args.session is None or args.maze is None:
        parser.error("--tag, --session and --maze are required unless --all-analyses")
    assignment = resolve_tag(args.tag)
    monkey = args.monkey or _monkey_for_session(args.session)
    plot_session(
        args.maze,
        args.session,
        view=args.view,
        trial_id=args.trial_id,
        monkey=monkey,
        assignment=assignment,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
