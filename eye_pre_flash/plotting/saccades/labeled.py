"""Pre-fixation saccade plots with pymovements event labels.

Same 2D views as `eye_pre_flash.plotting.saccades.traces`, with vertical borders at
each saccade/fixation onset and offset and a text label for the event type.

Usage:
    uv run python -m eye_pre_flash.plotting.saccades.labeled --session june_24_g0 --maze 1
    uv run python -m eye_pre_flash.plotting.saccades.labeled --session june_24_g0 --maze 1 --view trials
    uv run python -m eye_pre_flash.plotting.saccades.labeled --session june_24_g0 --maze 1 --view trial --trial-id 141
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

from data.loader import load_clean_eye_data, load_eye_behavioral_data, load_eye_data
from eye_pre_flash.plotting.plot_io import save_figure
from eye_pre_flash.plotting.saccades.traces import (
    _collect_maze_trials,
    _exit_x_refs,
    _exit_y_refs,
    _style_2d_axes,
    _time_window_label,
    _trace_stats,
    _trial_by_id,
)

VISUALIZER = "saccades/labeled"
EVENT_STYLE = {
    "saccade": {"color": "#c44e52"},
    "fixation": {"color": "#4c72b0"},
}


def _out_dir(session, maze):
    return f"{VISUALIZER}/{session}/maze_{maze}"


def _event_kind(name):
    name = str(name).lower()
    if "saccade" in name:
        return "saccade"
    if name == "fixation_idt":
        return "fixation"
    return None


def _event_lookup(events):
    lookup = {}
    for i in range(len(events["name"])):
        kind = _event_kind(events["name"][i])
        if kind is None:
            continue
        key = (str(events["session"][i]), int(events["trial_indices_all"][i]))
        lookup.setdefault(key, []).append(
            {
                "name": kind,
                "onset": float(events["onset"][i]),
                "offset": float(events["offset"][i]),
            }
        )
    return lookup


def _events_in_window(lookup, session, trial_id, *, t_min, t_max, fix_ms):
    spans = []
    for event in lookup.get((session, trial_id), ()):
        onset_rel = event["onset"] - fix_ms
        offset_rel = event["offset"] - fix_ms
        if offset_rel < t_min or onset_rel > t_max:
            continue
        spans.append(
            {
                "name": event["name"],
                "onset_rel": max(onset_rel, t_min),
                "offset_rel": min(offset_rel, t_max),
            }
        )
    return spans


def _attach_events(trials, session, events):
    lookup = _event_lookup(events)
    for trial in trials:
        trial["events"] = _events_in_window(
            lookup,
            session,
            trial["trial_id"],
            t_min=-trial["window_ms"],
            t_max=0.0,
            fix_ms=trial["fix_ms"],
        )
    return trials


def _draw_event_spans(ax_x, ax_y, events):
    for event in events:
        style = EVENT_STYLE[event["name"]]
        color = style["color"]
        for ax in (ax_x, ax_y):
            ax.axvspan(
                event["onset_rel"],
                event["offset_rel"],
                color=color,
                alpha=0.16,
                linewidth=0,
                zorder=0,
            )
            ax.axvline(event["onset_rel"], color=color, linewidth=1.1, zorder=3)
            ax.axvline(event["offset_rel"], color=color, linewidth=1.1, zorder=3)

        mid = 0.5 * (event["onset_rel"] + event["offset_rel"])
        trans = blended_transform_factory(ax_x.transData, ax_x.transAxes)
        ax_x.text(
            mid,
            0.98,
            event["name"],
            transform=trans,
            ha="center",
            va="top",
            fontsize=8,
            color=color,
            rotation=90,
            clip_on=True,
            zorder=4,
        )

    ax_x.legend(
        handles=[
            Patch(facecolor=EVENT_STYLE["saccade"]["color"], alpha=0.4, label="saccade"),
            Patch(
                facecolor=EVENT_STYLE["fixation"]["color"], alpha=0.4, label="fixation"
            ),
        ],
        loc="lower left",
        fontsize=8,
        framealpha=0.9,
    )


def plot_2d_saccades_average(
    maze, session, monkey="Faure", eye_data=None, behavioral=None, events=None
):
    """Mean eye_x / eye_y vs time; event labels are per-trial and omitted here."""
    del events  # trial-specific; not shown on the session average
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

    _style_2d_axes(ax_x, ax_y, x_refs, y_refs, stats["t_min"])
    fig.suptitle(
        f"{session} maze {maze} labeled pre-fixation saccades (2D average)\n"
        f"{_time_window_label(trials)}, n={len(trials)} trials"
    )
    save_figure(fig, "avg_2d", rel_dir=_out_dir(session, maze))
    plt.close(fig)
    return fig


def _save_2d_trial(maze, session, trial, trials):
    fig, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    ax_x.plot(trial["t_rel"], trial["x"], color="C0", linewidth=1.2, zorder=2)
    ax_y.plot(trial["t_rel"], trial["y"], color="C1", linewidth=1.2, zorder=2)
    _draw_event_spans(ax_x, ax_y, trial["events"])
    _style_2d_axes(
        ax_x,
        ax_y,
        _exit_x_refs(trial["h"][0], trial["h"][3]),
        _exit_y_refs(trial["h"][1], trial["h"][2], trial["h"][4], trial["h"][5]),
        -trial["window_ms"],
    )
    fig.suptitle(
        f"{session} maze {maze} trial {trial['trial_id']} (2D, labeled)\n"
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
    maze, session, monkey="Faure", eye_data=None, behavioral=None, events=None
):
    """One labeled 2D side-by-side plot per trial."""
    eye = eye_data if eye_data is not None else load_eye_data(monkey)
    behavioral = (
        behavioral if behavioral is not None else load_eye_behavioral_data(monkey)
    )
    events = events if events is not None else load_clean_eye_data(monkey)
    trials, _, _, _ = _collect_maze_trials(eye, behavioral, maze, session)
    if not trials:
        raise ValueError(f"No trials for session={session!r}, maze={maze}")
    _attach_events(trials, session, events)
    return [_save_2d_trial(maze, session, trial, trials) for trial in trials]


def plot_2d_saccade_trial(
    maze,
    session,
    trial_id,
    monkey="Faure",
    eye_data=None,
    behavioral=None,
    events=None,
):
    """One labeled 2D side-by-side plot for a single trial."""
    eye = eye_data if eye_data is not None else load_eye_data(monkey)
    behavioral = (
        behavioral if behavioral is not None else load_eye_behavioral_data(monkey)
    )
    events = events if events is not None else load_clean_eye_data(monkey)
    trials, _, _, _ = _collect_maze_trials(eye, behavioral, maze, session)
    _attach_events(trials, session, events)
    trial = _trial_by_id(trials, trial_id)
    return _save_2d_trial(maze, session, trial, trials)


def plot_saccades(
    maze,
    session,
    *,
    view="average",
    trial_id=None,
    monkey="Faure",
    eye_data=None,
    behavioral=None,
    events=None,
):
    """Dispatch to the requested 2D view."""
    if view == "trial" and trial_id is None:
        raise ValueError("--trial-id is required when --view trial")

    if view == "average":
        return plot_2d_saccades_average(
            maze, session, monkey, eye_data, behavioral, events
        )
    if view == "trials":
        return plot_2d_saccades_trials(
            maze, session, monkey, eye_data, behavioral, events
        )
    return plot_2d_saccade_trial(
        maze, session, trial_id, monkey, eye_data, behavioral, events
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
    args = parser.parse_args()

    plot_saccades(
        args.maze,
        args.session,
        view=args.view,
        trial_id=args.trial_id,
        monkey=args.monkey,
    )


if __name__ == "__main__":
    main()
