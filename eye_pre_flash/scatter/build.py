"""Presentation-ready fixation scatter plots: raw gaze, minimal chrome.

Two figures per monkey, sharing one trial pool and one pair of axis limits so
they're directly comparable:

    <monkey>_by_strategy.png   coloured by decoded strategy (H / S)
    <monkey>_by_maze.png       coloured by maze (1-6)

Deliberately not `eye_pre_flash.scatter.plots.pooled`, which this reuses data
from: no marginal-density panels, no KDE contours. Each group gets its raw
dots plus one small ringed dot at that group's mean -- nothing else -- per
request. The view is fixed at +-2.5 degrees in both x and y, centred on the
origin, rather than a percentile crop of the cloud -- checked empirically
against both monkeys' full `fix_start` pool (2026-09-10): the 100th
percentile of |x|, |y| tops out at 2.47 deg for Faure and 2.47 deg for
Nielsen, well inside the tracker's own +-30 deg range
(`data.builder.POSITION_LIMIT_DEG`), so +-2.5 shows the entire cloud for
either animal without clipping. The axis spines are moved to cross at (0, 0)
instead of sitting at the plot's edges, so the fixation target position is
directly legible.

Point is `fix_start` gaze (`eye_pre_flash.scatter.core.collect_fix_start`),
in raw `deg` space by default -- degrees from the fixation target, not warped
onto the maze-geometry-normalised `unith` space `mazestratpair`'s occupancy
features use. This is the untransformed measurement.

Must run where the eye/behavioral caches live (the cluster) -- see cloud.md.

Usage:
    uv run python -m eye_pre_flash.scatter.build
    uv run python -m eye_pre_flash.scatter.build --monkey Faure --source svm
    uv run python -m eye_pre_flash.scatter.build --scope publication

Writes eye_pre_flash/pres/scatter/<monkey>_by_<strategy|maze>.png for the
default scope (``allplus``); a non-default ``--scope`` (e.g. ``publication``)
gets its own `_<scope>` suffix instead of overwriting the default's files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eye_pre_flash.classifier.labels import strategy_label_lookup
from eye_pre_flash.label_sources import SOURCE_STEM, SOURCES
from eye_pre_flash.counts.census import source_sessions
from eye_pre_flash.scatter.core import (
    INK_SOFT,
    LABELS,
    MAZE_COLORS,
    MAZES,
    STRATEGY_COLORS,
    STRATEGY_NAMES,
    collect_fix_start,
    style_axes,
    subset,
)

OUT_ROOT = Path(__file__).resolve().parent / "out"
# `linewidths` > 0: the maze palette's `viridis` ramp (`scatter.core.MAZE_COLORS`)
# runs pale enough at its yellow end (~1.2:1 against this SURFACE) that an
# unstroked fill is close to invisible. A thin dark edge keeps every step
# legible as a ring regardless of how pale its fill is.
DOT = dict(s=6, alpha=0.55, linewidths=0.3, edgecolors=INK_SOFT)
MEAN_DOT = dict(s=60, edgecolors="white", linewidths=1.2, zorder=5)
VIEW = (-2.5, 2.5)  # fixed +-2.5 deg box, both axes -- covers the full cloud


def _mean_dot(ax, x, y, color):
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return
    ax.scatter([x.mean()], [y.mean()], color=color, **MEAN_DOT)


def _panel(trials, groups, *, xlim, ylim, title, axis_names):
    fig, ax = plt.subplots(figsize=(6.4, 6.4), layout="constrained")
    style_axes(ax)
    for name, color, mask in groups:
        ax.scatter(trials["x"][mask], trials["y"][mask], color=color, label=name, zorder=3, **DOT)
    for name, color, mask in groups:
        _mean_dot(ax, trials["x"][mask], trials["y"][mask], color)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")

    # Cross the spines at the origin rather than the plot's edges, so (0, 0)
    # -- the fixation target -- sits visibly inside the axes instead of only
    # being inferred from tick labels.
    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_position(("data", 0))
    # `set_xlabel`/`set_ylabel` anchor to the spine they belong to, which is
    # now the crossing line through the densest part of the cloud -- so the
    # axis name goes in a corner instead, fully in axes-fraction space and
    # therefore independent of where the spine sits.
    ax.text(0.99, 0.02, axis_names[0], transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    ax.text(0.02, 0.99, axis_names[1], transform=ax.transAxes, ha="left", va="top", rotation=90, fontsize=9)
    # Tick labels still sit on the moved spines, i.e. right where the cloud is
    # densest -- a light halo keeps them legible against the dots.
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_bbox(dict(facecolor="#fcfcfbcc", edgecolor="none", pad=1))

    ax.set_title(title, fontsize=12)
    legend = ax.legend(loc="upper right", fontsize=8, frameon=False, markerscale=2.5)
    handles = getattr(legend, "legend_handles", None) or getattr(legend, "legendHandles", [])
    for handle in handles:
        handle.set_alpha(1.0)
    return fig


DEFAULT_SCOPE = "allplus"


def build(monkey, *, source, scope, space):
    by_monkey, _dropped = source_sessions(source, scope)
    sessions = by_monkey.get(monkey, ())
    if not sessions:
        raise SystemExit(f"no {source} labels for {monkey} in scope {scope}")

    lookup = strategy_label_lookup(sessions, stem=SOURCE_STEM[source])
    trials = collect_fix_start(monkey, sessions=sessions, label_lookup=lookup, space=space)

    axis_unit = "unit H" if space == "unith" else "deg"
    axis_names = (f"eye x ({axis_unit})", f"eye y ({axis_unit})")
    xlim, ylim = VIEW, VIEW
    # Only a non-default scope earns a filename suffix, so the common
    # (allplus) case keeps the plain name already in use elsewhere -- same
    # drop-the-default-value convention as `scatter.core.output_rel_dir`.
    suffix = "" if scope == DEFAULT_SCOPE else f"_{scope}"

    maze_groups = [(f"maze {m}", MAZE_COLORS[m], trials["maze"] == m) for m in MAZES]
    fig = _panel(
        trials, maze_groups, xlim=xlim, ylim=ylim,
        title=f"{monkey} — fixation at flash, by maze", axis_names=axis_names,
    )
    out_dir = OUT_ROOT / monkey
    out_dir.mkdir(parents=True, exist_ok=True)
    path_maze = out_dir / f"by_maze{suffix}.png"
    fig.savefig(path_maze, dpi=300)
    plt.close(fig)
    print(f"Saved {path_maze}  (n={trials['x'].size} trials)")

    strat_trials = subset(trials, np.isfinite(trials["label"]))
    strat_groups = [
        (STRATEGY_NAMES[v], STRATEGY_COLORS[v], strat_trials["label"] == v) for v in LABELS
    ]
    fig = _panel(
        strat_trials, strat_groups, xlim=xlim, ylim=ylim,
        title=f"{monkey} — fixation at flash, by decoded strategy", axis_names=axis_names,
    )
    path_strategy = out_dir / f"by_strategy{suffix}.png"
    fig.savefig(path_strategy, dpi=300)
    plt.close(fig)
    print(f"Saved {path_strategy}  (n={strat_trials['x'].size} labelled trials)")
    return path_maze, path_strategy


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--monkey", nargs="*", default=["Faure", "Nielsen"], choices=("Faure", "Nielsen"))
    parser.add_argument("--source", default="dendro", choices=SOURCES,
                        help="label source for the strategy colouring (default: dendro, non-circular)")
    parser.add_argument("--scope", default=DEFAULT_SCOPE, choices=("publication", "all", "allplus"))
    parser.add_argument("--space", default="deg", choices=("deg", "unith"))
    args = parser.parse_args()

    for monkey in args.monkey:
        build(monkey, source=args.source, scope=args.scope, space=args.space)


if __name__ == "__main__":
    main()
