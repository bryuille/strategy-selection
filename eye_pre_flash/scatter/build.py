"""Presentation-ready fixation summaries: one mean per group, SD bars, no dots.

Two figures per monkey, sharing one trial pool and one pair of axis limits so
they're directly comparable:

    <monkey>/by_strategy.png   grouped by decoded strategy (H / S)
    <monkey>/by_maze.png       grouped by maze (1-6)

Each group is drawn as exactly one mark: a ringed dot at the group's mean
gaze -- itself a mean of each trial's own mean gaze over the `fix_start` to
`flash_one` window (`eye_pre_flash.scatter.core.collect_fix_start`) -- with a
horizontal bar spanning +-1 SD of x and a vertical bar spanning +-1 SD of y,
taken across trials. The per-trial dots an earlier version of this module drew
underneath are gone -- with several hundred trials per maze and six mazes
overlapping in the same half-degree of screen, the clouds occluded each other
and the means alike, and nothing in the figure was readable at all-pairs.
Cross bars are SD, not SEM or a CI (`scatter.core.centroid_sd`): the question
is how spread each group's gaze is, not how precisely its mean is pinned.

The view is fixed at +-2.6 degrees in both axes, centred on the origin, rather
than a percentile crop of the cloud. A dashed circle of radius 2.5 deg marks
the reference extent around the fixation target -- checked empirically against
both monkeys' full pool of single `fix_start` samples (2026-09-10), the 100th
percentile of |x|, |y| tops out at 2.47 deg for either animal, well inside the
tracker's own +-30 deg range (`data.builder.POSITION_LIMIT_DEG`). Averaging
each trial over the `fix_start`-`flash_one` window instead of reading that one
sample can only pull per-trial extremes toward the trial's own mean, so this
box should if anything be more conservative now, but it has not been
re-checked against the windowed pool. The axis spines cross at (0, 0) instead
of sitting at the plot's edges, so the fixation target position is directly
legible.

Point is mean gaze over the `fix_start`-`flash_one` window
(`eye_pre_flash.scatter.core.collect_fix_start`), in raw `deg` space by
default -- degrees from the fixation target, not warped onto the
maze-geometry-normalised `unith` space `mazestratpair`'s occupancy features
use. This is the untransformed measurement.

Scope resolution is `eye_pre_flash.label_sources.source_scope_sessions`, so
the session pool here is the same ranked, vetted pool the classifier figures
use rather than a raw `classifier.labels` pool scope. The default is
``svm``/``top_ten``: the 10 highest-`snr_auc` sessions per monkey. `top_ten`
is defined for the `svm` source only (see `label_sources.SOURCE_SCOPES`), and
an undefined ``--source``/``--scope`` pair is rejected there rather than
silently falling back.

Must run where the eye/behavioral caches live (the cluster) -- see cloud.md.

Usage:
    uv run python -m eye_pre_flash.scatter.build
    uv run python -m eye_pre_flash.scatter.build --monkey Faure
    uv run python -m eye_pre_flash.scatter.build --source dendro --scope top_four

Writes eye_pre_flash/scatter/out/<monkey>/by_<strategy|maze>.png for the
default source and scope; any other pair gets a `_<source>_<scope>` suffix
instead of overwriting the default's files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eye_pre_flash.label_sources import (
    SOURCE_SCOPES,
    SOURCES,
    source_lookup,
    source_scope_sessions,
)
from eye_pre_flash.scatter.core import (
    INK_MUTED,
    INK_SOFT,
    LABELS,
    MAZE_COLORS,
    MAZES,
    STRATEGY_COLORS,
    STRATEGY_NAMES,
    centroid_sd,
    collect_fix_start,
    style_axes,
    subset,
)

OUT_ROOT = Path(__file__).resolve().parent / "out"
# `edgecolors` on the mean dot: the maze palette's `viridis` ramp
# (`scatter.core.MAZE_COLORS`) runs pale enough at its yellow end (~1.2:1
# against this SURFACE) that an unstroked fill is close to invisible. A ring
# keeps every step legible regardless of how pale its fill is -- and now that a
# group *is* one dot, an invisible dot loses the group outright.
MEAN_DOT = dict(s=60, edgecolors="white", linewidths=1.2, zorder=5)
# Bars sit under the mean dot: the dot marks the estimate, the bars qualify it.
SD_BAR = dict(elinewidth=1.4, capsize=3, capthick=1.4, zorder=4, fmt="none")
VIEW = (-2.6, 2.6)  # fixed +-2.6 deg box, both axes
CIRCLE_RADIUS = 2.5  # dashed reference circle about the fixation target


def _group_stats(trials, mask):
    """``(mean_x, mean_y, sd_x, sd_y, n)`` for one group, NaN-safe.

    x and y are masked to the trials finite in *both*, so the mean is a real
    gaze position rather than a pair of coordinates averaged over two
    different trial subsets.
    """
    x = trials["x"][mask]
    y = trials["y"][mask]
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    mx, sx = centroid_sd(x)
    my, sy = centroid_sd(y)
    return mx, my, sx, sy, int(x.size)


def _panel(trials, groups, *, xlim, ylim, title, axis_names):
    fig, ax = plt.subplots(figsize=(6.4, 6.4), layout="constrained")
    style_axes(ax)
    ax.add_patch(
        plt.Circle(
            (0, 0),
            CIRCLE_RADIUS,
            fill=False,
            edgecolor=INK_MUTED,
            linewidth=0.9,
            linestyle=(0, (5, 4)),
            zorder=1,
        )
    )

    outside = []
    for name, color, mask in groups:
        mx, my, sx, sy, n = _group_stats(trials, mask)
        if not (np.isfinite(mx) and np.isfinite(my)):
            continue
        # A one-trial group has a mean but no SD; draw it as a bare dot rather
        # than dropping the group or implying zero spread with a zero-length bar.
        xerr = float(sx) if np.isfinite(sx) else 0.0
        yerr = float(sy) if np.isfinite(sy) else 0.0
        ax.errorbar(mx, my, xerr=xerr, yerr=yerr, ecolor=color, **SD_BAR)
        ax.scatter([mx], [my], color=color, label=f"{name} (n={n})", **MEAN_DOT)
        if (
            mx - xerr < xlim[0]
            or mx + xerr > xlim[1]
            or my - yerr < ylim[0]
            or my + yerr > ylim[1]
        ):
            outside.append(name)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")

    # Cross the spines at the origin rather than the plot's edges, so (0, 0)
    # -- the fixation target -- sits visibly inside the axes instead of only
    # being inferred from tick labels.
    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_position(("data", 0))
    # `set_xlabel`/`set_ylabel` anchor to the spine they belong to, which is
    # now the crossing line through the middle of the plot -- so the axis name
    # goes in a corner instead, fully in axes-fraction space and therefore
    # independent of where the spine sits.
    ax.text(0.99, 0.02, axis_names[0], transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    ax.text(0.02, 0.99, axis_names[1], transform=ax.transAxes, ha="left", va="top", rotation=90, fontsize=9)
    # Tick labels still sit on the moved spines, i.e. in the middle of the
    # figure -- a light halo keeps them legible against a bar crossing them.
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_bbox(dict(facecolor="#fcfcfbcc", edgecolor="none", pad=1))

    ax.set_title(title, fontsize=12)
    ax.text(
        0.99, 0.985,
        "mean ± 1 SD",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK_SOFT,
    )
    legend = ax.legend(loc="lower left", fontsize=8, frameon=False, markerscale=1.4)
    handles = getattr(legend, "legend_handles", None) or getattr(legend, "legendHandles", [])
    for handle in handles:
        handle.set_alpha(1.0)
    return fig, outside


DEFAULT_SOURCE = "svm"
DEFAULT_SCOPE = "top_ten"


def build(monkey, *, source, scope, space):
    by_monkey, _dropped = source_scope_sessions(source, scope)
    sessions = by_monkey.get(monkey, ())
    if not sessions:
        raise SystemExit(f"no {source} labels for {monkey} in scope {scope}")

    lookup = source_lookup(source, sessions)
    trials = collect_fix_start(monkey, sessions=sessions, label_lookup=lookup, space=space)
    have = sorted(set(map(str, trials["session"])))
    if len(have) < len(sessions):
        missing = sorted(set(sessions) - set(have))
        print(f"  {monkey}: {len(have)}/{len(sessions)} scope sessions have eye data; missing {missing}")

    axis_unit = "unit H" if space == "unith" else "deg"
    axis_names = (f"eye x ({axis_unit})", f"eye y ({axis_unit})")
    xlim, ylim = VIEW, VIEW
    # Name the session pool in the title: which scope a figure came from is not
    # recoverable from the marks, and the default scope deliberately carries no
    # filename suffix, so the figure itself is the only place it can be read.
    scope_label = f"{scope.replace('_', ' ')} scope"
    # Only a non-default source/scope earns a filename suffix, so the common
    # case keeps the plain name already in use elsewhere -- same
    # drop-the-default-value convention as `scatter.core.output_rel_dir`.
    default = source == DEFAULT_SOURCE and scope == DEFAULT_SCOPE
    suffix = "" if default else f"_{source}_{scope}"

    out_dir = OUT_ROOT / monkey
    out_dir.mkdir(parents=True, exist_ok=True)

    maze_groups = [(f"maze {m}", MAZE_COLORS[m], trials["maze"] == m) for m in MAZES]
    fig, outside = _panel(
        trials, maze_groups, xlim=xlim, ylim=ylim,
        title=f"{monkey} — mean pre-flash fixation, by maze ({scope_label})", axis_names=axis_names,
    )
    path_maze = out_dir / f"by_maze{suffix}.png"
    fig.savefig(path_maze, dpi=300)
    plt.close(fig)
    print(f"Saved {path_maze}  (n={trials['x'].size} trials)")
    if outside:
        print(f"  mean ± SD reaches outside the ±{VIEW[1]:g} box: {', '.join(outside)}")

    strat_trials = subset(trials, np.isfinite(trials["label"]))
    strat_groups = [
        (STRATEGY_NAMES[v], STRATEGY_COLORS[v], strat_trials["label"] == v) for v in LABELS
    ]
    fig, outside = _panel(
        strat_trials, strat_groups, xlim=xlim, ylim=ylim,
        title=f"{monkey} — mean pre-flash fixation, by decoded strategy ({scope_label})",
        axis_names=axis_names,
    )
    path_strategy = out_dir / f"by_strategy{suffix}.png"
    fig.savefig(path_strategy, dpi=300)
    plt.close(fig)
    print(f"Saved {path_strategy}  (n={strat_trials['x'].size} labelled trials)")
    if outside:
        print(f"  mean ± SD reaches outside the ±{VIEW[1]:g} box: {', '.join(outside)}")
    return path_maze, path_strategy


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--monkey", nargs="*", default=["Faure", "Nielsen"], choices=("Faure", "Nielsen"))
    parser.add_argument("--source", default=DEFAULT_SOURCE, choices=SOURCES,
                        help=f"label source for the strategy grouping (default: {DEFAULT_SOURCE})")
    parser.add_argument("--scope", default=DEFAULT_SCOPE, choices=sorted({s for ss in SOURCE_SCOPES.values() for s in ss}),
                        help=f"session pool, per `label_sources.SOURCE_SCOPES` (default: {DEFAULT_SCOPE})")
    parser.add_argument("--space", default="deg", choices=("deg", "unith"))
    args = parser.parse_args()

    for monkey in args.monkey:
        build(monkey, source=args.source, scope=args.scope, space=args.space)


if __name__ == "__main__":
    main()
