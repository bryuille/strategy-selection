"""Presentation-ready fixation summary: one mean per maze, SD bars, no dots.

One figure per monkey: <monkey>/by_maze.png

Each maze is drawn as exactly one mark: a ringed dot at the maze's mean gaze
-- itself a mean, across trials, of each trial's own mean gaze over the
`fix_start` to `flash_one` window (`eye_pre_flash.scatter.core`) -- with a
horizontal bar spanning +-1 SD of x and a vertical bar spanning +-1 SD of y.

The view is fixed at +-2.6 degrees in both axes, centred on the origin,
rather than a percentile crop of the cloud. A dashed circle of radius 2.5 deg
marks the reference extent around the fixation target -- checked empirically
against both monkeys' full pool of single `fix_start` samples (2026-09-10),
the 100th percentile of |x|, |y| tops out at 2.47 deg for either animal.

Must run where the eye/behavioral caches live (the cluster) -- see cloud.md.

Usage:
    uv run python -m eye_pre_flash.scatter.build
    uv run python -m eye_pre_flash.scatter.build --monkey Faure

Also writes a p-value table, <monkey>/maze_pvalues.png: for each ordered pair
of mazes (i, j), the one-sided p-value of the distance between maze i's and
maze j's mean gaze, scaled by maze j's own SD of trial-to-mean distance
(`eye_pre_flash.scatter.core.maze_pvalues`) -- directional and uncorrected for
multiple comparisons.

Writes eye_pre_flash/scatter/out/<monkey>/by_maze.png and
eye_pre_flash/scatter/out/<monkey>/maze_pvalues.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from eye_pre_flash.classifier.classifier_io import render_table_png
from eye_pre_flash.scatter.core import MAZES, collect_fix_start, maze_pvalues, maze_stats

OUT_ROOT = Path(__file__).resolve().parent / "out"

# --- Palette ---------------------------------------------------------------
# Maze 1-6 is *ordered* (1-3 hierarchical, 5-6 sequential, 4 the boundary), but
# a scatter needs *all-pairs* CVD separation -- any two mazes' dots can end up
# neighbors anywhere in the cloud -- not just adjacent-step separation. A
# two-hue diverging ramp (blue/grey/red, one prior version of this palette)
# already cleared that floor; requested instead is a full-spectrum "rainbow"
# for more visual variety across all six. Plain ROYGBIV fails outright --
# non-monotone lightness (yellow reads far lighter than blue at the same
# "step") and red/green sit close for red-green colorblindness -- so this
# samples `viridis` at t = 0, 0.2, ..., 1.0 instead: perceptually-uniform
# lightness end to end, and still six distinct hues. Validated all-pairs
# (light surface): worst pair Delta E 15.0 normal-vision (>= 15 floor), 10.5
# CVD (clear of the 6-8 floor). The one thing `viridis` doesn't clear on its
# own is light-end contrast: step 6 (`#fde725`, yellow) sits at 1.23:1 against
# the `#fcfcfb` surface -- a fill that faint is invisible outright, not just
# hard to tell apart, and no legend fixes an invisible mark. `MEAN_DOT` gives
# every point a thin dark edge for exactly this, so the pale steps still
# render as a visible ring.
MAZE_COLORS = {
    1: "#440154",
    2: "#414487",
    3: "#2a788e",
    4: "#22a884",
    5: "#7ad151",
    6: "#fde725",
}

INK_SOFT = "#52514e"
INK_MUTED = "#8a8880"
SURFACE = "#fcfcfb"


def style_axes(ax, *, grid=True):
    """Recessive frame and grid; data is the only thing with weight."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SOFT, labelsize=8, width=0.8, length=3)
    if grid:
        ax.grid(True, color="#e8e6df", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
    return ax


# --- Figure constants -------------------------------------------------------
# `edgecolors` on the mean dot: the maze palette's `viridis` ramp
# (`MAZE_COLORS`) runs pale enough at its yellow end (~1.2:1 against
# `SURFACE`) that an unstroked fill is close to invisible. A ring keeps every
# step legible regardless of how pale its fill is -- and now that a group
# *is* one dot, an invisible dot loses the group outright.
MEAN_DOT = dict(s=60, edgecolors="white", linewidths=1.2, zorder=5)
# Bars sit under the mean dot: the dot marks the estimate, the bars qualify it.
SD_BAR = dict(elinewidth=1.4, capsize=3, capthick=1.4, zorder=4, fmt="none")
VIEW = (-2.6, 2.6)  # fixed +-2.6 deg box, both axes
CIRCLE_RADIUS = 2.5  # dashed reference circle about the fixation target


def _panel(stats, *, xlim, ylim, title):
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

    for maze in MAZES:
        s = stats[maze]
        color = MAZE_COLORS[maze]
        ax.errorbar(s["mean_x"], s["mean_y"], xerr=s["sd_x"], yerr=s["sd_y"], ecolor=color, **SD_BAR)
        ax.scatter([s["mean_x"]], [s["mean_y"]], color=color, label=f"maze {maze} (n={s['n']})", **MEAN_DOT)

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
    ax.text(0.99, 0.02, "eye x (deg)", transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    ax.text(0.02, 0.99, "eye y (deg)", transform=ax.transAxes, ha="left", va="top", rotation=90, fontsize=9)
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
    return fig


def _fmt_p(value):
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _pvalue_table(pvalues, monkey):
    """p(row maze | column maze's own spread) -- directional, diagonal blank."""
    columns = ["maze"] + [str(j) for j in MAZES]
    rows = [
        [str(i)] + [_fmt_p(pvalues[i][j]) if i != j else "--" for j in MAZES]
        for i in MAZES
    ]
    return render_table_png(
        columns,
        rows,
        stem="maze_pvalues",
        rel_dir=monkey,
        title=f"{monkey}: maze p-values (z-test)",
        out_root=OUT_ROOT,
    )


def build(monkey):
    trials = collect_fix_start(monkey)
    stats = maze_stats(trials)

    out_dir = OUT_ROOT / monkey
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = _panel(stats, xlim=VIEW, ylim=VIEW, title=f"{monkey} — mean pre-flash fixation, by maze")
    path = out_dir / "by_maze.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}  (n={trials['x'].size} trials)")

    _pvalue_table(maze_pvalues(stats), monkey)

    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--monkey", nargs="*", default=["Faure", "Nielsen"], choices=("Faure", "Nielsen"))
    args = parser.parse_args()

    for monkey in args.monkey:
        build(monkey)


if __name__ == "__main__":
    main()
