"""Scatter of the gaze sample at `fix_start`, holding one variable fixed at a time.

Each view answers "does this variable move the eyes at fix_start?" by fixing
everything else it can:

    pooled_maze        every trial in one panel, coloured by maze, with the x
                       and y marginals on the margins
    pooled_strategy    the same panel coloured by the decoded strategy label
    maze_facets        one panel per maze -- maze held fixed, the pooled cloud
                       greyed behind it for reference
    strategy_facets    one panel per strategy label, same idea
    maze_x_strategy    the 6 x 2 grid: within a maze, does the label still move
                       the cloud?
    session_facets     one panel per session -- session (so calibration) held
                       fixed, coloured by maze
    centroids          group means with 95% CIs, mazes connected 1 -> 6. The
                       shifts are ~0.1-0.3 deg against a ~0.5 deg spread, so
                       this is the view any effect is actually readable in
    marginals          per-group x and y distributions, one row per axis

Usage:
    uv run python -m eye_pre_flash.scatter.plots --monkey Faure
    uv run python -m eye_pre_flash.scatter.plots --monkey Nielsen --space unith
    uv run python -m eye_pre_flash.scatter.plots --monkey Faure --center session
    uv run python -m eye_pre_flash.scatter.plots --all
    uv run python -m eye_pre_flash.scatter.plots --monkey Faure --views centroids marginals
"""

from __future__ import annotations

import argparse
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from eye_pre_flash.classifier.labels import (
    MIN_LABEL_AGREEMENT,
    SCOPES,
    scope_sessions,
    strategy_label_lookup,
)
from eye_pre_flash.scatter.core import (
    CLOUD,
    INK,
    INK_MUTED,
    INK_SOFT,
    LABELS,
    MAZE_COLORS,
    MAZES,
    SPACES,
    STRATEGY_COLORS,
    STRATEGY_NAMES,
    TOL_MS,
    center_by_session,
    centroid_ci,
    clipped_count,
    collect_fix_start,
    limits,
    output_rel_dir,
    subset,
    save_figure,
    save_trials_csv,
    style_axes,
)

VIEWS = (
    "pooled_maze",
    "pooled_strategy",
    "maze_facets",
    "strategy_facets",
    "maze_x_strategy",
    "session_facets",
    "centroids",
    "marginals",
)
DOT = dict(s=5, alpha=0.28, linewidths=0.0)
CLOUD_DOT = dict(s=4, alpha=0.35, linewidths=0.0, color=CLOUD, zorder=1)
# Fewest trials a 50%-mass KDE contour is drawn for.
MIN_CONTOUR_N = 40


# --- shared furniture ------------------------------------------------------


def _axis_names(space):
    unit = SPACES[space]["axis"]
    return f"eye x ({unit})", f"eye y ({unit})"


def _footer(trials, *, extra=""):
    n = trials["x"].size
    centered = trials.get("centered", False)
    bits = [
        f"{trials['monkey']}",
        f"n={n} trials",
        f"space={trials['space']}",
        f"sample within ±{TOL_MS:g} ms of fix_start",
    ]
    if centered:
        bits.append("session-mean centred")
    if extra:
        bits.append(extra)
    # Wrapped, because the provenance line is longer than a figure is wide and
    # matplotlib will happily run it off both edges.
    return textwrap.fill(" | ".join(bits), width=110)


def _finish(fig, trials, *, title, extra=""):
    """Title above, provenance line below -- both via the layout engine.

    `fig.text` at a fixed y is invisible to constrained layout and lands on top
    of the bottom row's axis labels, so the footer goes in as a supxlabel and
    the title as a suptitle, which the engine reserves room for.
    """
    fig.suptitle(title, fontsize=12, color=INK)
    fig.supxlabel(_footer(trials, extra=extra), fontsize=7.5, color=INK_SOFT)


def _origin(ax, xlim, ylim):
    """Mark the fixation target -- the point every trial is being pulled to."""
    if xlim[0] < 0 < xlim[1]:
        ax.axvline(0, color=INK_MUTED, linewidth=0.7, zorder=2)
    if ylim[0] < 0 < ylim[1]:
        ax.axhline(0, color=INK_MUTED, linewidth=0.7, zorder=2)


def _centroid(ax, x, y, color, *, z=6):
    """Ringed group mean with 95% CI whiskers. Returns (mx, my), or None."""
    mx, ex = centroid_ci(x)
    my, ey = centroid_ci(y)
    if not np.isfinite(mx) or not np.isfinite(my):
        return None
    ax.errorbar(
        mx,
        my,
        xerr=None if not np.isfinite(ex) else ex,
        yerr=None if not np.isfinite(ey) else ey,
        fmt="none",
        ecolor=color,
        elinewidth=1.6,
        capsize=2.5,
        zorder=z,
    )
    ax.scatter(
        [mx],
        [my],
        s=95,
        color=color,
        edgecolors="white",
        linewidths=1.6,
        zorder=z + 1,
    )
    return mx, my


# Diagonal-only: a point's own error bar runs exactly horizontal and exactly
# vertical from it, so an axis-aligned candidate direction would send the
# label straight down that same whisker. A diagonal clears both arms.
_COMPASS = tuple(
    np.array(v) / np.linalg.norm(v) for v in ((1, 1), (-1, 1), (-1, -1), (1, -1))
)


def _label_offsets(ax, anchors, *, radius=15.0, step=0.055, obstacles=()):
    """Per-anchor (dx, dy) label offset, in points, clear of other anchors.

    A fixed offset (as `_centroid` used to hand every label) overlaps its
    neighbour's marker, whisker or connecting line whenever two group means
    sit within a few tenths of a degree of each other, which happens often
    here. Instead, each anchor picks whichever of 8 compass directions is
    currently furthest (in axes-fraction space, so the x/y scales don't have
    to match) from every other anchor, any given `obstacles` point (e.g. a
    legend corner), and every label already placed -- so labels spread out
    instead of stacking on the same side.
    """
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xspan, yspan = xlim[1] - xlim[0], ylim[1] - ylim[0]

    def norm(pt):
        return np.array([(pt[0] - xlim[0]) / xspan, (pt[1] - ylim[0]) / yspan])

    points = [norm(a) for a in anchors]
    fixed_obstacles = [norm(o) for o in obstacles]
    placed = []
    offsets = []
    for i, p in enumerate(points):
        near = [q for j, q in enumerate(points) if j != i] + fixed_obstacles + placed
        best_dir, best_score = _COMPASS[0], -np.inf
        for d in _COMPASS:
            cand = p + d * step
            score = min((np.linalg.norm(cand - q) for q in near), default=1.0)
            if score > best_score:
                best_score, best_dir = score, d
        placed.append(p + best_dir * step)
        offsets.append(tuple(best_dir * radius))
    return offsets


def _annotate(ax, xy, text, offset, *, color=INK, z=8):
    """Place `text` at `xy` + `offset` (points), aligned away from the point."""
    dx, dy = offset
    ha = "center" if abs(dx) < 1e-6 else ("left" if dx > 0 else "right")
    va = "center" if abs(dy) < 1e-6 else ("bottom" if dy > 0 else "top")
    ax.annotate(
        text,
        xy,
        textcoords="offset points",
        xytext=offset,
        ha=ha,
        va=va,
        fontsize=8.5,
        color=color,
        fontweight="medium",
        zorder=z,
    )


def _panel(ax, trials, mask, color, *, xlim, ylim, title, cloud=True):
    style_axes(ax)
    if cloud:
        ax.scatter(trials["x"][~mask], trials["y"][~mask], **CLOUD_DOT)
        _contour(
            ax,
            trials["x"],
            trials["y"],
            "#b3b0a7",
            xlim=xlim,
            ylim=ylim,
            lw=1.2,
            zorder=4,
        )
    ax.scatter(
        trials["x"][mask], trials["y"][mask], color=color, zorder=3, **DOT
    )
    _contour(ax, trials["x"][mask], trials["y"][mask], color, xlim=xlim, ylim=ylim)
    _origin(ax, xlim, ylim)
    _centroid(ax, trials["x"][mask], trials["y"][mask], color)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9.5, color=INK)


def _kde_line(ax, v, color, *, span, label=None, lw=1.6, vertical=False):
    """Group density as a smooth line.

    Step histograms of six overlapping groups at this sample size are mostly
    bin noise -- the group means differ by ~0.2 deg while neighbouring bins
    differ by more than that -- so the marginals are drawn as Gaussian KDEs,
    which is the shift-of-the-mode claim these panels are making.
    """
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 5 or v.std() == 0:
        return
    grid = np.linspace(*span, 256)
    dens = gaussian_kde(v)(grid)
    if vertical:
        ax.plot(dens, grid, color=color, linewidth=lw, label=label, zorder=3)
    else:
        ax.plot(grid, dens, color=color, linewidth=lw, label=label, zorder=3)


def _mass_level(z, mass):
    """The density level whose super-level set holds `mass` of the total."""
    flat = np.sort(z.ravel())[::-1]
    csum = np.cumsum(flat)
    if csum[-1] <= 0:
        return None
    idx = int(np.searchsorted(csum / csum[-1], mass))
    return float(flat[min(idx, flat.size - 1)])


def _contour(ax, x, y, color, *, xlim, ylim, mass=0.5, lw=1.8, zorder=5):
    """The contour containing `mass` of the group's 2-D density.

    The point cloud is a single overlapping blob at every grouping, so the
    readable version of "this group sits further right" is where its central
    50% of trials falls, not which dots are on top.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    # Below this a single-level KDE contour traces the few points it has
    # rather than the group's shape, so the cell shows dots and a mean only.
    if x.size < MIN_CONTOUR_N:
        return
    try:
        # Slightly wider than Scott's rule: at a few hundred trials the
        # default bandwidth puts real but uninterpretable wiggle in a
        # single-level contour.
        kde = gaussian_kde(np.vstack([x, y]), bw_method=0.55)
    except np.linalg.LinAlgError:
        return
    gx = np.linspace(*xlim, 120)
    gy = np.linspace(*ylim, 120)
    mx, my = np.meshgrid(gx, gy)
    z = kde(np.vstack([mx.ravel(), my.ravel()])).reshape(mx.shape)
    level = _mass_level(z, mass)
    if level is None:
        return
    ax.contour(
        mx, my, z, levels=[level], colors=[color], linewidths=lw, zorder=zorder
    )


# --- views -----------------------------------------------------------------


def pooled(trials, *, by="maze", rel_dir="", stem=""):
    """One panel, every trial, coloured by `by`, with both marginals."""
    if by == "strategy":
        trials = subset(trials, np.isfinite(trials["label"]))
        groups = [
            (STRATEGY_NAMES[v], STRATEGY_COLORS[v], trials["label"] == v)
            for v in LABELS
        ]
        legend_title = "decoded strategy"
    else:
        groups = [
            (f"maze {m}", MAZE_COLORS[m], trials["maze"] == m) for m in MAZES
        ]
        legend_title = "maze"
    x, y = trials["x"], trials["y"]

    xlim, ylim = limits(x, y)
    fig = plt.figure(figsize=(7.6, 7.2), layout="constrained")
    gs = fig.add_gridspec(
        2, 2, width_ratios=(4.2, 1.0), height_ratios=(1.0, 4.2), wspace=0.04, hspace=0.04
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    style_axes(ax)
    style_axes(ax_top, grid=False)
    style_axes(ax_right, grid=False)

    for name, color, mask in groups:
        ax.scatter(x[mask], y[mask], color=color, label=name, zorder=3, **DOT)
        _kde_line(ax_top, x[mask], color, span=xlim, label=name)
        _kde_line(ax_right, y[mask], color, span=ylim, vertical=True)
    # Contours read as a shape comparison for two groups and as a hairball for
    # six, so the pooled maze panel leaves them to `maze_facets`, where each
    # maze gets its own contour against the greyed pooled one.
    if len(groups) <= 2:
        for name, color, mask in groups:
            _contour(ax, x[mask], y[mask], color, xlim=xlim, ylim=ylim)
    # Means last, so no group's identity depends on having been drawn on top.
    for name, color, mask in groups:
        _centroid(ax, x[mask], y[mask], color)

    _origin(ax, xlim, ylim)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    xname, yname = _axis_names(trials["space"])
    ax.set_xlabel(xname, fontsize=9, color=INK_SOFT)
    ax.set_ylabel(yname, fontsize=9, color=INK_SOFT)
    ax_top.set_yticks([])
    ax_right.set_xticks([])
    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False)
    ax_top.set_ylabel("density", fontsize=7.5, color=INK_SOFT)
    ax_right.set_xlabel("density", fontsize=7.5, color=INK_SOFT)

    leg = ax.legend(
        title=legend_title,
        loc="upper left",
        fontsize=8,
        title_fontsize=8,
        frameon=True,
        framealpha=0.92,
        markerscale=3.0,
        borderpad=0.5,
    )
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)
    leg.get_frame().set_edgecolor("#e0ded7")
    leg.get_title().set_color(INK_SOFT)

    clipped = clipped_count(x, y, xlim, ylim)
    _finish(
        fig,
        trials,
        title=f"{trials['monkey']} — gaze at fix_start by {legend_title}",
        extra=(
            ("contour = central 50% of each group, " if len(groups) <= 2 else "")
            + f"ringed mark = mean ±95% CI; {clipped} point(s) outside the plotted box"
        ),
    )
    save_figure(fig, stem or f"pooled_{by}", rel_dir=rel_dir)
    plt.close(fig)


def maze_facets(trials, *, rel_dir=""):
    """Maze held fixed: six panels over the same greyed pooled cloud."""
    xlim, ylim = limits(trials["x"], trials["y"])
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 8.0), layout="constrained")
    for ax, m in zip(axes.ravel(), MAZES):
        mask = trials["maze"] == m
        _panel(
            ax,
            trials,
            mask,
            MAZE_COLORS[m],
            xlim=xlim,
            ylim=ylim,
            title=f"maze {m}  (n={int(mask.sum())})",
        )
    xname, yname = _axis_names(trials["space"])
    for ax in axes[-1]:
        ax.set_xlabel(xname, fontsize=8.5, color=INK_SOFT)
    for ax in axes[:, 0]:
        ax.set_ylabel(yname, fontsize=8.5, color=INK_SOFT)
    _finish(
        fig,
        trials,
        title=f"{trials['monkey']} — gaze at fix_start, one panel per maze",
        extra=(
            "grey dots = the other mazes, grey contour = central 50% of all "
            "trials pooled; coloured contour = this maze's central 50%, "
            "ringed mark = its mean ±95% CI"
        ),
    )
    save_figure(fig, "maze_facets", rel_dir=rel_dir)
    plt.close(fig)


def strategy_facets(trials, *, rel_dir=""):
    """Strategy held fixed: one panel per decoded label."""
    sub = subset(trials, np.isfinite(trials["label"]))
    xlim, ylim = limits(sub["x"], sub["y"])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 5.4), layout="constrained")
    for ax, v in zip(axes, LABELS):
        mask = sub["label"] == v
        _panel(
            ax,
            sub,
            mask,
            STRATEGY_COLORS[v],
            xlim=xlim,
            ylim=ylim,
            title=f"{STRATEGY_NAMES[v]}  (n={int(mask.sum())})",
        )
    xname, yname = _axis_names(trials["space"])
    for ax in axes:
        ax.set_xlabel(xname, fontsize=8.5, color=INK_SOFT)
    axes[0].set_ylabel(yname, fontsize=8.5, color=INK_SOFT)
    _finish(
        fig,
        sub,
        title=f"{trials['monkey']} — gaze at fix_start, one panel per decoded strategy",
        extra=(
            "grey dots = the other label, grey contour = central 50% of both "
            "labels pooled; coloured contour = this label's central 50%, "
            "ringed mark = its mean ±95% CI"
        ),
    )
    save_figure(fig, "strategy_facets", rel_dir=rel_dir)
    plt.close(fig)


def maze_x_strategy(trials, *, rel_dir=""):
    """Both fixed: does the label move the cloud inside a single maze?"""
    trials = subset(trials, np.isfinite(trials["label"]))
    x, y = trials["x"], trials["y"]
    maze, label = trials["maze"], trials["label"]
    xlim, ylim = limits(x, y)
    fig, axes = plt.subplots(
        2, 6, figsize=(14.6, 5.6), layout="constrained", sharex=True, sharey=True
    )
    for r, v in enumerate(LABELS):
        for c, m in enumerate(MAZES):
            ax = axes[r, c]
            style_axes(ax)
            mask = (maze == m) & (label == v)
            ax.scatter(x[~mask], y[~mask], **CLOUD_DOT)
            ax.scatter(
                x[mask], y[mask], color=STRATEGY_COLORS[v], zorder=3, **DOT
            )
            _contour(
                ax, x[mask], y[mask], STRATEGY_COLORS[v], xlim=xlim, ylim=ylim
            )
            _origin(ax, xlim, ylim)
            _centroid(ax, x[mask], y[mask], STRATEGY_COLORS[v])
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.set_title(f"maze {m} · n={int(mask.sum())}", fontsize=8, color=INK)
            if c == 0:
                ax.set_ylabel(STRATEGY_NAMES[v], fontsize=9.5, color=INK)
    xname, _ = _axis_names(trials["space"])
    for ax in axes[-1]:
        ax.set_xlabel(xname, fontsize=8, color=INK_SOFT)
    _finish(
        fig,
        trials,
        title=f"{trials['monkey']} — gaze at fix_start, maze × decoded strategy",
        extra=(
            "rows are the strategy label, columns the maze; grey = every other "
            f"cell; contour drawn only for cells with ≥{MIN_CONTOUR_N} trials"
        ),
    )
    save_figure(fig, "maze_x_strategy", rel_dir=rel_dir)
    plt.close(fig)


def session_facets(trials, *, rel_dir="", max_panels=24):
    """Session held fixed, so tracker calibration cannot carry a maze effect."""
    sessions = list(dict.fromkeys(trials["session"].tolist()))
    dropped = max(0, len(sessions) - max_panels)
    sessions = sessions[:max_panels]
    ncols = min(4, max(1, len(sessions)))
    nrows = int(np.ceil(len(sessions) / ncols))
    xlim, ylim = limits(trials["x"], trials["y"])

    # A thin dedicated row for the legend. A figure-level legend fights the
    # suptitle above the panels and the footer line below them, and a spare
    # grid cell wastes a whole panel row when the grid comes out full.
    fig = plt.figure(figsize=(3.1 * ncols, 3.1 * nrows + 1.1), layout="constrained")
    gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[*([6.0] * nrows), 1.0])
    axes = np.array(
        [[fig.add_subplot(gs[r, c]) for c in range(ncols)] for r in range(nrows)]
    )
    for ax in axes.ravel()[len(sessions) :]:
        ax.set_visible(False)

    for ax, session in zip(axes.ravel(), sessions):
        style_axes(ax)
        in_session = trials["session"] == session
        for m in MAZES:
            mask = in_session & (trials["maze"] == m)
            if not mask.any():
                continue
            ax.scatter(
                trials["x"][mask],
                trials["y"][mask],
                color=MAZE_COLORS[m],
                label=f"maze {m}",
                zorder=3,
                **DOT,
            )
            _centroid(ax, trials["x"][mask], trials["y"][mask], MAZE_COLORS[m], z=5)
        _origin(ax, xlim, ylim)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_title(f"{session}  (n={int(in_session.sum())})", fontsize=8.5, color=INK)

    xname, yname = _axis_names(trials["space"])
    for col in range(ncols):
        visible = [ax for ax in axes[:, col] if ax.get_visible()]
        if visible:
            visible[-1].set_xlabel(xname, fontsize=8, color=INK_SOFT)
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel(yname, fontsize=8, color=INK_SOFT)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    legend_ax = fig.add_subplot(gs[nrows, :])
    legend_ax.set_axis_off()
    leg = legend_ax.legend(
        handles,
        labels,
        title="maze",
        loc="center",
        ncols=6,
        fontsize=8.5,
        title_fontsize=8.5,
        frameon=False,
        markerscale=3.0,
    )
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)
    leg.get_title().set_color(INK_SOFT)

    extra = "ringed marks are per-maze means"
    if dropped:
        extra += f"; {dropped} further session(s) not shown"
    _finish(
        fig,
        trials,
        title=f"{trials['monkey']} — gaze at fix_start by maze, one panel per session",
        extra=extra,
    )
    save_figure(fig, "session_facets", rel_dir=rel_dir)
    plt.close(fig)


def centroids(trials, *, rel_dir=""):
    """Group means with 95% CIs -- the view the effect sizes are visible in."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0), layout="constrained")
    xname, yname = _axis_names(trials["space"])

    ax = axes[0]
    style_axes(ax)
    path = []
    for m in MAZES:
        mask = trials["maze"] == m
        xy = _centroid(ax, trials["x"][mask], trials["y"][mask], MAZE_COLORS[m])
        path.append(xy)
    plotted_path = [xy for xy in path if xy is not None]
    if len(plotted_path) > 1:
        arr = np.asarray(plotted_path)
        ax.plot(arr[:, 0], arr[:, 1], color=INK_MUTED, linewidth=1.0, zorder=4)
    ax.set_title("maze means, joined 1 → 6", fontsize=10, color=INK)
    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", markersize=7,
            markerfacecolor=MAZE_COLORS[m], markeredgecolor="white", label=f"maze {m}"
        )
        for m in MAZES
    ]

    ax2 = axes[1]
    style_axes(ax2)
    keep = np.isfinite(trials["label"])
    strategy_handles = []
    strategy_path = []
    for v in LABELS:
        mask = keep & (trials["label"] == v)
        strategy_path.append(_centroid(ax2, trials["x"][mask], trials["y"][mask], STRATEGY_COLORS[v]))
        strategy_handles.append(
            plt.Line2D(
                [], [], marker="o", linestyle="", markersize=7,
                markerfacecolor=STRATEGY_COLORS[v], markeredgecolor="white",
                label=f"{STRATEGY_NAMES[v]} (n={int(mask.sum())})",
            )
        )
    ax2.set_title("decoded-strategy means", fontsize=10, color=INK)

    # The direct labels sit outside the marker, so the data box needs slack or
    # the outermost group's name runs off the panel.
    for a in axes:
        a.margins(0.3)
        a.autoscale_view()
        xl, yl = a.get_xlim(), a.get_ylim()
        _origin(a, xl, yl)
        a.set_xlabel(xname, fontsize=9, color=INK_SOFT)
        a.set_ylabel(yname, fontsize=9, color=INK_SOFT)

    # Labels are placed only once the axes are done rescaling for the margins
    # above, since `_label_offsets` reasons in axes-fraction space -- and
    # after the legend corners are known, so a label can steer clear of them.
    legend_corner = {"lower right": (0.82, 0.1), "lower left": (0.18, 0.1)}
    for a, hs, title, loc, anchors, texts, colors in (
        (axes[0], handles, "maze", "lower right", path, [str(m) for m in MAZES], [MAZE_COLORS[m] for m in MAZES]),
        (
            axes[1],
            strategy_handles,
            "decoded strategy",
            "lower left",
            strategy_path,
            [STRATEGY_NAMES[v] for v in LABELS],
            [STRATEGY_COLORS[v] for v in LABELS],
        ),
    ):
        leg = a.legend(
            handles=hs,
            title=title,
            loc=loc,
            ncols=2 if len(hs) > 2 else 1,
            fontsize=8,
            title_fontsize=8,
            frameon=True,
            framealpha=0.92,
        )
        leg.get_frame().set_edgecolor("#e0ded7")
        leg.get_title().set_color(INK_SOFT)
        kept = [(xy, t, c) for xy, t, c in zip(anchors, texts, colors) if xy is not None]
        if not kept:
            continue
        xl, yl = a.get_xlim(), a.get_ylim()
        obstacle = (
            xl[0] + legend_corner[loc][0] * (xl[1] - xl[0]),
            yl[0] + legend_corner[loc][1] * (yl[1] - yl[0]),
        )
        offsets = _label_offsets(a, [xy for xy, _, _ in kept], obstacles=[obstacle])
        for (xy, text, color), offset in zip(kept, offsets):
            _annotate(a, xy, text, offset, color=INK)
    _finish(
        fig,
        trials,
        title=f"{trials['monkey']} — where the mean eye sits at fix_start",
        extra="whiskers are 95% CIs of the mean; note the axis scale vs. the raw cloud",
    )
    save_figure(fig, "centroids", rel_dir=rel_dir)
    plt.close(fig)


def marginals(trials, *, rel_dir=""):
    """One axis per row, one grouping per column: the 1-D view of the same shift."""
    xlim, ylim = limits(trials["x"], trials["y"])
    keep = np.isfinite(trials["label"])
    xname, yname = _axis_names(trials["space"])

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.4), layout="constrained")
    specs = [
        ("maze", [(f"maze {m}", MAZE_COLORS[m], trials["maze"] == m) for m in MAZES]),
        (
            "decoded strategy",
            [
                (STRATEGY_NAMES[v], STRATEGY_COLORS[v], keep & (trials["label"] == v))
                for v in LABELS
            ],
        ),
    ]
    for col, (group_name, groups) in enumerate(specs):
        for row, (values, span, axis_name) in enumerate(
            ((trials["x"], xlim, xname), (trials["y"], ylim, yname))
        ):
            ax = axes[row, col]
            style_axes(ax)
            for name, color, mask in groups:
                _kde_line(ax, values[mask], color, span=span, label=name)
                mean, err = centroid_ci(values[mask])
                if np.isfinite(mean):
                    # Stops short of the top, so the mean rules never run up
                    # through the legend.
                    ax.axvline(
                        mean, ymax=0.7, color=color, linewidth=1.0, alpha=0.8, zorder=2
                    )
            ax.set_xlim(*span)
            ax.set_xlabel(axis_name, fontsize=9, color=INK_SOFT)
            ax.set_ylabel("density", fontsize=9, color=INK_SOFT)
            # Headroom, so the legend never lands on a curve.
            top = ax.get_ylim()[1]
            ax.set_ylim(0, top * 1.32)
            if row == 0:
                ax.set_title(f"by {group_name}", fontsize=10, color=INK)
                ax.legend(fontsize=7.5, frameon=False, ncols=2, loc="upper right")
    _finish(
        fig,
        trials,
        title=f"{trials['monkey']} — fix_start gaze marginals",
        extra="Gaussian KDE per group; vertical rules are group means",
    )
    save_figure(fig, "marginals", rel_dir=rel_dir)
    plt.close(fig)


# --- driver ----------------------------------------------------------------


def resolve_scope(scope, *, verbose=True):
    """`monkey -> sessions` for a classifier scope, reporting what it skipped."""
    by_monkey, missing, dropped = scope_sessions(scope)
    if verbose and missing:
        print(
            f"scope {scope!r}: {len(missing)} session(s) not labelled yet, "
            f"skipped: {', '.join(missing)}"
        )
    if verbose and dropped:
        print(
            f"scope {scope!r}: {len(dropped)} session(s) dropped for anchor-maze "
            f"label agreement below {MIN_LABEL_AGREEMENT}: "
            + ", ".join(f"{s} ({a:.2f})" for s, a in sorted(dropped.items()))
        )
    if not by_monkey:
        raise SystemExit(
            f"scope {scope!r}: no labelled sessions; run the labels stage first"
        )
    return by_monkey


def plot_all(
    monkey="Faure",
    *,
    scope="all",
    space="deg",
    center="none",
    views=VIEWS,
    all_sessions=False,
):
    """Every requested view for one monkey, one scope and one coordinate frame.

    By default the trial pool is the scope's labelled sessions, so the
    maze-only and strategy views are drawn on the *same* trials and a maze
    effect cannot be an artefact of a different session mix. `all_sessions`
    widens the maze-only views to every session the animal has.
    """
    by_monkey = resolve_scope(scope)
    sessions = by_monkey.get(monkey)
    if not sessions:
        raise SystemExit(f"scope {scope!r} has no labelled {monkey} sessions")
    print(f"{monkey} / {scope}: {len(sessions)} session(s) — {', '.join(sessions)}")

    lookup = strategy_label_lookup(sessions)
    trials = collect_fix_start(
        monkey,
        sessions=None if all_sessions else sessions,
        label_lookup=lookup,
        space=space,
    )
    print(f"  {trials['x'].size} trials at fix_start; dropped {trials['dropped']}")
    n_labelled = int(np.isfinite(trials["label"]).sum())
    print(f"  {n_labelled} of them carry a decoded strategy label")
    if center == "session":
        trials = center_by_session(trials)

    rel_dir = output_rel_dir(space, monkey, scope, centered=center == "session")
    save_trials_csv(trials, rel_dir=rel_dir)
    for view in views:
        if view == "pooled_maze":
            pooled(trials, by="maze", rel_dir=rel_dir)
        elif view == "pooled_strategy":
            pooled(trials, by="strategy", rel_dir=rel_dir)
        elif view == "maze_facets":
            maze_facets(trials, rel_dir=rel_dir)
        elif view == "strategy_facets":
            strategy_facets(trials, rel_dir=rel_dir)
        elif view == "maze_x_strategy":
            maze_x_strategy(trials, rel_dir=rel_dir)
        elif view == "session_facets":
            session_facets(trials, rel_dir=rel_dir)
        elif view == "centroids":
            centroids(trials, rel_dir=rel_dir)
        elif view == "marginals":
            marginals(trials, rel_dir=rel_dir)
        else:
            raise ValueError(f"unknown view {view!r}; choose from {VIEWS}")
    return trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monkey", default="Faure", choices=("Faure", "Nielsen"))
    parser.add_argument("--scope", default="all", choices=sorted(SCOPES))
    parser.add_argument(
        "--space", default="deg", choices=(*sorted(SPACES), "both")
    )
    parser.add_argument(
        "--center",
        default="none",
        choices=("none", "session", "both"),
        help="subtract each session's mean gaze before plotting",
    )
    parser.add_argument("--views", nargs="+", default=list(VIEWS), choices=VIEWS)
    parser.add_argument(
        "--all-sessions",
        action="store_true",
        help="use every session, not just the scope's labelled ones",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="both monkeys × both spaces × raw and session-centred",
    )
    args = parser.parse_args()

    monkeys = ("Faure", "Nielsen") if args.all else (args.monkey,)
    spaces = tuple(sorted(SPACES)) if args.all or args.space == "both" else (args.space,)
    centers = (
        ("none", "session")
        if args.all or args.center == "both"
        else (args.center,)
    )
    for monkey in monkeys:
        for space in spaces:
            for center in centers:
                plot_all(
                    monkey,
                    scope=args.scope,
                    space=space,
                    center=center,
                    views=tuple(args.views),
                    all_sessions=args.all_sessions,
                )


if __name__ == "__main__":
    main()
