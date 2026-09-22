"""The figure: p against codebook size K, one file per (monkey, maze).

Same layout as `comp_figures.p_vs_k_figure` -- six curves, one per (feature,
variant), feature picks the colour and variant the line style -- but the
number behind each point is different. There, a K's point is the mean of
however many (session, maze) cells cleared coverage, so the count of
contributing sessions varies point to point and is worth annotating. Here,
coverage is decided once per (monkey, maze) by the pooled label mask and does
not depend on K at all (only whether the *codebook* fits does, and that
almost never fails for maze 2-5 pooled across ten sessions) -- so a
"sessions contributing" annotation would be the same constant at every point
and say nothing, and none is drawn.

The permutation floor and the log y-axis are unchanged from `comp_figures`
for the same reasons given there.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from eye_pre_flash.maze_strategy_pairs.comp_figures import (
    FEATURE_COLOUR,
    VARIANT_MARKER,
    VARIANT_STYLE,
    curve_key,
)
from eye_pre_flash.maze_strategy_pairs.figures import METHOD_LABEL, SOURCE_LABEL

ALPHA_LINE = 0.05
FIG_W, FIG_H = 9.0, 5.0


def p_vs_k_figure(curves, *, title, p_floor=None):
    """One figure. `curves` maps (feature, variant) -> list of (k, p, z).

    Points arrive unsorted and are sorted here, so a caller may build them in
    whatever order its sweep happens to run.
    """
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), layout="constrained")

    for (feature, variant), points in sorted(curves.items()):
        if not points:
            continue
        pts = sorted(points)
        ks = [p[0] for p in pts]
        ps = [p[1] for p in pts]
        ax.plot(
            ks,
            ps,
            VARIANT_STYLE[variant],
            color=FEATURE_COLOUR[feature],
            marker=VARIANT_MARKER[variant],
            markersize=4.0,
            markerfacecolor="none",
            linewidth=1.4,
            alpha=0.85,
            label=curve_key(feature, variant),
        )

    ax.axhline(ALPHA_LINE, color="0.35", linewidth=0.9, zorder=0)
    ax.annotate(
        "p = 0.05",
        (1.0, ALPHA_LINE),
        xycoords=("axes fraction", "data"),
        textcoords="offset points",
        xytext=(-3, 3),
        ha="right",
        fontsize=7,
        color="0.35",
    )
    if p_floor:
        ax.axhline(p_floor, color="0.7", linewidth=0.9, linestyle=":", zorder=0)
        ax.annotate(
            f"floor {p_floor:.3g}",
            (1.0, p_floor),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
            xytext=(-3, 3),
            ha="right",
            fontsize=7,
            color="0.7",
        )

    ax.set_yscale("log")
    ax.set_xlabel("codebook size K")
    ax.set_ylabel("p (pooled estimate)")
    # K is an integer, so the default locator's 4.5 / 7.5 ticks name codebook
    # sizes that were never fitted. Tick on the swept values themselves.
    swept = sorted({k for points in curves.values() for k, _p, _z in points})
    if swept:
        ax.set_xticks(swept)
        ax.set_xticklabels([str(k) for k in swept], fontsize=8)
    # Centred on the figure rather than the axes: the legend sits outside on
    # the right, so an axes-centred title is off-centre on the canvas and a
    # long one overflows the left edge and is silently clipped at save time.
    fig.suptitle(title, fontsize=9.5)
    ax.grid(True, which="major", axis="both", alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(
        fontsize=7.5,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )
    return fig


def suptitle(*, monkey, maze, source, method, n_H, n_S, n_sessions):
    """Names every axis collapsed into the figure, over three short lines.

    Short lines on purpose: `fig.suptitle` is layout-managed but not wrapped,
    so one long line overflows the canvas width and is clipped at save time.
    """
    return (
        f"{monkey} - maze {maze} - {METHOD_LABEL[method]}\n"
        f"{SOURCE_LABEL[source]} labels; codebook fitted per (monkey, maze)\n"
        f"pooled across {n_sessions} sessions; n_H={n_H} n_S={n_S}"
    )
