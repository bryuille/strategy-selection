"""The figure: mean p against codebook size K, one file per (monkey, maze).

Six curves, one per (feature, variant): time occupancy and binary occupancy
crossed with `full`, `no_origin` and `mean_removed`. Feature picks the colour
and variant the line style, so the two distinctions stay separable by eye
without a six-entry rainbow.

The y axis is log-scaled, because the interesting range is the bottom of it --
the difference between p = 0.5 and p = 0.3 is noise, the difference between
p = 0.05 and p = 0.005 is the whole question. The permutation floor is drawn
in: with `n_perm` permutations no single session can report p below
``1 / (n_perm + 1)``, so a mean sitting on that line means "every contributing
session was at the floor", not "p = 0.001 exactly".

The point count per curve is not constant. A (session, maze) cell that fails
coverage contributes no p, and a K where no session clears it has no point at
all -- so a gap in a curve is a coverage gap, not a missing computation. The
per-point session count is annotated on the marker for exactly that reason.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from eye_pre_flash.maze_strategy_pairs.figures import METHOD_LABEL, SOURCE_LABEL

FEATURE_LABEL = {"occupancy": "time occupancy", "occupancy_bin": "binary occupancy"}
VARIANT_LABEL = {
    "full": "full",
    "no_origin": "no origin",
    "mean_removed": "mean removed",
}

# Feature -> colour, variant -> dash pattern. Two hues far apart in both hue
# and lightness, so the pairing survives greyscale printing and the common
# red-green confusions.
FEATURE_COLOUR = {"occupancy": "#1f5fa9", "occupancy_bin": "#c2570a"}
VARIANT_STYLE = {"full": "-", "no_origin": "--", "mean_removed": ":"}
# Marker as well as dash, because curves here routinely coincide: three
# variants of one feature can agree to the permutation floor, and two curves
# drawn on top of each other in the same hue are indistinguishable without it.
VARIANT_MARKER = {"full": "o", "no_origin": "s", "mean_removed": "^"}

ALPHA_LINE = 0.05
# Wide enough to carry the legend outside the axes without squeezing the plot.
FIG_W, FIG_H = 9.0, 5.0


def curve_key(feature, variant):
    return f"{FEATURE_LABEL[feature]} / {VARIANT_LABEL[variant]}"


def p_vs_k_figure(curves, *, title, p_floor=None, annotate_n=True):
    """One figure. `curves` maps (feature, variant) -> list of (k, p, n_sessions).

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
        ns = [p[2] for p in pts]
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
        if annotate_n:
            for k, p, n in zip(ks, ps, ns):
                ax.annotate(
                    str(n),
                    (k, p),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=5.5,
                    color=FEATURE_COLOUR[feature],
                    alpha=0.75,
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
    ax.set_ylabel("mean p over sessions")
    # K is an integer, so the default locator's 4.5 / 7.5 ticks name codebook
    # sizes that were never fitted. Tick on the swept values themselves.
    swept = sorted({k for points in curves.values() for k, _p, _n in points})
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


def suptitle(*, monkey, maze, source, method, n_sessions_scope):
    """Names every axis collapsed into the figure, over three short lines.

    Short lines on purpose: `fig.suptitle` is layout-managed but not wrapped,
    so one long line overflows the canvas width and is clipped at save time.
    """
    return (
        f"{monkey} - maze {maze} - {METHOD_LABEL[method]}\n"
        f"{SOURCE_LABEL[source]} labels; codebook fitted per (session, maze)\n"
        f"p averaged over up to {n_sessions_scope} sessions "
        f"(marker labels = sessions contributing)"
    )
