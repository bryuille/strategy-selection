"""The selection figure: held-out EV and partition stability against K.

Two stacked panels sharing the K axis, because the decision rests on both and
reading them apart invites the wrong conclusion.

**Top -- held-out explained variance.** One line per maze, shaded by the
across-split SD. The kneedle pick is a filled marker; the marginal-gain and
max-curvature picks are small ticks on the same line, so rule-dependence is
visible rather than buried in a CSV column. A dashed segment means the curve
was non-monotonic there and the running maximum kneedle actually reads differs
from the raw mean.

**Bottom -- partition stability.** Mean pairwise adjusted Rand index between
the assignments produced by each split's codebook. This is the panel that says
whether the EV elbow means anything: `occ_ms` depends on the codebook through a
categorical assignment, so an EV knee sitting where stability has already
collapsed is a knee in a quantity the features do not use. Stability is
expected to fall as K rises; what matters is whether it is still respectable at
the selected K.

The y axis is linear in both panels. Unlike `comp_figures`' p axis there is no
floor and no orders-of-magnitude range -- EV runs 0 to 1 by construction and
ARI over roughly the same span -- so a log scale would only compress the part
being read.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from eye_pre_flash.maze_strategy_pairs.figures import METHOD_LABEL, SOURCE_LABEL

# One hue per maze, ordered so adjacent mazes are far apart in lightness as
# well as hue -- four curves that routinely overlap need to survive greyscale.
MAZE_COLOUR = {2: "#1f5fa9", 3: "#c2570a", 4: "#2e7d32", 5: "#7b1fa2", 6: "#455a64", 1: "#b71c1c"}

RULE_MARKER = {"k_marginal_gain": "|", "k_max_curvature": "_"}

FIG_W, FIG_H = 9.0, 6.4


def _maze_colour(maze):
    return MAZE_COLOUR.get(int(maze), "0.4")


def ev_figure(curves, selections, *, title, stability=None):
    """One figure. `curves` maps maze -> the dict `elbow_select.ev_curve` returns.

    `selections` maps maze -> the dict `elbow_select.select_k` returns.
    `stability` maps maze -> the array `elbow_select.stability_curve` returns,
    or None to draw the EV panel alone.
    """
    n_rows = 2 if stability else 1
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(FIG_W, FIG_H if n_rows == 2 else FIG_H * 0.62),
        layout="constrained",
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    ax_ev = axes[0]

    all_ks = set()
    for maze in sorted(curves):
        curve = curves[maze]
        if curve is None:
            continue
        ks = np.asarray(curve["ks"], dtype=int)
        ev = np.asarray(curve["ev"], dtype=float)
        sd = np.asarray(curve["ev_sd"], dtype=float)
        colour = _maze_colour(maze)
        all_ks.update(ks.tolist())

        ax_ev.fill_between(ks, ev - sd, ev + sd, color=colour, alpha=0.13, linewidth=0)
        ax_ev.plot(
            ks, ev, "-", color=colour, linewidth=1.5, alpha=0.9,
            label=f"maze {maze}",
        )

        sel = selections.get(maze) or {}
        knee = sel.get("k")
        if knee is not None and knee in ks:
            j = int(np.flatnonzero(ks == knee)[0])
            ax_ev.plot(
                knee, ev[j], "o", color=colour, markersize=7.0,
                markeredgecolor="white", markeredgewidth=0.9, zorder=5,
            )
            ax_ev.annotate(
                f"K={knee}", (knee, ev[j]),
                textcoords="offset points", xytext=(0, -13),
                ha="center", fontsize=7.5, color=colour, fontweight="bold",
            )
        for rule, marker in RULE_MARKER.items():
            other = sel.get(rule)
            if other is None or other not in ks:
                continue
            j = int(np.flatnonzero(ks == other)[0])
            ax_ev.plot(
                other, ev[j], marker, color=colour, markersize=9.0,
                alpha=0.75, zorder=4,
            )

    ax_ev.set_ylabel("held-out explained variance")
    ax_ev.grid(True, alpha=0.18, linewidth=0.6)
    ax_ev.set_axisbelow(True)
    for side in ("top", "right"):
        ax_ev.spines[side].set_visible(False)
    ax_ev.legend(
        fontsize=7.5, frameon=False, loc="upper left",
        bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
    )

    if stability:
        ax_st = axes[1]
        for maze in sorted(stability):
            values = stability[maze]
            curve = curves.get(maze)
            if values is None or curve is None:
                continue
            ks = np.asarray(curve["ks"], dtype=int)
            ax_st.plot(
                ks, np.asarray(values, dtype=float), "-",
                color=_maze_colour(maze), linewidth=1.4, alpha=0.9,
            )
            sel = selections.get(maze) or {}
            knee = sel.get("k")
            if knee is not None and knee in ks:
                j = int(np.flatnonzero(ks == knee)[0])
                ax_st.plot(
                    knee, float(np.asarray(values, dtype=float)[j]), "o",
                    color=_maze_colour(maze), markersize=6.0,
                    markeredgecolor="white", markeredgewidth=0.9, zorder=5,
                )
        ax_st.set_ylabel("partition stability (ARI)")
        ax_st.grid(True, alpha=0.18, linewidth=0.6)
        ax_st.set_axisbelow(True)
        for side in ("top", "right"):
            ax_st.spines[side].set_visible(False)

    axes[-1].set_xlabel("codebook size K")
    if all_ks:
        ticks = sorted(all_ks)
        axes[-1].set_xticks(ticks)
        axes[-1].set_xticklabels([str(k) for k in ticks], fontsize=8)

    fig.suptitle(title, fontsize=9.5)
    return fig


def ev_suptitle(*, monkey, source, n_sessions, n_splits, marginal_gain):
    """Names every choice the curve depends on, over three short lines.

    `fig.suptitle` is layout-managed but not wrapped, so a single long line
    overflows the canvas and is clipped at save time.
    """
    return (
        f"{monkey} - held-out EV against codebook size K\n"
        f"{SOURCE_LABEL.get(source, source)} scope, {n_sessions} sessions; "
        f"codebook fitted per maze on in-scope sessions only\n"
        f"{n_splits} random 80/20 trial splits; filled marker = kneedle pick, "
        f"| = gain < {marginal_gain:g}, _ = max curvature"
    )


def selected_suptitle(*, monkey, maze, feature, variant, source, method, k):
    """The 2x2 panel's title, naming the selected K and how it was chosen."""
    return (
        f"{monkey} - maze {maze} - K = {k} (held-out EV elbow)\n"
        f"{variant}  ·  {SOURCE_LABEL.get(source, source)} labels  ·  "
        f"{METHOD_LABEL.get(method, method)}\n"
        f"{feature}; K selected label-blind before this test"
    )
