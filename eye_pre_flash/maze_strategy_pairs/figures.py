"""The merged-K figure: one 2x2 panel per K, side by side, one file.

Layout rests on one verified property of `layout="constrained"`: it reserves
figure-top margin from ``fig._suptitle.get_tightbbox()`` and per-axes margins
from ``ax.get_tightbbox()``, both of which measure *rendered multi-line* text.
So vertical overlap across suptitle / axes title / xlabel / supxlabel is
structurally impossible and **no manual ``y=`` or ``top=`` tuning belongs
here** -- adding it would fight the engine and reintroduce the collisions it
prevents. Two corollaries shape the code below:

* ``fig.text(...)`` is **not** layout-managed (only `_suptitle`, `_supxlabel`
  and `_supylabel` are consulted), so no header line may use it.
* Horizontal overflow is *not* handled -- a suptitle wider than the canvas is
  silently clipped at save time. Line lengths are budgeted against the 11.5 in
  two-panel width in `pair_suptitle`.

Per-panel statistics go in `ax.set_xlabel`, below the matrix, rather than in
the axes title. Stacking a 2-line suptitle over a 5-line axes title would
crowd every line into the top inch and squeeze the square panels; splitting
the text budget between top and bottom halves the crowding at each end. It is
also collision-proof against the in-axes cell labels, which are drawn at data
``y = i + 0.28`` inside ``ylim = (1.5, -0.5)``, while ticks and xlabel occupy
separate bands outside the spine.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from eye_pre_flash.plotting.similarities.common import BLUE_YELLOW

FEATURE_LABEL = {
    "occupancy": "occupancy",
    "occupancy_bin": "binary occupancy",
    "bigram": "state-bigram",
}
SOURCE_LABEL = {"dendro": "dendrogram", "svm": "SVM"}

# `.with_extremes` COPIES. Never call `BLUE_YELLOW.set_under` / `set_bad`:
# that colormap is a module-level object shared by every similarity figure in
# the repo, and mutating it would leak grey cells into `heatmap*.py`,
# `occupancy.py` and `transition.py`.
#
# `under` matters because the colour scale is fixed at 0-1 for every variant,
# while `mean_removed` makes negative correlations routine (its grand-mean
# subtraction leaves residuals that sum to ~zero across cells, so the average
# off-diagonal is pushed negative arithmetically). A negative cell would
# otherwise clamp to the darkest parula blue, indistinguishable from r = 0.01.
# Grey says "off scale"; the printed number still says what it is.
CMAP = BLUE_YELLOW.with_extremes(under="#6b6b6b", bad="0.92")

VMIN, VMAX = 0.0, 1.0
PANEL_W, FIG_H = 5.75, 6.8


def fmt(value, digits=3):
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def p_text(null):
    """`p`, with a tilde marking the permutation floor.

    ``p`` cannot go below ``1 / (n_perm + 1)``, so at the floor the printed
    value is an upper bound on the true p rather than an estimate of it -- a
    statement about the test's resolution. `z` is the number that says by how
    much.
    """
    p = null.get("p_two_sided", np.nan) if null else np.nan
    txt = fmt(p, 4)
    n_perm = (null or {}).get("n_perm")
    if np.isfinite(p) and n_perm and abs(p - 1.0 / (n_perm + 1)) < 1e-12:
        return f"~{txt}"
    return txt


def draw_cell_matrix(ax, mat, *, names, census, cmap=CMAP):
    """The 2x2 heatmap with its four numbers and the per-cell trial counts.

    Four *distinct* numbers: the off-diagonal is not symmetrised, so
    ``[H,S]`` and ``[S,H]`` differ. How close they sit is the QC read -- they
    are two exchangeable draws of one quantity, so a gap means the estimate is
    noisy at this `m`, not that the strategies are asymmetric.
    """
    im = ax.imshow(
        mat, origin="upper", cmap=cmap, vmin=VMIN, vmax=VMAX, aspect="equal"
    )
    ax.set_xticks(range(len(names)), labels=names, fontsize=13)
    ax.set_yticks(range(len(names)), labels=names, fontsize=13)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isfinite(val):
                ax.text(j, i, "—", ha="center", va="center", color="0.4", fontsize=13)
                continue
            if val < VMIN:
                colour = "white"  # the flat grey under-range swatch
            else:
                r, g, b, _ = im.cmap(im.norm(val))
                colour = (
                    "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "white"
                )
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=colour, fontsize=15)
            if i == j and census is not None:
                ax.text(
                    j, i + 0.28, f"n = {int(census[i])}",
                    ha="center", va="center", color=colour, fontsize=10,
                )
    return im


def attach_colorbar(fig, ax, im, *, label="r"):
    cbar = fig.colorbar(
        im, ax=ax, fraction=0.05, pad=0.03, shrink=0.78, aspect=18
    )
    cbar.set_ticks([VMIN, 0.5 * (VMIN + VMAX), VMAX])
    cbar.ax.set_yticklabels([f"{t:.1f}" for t in (VMIN, 0.5 * (VMIN + VMAX), VMAX)], fontsize=8)
    cbar.set_label(label, fontsize=9)
    return cbar


def panel_stats(panel):
    """The four-line block under one panel.

    Always four lines, including for a blanked panel: constrained layout sizes
    each axes from its own tightbbox, so a placeholder with a different line
    count would make the two square panels come out different heights.
    """
    null = panel.null or {}
    if panel.result is None or not panel.result.n_sessions:
        reason = panel.reason or "no usable session"
        return (
            "z = —    Δ = —    p = —\n"
            "beyond floor —    r_HH−r_SS = —\n"
            f"{reason}\n"
            f"n = 0 of {panel.n_in_scope} sessions    m = —    d = {panel.d}"
        )
    return (
        f"z = {fmt(null.get('z_perm'), 2)}    "
        f"Δ = {fmt(null.get('delta_obs'))}    "
        f"p = {p_text(null)}\n"
        f"beyond floor {fmt(null.get('delta_beyond_floor'))}    "
        f"r_HH−r_SS = {fmt(null.get('diag_imbalance'), 3)}\n"
        f"Δ_time: early/late {fmt(panel.delta_time_el)}, "
        f"odd/even {fmt(panel.delta_time_oe)}    "
        f"|HS−SH| = {fmt(null.get('cross_gap'))}\n"
        f"n = {panel.result.n_sessions} of {panel.n_in_scope} sessions    "
        f"m = {null.get('half_size_median', '—')}    d = {panel.d}    "
        f"r_pb = {fmt(panel.r_pb, 2)}"
    )


def pair_suptitle(*, monkey, maze, feature, variant, source, scope):
    """Two lines, one `Text`, so constrained layout can measure it.

    Width budget at 13 pt (~0.6 em average advance, ~7.8 pt/char): the worst
    case is line 2 at 67 characters, about 7.3 in against an 11.5 in canvas.
    Comfortable, so no wrapping and no ``bbox_inches="tight"`` -- every PNG in
    the deck stays the same pixel size.

    One `Text` means one fontsize, so the hierarchy comes from content order
    rather than type size. That is the overlap-safe trade.
    """
    return (
        f"{monkey} — maze {maze} — {FEATURE_LABEL.get(feature, feature)}\n"
        f"variant = {variant}    ·    "
        f"{SOURCE_LABEL.get(source, source)} labels, {scope} scope"
    )


def pair_footer(*, n_splits, n_perm, min_trials, min_half, seed, detrend=0):
    detrend_txt = f"detrended (order {detrend})   ·   " if detrend else ""
    return (
        f"{n_splits} splits (Fisher-z) × {n_perm} permutations   ·   "
        f"min_trials {min_trials}, min_half {min_half}   ·   {detrend_txt}"
        f"colour scale fixed 0–1   ·   off-diagonal NOT symmetrised   ·   seed {seed}"
    )


def pair_figure(panels, *, suptitle, footer, names, cmap=CMAP):
    """One figure, one 2x2 panel per K.

    `panels` is a list of objects carrying `k`, `d`, `result`, `null`,
    `n_in_scope`, `reason`, `delta_time_el`, `delta_time_oe`, `r_pb` (see
    `build.Panel`). A panel with no usable session is blanked rather than
    dropped, so a K that lost its coverage is visible instead of silently
    absent -- but the colorbar is still created for it, since otherwise the
    two axes come out different widths.
    """
    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(PANEL_W * len(panels), FIG_H),
        layout="constrained",
    )
    axes = np.atleast_1d(axes)
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.06, wspace=0.07)

    for ax, panel in zip(axes, panels):
        usable = panel.result is not None and panel.result.n_sessions
        if usable:
            im = draw_cell_matrix(
                ax, panel.result.mean, names=names,
                census=panel.result.census, cmap=cmap,
            )
            attach_colorbar(fig, ax, im)
        else:
            im = draw_cell_matrix(
                ax, np.full((2, 2), np.nan), names=names, census=None, cmap=cmap
            )
            ax.text(
                0.5, 0.5, panel.reason or "no usable session",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="0.35",
            )
            cbar = attach_colorbar(fig, ax, im, label="r (n/a)")
            cbar.ax.set_yticklabels([])
        ax.set_title(f"K = {panel.k}", fontsize=12, fontweight="bold", pad=6)
        ax.set_xlabel(panel_stats(panel), fontsize=9, linespacing=1.3, labelpad=8)

    fig.suptitle(suptitle, fontsize=13)
    fig.supxlabel(footer, fontsize=8, color="0.35")
    return fig
