"""The figure: one 2x2 panel per K, side by side, one file.

Deliberately sparse. The panel carries the four correlations, and above each
one the three numbers worth reading: `delta`, `z`, `p`. Everything a reader
needs in order to *interpret* those numbers lives in `CAVEATS.md`, not on the
figure -- a caption that has to be read before the picture can be trusted is
a caption nobody reads.

Layout rests on one property of ``layout="constrained"``: it reserves figure
margin from ``fig._suptitle.get_tightbbox()`` and per-axes margin from
``ax.get_tightbbox()``, both measuring *rendered multi-line* text. Vertical
collisions between the suptitle and the axes titles are therefore structurally
impossible, and no manual ``y=`` or ``top=`` tuning belongs here. Note
``fig.text(...)`` is **not** layout-managed, so no header may use it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from eye_pre_flash.plotting.similarities.common import BLUE_YELLOW

FEATURE_LABEL = {"occupancy": "occupancy", "occupancy_bin": "binary occupancy"}
SOURCE_LABEL = {"dendro": "dendrogram", "svm": "SVM"}
# Named on the figure, not just in the path: the two arms produce very
# different numbers from the same trials, so a PNG that does not say which
# one it is invites exactly the wrong comparison.
METHOD_LABEL = {
    "trial_by_trial": "trial-by-trial similarity",
    "block_means": "similarity of group means",
}
# What one cell actually is, which differs between the arms -- a mean over
# m*m trial-pair correlations, versus one correlation between two group means.
CBAR_LABEL = {
    "trial_by_trial": "mean trial-pair r",
    "block_means": "r between group means",
}

# `.with_extremes` COPIES. Never call `set_bad` on `BLUE_YELLOW` itself: it is
# a module-level object shared with every other similarity figure in the repo.
CMAP = BLUE_YELLOW.with_extremes(bad="0.92")

PANEL_W, FIG_H = 4.6, 5.2
# Constrained layout reserves vertical space for the suptitle but does NOT
# handle horizontal overflow -- a title wider than the canvas is silently
# clipped at save time. A one-panel figure is only PANEL_W wide, so it needs
# a floor independent of the panel count.
MIN_FIG_W = 7.2


def colour_limits(panels, *, vmax=None):
    """``(vmin, vmax, cmap)`` shared by every panel in one figure.

    Default spans the figure's own min-to-max. That is a deliberate change
    from the fixed 0-1 scale the mean-vs-mean estimator used, and the reason
    is that trial-pair correlations occupy a narrow band -- a real panel might
    run 0.28 to 0.37 -- so any wider fixed range renders all four cells as one
    flat colour and hides the diagonal-vs-off-diagonal gap the figure exists
    to show.

    The cost is that colour is a **within-figure** read only: a panel of pure
    noise gets the same visual contrast as a strong one, because the range
    adapts. That is what `delta`, `z` and `p` above the panel are for, and
    `CAVEATS.md` says so. The two K panels share one range so they stay
    comparable to each other.

    `vmax` forces a fixed 0-to-`vmax` scale instead, for when panels do need
    to be compared by eye across figures.
    """
    if vmax is not None:
        return 0.0, float(vmax), CMAP
    values = np.concatenate([np.ravel(p.result.q) for p in panels])
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 0.0, 1.0, CMAP
    low, high = float(finite.min()), float(finite.max())
    if high - low < 1e-9:  # four identical cells must not collapse the norm
        pad = max(abs(high), 1e-3) * 0.05
        low, high = low - pad, high + pad
    return low, high, CMAP


def draw_matrix(ax, q, *, names, im_kw, counts=None):
    """The 2x2 heatmap with its four numbers, and the pooled counts on the diagonal.

    `counts` is ``(n_H, n_S)``: how many trials of that strategy the groups
    were drawn from, pooled over every session. No trial is excluded, so this
    is the same for both K and every variant. It is not the number behind any
    one correlation -- each cell averages up to ``m x m`` pairs from groups of
    `m = min(n_H, n_S) // 2` -- so it says how much data stands behind the
    panel, not how much precision the cell has. `m` is in `results.csv`.
    """
    im = ax.imshow(q, origin="upper", aspect="equal", **im_kw)
    ax.set_xticks(range(len(names)), labels=names, fontsize=13)
    ax.set_yticks(range(len(names)), labels=names, fontsize=13)
    ax.tick_params(length=0)

    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            value = q[i, j]
            if not np.isfinite(value):
                ax.text(j, i, "—", ha="center", va="center", color="0.4", fontsize=14)
                continue
            r, g, b, _ = im.cmap(im.norm(value))
            colour = "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "white"
            # Nudged up when a count sits under it, so the pair reads as one
            # block centred in the cell rather than the value looking off-centre.
            offset = -0.10 if (i == j and counts is not None) else 0.0
            ax.text(
                j, i + offset, f"{value:.3f}",
                ha="center", va="center", color=colour, fontsize=16,
            )
            if i == j and counts is not None:
                ax.text(
                    j, i + 0.16, f"n = {int(counts[i])}",
                    ha="center", va="center", color=colour, fontsize=11,
                )
    return im


def panel_title(result, k):
    """``K = 6`` over the three numbers. Nothing else goes above the matrix."""
    p = f"{result.p:.4f}"
    if result.p_at_floor:
        # `p` cannot go below 1/(n_perm+1). At the floor the printed value is
        # the test's resolution, not a measurement -- "at least this
        # significant". `z` says by how much.
        p = f"~{p}"
    z = "—" if not np.isfinite(result.z) else f"{result.z:+.2f}"
    return f"K = {k}\nΔ = {result.delta:+.3f}    z = {z}    p = {p}"


def suptitle(*, monkey, maze, feature, variant, source, method):
    """Two lines, one `Text`, so a PNG pulled into a deck is not anonymous.

    Two lines rather than one because the single-panel case (see
    `pair_figure`) is only `PANEL_W` wide, and one line of this content
    overflows it and gets clipped. Split, the longest line is about 30
    characters and fits either width.
    """
    return (
        f"{monkey} — maze {maze} — {FEATURE_LABEL.get(feature, feature)}\n"
        f"{variant}  ·  {SOURCE_LABEL.get(source, source)} labels  ·  "
        f"{METHOD_LABEL.get(method, method)}"
    )


def pair_figure(panels, *, title, names, method, vmax=None):
    """One figure, one 2x2 panel per K.

    `panels` is a list of objects carrying `k` and `result` (an
    `estimator.Result`); `build` passes only the panels that have one. Since
    no trial is excluded, coverage no longer depends on K or variant, so in
    practice that is either both panels or no figure at all. The single-panel
    path is kept anyway: it costs nothing and a blanked panel would be worse.
    """
    vmin, vmax_eff, cmap = colour_limits(panels, vmax=vmax)
    im_kw = dict(cmap=cmap, vmin=vmin, vmax=vmax_eff)

    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(max(PANEL_W * len(panels), MIN_FIG_W), FIG_H),
        layout="constrained",
    )
    axes = np.atleast_1d(axes)
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.06, wspace=0.06)

    for ax, panel in zip(axes, panels):
        im = draw_matrix(
            ax, panel.result.q, names=names, im_kw=im_kw,
            counts=(panel.result.n_H, panel.result.n_S),
        )
        ax.set_title(panel_title(panel.result, panel.k), fontsize=11, pad=8)

    cbar = fig.colorbar(im, ax=list(axes), fraction=0.04, pad=0.02, shrink=0.62)
    cbar.set_label(CBAR_LABEL.get(method, "r"), fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=12)
    return fig
