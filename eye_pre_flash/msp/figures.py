"""The 2x2 panel figure, the codebook diagnostic figure, and `save_figure`.

One 2x2 per (monkey, variant, maze). Each carries the four correlations and,
above them, the three numbers worth reading: `delta`, `z`, `p`. How to
interpret them lives in `msp.md`, not on the figure.

`save_figure` is copied from `eye_pre_flash/plotting/plot_io.py`; the panel
drawing from `eye_pre_flash/maze_strategy_pairs/figures.py`, reduced to one
panel because K is fixed.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle

from eye_pre_flash.msp.colormap import BLUE_YELLOW

STRATEGY_COLOUR = {"H": "#1f77b4", "S": "#d62728"}
DENSITY_BINS = 60

# `.with_extremes` COPIES; never `set_bad` on `BLUE_YELLOW` itself.
CMAP = BLUE_YELLOW.with_extremes(bad="0.92")

FIG_W, FIG_H = 7.2, 5.2
CBAR_LABEL = "r between group means"


def save_figure(
    fig, stem, *, out_root, rel_dir=None, dpi=200, bbox_inches=None, pad_inches=0.02
):
    """Save `fig` under `out_root/[rel_dir/]<stem>.png`."""
    out_dir = out_root / rel_dir if rel_dir else out_root
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    print(f"Saved {path}")
    return path


def colour_limits(q, *, vmax=None):
    """``(vmin, vmax, cmap)``: the panel's own min-to-max unless `vmax` fixes 0-to-vmax.

    Colour is therefore a within-figure read only; judge strength by `z`.
    """
    if vmax is not None:
        return 0.0, float(vmax), CMAP
    values = np.ravel(q)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 0.0, 1.0, CMAP
    low, high = float(finite.min()), float(finite.max())
    if high - low < 1e-9:
        pad = max(abs(high), 1e-3) * 0.05
        low, high = low - pad, high + pad
    return low, high, CMAP


def draw_matrix(ax, q, *, names, im_kw, counts=None):
    """The 2x2 heatmap with its four numbers and the pooled counts on the diagonal."""
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
    p = f"{result.p:.4f}"
    if result.p_at_floor:
        p = f"~{p}"  # at the test's resolution: "at least this significant"
    z = "—" if not np.isfinite(result.z) else f"{result.z:+.2f}"
    return (
        f"K = {k}  (fixed: origin + 4 exits)\n"
        f"Δ = {result.delta:+.3f}    z = {z}    p = {p}"
    )


def suptitle(*, monkey, maze, variant):
    return (
        f"{monkey} — maze {maze} — binary occupancy\n"
        f"{variant}  ·  SVM labels  ·  similarity of group means"
    )


def panel_figure(result, *, k, title, names, vmax=None):
    """One figure, one 2x2 panel."""
    vmin, vmax_eff, cmap = colour_limits(result.q, vmax=vmax)
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.06)
    im = draw_matrix(
        ax, result.q, names=names,
        im_kw=dict(cmap=cmap, vmin=vmin, vmax=vmax_eff),
        counts=(result.n_H, result.n_S),
    )
    ax.set_title(panel_title(result, k), fontsize=11, pad=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.62)
    cbar.set_label(CBAR_LABEL, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=12)
    return fig


def _draw_balls(ax, codebook, radius, lim, *, colours=None, lw=1.3):
    """The five prototypes as crosses with their assignment circles."""
    for s, (x, y) in enumerate(np.asarray(codebook, dtype=float)):
        ec = colours(s) if colours is not None else "black"
        ax.add_patch(Circle((x, y), radius, fill=False, ec=ec, lw=lw, zorder=3))
        ax.plot(x, y, "x", color="black", ms=7, mew=1.4, zorder=4)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("unit-H x")
    ax.set_ylabel("unit-H y")


def codebook_figure(fix_xy, fix_state, *, codebook, names, radius, lim, title):
    """The five prototypes and their assignment balls over every fixation
    centroid, coloured by the state each was assigned to (grey =
    unassigned). One dot is one I-DT fixation, drawn at the mean warped
    position of its valid in-window samples -- the point the assignment
    used. The legend gives each state's share of all fixations, which is the
    coverage number `msp.md` warns about."""
    fix_xy = np.asarray(fix_xy, dtype=float)
    fix_state = np.asarray(fix_state, dtype=int)
    n = max(fix_state.size, 1)
    colours = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(6.6, 6.6), layout="constrained")
    ax.axhline(0, color="0.88", lw=0.6, zorder=0)
    ax.axvline(0, color="0.88", lw=0.6, zorder=0)

    un = fix_state < 0
    ax.scatter(
        fix_xy[un, 0], fix_xy[un, 1], s=2, c="0.72", alpha=0.35, linewidths=0,
        label=f"unassigned  {un.sum() / n:.1%}", rasterized=True,
    )
    for s, name in enumerate(names):
        hit = fix_state == s
        ax.scatter(
            fix_xy[hit, 0], fix_xy[hit, 1], s=2, color=colours(s), alpha=0.45,
            linewidths=0, label=f"{name}  {hit.sum() / n:.1%}", rasterized=True,
        )
    _draw_balls(ax, codebook, radius, lim, colours=colours)
    ax.legend(
        loc="upper right", fontsize=8, markerscale=5, framealpha=0.92,
        title=f"share of {fix_state.size} fixations\n(one dot = one fixation's centroid)",
        title_fontsize=8,
    )
    ax.set_title(title, fontsize=11)
    return fig


def _state_shares(fix_state, k):
    """Share of fixations per state, unassigned last; sums to 1 (or NaN if none)."""
    fix_state = np.asarray(fix_state, dtype=int)
    n = fix_state.size
    if n == 0:
        return np.full(k + 1, np.nan)
    counts = np.r_[np.bincount(fix_state[fix_state >= 0], minlength=k)[:k], (fix_state < 0).sum()]
    return counts / n


def strategy_figure(
    fix, occ, *, codebook, names, radius, lim, title
):
    """How fixation concentration at each fixed state differs between the
    two strategies of one maze.

    `fix` is ``{"H": (xy, state), "S": (xy, state)}``: every fixation centroid
    of that strategy's pooled trials, with its assigned state. `occ` is
    ``{"H": occ_bin rows, "S": occ_bin rows}`` for the same trials.

    **Everything is normalised within strategy**, because H and S trial
    counts differ by up to an order of magnitude in some mazes. The two
    density panels show, per spatial bin, the *fraction of that strategy's
    fixations* (each panel sums to 1), on one shared colour scale. The bar
    panels show each state's share of that strategy's fixations, and the
    fraction of that strategy's trials that visit the state at least once.
    Trial and fixation counts are printed so the imbalance is visible.
    """
    k = len(names)
    colours = plt.get_cmap("tab10")
    edges = np.linspace(-lim, lim, DENSITY_BINS + 1)

    dens = {}
    for tag in ("H", "S"):
        xy, _state = fix[tag]
        xy = np.asarray(xy, dtype=float)
        h, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[edges, edges])
        dens[tag] = h / max(xy.shape[0], 1)
    vmax = max(float(np.nanmax(d)) if d.size else 0.0 for d in dens.values()) or 1.0
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 11.0), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.08, h_pad=0.08)

    for ax, tag in zip(axes[0], ("H", "S")):
        xy, state = fix[tag]
        n_trials = int(np.asarray(occ[tag]).shape[0])
        im = ax.pcolormesh(
            edges, edges, dens[tag].T, cmap="Blues", norm=norm, rasterized=True
        )
        _draw_balls(ax, codebook, radius, lim, colours=colours)
        ax.set_title(
            f"{tag}: {n_trials} trials, {len(state)} fixations "
            f"({len(state) / max(n_trials, 1):.2f} per trial)",
            fontsize=11, color=STRATEGY_COLOUR[tag],
        )
    cbar = fig.colorbar(im, ax=list(axes[0]), fraction=0.03, pad=0.02, shrink=0.8)
    cbar.set_label("fraction of that strategy's fixations per bin", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Bars: state share of fixations, and per-trial visit rate.
    labels = list(names) + ["unassigned"]
    x = np.arange(len(labels))
    w = 0.38
    share = {tag: _state_shares(fix[tag][1], k) for tag in ("H", "S")}
    visit = {}
    for tag in ("H", "S"):
        rows = np.asarray(occ[tag], dtype=float)
        v = rows.mean(axis=0) if rows.shape[0] else np.full(k, np.nan)
        visit[tag] = np.r_[v, np.nan]  # no per-trial notion of "unassigned"

    for ax, data, ylabel, ttl in (
        (axes[1][0], share, "share of that strategy's fixations",
         "where each strategy's fixations land"),
        (axes[1][1], visit, "fraction of that strategy's trials visiting the state",
         "how often a trial visits each state"),
    ):
        for i, tag in enumerate(("H", "S")):
            vals = data[tag]
            bars = ax.bar(
                x + (i - 0.5) * w, np.nan_to_num(vals), w,
                color=STRATEGY_COLOUR[tag], alpha=0.85, label=tag,
            )
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(
                        b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8,
                    )
        diff = data["S"] - data["H"]
        tick_labels = [
            f"{lab}\nS−H {dv:+.2f}" if np.isfinite(dv) else lab
            for lab, dv in zip(labels, diff)
        ]
        ax.set_xticks(x, labels=tick_labels, fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ttl, fontsize=11)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True)

    fig.suptitle(title, fontsize=13)
    return fig
