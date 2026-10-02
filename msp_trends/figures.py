"""Per-maze 2x2 / profile / codebook figures, and the one-per-monkey paper figure.

The 2x2 is unsigned: it says H and S gaze differ, not which way. Every
figure that shows a 2x2 therefore pairs it with the H vs S profile over the
merged states (`draw_profile`), which shows the direction.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from msp import figures as msp_fig
from msp_trends.estimator import low_n
from msp_trends.config import (
    BLOCK_LABEL,
    BLOCK_YLABEL,
    CMAP,
    LOW_N_MARK,
    LOW_N_NOTE,
    EXIT_HALF_WIDTH,
    PERIMETER_RADIUS,
    STATE_COLOUR,
    STEM_HALF_WIDTH,
    STEM_TOP,
    STRATEGY_COLOUR,
)

save_figure = msp_fig.save_figure
CBAR_LABEL = "r between group means"
MATRIX_NAMES = ("H", "S")


def stats_line(result):
    p = f"{result.p:.4f}"
    if result.p_at_floor:
        p = f"~{p}"  # at the test's resolution
    z = "—" if not np.isfinite(result.z) else f"{result.z:+.2f}"
    mark = f" {LOW_N_MARK}" if low_n(result) else ""
    return f"Δ = {result.delta:+.3f}   z = {z}   p = {p}{mark}"


def _limits(qs, vmax):
    if vmax is not None:
        return 0.0, float(vmax)
    lo, hi, _ = msp_fig.colour_limits(np.stack([np.asarray(q) for q in qs]))
    return lo, hi


def _matrix(ax, result, vmin, vmax):
    return msp_fig.draw_matrix(
        ax, result.q, names=MATRIX_NAMES,
        im_kw=dict(cmap=CMAP, vmin=vmin, vmax=vmax),
        counts=(result.n_H, result.n_S),
    )


# ---- profiles ----------------------------------------------------------------


def profile(X):
    """``(mean, se)`` per merged state over the rows of one strategy."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n == 0:
        return np.full(X.shape[1], np.nan), np.full(X.shape[1], np.nan)
    se = X.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(X.shape[1])
    return X.mean(axis=0), se


def draw_profile(ax, prof, *, names, block, ylim=(0.0, 1.0), legend=True):
    """Grouped bars, H vs S, per state; `prof` is ``{"H": (m, se), "S": ...}``."""
    x = np.arange(len(names))
    w = 0.38
    for i, tag in enumerate(("H", "S")):
        mean, se = prof[tag]
        ax.bar(
            x + (i - 0.5) * w, np.nan_to_num(mean), w * 0.94, yerr=se,
            color=STRATEGY_COLOUR[tag], alpha=0.9, label=tag,
            error_kw=dict(lw=0.9, capsize=2, ecolor="0.25"),
        )
    ax.set_xticks(x, labels=names, fontsize=9)
    ax.set_ylim(*ylim)
    ax.set_ylabel(BLOCK_YLABEL[block], fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if legend:
        ax.legend(fontsize=8, frameon=False, loc="upper right")


def pooled_difference(profiles, weights):
    """Weighted mean over mazes of the within-maze H − S, and its SE."""
    w = np.asarray(weights, dtype=float)
    diff = sum(wi * (p["H"][0] - p["S"][0]) for wi, p in zip(w, profiles))
    var = sum(wi**2 * (p["H"][1] ** 2 + p["S"][1] ** 2) for wi, p in zip(w, profiles))
    return diff, np.sqrt(var)


def draw_difference(ax, diff, se, *, names, block):
    x = np.arange(len(names))
    ax.bar(
        x, diff, 0.6, yerr=se,
        color=[STATE_COLOUR[n] for n in names], alpha=0.9,
        error_kw=dict(lw=0.9, capsize=2, ecolor="0.25"),
    )
    ax.axhline(0, color="0.3", lw=0.8)
    lim = max(0.05, float(np.nanmax(np.abs(diff) + se)) * 1.25)
    ax.set_ylim(-lim, lim)
    ax.set_xticks(x, labels=names, fontsize=9)
    ax.set_ylabel(f"H − S, {BLOCK_YLABEL[block]}", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ---- per-maze figures --------------------------------------------------------


def panel_figure(result, *, codebook, title, vmax=None):
    """One 2x2, as in msp, titled with the codebook's states."""
    vmin, vmax_eff = _limits([result.q], vmax)
    fig, ax = plt.subplots(figsize=(msp_fig.FIG_W, msp_fig.FIG_H), layout="constrained")
    im = _matrix(ax, result, vmin, vmax_eff)
    ax.set_title(f"K = {codebook.k}  ({', '.join(codebook.names)})\n{stats_line(result)}",
                 fontsize=10, pad=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.62)
    cbar.set_label(CBAR_LABEL, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    if low_n(result):
        title = f"{title}\n{LOW_N_NOTE}"
    fig.suptitle(title, fontsize=12)
    return fig


def profile_figure(prof, *, codebook, block, title):
    fig, ax = plt.subplots(figsize=(4.8 + 0.5 * (codebook.k - 3), 3.8), layout="constrained")
    draw_profile(ax, prof, names=codebook.names, block=block)
    ax.set_title(title, fontsize=10)
    return fig


def _draw_regions(ax, codebook, codebook5, radius):
    """Balls for ``balls``; origin ball + maze perimeter + split lines otherwise."""
    smap = codebook.map_array
    if not codebook.region:
        for b, (x, y) in enumerate(np.asarray(codebook5, dtype=float)):
            colour = STATE_COLOUR[codebook.names[smap[b]]]
            ax.add_patch(Circle((x, y), float(radius), fill=False, ec=colour, lw=1.4, zorder=3))
            ax.plot(x, y, "x", color="black", ms=7, mew=1.4, zorder=4)
        return
    half, pad = EXIT_HALF_WIDTH, PERIMETER_RADIUS
    ax.add_patch(FancyBboxPatch(
        (-half, -half), 2 * half, 2 * half, boxstyle=f"round,pad={pad}",
        fill=False, ec="0.2", lw=1.3, zorder=3,
    ))
    ax.add_patch(Circle((0, 0), float(radius), fill=False,
                        ec=STATE_COLOUR["origin"], lw=1.6, zorder=4))
    reach = half + pad
    if codebook.stem:
        ax.add_patch(Rectangle(
            (-STEM_HALF_WIDTH, 0.0), 2 * STEM_HALF_WIDTH, STEM_TOP,
            fill=False, ec=STATE_COLOUR["stem"], lw=1.6, zorder=4,
        ))
        ax.plot([0, 0], [STEM_TOP, reach], color="0.2", lw=1.0, zorder=3)
    else:
        ax.plot([0, 0], [radius, reach], color="0.2", lw=1.0, zorder=3)
    ax.plot([0, 0], [-reach, -radius], color="0.2", lw=1.0, zorder=3)
    if codebook.key.startswith("quads"):
        ax.plot([-reach, -radius], [0, 0], color="0.2", lw=1.0, zorder=3)
        ax.plot([radius, reach], [0, 0], color="0.2", lw=1.0, zorder=3)
    for x, y in np.asarray(codebook5, dtype=float)[1:]:
        ax.plot(x, y, "x", color="black", ms=6, mew=1.2, zorder=4)


def codebook_figure(fix_xy, fix_state, *, codebook, codebook5, radius, lim, title):
    """Fixation centroids coloured by `codebook` state, with its regions drawn."""
    fix_xy = np.asarray(fix_xy, dtype=float)
    fix_state = np.asarray(fix_state, dtype=int)
    n = max(fix_state.size, 1)
    fig, ax = plt.subplots(figsize=(6.6, 6.6), layout="constrained")
    ax.axhline(0, color="0.93", lw=0.6, zorder=0)
    ax.axvline(0, color="0.93", lw=0.6, zorder=0)
    un = fix_state < 0
    ax.scatter(
        fix_xy[un, 0], fix_xy[un, 1], s=2, c="0.75", alpha=0.35, linewidths=0,
        label=f"unassigned  {un.sum() / n:.1%}", rasterized=True,
    )
    for s_, name in enumerate(codebook.names):
        hit = fix_state == s_
        ax.scatter(
            fix_xy[hit, 0], fix_xy[hit, 1], s=2, color=STATE_COLOUR[name], alpha=0.45,
            linewidths=0, label=f"{name}  {hit.sum() / n:.1%}", rasterized=True,
        )
    _draw_regions(ax, codebook, codebook5, radius)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("unit-H x")
    ax.set_ylabel("unit-H y")
    ax.legend(
        loc="upper right", fontsize=8, markerscale=5, framealpha=0.92,
        title=f"share of {fix_state.size} fixations", title_fontsize=8,
    )
    ax.set_title(title, fontsize=11)
    return fig


# ---- paper figure ------------------------------------------------------------


def paper_figure(*, monkey, mazes, results, pooled_result, profiles, weights,
                 block, variant, tag_label, labels, trend, codebook, vmax=None,
                 skipped=None):
    """One monkey: 2x2 per maze plus pooled (top), H vs S profile per maze plus
    the pooled within-maze H − S (bottom). One colour scale for the top row."""
    n_col = len(mazes) + 1
    fig, axes = plt.subplots(
        2, n_col, figsize=((3.1 + 0.35 * (codebook.k - 3)) * n_col + 0.8, 6.6),
        layout="constrained",
        gridspec_kw=dict(height_ratios=[1.0, 0.78]),
    )
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.06, wspace=0.04)
    shown = [r for r in results if r is not None] + [pooled_result]
    vmin, vmax_eff = _limits([r.q for r in shown if r is not None], vmax)

    im = None
    prof_hi = max(
        float(np.nanmax(p[t][0] + p[t][1]))
        for p in profiles if p is not None for t in ("H", "S")
    )
    prof_hi = 1.0 if block == "binary" else min(1.0, np.ceil(prof_hi * 10) / 10 + 0.05)

    for col, maze in enumerate(mazes):
        ax_m, ax_p = axes[0, col], axes[1, col]
        result = results[col]
        if result is None:
            ax_m.axis("off")
            n_h, n_s = (len(skipped[col]["H"]), len(skipped[col]["S"])) if skipped else ("?", "?")
            ax_m.set_title(f"maze {maze}\nnot tested", fontsize=9.5, color="0.4")
            ax_m.text(0.5, 0.5, f"cannot split\nn_H = {n_h}, n_S = {n_s}\n(need ≥ 2 of each)",
                      ha="center", va="center", transform=ax_m.transAxes, color="0.4")
        else:
            im = _matrix(ax_m, result, vmin, vmax_eff)
            ax_m.set_title(f"maze {maze}\n{stats_line(result)}", fontsize=9.5)
        if profiles[col] is not None:
            draw_profile(ax_p, profiles[col], names=codebook.names, block=block,
                         ylim=(0, prof_hi), legend=(col == 0))
        if col:
            ax_p.set_ylabel("")

    ax_m, ax_p = axes[0, -1], axes[1, -1]
    if pooled_result is not None:
        im = _matrix(ax_m, pooled_result, vmin, vmax_eff)
        ax_m.set_title(
            f"mazes {', '.join(str(m) for m, r in zip(mazes, results) if r is not None and np.isfinite(r.delta))}"
            f" pooled\n{stats_line(pooled_result)}",
            fontsize=9.5, fontweight="bold",
        )
        kept = [p for p, r in zip(profiles, results)
                if r is not None and np.isfinite(r.delta) and p is not None]
        diff, se = pooled_difference(kept, weights)
        draw_difference(ax_p, diff, se, names=codebook.names, block=block)
    else:
        ax_m.axis("off")
        ax_p.axis("off")

    if im is not None:
        cbar = fig.colorbar(im, ax=list(axes[0]), fraction=0.02, pad=0.01, shrink=0.8)
        cbar.set_label(CBAR_LABEL, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    note = f"   ·   {LOW_N_NOTE}" if any(low_n(r) for r in results) else ""
    fig.suptitle(
        f"{monkey} — H: {trend['H']}, S: {trend['S']} — {codebook.label}\n"
        f"{BLOCK_LABEL[block]} · {variant} · {tag_label} · {labels}\n"
        f"pooled Δ = trial-weighted mean of maze Δ, within-(session, maze) shuffle null{note}",
        fontsize=11,
    )
    return fig
