"""2x2 panels, H vs S profiles, codebook maps, and the per-monkey paper figure.
Copied and generalised from `msp.figures` / `msp_trends.figures`."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch

from msp_revised.codebooks import EXIT_HALF_WIDTH, PERIMETER_RADIUS, UNIT_LANDMARKS, deg_landmarks
from msp_revised.config import (
    CBAR_LABEL,
    CMAP,
    DEG_LIM,
    FIG_H,
    FIG_W,
    LOW_N_MARK,
    LOW_N_NOTE,
    STEM_DEG,
    STRATEGY_COLOUR,
    UNIT_LIM,
    state_colour,
)
from msp_revised.estimator import low_n

MATRIX_NAMES = ("H", "S")
KIND_YLABEL = {
    "binary": "fraction of trials visiting",
    "dwell": "mean share of assigned dwell",
    "count": "mean share of assigned fixations",
    "first": "fraction of trials whose first fixation is here",
    "last": "fraction of trials whose last fixation is here",
    "both": "binary (fraction) | dwell (share)",
}


def save_figure(fig, stem, *, out_root, rel_dir=None, dpi=200, bbox_inches=None, pad_inches=0.02):
    out_dir = out_root / rel_dir if rel_dir else out_root
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    return path


def stats_line(result):
    p = f"{result.p:.4f}"
    if result.p_at_floor:
        p = f"~{p}"
    z = "—" if not np.isfinite(result.z) else f"{result.z:+.2f}"
    mark = f" {LOW_N_MARK}" if low_n(result) else ""
    return f"Δ = {result.delta:+.3f}   z = {z}   p = {p}{mark}"


def colour_limits(qs, vmax=None):
    if vmax is not None:
        return 0.0, float(vmax)
    values = np.concatenate([np.ravel(q) for q in qs])
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 0.0, 1.0
    lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-9:
        pad = max(abs(hi), 1e-3) * 0.05
        lo, hi = lo - pad, hi + pad
    return lo, hi


def draw_matrix(ax, result, vmin, vmax):
    q = result.q
    im = ax.imshow(q, origin="upper", aspect="equal", cmap=CMAP, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(2), labels=MATRIX_NAMES, fontsize=13)
    ax.set_yticks(range(2), labels=MATRIX_NAMES, fontsize=13)
    ax.tick_params(length=0)
    counts = (result.n_H, result.n_S)
    for i in range(2):
        for j in range(2):
            v = q[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "—", ha="center", va="center", color="0.4", fontsize=14)
                continue
            r, g, b, _ = im.cmap(im.norm(v))
            colour = "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "white"
            if i == j:
                ax.text(j, i - 0.08, f"{v:.3f}", ha="center", va="center", color=colour, fontsize=15)
                ax.text(j, i + 0.22, f"n = {counts[i]}", ha="center", va="center", color=colour, fontsize=10)
            else:
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", color=colour, fontsize=15)
    return im


# ---- profiles ----------------------------------------------------------------


def profile(X):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n == 0:
        return np.full(X.shape[1], np.nan), np.full(X.shape[1], np.nan)
    se = X.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(X.shape[1])
    return X.mean(axis=0), se


def draw_profile(ax, prof, *, names, kind, ylim=None, legend=True):
    x = np.arange(len(names))
    w = 0.38
    for i, tag in enumerate(("H", "S")):
        mean, se = prof[tag]
        ax.bar(x + (i - 0.5) * w, np.nan_to_num(mean), w * 0.94, yerr=se,
               color=STRATEGY_COLOUR[tag], alpha=0.9, label=tag,
               error_kw=dict(lw=0.9, capsize=2, ecolor="0.25"))
    ax.set_xticks(x, labels=names, fontsize=8, rotation=30 if len(names) > 6 else 0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(0, color="0.3", lw=0.6)
    ax.set_ylabel(KIND_YLABEL.get(kind, kind), fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if legend:
        ax.legend(fontsize=8, frameon=False, loc="upper right")


def pooled_difference(profiles, weights):
    w = np.asarray(weights, dtype=float)
    diff = sum(wi * (p["H"][0] - p["S"][0]) for wi, p in zip(w, profiles))
    var = sum(wi**2 * (p["H"][1] ** 2 + p["S"][1] ** 2) for wi, p in zip(w, profiles))
    return diff, np.sqrt(var)


def draw_difference(ax, diff, se, *, names, kind):
    x = np.arange(len(names))
    ax.bar(x, diff, 0.6, yerr=se, color=[state_colour(i) for i in range(len(names))], alpha=0.9,
           error_kw=dict(lw=0.9, capsize=2, ecolor="0.25"))
    ax.axhline(0, color="0.3", lw=0.8)
    lim = max(0.05, float(np.nanmax(np.abs(diff) + se)) * 1.25)
    ax.set_ylim(-lim, lim)
    ax.set_xticks(x, labels=names, fontsize=8, rotation=30 if len(names) > 6 else 0)
    ax.set_ylabel(f"H − S, {KIND_YLABEL.get(kind, kind)}", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ---- single panels -----------------------------------------------------------


def panel_figure(result, *, names, title, vmax=None):
    vmin, vmax_eff = colour_limits([result.q], vmax)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), layout="constrained")
    im = draw_matrix(ax, result, vmin, vmax_eff)
    ax.set_title(f"d = {len(names)}  ({', '.join(names)})\n{stats_line(result)}", fontsize=9, pad=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.62)
    cbar.set_label(CBAR_LABEL, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    if low_n(result):
        title = f"{title}\n{LOW_N_NOTE}"
    fig.suptitle(title, fontsize=11)
    return fig


def profile_figure(prof, *, names, kind, title):
    fig, ax = plt.subplots(figsize=(4.8 + 0.45 * max(len(names) - 3, 0), 3.8), layout="constrained")
    draw_profile(ax, prof, names=names, kind=kind)
    ax.set_title(title, fontsize=9)
    return fig


# ---- codebook map ------------------------------------------------------------


def _draw_geometry(ax, codebook, h_typical):
    """Landmarks and the assignment regions that can be drawn."""
    if codebook.frame == "unit":
        lm = UNIT_LANDMARKS
        for a, b in (((-1, 0), (1, 0)), ((-1, -1), (-1, 1)), ((1, -1), (1, 1))):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="0.75", lw=0.8, zorder=2)
        ax.add_patch(FancyBboxPatch((-EXIT_HALF_WIDTH, -EXIT_HALF_WIDTH), 2 * EXIT_HALF_WIDTH,
                                    2 * EXIT_HALF_WIDTH, boxstyle=f"round,pad={PERIMETER_RADIUS}",
                                    fill=False, ec="0.75", lw=0.8, ls="--", zorder=2))
    else:
        lm = deg_landmarks(np.asarray(h_typical, dtype=float)[None])[0]
        h1, h2, h3, h4, h5, h6 = h_typical
        # the H: crossbar through the fixation point, bars at x = -h1 and x = h4
        for a, b in (((-h1, 0), (h4, 0)), ((-h1, -h3), (-h1, h2)), ((h4, -h6), (h4, h5))):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="0.75", lw=0.8, zorder=2)
    for x, y in lm:
        ax.plot(x, y, "x", color="black", ms=6, mew=1.2, zorder=4)
    if codebook.radius is not None:
        if codebook.key.startswith("balls"):
            n_balls = codebook.k
        elif codebook.key.startswith("oquads_balls"):
            n_balls = 5
        else:
            n_balls = 1
        for i in range(min(n_balls, len(lm))):
            ax.add_patch(Circle(tuple(lm[i]), codebook.radius, fill=False, ec="0.2", lw=1.0, zorder=3))


def codebook_figure(fix_xy, fix_state, *, codebook, h_typical, title):
    fix_xy = np.asarray(fix_xy, dtype=float)
    fix_state = np.asarray(fix_state, dtype=int)
    n = max(fix_state.size, 1)
    lim = UNIT_LIM if codebook.frame == "unit" else DEG_LIM
    fig, ax = plt.subplots(figsize=(6.6, 6.6), layout="constrained")
    ax.axhline(0, color="0.93", lw=0.6, zorder=0)
    ax.axvline(0, color="0.93", lw=0.6, zorder=0)
    un = fix_state < 0
    ax.scatter(fix_xy[un, 0], fix_xy[un, 1], s=2, c="0.75", alpha=0.35, linewidths=0,
               label=f"unassigned  {un.sum() / n:.1%}", rasterized=True)
    for s_, name in enumerate(codebook.names):
        hit = fix_state == s_
        ax.scatter(fix_xy[hit, 0], fix_xy[hit, 1], s=2, color=state_colour(s_), alpha=0.45,
                   linewidths=0, label=f"{name}  {hit.sum() / n:.1%}", rasterized=True)
    _draw_geometry(ax, codebook, h_typical)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    unit = "unit-H" if codebook.frame == "unit" else "deg"
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")
    ax.legend(loc="upper right", fontsize=7, markerscale=5, framealpha=0.92,
              title=f"share of {fix_state.size} fixations", title_fontsize=7)
    ax.set_title(title, fontsize=10)
    return fig


# ---- paper figure ------------------------------------------------------------


def paper_figure(*, monkey, mazes, results, pooled_result, profiles, weights,
                 names, kind, variant, codebook_label, vmax=None, skipped=None):
    """Top: 2x2 per maze plus the pooled estimate. Bottom: H vs S profile per
    maze and the pooled within-maze H − S. One colour scale across the top."""
    n_col = len(mazes) + 1
    k = len(names)
    fig, axes = plt.subplots(2, n_col, figsize=((2.6 + 0.25 * max(k - 3, 0)) * n_col + 0.8, 6.4),
                             layout="constrained", gridspec_kw=dict(height_ratios=[1.0, 0.8]))
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.06, wspace=0.04)
    shown = [r for r in results if r is not None] + ([pooled_result] if pooled_result is not None else [])
    vmin, vmax_eff = colour_limits([r.q for r in shown] or [np.zeros((2, 2))], vmax)
    im = None
    finite_prof = [p for p in profiles if p is not None]
    hi = max((float(np.nanmax(p[t][0] + p[t][1])) for p in finite_prof for t in ("H", "S")), default=1.0)
    lo = min((float(np.nanmin(p[t][0] - p[t][1])) for p in finite_prof for t in ("H", "S")), default=0.0)
    ylim = (min(lo, 0.0) * 1.1 - 0.02, max(hi, 0.05) * 1.1)

    for col, maze in enumerate(mazes):
        ax_m, ax_p = axes[0, col], axes[1, col]
        result = results[col]
        if result is None:
            ax_m.axis("off")
            n_h, n_s = skipped[col] if skipped else ("?", "?")
            ax_m.set_title(f"maze {maze}\nnot tested", fontsize=9, color="0.4")
            ax_m.text(0.5, 0.5, f"cannot split\nn_H = {n_h}, n_S = {n_s}", ha="center", va="center",
                      transform=ax_m.transAxes, color="0.4", fontsize=8)
        else:
            im = draw_matrix(ax_m, result, vmin, vmax_eff)
            ax_m.set_title(f"maze {maze}\n{stats_line(result)}", fontsize=8.5)
        if profiles[col] is not None:
            draw_profile(ax_p, profiles[col], names=names, kind=kind, ylim=ylim, legend=(col == 0))
        else:
            ax_p.axis("off")
        if col:
            ax_p.set_ylabel("")

    ax_m, ax_p = axes[0, -1], axes[1, -1]
    if pooled_result is None:
        ax_m.axis("off")
        ax_p.axis("off")
    else:
        im = draw_matrix(ax_m, pooled_result, vmin, vmax_eff)
        used = [m for m, r in zip(mazes, results) if r is not None and np.isfinite(r.delta)]
        ax_m.set_title(f"mazes {', '.join(map(str, used))} pooled\n{stats_line(pooled_result)}",
                       fontsize=8.5, fontweight="bold")
        kept = [p for p, r in zip(profiles, results) if r is not None and np.isfinite(r.delta) and p is not None]
        if kept and len(weights) == len(kept):
            diff, se = pooled_difference(kept, weights)
            draw_difference(ax_p, diff, se, names=names, kind=kind)
        else:
            ax_p.axis("off")

    if im is not None:
        cbar = fig.colorbar(im, ax=list(axes[0]), fraction=0.02, pad=0.01, shrink=0.8)
        cbar.set_label(CBAR_LABEL, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    note = f"   ·   {LOW_N_NOTE}" if any(low_n(r) for r in results) else ""
    fig.suptitle(f"{monkey} — {codebook_label}\n{kind} · {variant} · dendrogram labels, publication sessions · "
                 f"pooled Δ = trial-weighted mean of maze Δ, within-(session, maze) shuffle null{note}",
                 fontsize=10)
    return fig
