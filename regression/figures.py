"""The one figure: every coefficient of the regression on a common log-odds axis.

One panel per monkey. A horizontal line at 0 runs through the middle; bars go
up for "more sequential (S)" and down for "more hierarchical (H)". Left of the
gap: the per-maze offsets (each maze's baseline). Right of it: the gaze
predictors, grouped by predictor, one bar per codebook state, with 95% Wald
intervals. A bar is solid when its interval excludes 0 (faded otherwise); a
star marks coefficients that also survive Holm correction across all gaze
coefficients. A cross at 0 marks a predictor that could not be fitted.
"""

from __future__ import annotations

import numpy as np

from regression.config import ALPHA, PREDICTORS, STATE_NAMES

# Okabe-Ito, fixed order: colour follows the state, and the state name is also
# printed under every bar, so identity never rests on colour alone.
STATE_COLORS = {
    "origin": "#009E73", "LU": "#0072B2", "LD": "#56B4E9", "RU": "#D55E00", "RD": "#E69F00",
}
OFFSET_COLOR = "#3b3b3b"
INK, INK_2, GRID = "#1a1a1a", "#555555", "#e4e4e4"
PRED_TITLES = {
    "occupied": "bin occupancy\n(visited vs not)",
    "visits": "visit count\n(per SD)",
    "duration": "mean fixation duration\n(per SD)",
}


def _layout(maze_ids):
    """x position of each offset and each (predictor, state)."""
    x, pos = 0, {}
    for m in maze_ids:
        pos[("offset", m)] = x
        x += 1
    x += 1.5  # gap between offsets and gaze
    centers = {}
    for key, _ in PREDICTORS:
        start = x
        for s in STATE_NAMES:
            pos[(key, s)] = x
            x += 1
        centers[key] = (start + x - 1) / 2
        x += 1
    return pos, centers, x - 1


def _panel(ax, rows, title):
    maze_ids = sorted({int(r["maze"]) for r in rows if r["kind"] == "offset"})
    pos, centers, x_end = _layout(maze_ids)
    key_of = lambda r: ("offset", int(r["maze"])) if r["kind"] == "offset" else (r["predictor"], r["state"])

    fitted = [r for r in rows if r["status"] == "fit" and r["kind"] != "window"]
    lo = min(r["ci_lo"] for r in fitted)
    hi = max(r["ci_hi"] for r in fitted)
    pad = 0.12 * (hi - lo)
    ymin, ymax = min(lo, 0) - pad, max(hi, 0) + pad

    ax.axhline(0, color=INK, lw=1.0, zorder=3)
    ticks, labels = [], []
    for r in rows:
        x = pos.get(key_of(r))
        if x is None:
            continue
        if r["status"] != "fit":
            ax.plot(x, 0, marker="x", color="#999999", ms=6, mew=1.2, zorder=4)
            ticks.append(x), labels.append(r["state"])
            continue
        is_off = r["kind"] == "offset"
        color = OFFSET_COLOR if is_off else STATE_COLORS[r["state"]]
        sig = is_off or (r["ci_lo"] > 0 or r["ci_hi"] < 0)
        ax.bar(x, r["beta"], width=0.7, color=color, alpha=1.0 if sig else 0.35, zorder=2)
        ax.plot([x, x], [r["ci_lo"], r["ci_hi"]], color=INK, lw=1.0, zorder=4)
        if not is_off and r["p_holm"] < ALPHA:
            up = r["beta"] >= 0
            ax.text(x, r["ci_hi"] + 0.01 * (ymax - ymin) if up else r["ci_lo"] - 0.01 * (ymax - ymin),
                    "★", ha="center", va="bottom" if up else "top", fontsize=9, color=INK)
        ticks.append(x)
        labels.append(f"m{int(r['maze'])}" if is_off else r["state"])

    ax.set_xticks(ticks, labels, fontsize=7.5, color=INK_2)
    ax.tick_params(axis="x", length=0, pad=3)
    tr = ax.get_xaxis_transform()
    for key, _ in PREDICTORS:
        ax.text(centers[key], -0.095, PRED_TITLES[key], transform=tr, ha="center", va="top",
                fontsize=8.5, color=INK)
    off_c = (len(maze_ids) - 1) / 2
    ax.text(off_c, 1.02, "maze offsets\n(baseline per maze)", transform=tr, ha="center",
            va="bottom", fontsize=8.5, color=INK)
    ax.text(np.mean(list(centers.values())), 1.02, "gaze predictors", transform=tr,
            ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xlim(-0.8, x_end + 0.8)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("log-odds of sequential (S)\n↑ more S    ↓ more H", fontsize=9, color=INK)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=34)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # same axis in odds-ratio units on the right
    rt = ax.twinx()
    rt.set_ylim(ymin, ymax)
    yt = [t for t in ax.get_yticks() if ymin <= t <= ymax]
    rt.set_yticks(yt, [f"×{np.exp(t):.2g}" for t in yt], fontsize=8, color=INK_2)
    rt.set_ylabel("odds of S relative to baseline", fontsize=9, color=INK_2)
    for side in ("top", "left"):
        rt.spines[side].set_visible(False)


def coefficient_figure(rows_by_monkey, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows_by_monkey)
    fig, axes = plt.subplots(n, 1, figsize=(11.5, 4.9 * n), squeeze=False)
    for ax, (monkey, rows) in zip(axes[:, 0], rows_by_monkey.items()):
        n_tr = sum(1 for r in rows if r["kind"] == "offset")  # mazes kept
        _panel(ax, rows, f"{monkey}: {n_tr} mazes")
    fig.text(
        0.01, 0.003,
        "Solid = 95% interval excludes 0; faded = it does not. ★ = survives Holm correction over all gaze "
        "coefficients. × = predictor not fitted (see coefficients.csv). Offsets are the log-odds of S in "
        "that maze at average gaze.",
        fontsize=7.5, color=INK_2, ha="left", va="bottom", wrap=True,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1), h_pad=5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
