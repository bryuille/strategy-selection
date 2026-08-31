"""Figures for the counterfactual-saccade model comparison.

Two figures, both keyed to the one interpretable metric -- how often a model
names the exit the animal actually looked at:

  model_accuracy.png    Ranked bars, both animals on one axis. The chance
                        line (1/3) and each animal's achievable ceiling are
                        drawn, so a bar's position between them is readable
                        without consulting a table. Hatched bars are fits
                        with a parameter at its limit or an implausible
                        timing noise -- their height is bought, not earned.
  cell_agreement.png    Per path type, whether the best model named the exit
                        the saccades actually favoured, against how lopsided
                        that cell's saccades were. Shows *where* a model
                        fails, which the single accuracy number hides.

Colors are the validated categorical slots 1-2 (blue Faure / orange Nielsen)
on the light chart surface; identity is carried by the legend and by direct
value labels, never by hue alone.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from eye_post_flash.model_comparison import _predict_fn

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834")
WARN = "#fab219"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=1.0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)


def plot_accuracy(results, out_dir):
    """Ranked accuracy bars, one group per model, one bar per animal."""
    # Key rows on the model *base*: each animal picks its own best revision
    # variant, and two half-empty rows would read as two different models
    # rather than one fitted twice. The variant rides on the bar label.
    by_monkey = {
        res["monkey"]: {r["base"]: r for r in res["records"]}
        for res in results
    }
    mean_rate = {}
    for res in results:
        for r in res["records"]:
            if r["base"] == "saturated":
                continue
            mean_rate.setdefault(r["base"], []).append(r["hit_rate"])
    labels = sorted(mean_rate, key=lambda b: -float(np.mean(mean_rate[b])))

    n = len(labels)
    height = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 0.52 * n + 2.0))
    fig.patch.set_facecolor(SURFACE)

    ys = np.arange(n)
    for i, (res, color) in enumerate(zip(results, SERIES)):
        rec = by_monkey[res["monkey"]]
        vals, flagged, notes = [], [], []
        for lbl in labels:
            r = rec.get(lbl)
            vals.append(np.nan if r is None else r["hit_rate"])
            flagged.append(bool(r and r["flags"]))
            variant = (r or {}).get("params", {}).get("variant", "")
            notes.append(f" {variant}" if variant else "")
        offset = (i - 0.5) * (height + 0.04)
        bars = ax.barh(
            ys + offset,
            vals,
            height=height,
            color=color,
            label=res["monkey"],
            zorder=3,
        )
        for bar, v, flag, note in zip(bars, vals, flagged, notes):
            if not np.isfinite(v):
                continue
            if flag:
                bar.set_hatch("////")
                bar.set_edgecolor(SURFACE)
                bar.set_linewidth(0.8)
            ax.text(
                v + 0.006,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.0%}" + (" †" if flag else "") + note,
                va="center",
                ha="left",
                fontsize=7.5,
                color=INK_2,
            )
        ceiling = res["ceiling"]
        ax.plot(
            [ceiling, ceiling],
            [-0.62, n - 0.4],
            color=color,
            lw=1.6,
            ls=(0, (4, 3)),
            zorder=2,
        )
        ax.text(
            ceiling,
            -0.72 - 0.34 * i,
            f"{res['monkey']} best possible {ceiling:.0%}",
            fontsize=7.5,
            color=color,
            ha="right" if i == 0 else "left",
            va="center",
        )

    chance = results[0]["chance"]
    ax.axvline(chance, color=MUTED, lw=1.4, zorder=2)
    ax.text(
        chance,
        -1.05,
        f"chance {chance:.0%}",
        fontsize=7.5,
        color=MUTED,
        ha="center",
        va="top",
    )

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(r["ceiling"] for r in results) * 1.22)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _p: f"{v:.0%}")
    )
    ax.set_xlabel(
        "saccades whose target the model named correctly",
        fontsize=9,
        color=INK_2,
    )
    ax.set_title(
        "Which model predicts where the counterfactual saccade goes?",
        fontsize=11.5,
        color=INK,
        pad=26,
        loc="left",
    )
    ax.grid(axis="x", color=AXIS, lw=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    _style(ax)

    handles = [
        Patch(facecolor=SERIES[i], label=res["monkey"])
        for i, res in enumerate(results)
    ]
    handles.append(
        Patch(
            facecolor="white",
            edgecolor=MUTED,
            hatch="////",
            label="† parameter at its limit / implausible timing noise",
        )
    )
    # opaque surface panel so the dashed ceiling rules do not run through
    # the swatches and labels
    leg = ax.legend(
        handles=handles,
        frameon=True,
        fontsize=8,
        loc="lower right",
        labelcolor=INK_2,
    )
    leg.get_frame().set_facecolor(SURFACE)
    leg.get_frame().set_edgecolor("none")
    leg.set_zorder(6)
    fig.tight_layout()
    path = out_dir / "model_accuracy.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {path}")


def plot_cell_agreement(results, out_dir):
    """Per path type: did the best model name the favoured exit?"""
    fig, axes = plt.subplots(
        1, len(results), figsize=(4.9 * len(results), 4.3), sharey=True
    )
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)

    for ax, res, color in zip(axes, results, SERIES):
        decision = [
            r for r in res["records"] if r["model"] != "saturated"
        ]
        winner = max(decision, key=lambda r: r["hit_rate"])
        predict = _predict_fn(
            winner["base"], winner["params"], res["nodes"]
        )
        xs, ys, hit = [], [], []
        for cell in res["cells"]:
            pt, _chosen, _c, _iv, _pos, counts = cell
            n = counts.sum()
            obs = int(np.argmax(counts))
            pred = int(np.argmax(predict(cell)))
            xs.append(pt)
            ys.append(counts[obs] / n)
            hit.append(obs == pred)
        hit = np.array(hit)
        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)
        ax.scatter(
            xs[hit], ys[hit], s=54, color=color, zorder=3,
            edgecolor=SURFACE, linewidth=1.2, label="model named it",
        )
        ax.scatter(
            xs[~hit], ys[~hit], s=54, facecolor="white", zorder=3,
            edgecolor=color, linewidth=1.8, label="model named another exit",
        )
        ax.axhline(1 / 3, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.text(
            0.6, 1 / 3, " even split", fontsize=7.5, color=MUTED,
            va="bottom", ha="left",
        )
        ax.set_title(
            f"{res['monkey']} — {winner['model']}, "
            f"{hit.sum()}/{len(hit)} cells",
            fontsize=10, color=INK, loc="left",
        )
        ax.set_xlabel("path type", fontsize=9, color=INK_2)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, 25)
        ax.grid(axis="y", color=AXIS, lw=0.6, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=7.5, loc="lower right",
                  labelcolor=INK_2)
        _style(ax)

    axes[0].set_ylabel(
        "share of the cell's saccades going to its\nmost-looked-at exit",
        fontsize=9, color=INK_2,
    )
    axes[0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _p: f"{v:.0%}")
    )
    fig.suptitle(
        "Where the best model gets it right, path type by path type",
        fontsize=11.5, color=INK, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out_dir / "cell_agreement.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {path}")


def plot_all(results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(results) < 2:
        print("(accuracy figure compares both animals; run without --monkey)")
    plot_accuracy(results, out_dir)
    plot_cell_agreement(results, out_dir)
