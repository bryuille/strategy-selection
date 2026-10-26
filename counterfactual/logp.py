"""Absolute fit of the counterfactual saccades: mean log P per trial under H,
S and a uniform reference.

`counterfactual.compare` reports log P_H - log P_S, which says which model is
less wrong but not whether either explains the looks. Here each model is
scored on its own: per trial, the Plackett-Luce log probability of the
ordered looks at unchosen exits (`compare.plackett_luce`, the model's
unchosen-exit distribution with `MODEL_EPS` smoothing). Uniform puts 1/3 on
each unchosen exit; it is scored the same way, so it is the chance level of
each unit (in the all-saccades variant the chance level depends on how many
distinct exits were looked at, so it differs between units).

Per label group (H-labelled, S-labelled) and unit (the monkey and each
session): mean log P per trial with a 95% bootstrap CI. Higher is better.

Output under ``counterfactual/out/<Monkey>/``: ``logp_results.csv`` and
``logp_sessions_{all,first}.png``.

    uv run python -m counterfactual.logp [--monkey ...]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from counterfactual import models as M  # noqa: E402
from counterfactual.compare import GRID, H_LABEL, INK, MUTED, N_BOOT, S_LABEL, counterfactual_items, plackett_luce  # noqa: E402
from counterfactual.config import MONKEYS, OUT_ROOT  # noqa: E402
from counterfactual.extract import load  # noqa: E402

MODELS = {"h": "H model", "s": "S model", "uniform": "uniform"}
MODEL_COLOUR = {"h": "#1f77b4", "s": "#d62728", "uniform": "#8c8c8c"}
LABELS = {H_LABEL: "H-labelled", S_LABEL: "S-labelled"}
VARIANTS = ("all", "first")


def trial_logp(exits, models, *, first_only=False):
    """Per labelled trial with >= 1 counterfactual saccade: session, label and
    log P of the counterfactual sequence under each of `MODELS`."""
    key = {(str(s), int(t)): j for j, (s, t) in enumerate(zip(models["session"], models["trial_indices_all"]))}
    prob = {m: models[f"prob_{m}"].astype(float) for m in ("h", "s")}
    rows = []
    for i, (s, t) in enumerate(zip(exits["session"], exits["trial_indices_all"])):
        j = key.get((str(s), int(t)))
        label = float(exits["strategy"][i])
        if j is None or not np.isfinite(label):
            continue
        chosen = int(exits["chosen_exit"][i])
        items = counterfactual_items(exits["exit_seq"][i])
        if not items:
            continue
        if first_only:
            items = items[:1]
        row = dict(session=str(s), label=int(label))
        for m in ("h", "s"):
            row[m] = plackett_luce(M.counterfactual_dist(prob[m][j], [chosen])[0], items)
        uniform = np.ones(4)
        uniform[chosen - 1] = 0.0
        row["uniform"] = plackett_luce(uniform, items)
        rows.append(row)
    return rows


def summarise(monkey, rows_by_variant, *, seed=0):
    rng = np.random.default_rng(seed)
    results = []
    for variant, rows in rows_by_variant.items():
        units = [("monkey", monkey, rows)] + [("session", s, [r for r in rows if r["session"] == s])
                                              for s in sorted({r["session"] for r in rows})]
        for level, unit, unit_rows in units:
            for label, lname in LABELS.items():
                group = [r for r in unit_rows if r["label"] == label]
                if not group:
                    continue
                idx = rng.integers(0, len(group), (N_BOOT, len(group)))
                for m in MODELS:
                    v = np.array([r[m] for r in group])
                    means = v[idx].mean(axis=1)
                    results.append(dict(variant=variant, level=level, unit=unit, group=lname, model=m, n=len(group),
                                        mean=float(v.mean()), ci=(float(np.quantile(means, 0.025)),
                                                                 float(np.quantile(means, 0.975)))))
    return results


def write_results(path, results):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["saccades", "level", "unit", "group", "model", "n", "mean_logp", "ci_lo", "ci_hi"])
        for r in results:
            w.writerow([r["variant"], r["level"], r["unit"], r["group"], r["model"], r["n"],
                        f"{r['mean']:.4f}", f"{r['ci'][0]:.4f}", f"{r['ci'][1]:.4f}"])


def figure(path, monkey, results, variant, *, dpi):
    """Rows = unit x label group; one marker per model."""
    rs = [r for r in results if r["variant"] == variant]
    rows = list(dict.fromkeys((r["unit"], r["group"]) for r in rs))
    ys = {row: y for y, row in zip(np.arange(len(rows))[::-1], rows)}
    offsets = {"h": 0.18, "s": 0.0, "uniform": -0.18}
    fig, ax = plt.subplots(figsize=(8, 0.75 * len(rows) + 1.6))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for r in rs:
        y = ys[(r["unit"], r["group"])] + offsets[r["model"]]
        col = MODEL_COLOUR[r["model"]]
        ax.plot(r["ci"], [y, y], color=col, lw=1.7, zorder=2)
        ax.plot([r["mean"]], [y], "o", color=col, ms=5.5, mec="white", mew=1, zorder=3)
    for (unit, group), y in ys.items():
        n = next(r["n"] for r in rs if r["unit"] == unit and r["group"] == group)
        ax.text(1.01, y, f"n {n}", transform=ax.get_yaxis_transform(), va="center", fontsize=7.5, color=MUTED)
    for prev, row in zip(rows, rows[1:]):
        if row[0] != prev[0]:  # rule between units
            ax.axhline(ys[row] + 0.5, color=GRID, lw=0.8)
    ax.set_yticks(list(ys.values()))
    ax.set_yticklabels([f"{unit} · {group}" for unit, group in rows], fontsize=8.5, color=INK)
    ax.set_xlabel("mean log P of the counterfactual saccades per trial (higher is better)", fontsize=9, color=INK)
    ax.tick_params(axis="x", labelsize=8, colors=MUTED)
    handles = [plt.Line2D([], [], color=MODEL_COLOUR[m], marker="o", lw=1.7, mec="white") for m in MODELS]
    fig.legend(handles, list(MODELS.values()), frameon=False, fontsize=8.5, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.01))
    ax.set_title(f"{monkey}: absolute fit, "
                 + ("all counterfactual saccades" if variant == "all" else "first counterfactual saccade only")
                 + "\nmean ± 95% bootstrap CI; uniform = chance", fontsize=9.5, color=INK, loc="left")
    fig.tight_layout(rect=(0, 0.04, 0.96, 1))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Mean log P of the counterfactual saccades under H, S and uniform.")
    parser.add_argument("--monkey", nargs="*", default=list(MONKEYS), choices=list(MONKEYS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    for monkey in args.monkey:
        exits, models = load(monkey), M.load(monkey)
        out = args.out_root / monkey
        out.mkdir(parents=True, exist_ok=True)
        rows = {"all": trial_logp(exits, models), "first": trial_logp(exits, models, first_only=True)}
        results = summarise(monkey, rows, seed=args.seed)
        for r in results:
            if r["level"] == "monkey":
                print(f"{monkey} {r['variant']:5s} {r['group']:10s} {MODELS[r['model']]:8s} n {r['n']:4d}  "
                      f"mean log P {r['mean']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]")
        write_results(out / "logp_results.csv", results)
        for variant in VARIANTS:
            figure(out / f"logp_sessions_{variant}.png", monkey, results, variant, dpi=args.dpi)
        print(f"wrote {out}/logp_* (results.csv, 2 figures)")


if __name__ == "__main__":
    main()
