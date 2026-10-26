"""Absolute metric: how far above chance does each strategy's conditioned
belief score the counterfactual looks?

Per labelled trial, the gain over chance of model M is
``log P_M(looks) - log P_uniform(looks)``, both Plackett-Luce
(`counterfactual.compare.plackett_luce`), uniform = 1/3 on each unchosen exit. Chance is
0 whatever the number of exits looked at (a raw log P falls with every
extra look). Per label group and unit (the monkey and each session): mean
gain of H and of S with a 95% bootstrap CI. Higher is better; above 0 means
better than chance.

Output under ``counterfactual/out/<Monkey>/``: ``logp_results.csv``, ``logp_sessions.png``;
with ``--uncond`` (unconditioned choice probabilities, as `counterfactual.compare`) the
prefix ``uncond_``.

    uv run python -m counterfactual.logp [--monkey ...]
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from counterfactual.compare import GRID, INK, MUTED, trial_table, units  # noqa: E402
from counterfactual.config import H_LABEL, MODELS, MONKEYS, N_BOOT, S_LABEL, out_dir  # noqa: E402

MODEL_COLOUR = {"H": "#1f77b4", "S": "#d62728"}
MODEL_TEXT = {"H": "H model", "S": "S model"}
LABELS = {H_LABEL: "H-labelled", S_LABEL: "S-labelled"}


def results(df, monkey, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for unit, g in units(df, monkey):
        for label, lname in LABELS.items():
            k = g[g.label == label]
            if k.empty:
                continue
            idx = rng.integers(0, len(k), (N_BOOT, len(k)))
            for m in MODELS:
                gain = (k[f"logp_{m}"] - k.logp_uniform).to_numpy()
                means = gain[idx].mean(axis=1)
                rows.append(dict(unit=unit, group=lname, model=m, n=len(k), mean_gain=gain.mean(),
                                 ci_lo=np.quantile(means, 0.025), ci_hi=np.quantile(means, 0.975)))
    return pd.DataFrame(rows)


def figure(path, monkey, res, uncond=False):
    rows = list(dict.fromkeys(zip(res.unit, res.group)))
    ys = {row: y for y, row in zip(np.arange(len(rows))[::-1], rows)}
    offsets = {"H": 0.12, "S": -0.12}
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(rows) + 1.6))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axvline(0, color=MUTED, lw=1, ls="--")
    for r in res.itertuples():
        y = ys[(r.unit, r.group)] + offsets[r.model]
        col = MODEL_COLOUR[r.model]
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=col, lw=1.7, zorder=2)
        ax.plot([r.mean_gain], [y], "o", color=col, ms=5.5, mec="white", mew=1, zorder=3)
    for (unit, group), y in ys.items():
        n = res[(res.unit == unit) & (res.group == group)].n.iloc[0]
        ax.text(1.01, y, f"n {n}", transform=ax.get_yaxis_transform(), va="center", fontsize=7.5, color=MUTED)
    for prev, row in zip(rows, rows[1:]):
        if row[0] != prev[0]:
            ax.axhline(ys[row] + 0.5, color=GRID, lw=0.8)
    ax.set_yticks(list(ys.values()))
    ax.set_yticklabels([f"{u} · {g}" for u, g in rows], fontsize=8.5, color=INK)
    ax.set_xlabel("log P(model) − log P(uniform), per trial (0 = chance, higher is better)", fontsize=9, color=INK)
    ax.tick_params(axis="x", labelsize=8, colors=MUTED)
    handles = [plt.Line2D([], [], color=MODEL_COLOUR[m], marker="o", lw=1.7, mec="white") for m in MODELS]
    fig.legend(handles, [MODEL_TEXT[m] for m in MODELS], frameon=False, fontsize=8.5, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.01))
    ax.set_title(f"{monkey}: gain over chance, {'unconditioned (Figure Five)' if uncond else 'choice-conditioned'} "
                 "models, all counterfactual looks\nmean ± 95% bootstrap CI",
                 fontsize=9.5, color=INK, loc="left")
    fig.tight_layout(rect=(0, 0.04, 0.96, 1))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Absolute metric: gain over chance of H and S.")
    p.add_argument("--monkey", nargs="*", default=list(MONKEYS), choices=MONKEYS)
    p.add_argument("--uncond", action="store_true", help="unconditioned choice probabilities (prefix uncond_)")
    args = p.parse_args()
    pre = "uncond_" if args.uncond else ""
    for m in args.monkey:
        res = results(trial_table(m, uncond=args.uncond), m)
        d = out_dir(m)
        res.to_csv(d / f"{pre}logp_results.csv", index=False)
        figure(d / f"{pre}logp_sessions.png", m, res, args.uncond)
        print(res[res.unit == m].round(3).to_string(index=False))
        print(f"wrote {d}/{pre}logp_results.csv, {pre}logp_sessions.png")


if __name__ == "__main__":
    main()
