"""Relative metric: within each label group, does the matched strategy's
conditioned belief explain the counterfactual looks better than the other?

Per labelled trial with >= 1 counterfactual look, ``lr = log P_H - log P_S``,
P the Plackett-Luce probability of the ordered unchosen-exit looks under q_H
and q_S (`counterfactual.beliefs`). ``d`` is ``lr`` oriented so positive favours the
label's model. Groups: H-labelled (d = lr), S-labelled (d = -lr), all (both
pooled: matched - mismatched). Units: the monkey and each session.

Per row: mean d, 95% bootstrap CI (`N_BOOT` resamples of trials), one-sided
paired t-test p (d > 0), and the share of trials where the matched model
wins.

Output under ``counterfactual/out/<Monkey>/``: ``compare_results.csv``,
``compare_sessions.png``. With ``--uncond`` q is instead each model's
unconditioned choice probability (the Figure Five / v1 q, commit `09e6fea`,
`unconditioned_q`) and the files get the prefix ``uncond_``.

    uv run python -m counterfactual.compare [--monkey ...]
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ttest_1samp  # noqa: E402

from counterfactual import beliefs, extract  # noqa: E402
from counterfactual.config import EPS, H_LABEL, MODELS, MONKEYS, N_BOOT, S_LABEL, out_dir  # noqa: E402

INK, MUTED, GRID = "#222222", "#6b6b6b", "#dddddd"
GROUP_COLOUR = {"H": "#1f77b4", "S": "#d62728", "all": INK}
GROUP_TEXT = {"H": "H-labelled: H − S", "S": "S-labelled: S − H", "all": "all: matched − mismatched"}


def plackett_luce(q, items):
    """log P(items in this order | sampling without replacement from q, over
    exits 1-4); a return to an exit already visited adds nothing."""
    q = np.asarray(q, dtype=float).copy()
    ll, seen = 0.0, set()
    for e in items:
        if e in seen:
            continue
        ll += np.log(q[e - 1] / q.sum())
        q[e - 1] = 0.0
        seen.add(e)
    return ll


def unconditioned_q(choice, c, eps=EPS):
    """The model's probability of choosing each exit, chosen exit zeroed,
    `eps` added to the other three, renormalised (v1's q, commit `09e6fea`)."""
    q = np.asarray(choice, dtype=float) + eps
    q[c - 1] = 0.0
    return q / q.sum()


def trial_table(monkey, models=MODELS, uncond=False):
    """One row per labelled trial with >= 1 counterfactual look: session,
    label, outcome, number of distinct exits looked at, and log P of the
    looks under each of `models` and uniform. `uncond` scores the
    unconditioned choice probabilities instead of the conditioned beliefs."""
    tr, q = extract.load(monkey), beliefs.load(monkey)
    rows = []
    for r in range(len(tr["session"])):
        label = tr["label"][r]
        items = extract.counterfactual_items(tr, r)
        if not np.isfinite(label) or not items:
            continue
        c = int(tr["chosen"][r])
        uniform = np.ones(4)
        uniform[c - 1] = 0.0
        rows.append(dict(session=str(tr["session"][r]), trial=int(tr["trial"][r]), label=int(label),
                         outcome="correct" if c == int(tr["correct"][r]) else "error",
                         n_exits=len(set(items)), items=" ".join(map(str, items)),
                         **{f"logp_{m}": plackett_luce(unconditioned_q(q[f"choice_{m}"][r], c) if uncond
                                                       else q[f"q_{m}"][r], items) for m in models},
                         logp_uniform=plackett_luce(uniform, items)))
    return pd.DataFrame(rows)


def summary(d, rng):
    """mean, 95% bootstrap CI, one-sided t p, share > 0."""
    d = np.asarray(d, dtype=float)
    out = dict(n=len(d), mean=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan, frac_matched=np.nan)
    if len(d) < 2:
        return out
    means = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(axis=1)
    out.update(mean=d.mean(), ci_lo=np.quantile(means, 0.025), ci_hi=np.quantile(means, 0.975),
               frac_matched=np.mean(d > 0))
    if np.ptp(d) > 0:
        out["p"] = ttest_1samp(d, 0.0, alternative="greater").pvalue
    return out


def units(df, monkey):
    return [(monkey, df)] + [(s, g) for s, g in df.groupby("session")]


def results(df, monkey, seed=0):
    rng = np.random.default_rng(seed)
    lr = df.logp_H - df.logp_S
    sign = np.where(df.label == H_LABEL, 1.0, -1.0)
    rows = []
    for unit, g in units(df.assign(d=sign * lr), monkey):
        for group, k in (("H", g.label == H_LABEL), ("all", slice(None)), ("S", g.label == S_LABEL)):
            rows.append(dict(unit=unit, group=group, **summary(g.d[k], rng)))
    return pd.DataFrame(rows)


def figure(path, monkey, res, uncond=False):
    names = list(dict.fromkeys(res.unit))
    ys = {name: y for y, name in zip(np.arange(len(names))[::-1], names)}
    offsets = {"H": 0.24, "all": 0.0, "S": -0.24}
    fig, ax = plt.subplots(figsize=(8, 0.8 * len(names) + 1.6))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axvline(0, color=MUTED, lw=0.8)
    for r in res.itertuples():
        y = ys[r.unit] + offsets[r.group]
        col = GROUP_COLOUR[r.group]
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=col, lw=1.7, zorder=2)
        ax.plot([r.mean], [y], "o", color=col, ms=6.5 if r.group == "all" else 5.5, mec="white", mew=1, zorder=3)
        ax.text(1.01, y, f"n {r.n}  p {r.p:.3f}  wins {r.frac_matched:.0%}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=7.5, color=col)
    ax.set_yticks(list(ys.values()))
    ax.set_yticklabels(names, fontsize=8.5, color=INK)
    ax.set_xlabel("log P(label's model) − log P(other), per trial", fontsize=9, color=INK)
    ax.tick_params(axis="x", labelsize=8, colors=MUTED)
    handles = [plt.Line2D([], [], color=GROUP_COLOUR[g], marker="o", lw=1.7, mec="white") for g in ("H", "all", "S")]
    fig.legend(handles, [GROUP_TEXT[g] for g in ("H", "all", "S")], frameon=False, fontsize=8.5,
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    ax.set_title(f"{monkey}: {'unconditioned (Figure Five)' if uncond else 'choice-conditioned'} H vs S, "
                 "all counterfactual looks"
                 "\nmean ± 95% bootstrap CI, p = one-sided paired t-test, wins = share of trials the label's model wins",
                 fontsize=9.5, color=INK, loc="left")
    fig.tight_layout(rect=(0, 0.04, 0.96, 1))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Relative metric: H vs S per label group.")
    p.add_argument("--monkey", nargs="*", default=list(MONKEYS), choices=MONKEYS)
    p.add_argument("--uncond", action="store_true", help="unconditioned choice probabilities (prefix uncond_)")
    args = p.parse_args()
    pre = "uncond_" if args.uncond else ""
    for m in args.monkey:
        res = results(trial_table(m, uncond=args.uncond), m)
        d = out_dir(m)
        res.to_csv(d / f"{pre}compare_results.csv", index=False)
        figure(d / f"{pre}compare_sessions.png", m, res, args.uncond)
        print(res[res.unit == m].round(3).to_string(index=False))
        print(f"wrote {d}/{pre}compare_results.csv, {pre}compare_sessions.png")


if __name__ == "__main__":
    main()
