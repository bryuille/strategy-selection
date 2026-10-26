"""Paired model comparison: within each label group, does the matched
strategy's exit ranking explain the counterfactual saccades better than the
other strategy's?

Per trial, the ordered looks at unchosen exits (`counterfactual.extract`) are
scored under the hierarchical model H and an alternative A (the sequential
model S; `--alt total` / `--alt optimal` swap in Figure Five's Total Time or
Optimal model as an experiment, whose outputs are not kept):

    lr = log P_H(sequence) - log P_A(sequence)

where P is the Plackett-Luce probability of drawing those exits in that order
without replacement from the model's unchosen-exit probabilities q (the
model's choice probability renormalised over the three unchosen exits, with
`MODEL_EPS` smoothing). With one counterfactual saccade this is
log q_H(e) - log q_A(e).

Per label group and unit:

- H-labelled trials: is the mean of (H - A) above 0?
- S-labelled trials: is the mean of (A - H) above 0?
- all trials of the unit: matched - mismatched above 0 (the two pooled).

**p values.** ``d_i`` is trial *i*'s difference oriented so that positive
favours the label's model. The reported p is a one-sided paired t-test
against 0 (as Figure Five's neural-label vs comparator NLL test):
``t = mean(d) / (sd(d) / sqrt(n))``, ``p = P(T_{n-1} >= t)``. In brackets, a
sign-flip permutation p: the sign of each ``d_i`` is flipped at random
(``--n-perm`` times), the null distribution of the mean is built, and
``p = (#{null >= observed} + 1) / (n_perm + 1)``; no normality assumption.
The CI is the 2.5-97.5% percentile interval of 2,000 bootstrap resamples of
the trials. "Favour" is the share of trials where the label's model wins.

Units: each session and the whole monkey (``sessions`` figure), and each
session x maze stratum (``strata`` figure). Two variants: all counterfactual
saccades (``all``) and the first one only (``first``).

Output under ``counterfactual/out/<Monkey>/``: ``compare_<alt>_results.csv``,
``compare_<alt>_summary.txt`` and ``compare_<alt>_{sessions,strata}_{all,first}.png``.

    uv run python -m counterfactual.compare [--monkey ...] [--n-perm 10000] [--alt s total optimal]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import ttest_1samp  # noqa: E402

from counterfactual import models as M  # noqa: E402
from counterfactual.config import MODEL_EPS, MONKEYS, OUT_ROOT  # noqa: E402
from counterfactual.extract import load  # noqa: E402

GROUPS = ("H", "S", "all")
VARIANTS = ("all", "first")
ALTS = {"s": "S", "total": "Total Time", "optimal": "Optimal"}
H_LABEL, S_LABEL = 0, 1  # dendrogram label codes
N_BOOT = 2000
INK, MUTED, GRID = "#222222", "#6b6b6b", "#dddddd"
GROUP_COLOUR = {"H": "#1f77b4", "S": "#d62728", "all": INK}


# ---- per-trial scores ----------------------------------------------------------------


def counterfactual_items(exit_seq):
    """Post-choice exit items with repeats of the chosen exit removed."""
    chosen = int(exit_seq[0])
    return [int(e) for e in exit_seq[1:] if int(e) != chosen]


def plackett_luce(q, items):
    """log P(items in this order | sampling without replacement from q);
    a return to an exit already visited adds nothing."""
    q = np.asarray(q, dtype=float).copy()
    ll = 0.0
    seen = set()
    for e in items:
        if e in seen:
            continue
        ll += np.log(q[e - 1] / q.sum())
        q[e - 1] = 0.0
        seen.add(e)
    return ll


def trial_scores(exits, models, alt, *, first_only=False, eps=MODEL_EPS):
    """Per labelled trial with >= 1 counterfactual saccade: session, maze,
    label and ``lr`` = log P_H - log P_alt of the counterfactual sequence."""
    key = {(str(s), int(t)): j for j, (s, t) in enumerate(zip(models["session"], models["trial_indices_all"]))}
    prob_h = models["prob_h"].astype(float)
    prob_a = models[f"prob_{alt}"].astype(float)
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
        q_h = M.counterfactual_dist(prob_h[j], [chosen], eps=eps)[0]
        q_a = M.counterfactual_dist(prob_a[j], [chosen], eps=eps)[0]
        rows.append(dict(session=str(s), maze=int(exits["maze_id"][i]), trial=int(t), label=int(label),
                         lr=plackett_luce(q_h, items) - plackett_luce(q_a, items)))
    return rows


# ---- the paired test at one unit ---------------------------------------------------------


def test(rows, group, *, n_perm, rng):
    """One-sided paired comparison. `group` H: mean(H - A) > 0; S: mean(A - H) > 0;
    all: matched - mismatched > 0."""
    lr = np.array([r["lr"] for r in rows])
    label = np.array([r["label"] for r in rows])
    if group == "H":
        d = lr[label == H_LABEL]
    elif group == "S":
        d = -lr[label == S_LABEL]
    else:
        d = np.where(label == H_LABEL, 1.0, -1.0) * lr
    out = dict(n=int(d.size), mean=np.nan, ci=(np.nan, np.nan), p_t=np.nan, p_signflip=np.nan,
               frac_matched=np.nan, sum_lr=np.nan)
    if d.size < 2:
        return out
    out["mean"] = float(d.mean())
    idx = rng.integers(0, d.size, (N_BOOT, d.size))
    means = d[idx].mean(axis=1)
    out["ci"] = (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))
    if np.ptp(d) > 0:
        out["p_t"] = float(ttest_1samp(d, 0.0, alternative="greater").pvalue)
        flips = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
        null = (flips * d[None, :]).mean(axis=1)
        out["p_signflip"] = float((np.sum(null >= out["mean"]) + 1) / (n_perm + 1))
    out["frac_matched"] = float(np.mean(d > 0))
    out["sum_lr"] = float(d.sum())
    return out


def run(monkey, rows_by_variant, *, n_perm, min_trials, seed=0):
    rng = np.random.default_rng(seed)
    results = []

    def add(variant, level, session, maze, rows):
        for group in GROUPS:
            t = test(rows, group, n_perm=n_perm, rng=rng)
            if t["n"] >= min_trials:
                results.append(dict(variant=variant, level=level, monkey=monkey, session=session, maze=maze,
                                    group=group, **t))

    for variant, rows in rows_by_variant.items():
        for key in sorted({(r["session"], r["maze"]) for r in rows}):
            add(variant, "stratum", key[0], key[1], [r for r in rows if (r["session"], r["maze"]) == key])
        for session in sorted({r["session"] for r in rows}):
            add(variant, "session", session, "", [r for r in rows if r["session"] == session])
        add(variant, "monkey", "", "", rows)
    return results


# ---- reporting -------------------------------------------------------------------------------


def _fmt(v, f="{:.4f}"):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f.format(v)


def write_results(path, results):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["saccades", "level", "monkey", "session", "maze", "group", "n", "mean", "ci_lo", "ci_hi",
                    "p_paired_t_one_sided", "p_signflip", "frac_trials_matched_better", "sum_log_ratio"])
        for r in results:
            w.writerow([r["variant"], r["level"], r["monkey"], r["session"], r["maze"], r["group"], r["n"],
                        _fmt(r["mean"]), _fmt(r["ci"][0]), _fmt(r["ci"][1]), _fmt(r["p_t"]), _fmt(r["p_signflip"]),
                        _fmt(r["frac_matched"]), _fmt(r["sum_lr"])])


def _unit(r):
    return {"stratum": f"{r['session']} maze {r['maze']}", "session": r["session"], "monkey": r["monkey"]}[r["level"]]


def group_text(alt):
    a = ALTS[alt]
    return {"H": f"H-labelled: H - {a}", "S": f"S-labelled: {a} - H", "all": "all: matched - mismatched"}


def summary_text(monkey, alt, results, *, n_perm):
    gt = group_text(alt)
    L = [f"{monkey}: do the counterfactual saccades favour the label's model? H vs {ALTS[alt]}",
         "mean per-trial log P(label's model) - log P(other) with 95% bootstrap CI; p = one-sided paired t-test "
         f"against 0 (Figure Five's test), sign-flip permutation p ({n_perm} flips) in brackets;",
         "'favour' = share of trials where the label's model wins", ""]
    for variant, vname in (("all", "ALL COUNTERFACTUAL SACCADES"), ("first", "FIRST COUNTERFACTUAL SACCADE ONLY")):
        L.append(vname)
        for level in ("monkey", "session", "stratum"):
            for r in results:
                if r["variant"] != variant or r["level"] != level:
                    continue
                L.append(f"  {level:8s} {_unit(r):22s} {gt[r['group']]:30s} n {r['n']:4d}  "
                         f"mean {r['mean']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  "
                         f"p {_fmt(r['p_t'], '{:.4f}') or 'n/a':6s} [{_fmt(r['p_signflip'], '{:.4f}') or 'n/a'}]  "
                         f"favour {_fmt(r['frac_matched'], '{:.0%}') or 'n/a'}  sum log ratio {r['sum_lr']:+.1f}")
        L.append("")
    return "\n".join(L) + "\n"


def figure(path, monkey, alt, results, variant, levels, *, dpi):
    """Rows = units of `levels`; three markers per row (H-labelled, all, S-labelled)."""
    rs = [r for r in results if r["variant"] == variant and r["level"] in levels]
    order = {"monkey": 0, "session": 1, "stratum": 2}
    units = sorted({(order[r["level"]], r["session"], str(r["maze"]), _unit(r)) for r in rs})
    names = [u[3] for u in units]
    ys = {name: y for y, name in zip(np.arange(len(names))[::-1], names)}
    row_h = 0.62 if "stratum" in levels else 0.8
    fig, ax = plt.subplots(figsize=(8, row_h * len(names) + 1.6))
    gt = group_text(alt)
    offsets = {"H": 0.24, "all": 0.0, "S": -0.24}
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axvline(0, color=MUTED, lw=0.8)
    for r in rs:
        y = ys[_unit(r)] + offsets[r["group"]]
        col = GROUP_COLOUR[r["group"]]
        ax.plot(r["ci"], [y, y], color=col, lw=1.7, zorder=2)
        ax.plot([r["mean"]], [y], "o", color=col, ms=6.5 if r["group"] == "all" else 5.5, mec="white", mew=1, zorder=3)
        ax.text(1.01, y, f"n {r['n']}  p {_fmt(r['p_t'], '{:.3f}') or 'n/a'}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=7.5, color=col)
    ax.set_yticks(list(ys.values()))
    ax.set_yticklabels(names, fontsize=8.5, color=INK)
    ax.set_xlabel("log P(label's model) − log P(other), per trial", fontsize=9, color=INK)
    ax.tick_params(axis="x", labelsize=8, colors=MUTED)
    handles = [plt.Line2D([], [], color=GROUP_COLOUR[g], marker="o", lw=1.7, mec="white") for g in ("H", "all", "S")]
    fig.legend(handles, [gt["H"], gt["all"], gt["S"]], frameon=False, fontsize=8.5, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.01))
    ax.set_title(f"{monkey}: H vs {ALTS[alt]}, "
                 + ("all counterfactual saccades" if variant == "all" else "first counterfactual saccade only")
                 + "\nmean ± 95% CI, p = one-sided paired t-test", fontsize=9.5, color=INK, loc="left")
    fig.tight_layout(rect=(0, 0.04, 0.96, 1))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Paired H vs alternative model comparison of the counterfactual saccades.")
    parser.add_argument("--monkey", nargs="*", default=list(MONKEYS), choices=list(MONKEYS))
    parser.add_argument("--alt", nargs="*", default=["s"], choices=list(ALTS),
                        help="alternative(s) to H: s (default); total, optimal are experiments and are not run unless asked")
    parser.add_argument("--n-perm", type=int, default=10_000, help="sign flips for the permutation p")
    parser.add_argument("--min-trials", type=int, default=5, help="skip units / label groups with fewer trials")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    for monkey in args.monkey:
        exits, models = load(monkey), M.load(monkey)
        out = args.out_root / monkey
        out.mkdir(parents=True, exist_ok=True)
        for alt in args.alt:
            rows = {"all": trial_scores(exits, models, alt), "first": trial_scores(exits, models, alt, first_only=True)}
            results = run(monkey, rows, n_perm=args.n_perm, min_trials=args.min_trials, seed=args.seed)
            text = summary_text(monkey, alt, results, n_perm=args.n_perm)
            print("\n".join(line for line in text.split("\n") if "  monkey " in line or line.startswith(monkey)))
            write_results(out / f"compare_{alt}_results.csv", results)
            (out / f"compare_{alt}_summary.txt").write_text(text)
            for variant in VARIANTS:
                figure(out / f"compare_{alt}_sessions_{variant}.png", monkey, alt, results, variant,
                       ("monkey", "session"), dpi=args.dpi)
                figure(out / f"compare_{alt}_strata_{variant}.png", monkey, alt, results, variant,
                       ("stratum",), dpi=args.dpi)
            print(f"wrote {out}/compare_{alt}_* (results.csv, summary.txt, 4 figures)")


if __name__ == "__main__":
    main()
