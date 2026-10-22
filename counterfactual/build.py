"""Per-monkey outputs: exit sequences, the counterfactual test and figures.

    uv run python -m counterfactual.build [--monkey ...] [--dry-run] [--refresh]
                                          [--n-perm 10000] [--eps 1e-3] [--no-sensitivity]

Writes to ``counterfactual/out/<Monkey>/``:

- ``exit_sequences.csv``  one row per trial: the exit list as codes and names
- ``trials.csv``          one row per labelled trial: label, e1, q_H, q_S, LLs
- ``results.csv``         the statistics (long format, with CIs and p values)
- ``summary.txt``         extraction tallies + the analysis summary
- ``delta_by_label.png``  matched - mismatched log likelihood per label with the
                          permutation null
- ``top_accuracy.png``    first counterfactual exit == each model's top unchosen
                          exit, by label, on the disagreement trials
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from counterfactual import analysis as A  # noqa: E402
from counterfactual import models as M  # noqa: E402
from counterfactual.config import EXIT_NAMES, MODEL_EPS, MONKEYS, OUT_ROOT, SESSIONS, STATE_NAMES  # noqa: E402
from counterfactual.extract import load  # noqa: E402

STRATEGY_COLOUR = {"H": "#1f77b4", "S": "#d62728"}  # as msp / classifier
INK, MUTED, GRID = "#222222", "#6b6b6b", "#dddddd"


def exit_name(code):
    return EXIT_NAMES[int(code) - 1] if 1 <= int(code) <= 4 else STATE_NAMES.get(int(code), "?")


# ---- extraction summary (as before) ------------------------------------------------


def extraction_summary(monkey, data):
    seqs = [list(q) for q in data["exit_seq"]]
    chosen = data["chosen_exit"].astype(int)
    correct = data["correct_exit"].astype(int)
    lengths = np.array([len(q) for q in seqs])
    with_fix = [k for k, q in enumerate(seqs) if len(q) > 1]
    landing = np.asarray(data["landing_on_chosen"], dtype=bool)
    lines = [f"{monkey}: post-choice exit sequences (looks from timestamp_preprocessing)", ""]
    sessions = data["sessions"].astype(str).tolist()
    dropped = np.asarray(data["dropped"], dtype=int)
    reasons = data["drop_reasons"].astype(str).tolist()
    for j, s in enumerate(sessions):
        n = int((data["session"].astype(str) == s).sum())
        drops = ", ".join(f"{r}={dropped[j, i]}" for i, r in enumerate(reasons))
        lines.append(f"  {s}: {n} trials kept; dropped {drops}")
    lines += [
        "",
        f"trials: {len(seqs)}; looks in window: {len(data['fix_row'])}; looks straddling the window start: "
        f"{int(data['n_straddle'].sum())}",
        "exit-list length (incl. chosen): "
        + ", ".join(f"{k}: {int((lengths == k).sum())}" for k in sorted(set(lengths.tolist()))),
        f"trials with >= 1 post-choice exit look: {len(with_fix)} ({len(with_fix) / max(len(seqs), 1):.0%})",
        f"choice-saccade landing look on the decoded chosen exit: {int(landing.sum())}/{len(seqs)} "
        f"({landing.mean() if landing.size else 0:.0%})  <- sanity check of the LR/LR2 decoding; that look is item 0",
        f"trials whose list visits an exit other than the chosen one: "
        f"{sum(1 for q in seqs if any(e != q[0] for e in q[1:]))}",
        f"chosen == correct: {int((chosen == correct).sum())}/{len(seqs)}",
        "post-choice look states: "
        + ", ".join(f"{STATE_NAMES[int(k)]}={v}" for k, v in sorted(Counter(data["fix_state"].astype(int).tolist()).items())),
        f"strategy label present: {int(np.isfinite(data['strategy']).sum())}/{len(seqs)} (0 = H, 1 = S)",
    ]
    return "\n".join(lines) + "\n"


def write_exit_sequences(path, data):
    seqs = [list(q) for q in data["exit_seq"]]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "trial", "maze", "chosen", "correct", "strategy", "window_lo_ms", "window_hi_ms",
                    "n_looks", "landing_on_chosen", "exit_seq", "exit_seq_names", "state_seq"])
        for k in range(len(seqs)):
            strat = data["strategy"][k]
            w.writerow([
                data["session"][k], int(data["trial_indices_all"][k]), int(data["maze_id"][k]),
                int(data["chosen_exit"][k]), int(data["correct_exit"][k]),
                "" if not np.isfinite(strat) else int(strat),
                f"{float(data['window_lo_ms'][k]):.0f}", f"{float(data['window_hi_ms'][k]):.0f}",
                int(data["n_fix"][k]), int(data["landing_on_chosen"][k]),
                " ".join(str(e) for e in seqs[k]), " ".join(exit_name(e) for e in seqs[k]),
                " ".join(str(s) for s in data["state_seq"][k]),
            ])


def write_trials(path, rows):
    cols = ["session", "trial", "maze", "label", "chosen", "correct", "is_correct", "n_cf", "e1", "items",
            "top_h", "top_s", "disagree", "same_side", "ll_h", "ll_s", "ll_prox", "ll_unif",
            "seq_ll_h", "seq_ll_s", "q_h", "q_s"]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([
                r["session"], r["trial"], r["maze"], "H" if r["label"] == A.H else "S", r["chosen"], r["correct"],
                int(r["is_correct"]), r["n_cf"], r["e1"], " ".join(map(str, r["items"])), r["top_h"], r["top_s"],
                int(r["disagree"]),
                "" if r["n_cf"] == 0 else int(r["same_side"]),
                *("" if r["n_cf"] == 0 else f"{r[k]:.4f}" for k in ("ll_h", "ll_s", "ll_prox", "ll_unif", "seq_ll_h", "seq_ll_s")),
                " ".join(f"{v:.4f}" for v in r["q_h"]), " ".join(f"{v:.4f}" for v in r["q_s"]),
            ])


def write_results(path, rows):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["monkey", "statistic", "value", "ci_lo", "ci_hi", "p", "n"])
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()})


# ---- figures --------------------------------------------------------------------------------


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def figure_delta(path, monkey, stats, *, dpi):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _style(ax)
    xs = [0, 1, 2]
    labels = ["all trials", "H-labelled\n(log q_H − log q_S)", "S-labelled\n(log q_S − log q_H)"]
    means = [stats["delta_mean"], stats["delta_H_mean"], stats["delta_S_mean"]]
    cis = [stats["delta_ci"], stats["delta_H_ci"], stats["delta_S_ci"]]
    cols = [INK, STRATEGY_COLOUR["H"], STRATEGY_COLOUR["S"]]
    null = stats["delta_null"]
    ax.axhspan(np.quantile(null, 0.025), np.quantile(null, 0.975), color=GRID, alpha=0.6, lw=0,
               label="permutation null, 95% (all trials)")
    ax.axhline(0, color=MUTED, lw=0.8)
    for x, m, (lo, hi), c in zip(xs, means, cis, cols):
        ax.plot([x, x], [lo, hi], color=c, lw=2)
        ax.plot([x], [m], "o", color=c, ms=8, mec="white", mew=1.5)
        ax.text(x + 0.08, m, f"{m:+.2f}", va="center", fontsize=9, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9, color=INK)
    ax.set_ylabel("matched − mismatched  log q(first counterfactual exit)", fontsize=9, color=INK)
    ax.set_title(f"{monkey}: n {stats['n_cf']}, one-sided permutation p {stats['delta_p_one']:.3f}",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def figure_top_accuracy(path, monkey, stats, *, dpi):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    _style(ax)
    groups = [("H-labelled", stats["top_acc_h_H"], stats["top_acc_s_H"], stats["n_disagree_H"]),
              ("S-labelled", stats["top_acc_h_S"], stats["top_acc_s_S"], stats["n_disagree_S"])]
    width = 0.34
    for g, (name, acc_h, acc_s, n) in enumerate(groups):
        for k, (acc, model) in enumerate(((acc_h, "H"), (acc_s, "S"))):
            x = g + (k - 0.5) * (width + 0.04)
            ax.bar(x, acc, width=width, color=STRATEGY_COLOUR[model], lw=0)
            ax.text(x, acc + 0.015, f"{acc:.0%}", ha="center", fontsize=9, color=INK)
        ax.text(g, -0.09, f"n {n}", ha="center", fontsize=8, color=MUTED, transform=ax.get_xaxis_transform())
    ax.axhline(1 / 3, color=MUTED, lw=0.8, ls=":")
    ax.text(1.55, 1 / 3 + 0.01, "chance 1/3", fontsize=8, color=MUTED)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g[0] for g in groups], fontsize=9, color=INK)
    ax.set_ylim(0, 1)
    ax.set_ylabel("first counterfactual exit == model's top unchosen exit", fontsize=9, color=INK)
    ax.set_title(f"{monkey}: disagreement trials only (H and S rank a different exit first)", fontsize=10,
                 color=INK, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=STRATEGY_COLOUR[m]) for m in ("H", "S")]
    ax.legend(handles, ["H model's top", "S model's top"], frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---- main -------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Counterfactual saccade analysis.")
    parser.add_argument("--monkey", nargs="*", default=list(MONKEYS), choices=list(MONKEYS))
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--n-perm", type=int, default=10_000)
    parser.add_argument("--eps", type=float, default=MODEL_EPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--no-sensitivity", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="rebuild the exit and model caches")
    parser.add_argument("--dry-run", action="store_true", help="print the summaries, write nothing")
    args = parser.parse_args()
    for monkey in args.monkey:
        exits = load(monkey, refresh=args.refresh)
        models = M.load(monkey, refresh=args.refresh)
        text = extraction_summary(monkey, exits)
        rows = A.trial_table(exits, models, eps=args.eps)
        stats = A.summarize(rows, n_perm=args.n_perm, seed=args.seed)
        sens = {} if args.no_sensitivity else A.sensitivity(exits, models, monkey, seed=args.seed)
        text += "\n" + A.summary_text(monkey, stats, sens)
        print(text)
        if args.dry_run:
            continue
        out = args.out_root / monkey
        out.mkdir(parents=True, exist_ok=True)
        write_exit_sequences(out / "exit_sequences.csv", exits)
        write_trials(out / "trials.csv", rows)
        write_results(out / "results.csv", A.results_rows(monkey, stats, sens))
        (out / "summary.txt").write_text(text)
        if "delta_mean" in stats:
            figure_delta(out / "delta_by_label.png", monkey, stats, dpi=args.dpi)
            figure_top_accuracy(out / "top_accuracy.png", monkey, stats, dpi=args.dpi)
        print(f"wrote {out}/ (exit_sequences.csv, trials.csv, results.csv, summary.txt, 2 figures; "
              f"sessions {', '.join(SESSIONS[monkey])})")


if __name__ == "__main__":
    main()
