"""Presentation figures: within-maze AUC bars, each against its own shuffle null.

    uv run python -m classifier.pres                 # per session: one bar per (monkey, session)
    uv run python -m classifier.pres --merged        # sessions merged: one bar per monkey
    uv run python -m classifier.pres --merged --model gbm --labels svm
    uv run python -m classifier.pres --variant regions/both_per_session/logreg
    uv run python -m classifier.pres --merged --variant exits/both/logreg

Without `--variant` the run is the `--model` region / exits run of the
`--labels` scope at the default shuffle count (``<variant>_per_session``, or
with `--merged` the plain ``both`` / ``dwell`` / ``binary`` / ``all``) whose
weaker monkey has the higher z, so one strong monkey cannot carry it; per
session that z is the monkey's pair-weighted combination in results.csv.
Writes ``session_auc.{png,svg}`` or ``merged_auc.{png,svg}`` next to the
run's own figures (out/<scope>/<family>/<variant>/<model>/); the plotted
numbers are printed, and the run's description (features, model, shuffles)
with them for the caption rather than drawn.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from classifier import config as cfg  # noqa: E402

MONKEY_COLOUR = {"Faure": "#2a78d6", "Nielsen": "#eb6834"}
INK, MUTED, NULL_FILL = "#1f1f1e", "#6b6a63", "#dcdbd5"
FAMILY_LABEL = {"regions": "gaze regions", "regions_plus": "gaze regions + pupil",
                "exits": "exit balls", "lor": "origin ball + maze halves", "balls": "origin ball + exit balls",
                "capsules": "origin ball + exit capsules + elsewhere",
                "capsules_tight": "smaller origin ball + exit capsules + elsewhere"}
VARIANT_LABEL = {"both": "dwell share + visited", "dwell": "dwell share", "binary": "visited",
                 "all": "dwell share + visited + visit count"}
MODEL_LABEL = {"logreg": "logistic regression", "gbm": "boosted trees"}


def read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def monkey_scores(run, merged):
    """``{monkey: (z, p, auc)}``: the pooled fit's row, or per session the
    combined row."""
    want = (lambda r: r["maze"] == "within-maze pooled") if merged else (lambda r: r.get("session") == "combined")
    return {r["monkey"]: (float(r["z"]), float(r["p"]), float(r["auc"]))
            for r in read_rows(run / "results.csv") if want(r)}


def candidates(root, merged, model):
    """Every finished default-shuffle region / exits run of the requested kind
    and model, best first."""
    runs = []
    for res in sorted(p for fam in ("regions*", "exits*", "lor*", "balls*", "capsules*")
                      for p in root.glob(f"{fam}/*/{model}/results.csv")):
        variant = res.parent.parent.name
        if variant not in (set(VARIANT_LABEL) if merged else {f"{v}_per_session" for v in VARIANT_LABEL}):
            continue
        sc = monkey_scores(res.parent, merged)
        if set(sc) >= set(cfg.MONKEYS):
            runs.append((min(sc[m][0] for m in cfg.MONKEYS), res.parent, sc))
    return sorted(runs, key=lambda t: -t[0])


def p_text(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def session_label(s):
    """``june_24_g0`` -> ``Jun 24``; ``Nov_3_g0`` -> ``Nov 3``."""
    month, day = s.split("_g0")[0].split("_")[:2]
    return f"{month[:3].capitalize()} {int(day)}"


def session_order(s):
    month, day = s.split("_g0")[0].split("_")[:2]
    return ("janfebmaraprmayjunjulaugsepoctnovdec".index(month[:3].lower()), int(day))


def bar(path, monkey, session=None):
    d = np.load(path, allow_pickle=True)
    null = d["null"][np.isfinite(d["null"])]
    y = d["y"]
    return dict(monkey=monkey, session=session or "merged", auc=float(d["auc"]), p=float(d["p"]), z=float(d["z"]),
                null_lo=float(np.percentile(null, 5)), null_hi=float(np.percentile(null, 95)),
                null_mean=float(null.mean()), n_trials=int(y.size), n_H=int((y == 0).sum()),
                n_S=int((y == 1).sum()), n_perm=int(null.size))


def load(run, merged):
    if merged:
        return [bar(run / f"eval_{m}.npz", m) for m in cfg.MONKEYS]
    bars = []
    for m in cfg.MONKEYS:
        for s in sorted((p.stem.removeprefix(f"eval_{m}_") for p in run.glob(f"eval_{m}_*.npz")), key=session_order):
            bars.append(bar(run / f"eval_{m}_{s}.npz", m, s))
    return bars


def figure(bars, scores, merged):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.linewidth": 0.6, "ytick.major.width": 0.6,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(2.2 if merged else 3.4, 2.8), layout="constrained")
    # one slot per bar; per session, a wider gap between monkeys
    xs, x, prev = [], 0.0, None
    for b in bars:
        if prev is not None:
            x += 1.0 if (merged or b["monkey"] == prev) else 1.5
        xs.append(x)
        prev = b["monkey"]
    base, width = 0.5, 0.6
    for xi, b in zip(xs, bars):
        ax.add_patch(plt.Rectangle((xi - width / 2 - 0.07, b["null_lo"]), width + 0.14, b["null_hi"] - b["null_lo"],
                                   facecolor=NULL_FILL, edgecolor="none", zorder=1))
        ax.bar(xi, b["auc"] - base, bottom=base, width=width, color=MONKEY_COLOUR[b["monkey"]], zorder=2)
        sig = b["p"] < 0.05
        ax.text(xi, max(b["auc"], b["null_hi"]) + 0.01, p_text(b["p"]), ha="center", va="bottom", fontsize=6.5,
                color=INK if sig else MUTED, fontweight="bold" if sig else "normal")
    ax.axhline(base, color=INK, lw=0.6, zorder=3)

    if merged:
        ax.set_xticks(xs, [f"Monkey {b['monkey'][0]}" for b in bars], fontsize=8)
    else:
        ax.set_xticks(xs, [session_label(b["session"]) for b in bars], fontsize=7)
        for m in cfg.MONKEYS:
            idx = [i for i, b in enumerate(bars) if b["monkey"] == m]
            if idx:
                ax.text((xs[idx[0]] + xs[idx[-1]]) / 2, -0.09, f"Monkey {m[0]}\ncombined {p_text(scores[m][1])}",
                        transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7, color=INK,
                        linespacing=1.3)
    ax.tick_params(axis="x", length=0, pad=3, colors=INK)
    ax.tick_params(axis="y", colors=INK, labelsize=7)
    top = max(max(b["auc"], b["null_hi"]) for b in bars)
    lo = min(min(b["auc"], b["null_lo"]) for b in bars)
    ax.set_ylim(np.floor(lo * 10) / 10, top + 0.06)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.set_ylabel("Within-maze AUC", color=INK)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=NULL_FILL, edgecolor="none", label="shuffle null, 5th–95th percentile")], fontsize=6.5,
              frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2, borderaxespad=0.1)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", choices=list(cfg.LABEL_SCOPES), default=cfg.DEFAULT_LABELS)
    ap.add_argument("--model", choices=cfg.MODELS, default="logreg")
    ap.add_argument("--merged", action="store_true", help="the pooled-session fit: one bar per monkey")
    ap.add_argument("--variant", help="<family>/<variant>/<model> under out/<labels>/; default: the best run")
    args = ap.parse_args()
    args.root = cfg.OUT_ROOT / args.labels

    runs = candidates(args.root, args.merged, args.model)
    print(f"{args.labels} {args.model} {'merged' if args.merged else 'per-session'} runs, ranked by the weaker monkey's z:")
    for zmin, run, sc in runs:
        print(f"  {zmin:+.2f}  {run.relative_to(args.root)}  " +
              "  ".join(f"{m} AUC {sc[m][2]:.3f} p {sc[m][1]:.4f}" for m in cfg.MONKEYS))
    run = args.root / args.variant if args.variant else runs[0][1]
    bars = load(run, args.merged)
    family, variant, model = run.relative_to(args.root).parts
    stem = "merged_auc" if args.merged else "session_auc"
    fig = figure(bars, monkey_scores(run, args.merged), args.merged)
    for ext in ("png", "svg"):
        fig.savefig(run / f"{stem}.{ext}", dpi=300)
    plt.close(fig)
    for b in bars:
        print(f"  {b['monkey']} {b['session']}: AUC {b['auc']:.3f} null 5-95% {b['null_lo']:.3f}-{b['null_hi']:.3f} "
              f"z {b['z']:+.2f} p {b['p']:.4f} ({b['n_H']} H, {b['n_S']} S)")
    print(f"caption: {FAMILY_LABEL[family]}: {VARIANT_LABEL[variant.split('_')[0]]}, {MODEL_LABEL[model]}; "
          f"{bars[0]['n_perm']} shuffles{'' if args.merged else ' per session'}")
    print(f"{run.relative_to(args.root)} -> {run}/{stem}.{{png,svg}}")


if __name__ == "__main__":
    main()
