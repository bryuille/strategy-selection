"""Fit and evaluate both branches for both monkeys.

    uv run python -m classifier.build                      # both branches, both models
    uv run python -m classifier.build --branch engineered --model logreg
    uv run python -m classifier.build --branch regions     # dwell share + visited over nine regions
    uv run python -m classifier.build --labels svm         # top-ten sessions, SVM labels, mazes 2-5
    uv run python -m classifier.build --n-perm 20          # smoke test
    uv run python -m classifier.build --order-control      # trial-order covariates, conditional permutation null
    uv run python -m classifier.build --branch order       # trial order alone (the confound's size)

Writes out/<branch>[_svm]/<model>/<Monkey>.png (within-maze AUC per maze and
pooled against the shuffle null; feature weights), results.csv (per monkey
and maze, per held-out session, and pooled) and features.csv (weights /
importances).
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
from classifier import features as F  # noqa: E402
from classifier import model as M  # noqa: E402


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def figure(monkey, branch, kind, ev, weights, names, signs=None, *, scope_label=""):
    per = ev.per_maze
    mazes = sorted(per)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.4), layout="constrained",
                                   gridspec_kw=dict(width_ratios=[1.15, 1.4]))
    # left: AUC per maze + pooled, with the null band
    labels = [f"maze {m}\n{per[m][1]} H\n{per[m][2]} S" for m in mazes] + ["pooled\n(within\nmaze)"]
    vals = [per[m][0] for m in mazes] + [ev.auc]
    fin = ev.null[np.isfinite(ev.null)]
    lo, hi = np.percentile(fin, [2.5, 97.5]) if fin.size else (np.nan, np.nan)
    ax1.axhspan(lo, hi, color="#e6e6e6", zorder=0)
    ax1.axhline(0.5, color="0.4", lw=0.8)
    # red / blue are reserved for S / H (right panel); these are maze-status colours
    SOLID, THIN, POOLED, NULL = "#2a9d8f", "#e9c46a", "#7b4fb5", "#e6e6e6"
    colours = [THIN if min(per[m][1], per[m][2]) < cfg.SOLID_N else SOLID for m in mazes] + [POOLED]
    ax1.bar(range(len(vals)), vals, color=colours)
    from matplotlib.patches import Patch
    handles = [Patch(color=NULL, label="shuffle null 95%"),
               Patch(color=SOLID, label=f"maze with ≥ {cfg.SOLID_N} of each strategy"),
               Patch(color=THIN, label=f"maze with < {cfg.SOLID_N} of one strategy"),
               Patch(color=POOLED, label="within-maze pooled")]
    ax1.set_xticks(range(len(vals)), labels=labels, fontsize=7.5)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlim(-0.6, len(vals) - 0.4)
    ax1.set_ylabel("within-maze AUC (out of fold)")
    ho = holdout_text(ev)
    ax1.set_title(f"within-maze AUC = {ev.auc:.3f}   z = {ev.z:+.2f}   p = {ev.p:.3f}\n"
                  f"(ordinary AUC incl. maze = {ev.pooled_auc:.3f})\n{ho}", fontsize=9)
    ax1.legend(handles=handles, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
               frameon=False)
    # right: weights
    top = sorted(weights.items(), key=lambda kv: -abs(kv[1]))[:20][::-1]
    signs = signs or weights
    ax2.barh([k for k, _ in top], [v for _, v in top],
             color=["#1f77b4" if signs.get(k, v) < 0 else "#d62728" for k, v in top])
    ax2.axvline(0, color="0.3", lw=0.8)
    ax2.set_xlabel("standardised logistic coefficient (red: + → S, blue: − → H)" if kind == "logreg"
                   else "permutation importance (drop in within-maze AUC; red where the fitted sign is + → S)")
    ax2.tick_params(labelsize=7)
    ax2.set_title(f"top features of {len(names)}", fontsize=9)
    order = ", trial order in the design, null = gaze residuals shuffled given order" if M.ORDER_CONTROL else ""
    fig.suptitle(f"{monkey} — {branch} features{' + gaze x maze interactions' if M.INTERACTIONS else ''} — {kind}\n"
                 f"{scope_label}, maze one-hot in every model{order}, "
                 f"{cfg.N_REPEATS}x{cfg.N_FOLDS}-fold CV stratified on (session, maze, label)", fontsize=10)
    return fig


def holdout_text(ev):
    """Each held-out session (trained on the others), or for many sessions
    the pooled leave-one-session-out AUC and the per-session range."""
    ho = ev.session_holdout
    if not ho:
        return "no held-out-session fit"
    short = lambda s: s.split("_g0")[0]
    if len(ho) <= 2:
        return "; ".join(f"train {short(a)} → test {short(b)}: {v:.2f}" if len(ho) == 2 else f"held-out {short(b)}: {v:.2f}"
                         for b, v in ho.items() for a in [next((k for k in ho if k != b), b)])
    vals = np.asarray(list(ho.values()), dtype=float)
    return (f"leave-one-session-out ({len(ho)} sessions): pooled {ev.holdout_auc:.2f}, "
            f"per session {np.nanmin(vals):.2f}–{np.nanmax(vals):.2f}")


def rows_from_eval(monkey, branch, kind, out):
    """results.csv and features.csv rows rebuilt from eval_<Monkey>.npz."""
    d = np.load(out / f"eval_{monkey}.npz", allow_pickle=True)
    rows = [dict(monkey=monkey, branch=branch, model=kind, maze=int(m), auc=float(a), n_H=int(nh), n_S=int(ns))
            for m, a, nh, ns in d["per_maze"]]
    ho, ho_auc = _holdout_from_npz(d)
    rows += holdout_rows(monkey, branch, kind, ho, ho_auc)
    null = d["null"][np.isfinite(d["null"])]
    y = d["y"]
    rows.append(dict(monkey=monkey, branch=branch, model=kind, maze="within-maze pooled", auc=float(d["auc"]),
                     null_mean=null.mean(), null_sd=null.std(ddof=1), z=float(d["z"]), p=float(d["p"]),
                     ordinary_auc=float(d["pooled_auc"]), n_H=int((y == 0).sum()), n_S=int((y == 1).sum()),
                     n_features=len(d["names"]), n_perm=int(null.size)))
    frows = [dict(monkey=monkey, branch=branch, model=kind, feature=str(k), weight=float(v))
             for k, v in zip(d["names"], d["weights"])]
    return rows, frows


def _holdout_from_npz(d):
    """``({held-out session: AUC}, pooled AUC)``; reads the retired
    ``(train, test, auc)`` triples of older eval files too."""
    ho = np.asarray(d["holdout"], dtype=object).reshape(-1, d["holdout"].shape[-1]) if d["holdout"].size else []
    if len(ho) and len(ho[0]) == 3:
        per = {str(b): float(v) for _, b, v in ho}
    else:
        per = {str(b): float(v) for b, v in ho}
    return per, float(d["holdout_auc"]) if "holdout_auc" in d.files else np.nan


def holdout_rows(monkey, branch, kind, ho, ho_auc):
    rows = [dict(monkey=monkey, branch=branch, model=kind, maze=f"held-out {b}", auc=float(v))
            for b, v in ho.items()]
    if np.isfinite(ho_auc):
        rows.append(dict(monkey=monkey, branch=branch, model=kind, maze="leave-one-session-out pooled",
                         auc=float(ho_auc)))
    return rows


def replot(monkey, branch, kind, out, names, *, scope_label=""):
    """Rebuild the figure from eval_<Monkey>.npz."""
    d = np.load(out / f"eval_{monkey}.npz", allow_pickle=True)
    per = {int(m): (float(a), int(nh), int(ns)) for m, a, nh, ns in d["per_maze"]}
    holdout, holdout_auc = _holdout_from_npz(d)
    ev = M.Evaluation(float(d["auc"]), per, d["null"], float(d["z"]), float(d["p"]), d["oof"],
                      float(d["pooled_auc"]), holdout, holdout_auc)
    names = list(d["names"])
    return figure(monkey, branch, kind, ev, dict(zip(names, d["weights"])), names, dict(zip(names, d["signs"])),
                  scope_label=scope_label)


def out_branch_name(branch, bin_ms, *, labels=cfg.DEFAULT_LABELS, interactions=False, order_control=False):
    """``raw`` / ``raw_b<ms>`` / ``engineered``, then ``_svm`` for that scope,
    then ``_x_maze`` with interactions, then ``_order`` with order control."""
    name = branch
    if branch == "raw" and bin_ms != cfg.RAW_BIN_MS:
        name = f"raw_b{bin_ms:g}"
    tag = cfg.LABEL_SCOPES[labels]["tag"]
    if tag:
        name += f"_{tag}"
    if interactions:
        name += "_x_maze"
    if order_control:
        name += "_order"
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branch", nargs="*", default=["raw", "engineered", "regions"],
                    choices=["raw", "engineered", *F.REGIONS_KINDS, "order"],
                    help="raw traces; 26 engineered features; `regions` = per-region dwell share + visited "
                         "over origin-disc quadrants, exit balls and elsewhere (`regions_dwell` / "
                         "`regions_binary` for one set); `order` = trial order alone")
    ap.add_argument("--model", nargs="*", default=list(cfg.MODELS), choices=cfg.MODELS)
    ap.add_argument("--monkey", nargs="*", default=list(cfg.MONKEYS), choices=cfg.MONKEYS)
    ap.add_argument("--n-perm", type=int, default=cfg.N_PERM)
    ap.add_argument("--out-root", type=Path, default=cfg.OUT_ROOT)
    ap.add_argument("--bin-ms", type=float, default=cfg.RAW_BIN_MS,
                    help="raw branch: trace bin width in ms (1 = the whole 1 kHz trace); "
                         "non-default widths go to out/raw_b<ms>/")
    ap.add_argument("--labels", choices=list(cfg.LABEL_SCOPES), default=cfg.DEFAULT_LABELS,
                    help="label scope: `dendro` = dendrogram labels on the publication sessions, mazes 1-6 "
                         "(default); `svm` = SVM labels on the top-ten sessions, mazes 2-5 (out/<branch>_svm/)")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--interactions", action="store_true",
                    help="add gaze x maze interaction terms (per-maze weights shrunk toward shared ones); "
                         "output goes to out/<branch>_x_maze/")
    ap.add_argument("--order-control", action="store_true",
                    help="control trial order: order covariates in the design, and a null that keeps labels "
                         "and each feature's order trend and shuffles only its residual within (session, maze); "
                         "output out/<branch>[_svm]_order/")
    ap.add_argument("--replot", action="store_true",
                    help="redraw figures from the saved eval_<Monkey>.npz without refitting")
    args = ap.parse_args()

    M.INTERACTIONS = bool(args.interactions)
    M.ORDER_CONTROL = bool(args.order_control)
    scope_label = cfg.LABEL_SCOPES[args.labels]["label"]
    refreshed = set()  # (cache, monkey) already rebuilt under --refresh: once, not once per branch
    for branch in args.branch:
        cache = "raw" if branch == "raw" else "fix"
        data = {m: F.dataset(m, branch, labels=args.labels, bin_ms=args.bin_ms,
                             refresh=args.refresh and (cache, m) not in refreshed) for m in args.monkey}
        refreshed.update((cache, m) for m in args.monkey)
        out_branch = out_branch_name(branch, args.bin_ms, labels=args.labels,
                                     interactions=args.interactions, order_control=args.order_control)
        for kind in args.model:
            rows, frows = [], []
            for monkey in args.monkey:
                X, names, meta = data[monkey]
                y = meta["y"]
                out = args.out_root / out_branch / kind
                out.mkdir(parents=True, exist_ok=True)
                if args.replot:
                    fig = replot(monkey, out_branch, kind, out, names, scope_label=scope_label)
                    fig.savefig(out / f"{monkey}.png", dpi=args.dpi)
                    plt.close(fig)
                    r_, f_ = rows_from_eval(monkey, out_branch, kind, out)
                    rows += r_
                    frows += f_
                    continue
                print(f"\n=== {monkey} / {out_branch} / {kind}: {len(y)} trials ({(y == 0).sum()} H, {(y == 1).sum()} S), "
                      f"{np.unique(meta['session']).size} sessions, {X.shape[1]} features ===", flush=True)
                ev = M.evaluate(X, y, meta, kind, n_perm=args.n_perm)
                signs = M.coefficients(X, y, meta, names)
                weights = signs if kind == "logreg" else M.gbm_importance(X, y, meta, names)
                fin = ev.null[np.isfinite(ev.null)]
                print(f"  within-maze AUC {ev.auc:.3f} (null mean {fin.mean():.3f} sd {fin.std(ddof=1):.3f}) "
                      f"z={ev.z:+.2f} p={ev.p:.3f}; ordinary AUC {ev.pooled_auc:.3f}")
                for m, (a, nh, ns) in sorted(ev.per_maze.items()):
                    print(f"    maze {m}: AUC {a:.3f} ({nh} H, {ns} S)")
                    rows.append(dict(monkey=monkey, branch=branch, model=kind, maze=m, auc=a, n_H=nh, n_S=ns))
                for b, v in ev.session_holdout.items():
                    print(f"    held-out {b}: within-maze AUC {v:.3f}")
                if len(ev.session_holdout) > 2:
                    print(f"    leave-one-session-out pooled: {ev.holdout_auc:.3f}")
                rows += holdout_rows(monkey, branch, kind, ev.session_holdout, ev.holdout_auc)
                rows.append(dict(monkey=monkey, branch=branch, model=kind, maze="within-maze pooled", auc=ev.auc,
                                 null_mean=fin.mean(), null_sd=fin.std(ddof=1), z=ev.z, p=ev.p,
                                 ordinary_auc=ev.pooled_auc, n_H=int((y == 0).sum()), n_S=int((y == 1).sum()),
                                 n_features=X.shape[1], n_perm=args.n_perm))
                frows += [dict(monkey=monkey, branch=branch, model=kind, feature=k, weight=v) for k, v in weights.items()]
                np.savez(out / f"eval_{monkey}.npz", auc=ev.auc, z=ev.z, p=ev.p, pooled_auc=ev.pooled_auc,
                         null=ev.null, oof=ev.oof, y=y, maze=meta["maze"], session=meta["session"],
                         per_maze=np.array([(m, a, nh, ns) for m, (a, nh, ns) in sorted(ev.per_maze.items())]),
                         holdout=np.array([(b, v) for b, v in ev.session_holdout.items()], dtype=object),
                         holdout_auc=ev.holdout_auc, labels=args.labels,
                         names=np.array(names), weights=np.array([weights[n] for n in names]),
                         signs=np.array([signs[n] for n in names]))
                fig = figure(monkey, out_branch, kind, ev, weights, names, signs, scope_label=scope_label)
                fig.savefig(out / f"{monkey}.png", dpi=args.dpi)
                plt.close(fig)
            write_csv(args.out_root / out_branch / kind / "results.csv", rows)
            write_csv(args.out_root / out_branch / kind / "features.csv", frows)


if __name__ == "__main__":
    main()
