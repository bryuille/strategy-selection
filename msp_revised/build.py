"""One codebook, every maze: per-maze 2x2s and the pooled estimate, one paper
figure per monkey and variant, one codebook map per monkey, one results.csv.

Usage:
    uv run python -m msp_revised.build                                  # balls_u0.5, binary+dwell
    uv run python -m msp_revised.build --codebook segments_d --kind binary --pupil
    uv run python -m msp_revised.build --codebook balls6_d3 --kind binary dwell first
    uv run python -m msp_revised.build --monkey Faure --n-perm 200      # smoke test
    uv run python -m msp_revised.build --list                           # codebooks

Scope is fixed (config): four publication sessions, dendrogram labels,
mazes 1-6, window maze onset -> fixation onset.

Each maze is scored on its own (labels shuffled within session). ``pooled``
keeps every maze's own 2x2: Δ is the trial-weighted mean of the per-maze Δs
and the null the same mean of the per-maze draws, so no block mean ever
mixes trials from different mazes and a unit-H scale difference between
mazes cannot enter. Mazes with < 2 trials of a strategy cannot be split and
drop out of the pool. ``mean_removed`` centres each maze on its own grand mean.

Writes under out/<codebook>__<kind>[+pupil]/:
    codebook_<Monkey>.png            every fixation centroid, coloured by state
    paper/<variant>/<Monkey>.png     2x2 per maze + pooled, H vs S profiles
    results.csv                      every monkey, variant, maze and pooled row
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

from msp_revised import codebooks as cbmod
from msp_revised import config as cfg
from msp_revised import estimator
from msp_revised import features
from msp_revised import figures as figmod

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def cell_row(result, reason, **base):
    if result is None:
        return {**base, "reason_skipped": reason}
    r = result
    return {
        **base,
        "r_HH": r.q[0, 0], "r_SS": r.q[1, 1], "r_HS": r.q[0, 1], "r_SH": r.q[1, 0],
        "delta": r.delta, "z": r.z, "p": r.p, "p_at_floor": int(r.p_at_floor),
        "null_mean": r.null_mean, "null_sd": r.null_sd,
        "n_H": r.n_H, "n_S": r.n_S, "m_H": r.m_H, "m_S": r.m_S, "d": r.d,
        "n_sessions": r.n_sessions, "n_degenerate": r.n_degenerate,
        "n_rounds": r.n_rounds, "n_perm": r.n_perm, "low_n": int(estimator.low_n(r)),
        "reason_skipped": "",
    }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def strategy_rows(X, y):
    return {"H": X[y == estimator.H], "S": X[y == estimator.S]}


def estimates(data, y_full, X_all, variant, *, mazes, seed, n_rounds, n_perm, min_trials):
    """Per-maze runs and the pooled result for one (codebook, kind, variant)."""
    runs, reasons = {}, {}
    for maze in mazes:
        mask, y, sess = features.pool_maze(data, y_full, maze)
        X = features.apply_variant(X_all[mask], variant)
        run, reason = estimator.run(X, y, sess, seed=estimator.maze_seed(seed, maze),
                                    n_rounds=n_rounds, n_perm=n_perm, min_trials=min_trials)
        runs[maze], reasons[maze] = run, reason
    pooled = estimator.pooled([r for r in runs.values() if r is not None])
    return runs, reasons, pooled


def build_monkey(monkey, cb, args):
    """Estimates for every (kind, variant); writes the codebook map and returns
    ``(paper inputs keyed by (kind, variant), csv rows)``."""
    data = features.load(monkey, refresh=args.refresh)
    lookup = features.label_lookup(cfg.SESSIONS[monkey])
    y_full = features.labels_for(data, lookup)
    states = features.fixation_states(data, cb)
    h_typ = np.median(np.asarray(data["h"], dtype=float), axis=0)
    xy = data["fix_xy_u"] if cb.frame == "unit" else data["fix_xy_d"]
    print(f"\n=== {monkey} [{cb.key}] {len(data['session'])} trials, "
          f"{np.isfinite(y_full).sum()} labelled, {len(states)} fixations, "
          f"{(states >= 0).mean():.1%} assigned ===")

    paper, rows_csv = {}, []
    for kind in args.kind:
        tag = cfg.tag(cb.key, kind, args.pupil)
        fig = figmod.codebook_figure(
            xy, states, codebook=cb, h_typical=h_typ,
            title=f"{monkey} — {cb.label}\nfixation centroids, mazes 1–6 ({cfg.WINDOW_LABEL})")
        figmod.save_figure(fig, f"codebook_{monkey}", out_root=args.out_root, rel_dir=tag, dpi=args.dpi)
        plt.close(fig)

        names = features.feature_names(cb, kind, args.pupil)
        X_all = features.trial_features(data, cb, kind, pupil=args.pupil, states=states)
        profiles, skipped = {}, {}
        for maze in args.maze:
            mask, y, _ = features.pool_maze(data, y_full, maze)
            rows = strategy_rows(X_all[mask], y)
            profiles[maze] = {t: figmod.profile(rows[t]) for t in ("H", "S")}
            skipped[maze] = (len(rows["H"]), len(rows["S"]))

        for variant in args.variant:
            runs, reasons, pooled = estimates(
                data, y_full, X_all, variant, mazes=args.maze, seed=args.seed,
                n_rounds=args.n_rounds, n_perm=args.n_perm, min_trials=args.min_trials)
            base = dict(codebook=cb.key, kind=kind, pupil=int(args.pupil), monkey=monkey, variant=variant)
            used = [m for m in args.maze if runs[m] is not None and np.isfinite(runs[m].result.delta)]
            weights = estimator.weights_for([runs[m] for m in used]) if used else np.empty(0)
            for maze in args.maze:
                run = runs[maze]
                result = None if run is None else run.result
                row = cell_row(result, reasons[maze], **base, maze=maze)
                if maze in used:
                    row["weight"] = f"{weights[used.index(maze)]:.4f}"
                rows_csv.append(row)
                if run is None:
                    print(f"  {kind}/{variant}/maze{maze}: {reasons[maze]}")
                else:
                    print(f"  {kind}/{variant}/maze{maze}: Δ={result.delta:+.4f} z={result.z:+.2f} "
                          f"p={result.p:.4f} n_H={result.n_H} n_S={result.n_S}"
                          + (f" {cfg.LOW_N_MARK}" if estimator.low_n(result) else ""))
            row = cell_row(pooled, "no maze qualified", **base, maze="pooled")
            row["mazes_pooled"] = " ".join(map(str, used))
            rows_csv.append(row)
            if pooled is None:
                print(f"  {kind}/{variant}/pooled: no maze qualified")
            else:
                print(f"  {kind}/{variant}/pooled{used}: Δ={pooled.delta:+.4f} z={pooled.z:+.2f} "
                      f"p={pooled.p:.4f} n_H={pooled.n_H} n_S={pooled.n_S}")
            paper[(kind, variant)] = dict(
                results=[None if runs[m] is None else runs[m].result for m in args.maze],
                pooled_result=pooled, profiles=[profiles[m] for m in args.maze], weights=weights,
                skipped=[skipped[m] for m in args.maze], names=names)
    return paper, rows_csv


def paper_figures(per_monkey, cb, args):
    for kind in args.kind:
        tag = cfg.tag(cb.key, kind, args.pupil)
        for variant in args.variant:
            for monkey, (p, _rows) in per_monkey.items():
                d = p[(kind, variant)]
                fig = figmod.paper_figure(monkey=monkey, mazes=list(args.maze), kind=kind, variant=variant,
                                          codebook_label=cb.label, vmax=args.vmax, **d)
                figmod.save_figure(fig, monkey, out_root=args.out_root, rel_dir=cfg.paper_dir(tag, variant), dpi=args.dpi)
                plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--codebook", default=cbmod.DEFAULT_KEY, choices=list(cbmod.REGISTRY))
    parser.add_argument("--kind", nargs="*", default=["binary", "dwell"], choices=features.KINDS)
    parser.add_argument("--pupil", action="store_true", help="append the two pupil features")
    parser.add_argument("--variant", nargs="*", default=list(cfg.VARIANTS), choices=cfg.VARIANTS)
    parser.add_argument("--monkey", nargs="*", default=list(cfg.MONKEYS), choices=cfg.MONKEYS)
    parser.add_argument("--maze", type=int, nargs="*", default=list(cfg.MAZES), choices=cfg.MAZES)
    parser.add_argument("--n-rounds", type=int, default=estimator.N_ROUNDS)
    parser.add_argument("--n-perm", type=int, default=estimator.N_PERM)
    parser.add_argument("--min-trials", type=int, default=cfg.MIN_TRIALS)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--out-root", type=Path, default=cfg.OUT_ROOT)
    parser.add_argument("--refresh", action="store_true", help="rebuild the fixation-table cache")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--list", action="store_true", help="list codebooks and exit")
    args = parser.parse_args()
    if args.list:
        for key, cb in cbmod.REGISTRY.items():
            print(f"{key:18s} k={cb.k:2d} {cb.frame:4s} {cb.label}")
        return
    args.maze = sorted(set(args.maze))
    cb = cbmod.get(args.codebook)
    per_monkey = {m: build_monkey(m, cb, args) for m in args.monkey}
    paper_figures(per_monkey, cb, args)
    for kind in args.kind:
        tag = cfg.tag(cb.key, kind, args.pupil)
        write_csv(cfg.results_csv(args.out_root, tag),
                  [r for _p, rows in per_monkey.values() for r in rows if r["kind"] == kind])


if __name__ == "__main__":
    main()
