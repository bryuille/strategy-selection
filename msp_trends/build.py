"""msp's 2x2 on the msp_trends codebooks (balls / quads / halves), per maze and pooled.

Every maze runs individually, exactly as in `msp.build` (same estimator,
labels, scope, seeds-per-maze aside), and writes its own figure and csv row.
The pooled row reuses those mazes' null draws: pooled Δ is the trial-weighted
mean of the per-maze Δs, and its null the same mean of the aligned draws. A
separate paper figure per monkey puts mazes 2-5 and the pooled estimate side
by side with the H vs S profile underneath. See `msp_trends.md`.

Usage:
    uv run python -m msp_trends.build --dry-run --all-tags
    uv run python -m msp_trends.build                       # r1
    uv run python -m msp_trends.build --radius 0.5
    uv run python -m msp_trends.build --radius 0.5 --labels dendro
    uv run python -m msp_trends.build --radius 0.5 --codebook quads   # r0.5quads
    uv run python -m msp_trends.build --all-tags
    uv run python -m msp_trends.build --monkey Faure --n-perm 200   # smoke test

Writes under out/<tag>/:
    <Monkey>/codebook.png, <Monkey>/codebook_maze<M>.png
    <Monkey>/<block>/strategy_maze<M>.png
    <Monkey>/<block>/<variant>/maze<M>.png, results.csv
    paper/<block>/<variant>/<Monkey>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from msp import build as msp_build
from msp import config as msp_cfg
from msp import estimator as msp_est
from msp_trends import config as cfg
from msp_trends import estimator
from msp_trends import features
from msp_trends import figures as figmod


def tag_label(spec):
    return f"radius {spec.radius:g}, {cfg.WINDOW_LABEL}"


def labels_label(tag):
    return cfg.LABEL_SCOPES[cfg.parse_tag(tag).scope]["label"]


def strategy_rows(X, y):
    return {"H": X[y == msp_est.H], "S": X[y == msp_est.S]}


def codebook_figures(data, *, monkey, tag, spec, mazes, args):
    cb = cfg.parse_tag(tag).codebook
    fix_xy = np.asarray(data["fix_xy"])
    fs3 = features.merged_fix_state(data, cb)
    fix_maze = np.asarray(data["maze_id"], dtype=int)[np.asarray(data["fix_row"], dtype=int)]
    for maze in [None, *mazes]:
        sel = np.ones(fs3.size, bool) if maze is None else fix_maze == maze
        scope_txt = "mazes 1–6" if maze is None else f"maze {maze}"
        fig = figmod.codebook_figure(
            fix_xy[sel], fs3[sel], codebook=cb, codebook5=data["codebook_xy"],
            radius=spec.radius, lim=msp_cfg.MAZE_SCREEN_LIM,
            title=(
                f"{monkey} — {cb.label}\n{tag_label(spec)}\n"
                f"every in-window fixation centroid, all sessions, {scope_txt}"
            ),
        )
        name = "codebook" if maze is None else f"codebook_maze{maze}"
        figmod.save_figure(fig, name, out_root=args.out_root,
                           rel_dir=cfg.monkey_dir(tag, monkey), dpi=args.dpi)
        plt.close(fig)


def build_monkey(monkey, tag, spec, args):
    """Per-maze figures and csvs for every (block, variant); returns what the
    paper figure needs, keyed by (block, variant)."""
    keep, lookup = features.scope(monkey, tag)
    cb = cfg.parse_tag(tag).codebook
    print(f"\n=== {monkey} [{tag}] ({len(keep)} sessions in scope) ===")
    data = features.load(monkey, tag, refresh=args.refresh)
    codebook_figures(data, monkey=monkey, tag=tag, spec=spec, mazes=args.maze, args=args)

    pooled_in = {}
    for block in args.block:
        X_all = features.merge_block(data, block, cb)
        cells = {}
        for maze in args.maze:
            mask, y, sess = features.pool_maze(
                data, maze=maze, keep_sessions=keep, lookup=lookup
            )
            X = X_all[mask]
            rows = strategy_rows(X, y)
            prof = {t: figmod.profile(rows[t]) for t in ("H", "S")}
            cells[maze] = (X, y, sess, prof)
            fig = figmod.profile_figure(
                prof, codebook=cb, block=block,
                title=f"{monkey} — maze {maze} — {cfg.BLOCK_LABEL[block]}\n"
                      f"n_H = {len(rows['H'])}, n_S = {len(rows['S'])} · mean ± SE",
            )
            figmod.save_figure(fig, f"strategy_maze{maze}", out_root=args.out_root,
                               rel_dir=cfg.block_dir(tag, monkey, block), dpi=args.dpi)
            plt.close(fig)

        for variant in args.variant:
            runs, results, rows_csv = [], [], []
            for maze in args.maze:
                X, y, sess, _prof = cells[maze]
                run, reason = estimator.run_maze(
                    features.apply_variant(X, variant), y, sess,
                    seed=estimator.maze_seed(args.seed, maze),
                    n_rounds=args.n_rounds, n_perm=args.n_perm,
                    min_trials=args.min_trials,
                )
                result = None if run is None else run.result
                row = msp_build.cell_row(result, reason, monkey=monkey, variant=variant,
                                         maze=maze, radius=tag)
                row["low_n"] = int(estimator.low_n(result))
                if run is not None:
                    row["n_null_undefined"] = run.n_null_undefined
                rows_csv.append({"block": block, **row})
                results.append(result)
                target = cfg.figure_png(args.out_root, tag, monkey, block, variant, maze)
                if run is None:
                    print(f"  {block}/{variant}/maze{maze}: {reason}")
                    if target.exists():
                        target.unlink()
                        print(f"    removed stale {target}")
                    continue
                if np.isfinite(result.delta):
                    runs.append(run)  # an undefined observed score cannot be pooled
                fig = figmod.panel_figure(
                    result, codebook=cb, vmax=args.vmax,
                    title=f"{monkey} — maze {maze} — {cfg.BLOCK_LABEL[block]}\n"
                          f"{variant} · {tag_label(spec)} · {labels_label(tag)}",
                )
                figmod.save_figure(fig, cfg.stem(maze), out_root=args.out_root,
                                   rel_dir=cfg.variant_dir(tag, monkey, block, variant),
                                   dpi=args.dpi, bbox_inches="tight", pad_inches=0.08)
                plt.close(fig)
                print(f"    {block}/{variant}/maze{maze}: Δ={result.delta:+.4f} "
                      f"z={result.z:+.2f} p={result.p:.4f} n_H={result.n_H} n_S={result.n_S}"
                      + (f" {cfg.LOW_N_MARK}" if estimator.low_n(result) else ""))

            pooled = estimator.pooled(runs)
            weights = estimator.weights_for(runs) if runs else np.empty(0)
            used = [m for m, r in zip(args.maze, results)
                    if r is not None and np.isfinite(r.delta)]
            if pooled is None:
                prow = {"block": block, "monkey": monkey, "variant": variant,
                        "maze": cfg.POOLED, "radius": tag,
                        "reason_skipped": "no maze qualified"}
            else:
                prow = {"block": block, **msp_build.cell_row(
                    pooled, "", monkey=monkey, variant=variant, maze=cfg.POOLED, radius=tag)}
                prow["mazes_pooled"] = " ".join(map(str, used))
                prow["n_null_undefined"] = int(sum(
                    (~np.isfinite(r.draws)) for r in runs).astype(bool).sum())
                prow["weight"] = " ".join(f"{w:.4f}" for w in weights)
                print(f"    {block}/{variant}/pooled{used}: Δ={pooled.delta:+.4f} "
                      f"z={pooled.z:+.2f} p={pooled.p:.4f}")
            for row, maze in zip(rows_csv, args.maze):
                if maze in used:
                    row["weight"] = f"{weights[used.index(maze)]:.4f}"
            msp_build.write_csv(
                cfg.results_csv(args.out_root, tag, monkey, block, variant),
                rows_csv + [prow],
            )
            pooled_in[(block, variant)] = dict(
                results=results, pooled_result=pooled, weights=weights,
                profiles=[cells[m][3] for m in args.maze],
                skipped=[strategy_rows(cells[m][0], cells[m][1]) for m in args.maze],
            )
    return pooled_in


def paper_figures(per_monkey, tag, spec, args):
    for block in args.block:
        for variant in args.variant:
            for monkey, pin in per_monkey.items():
                p = pin[(block, variant)]
                fig = figmod.paper_figure(
                    monkey=monkey, mazes=list(args.maze), block=block, variant=variant,
                    tag_label=tag_label(spec), labels=labels_label(tag),
                    trend=cfg.TREND[monkey], codebook=cfg.parse_tag(tag).codebook,
                    vmax=args.vmax,
                    **p,
                )
                rel = cfg.paper_dir(tag, block, variant)
                figmod.save_figure(fig, monkey, out_root=args.out_root, rel_dir=rel,
                                   dpi=args.dpi)
                plt.close(fig)


def dry_run(tag, args):
    for monkey in args.monkey:
        keep, lookup = features.scope(monkey, tag)
        cb = cfg.parse_tag(tag).codebook
        data = features.load(monkey, tag, refresh=args.refresh)
        print(f"\n{monkey} [{tag}]  cache rows {len(data['session'])}, {len(keep)} sessions")
        for block in args.block:
            X_all = features.merge_block(data, block, cb)
            for maze in args.maze:
                mask, y, sess = features.pool_maze(
                    data, maze=maze, keep_sessions=keep, lookup=lookup
                )
                n_h, n_s, m_h, m_s, reason = msp_est.coverage(y, min_trials=args.min_trials)
                rows = strategy_rows(X_all[mask], y)
                fmt = lambda v: "/".join(f"{x:.2f}" for x in v)
                h = rows["H"].mean(axis=0) if len(rows["H"]) else np.full(cb.k, np.nan)
                s = rows["S"].mean(axis=0) if len(rows["S"]) else np.full(cb.k, np.nan)
                zero = float((X_all[mask].sum(axis=1) == 0).mean()) if mask.any() else np.nan
                print(f"  {block:6s} maze {maze}: n_H={n_h:4d} n_S={n_s:4d}  "
                      f"{'/'.join(cb.names)}  H {fmt(h)}  S {fmt(s)}  all-zero {zero:.1%}"
                      + (f"  SKIPPED ({reason})" if reason else "")
                      + (f"  {cfg.LOW_N_MARK}" if not reason and min(n_h, n_s) <= cfg.LOW_N else ""))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--monkey", nargs="*", default=list(cfg.MONKEYS), choices=cfg.MONKEYS)
    parser.add_argument("--maze", type=int, nargs="*", default=list(cfg.MAZES), choices=cfg.MAZES)
    parser.add_argument("--block", nargs="*", default=list(cfg.BLOCK_NAMES), choices=cfg.BLOCK_NAMES)
    parser.add_argument("--variant", nargs="*", default=list(cfg.VARIANTS), choices=cfg.VARIANTS)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument(
        "--labels", default=cfg.DEFAULT_LABELS, choices=list(cfg.LABEL_SCOPES),
        help="svm: SVM labels, top-ten sessions; dendro: dendrogram labels, "
             "publication sessions (tag suffix _dendro)",
    )
    parser.add_argument(
        "--codebook", default=cfg.DEFAULT_CODEBOOK, choices=list(cfg.CODEBOOKS),
        help="balls: merged exit balls; quads / halves: origin ball + maze "
             "quadrants / halves (tag suffix quads / halves)",
    )
    parser.add_argument("--all-tags", action="store_true",
                        help=f"run every tag in config.TAGS: {', '.join(cfg.TAGS)}")
    parser.add_argument("--n-rounds", type=int, default=msp_est.N_ROUNDS)
    parser.add_argument("--n-perm", type=int, default=msp_est.N_PERM)
    parser.add_argument("--min-trials", type=int, default=cfg.MIN_TRIALS,
                        help="per strategy; below this the split-half is impossible")
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--out-root", type=Path, default=cfg.OUT_ROOT)
    parser.add_argument("--refresh", action="store_true", help="rebuild the msp feature cache")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.maze = sorted(set(args.maze))

    if args.all_tags:
        tags = list(cfg.TAGS)
    else:
        radius = msp_cfg.ASSIGN_RADIUS if args.radius is None else args.radius
        tags = [cfg.make_tag(radius, args.codebook, args.labels)]

    for tag in tags:
        spec = features.assignment_for(tag)
        if args.dry_run:
            dry_run(tag, args)
            continue
        per_monkey = {m: build_monkey(m, tag, spec, args) for m in args.monkey}
        paper_figures(per_monkey, tag, spec, args)


if __name__ == "__main__":
    main()
