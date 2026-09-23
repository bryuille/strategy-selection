"""Maze-strategy pairs at a K chosen per maze from held-out data.

`comp_pooled_build.py` sweeps K over 13 values and reports `best_k` = the K
minimising `p`. That picks the codebook size *on the outcome statistic*, so the
reported `p` is a minimum over 13 correlated tests with no correction. This
module replaces that selection with a label-blind one: `elbow_select` splits
each maze's trials into train and test, takes the elbow of the held-out
explained-variance curve, and fixes K there **before** `estimator.estimate` is
called even once. The resulting `p` needs no correction over K.

Everything downstream of the selection is `comp_pooled_build`'s pattern,
unchanged: one codebook per maze, every in-scope session's trials of that maze
pooled into a single estimate, `estimator.estimate` and `variants.apply_variant`
reused verbatim. `build.py`, `comp_build.py`, `comp_pooled_build.py`,
`estimator.py`, `variants.py` and `classifier.features` are all left completely
alone; this module and its `elbow_*` siblings are new files writing to a new
`out/elbow/` tree and their own cache stem.

Three things follow and are worth knowing before reading any output:

* **The codebook is scope-restricted.** `comp_pooled_features` fits label-blind
  over every session that clears QC (up to 27 for Faure); this fits on the
  in-scope sessions only, so the selected K sizes the population it is applied
  to. The consequence is that `p` here is **not** comparable cell-for-cell to
  `comp_pooled`'s: the two rest on codebooks fitted from different pools.
* **Fixing K does not make this one test.** 4 mazes x 2 features x 3 variants
  is 24 cells. The pre-committed primary cell and the correction family over
  mazes are stated in `elbow.md`, which must be read first -- without them the
  multiplicity objection simply moves from K to the other axes.
* **The elbow may not be identified.** `selected_k.csv` carries the modal
  kneedle pick across the individual splits and how many distinct values they
  produced. If that spread is wide, "the elbow" is not a measurement and no
  amount of downstream significance changes that.

Usage:
    uv run python -m eye_pre_flash.maze_strategy_pairs.elbow_build --dry-run
    uv run python -m eye_pre_flash.maze_strategy_pairs.elbow_build
    uv run python -m eye_pre_flash.maze_strategy_pairs.elbow_build --monkey Faure --maze 5

Writes out/elbow/<source>/<monkey>/{ev_curves.png,selected_k.csv,
maze<M>_<feature>_<variant>.png,results.csv} plus a summary_<monkey>.md.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eye_pre_flash.classifier.labels import labels_for_rows
from eye_pre_flash.label_sources import source_lookup, source_scope_sessions
from eye_pre_flash.maze_strategy_pairs import elbow_features
from eye_pre_flash.maze_strategy_pairs import elbow_figures as elbowfig
from eye_pre_flash.maze_strategy_pairs import elbow_paths as elbowpath
from eye_pre_flash.maze_strategy_pairs import elbow_select
from eye_pre_flash.maze_strategy_pairs import estimator
from eye_pre_flash.maze_strategy_pairs import figures as figmod
from eye_pre_flash.maze_strategy_pairs import variants as variantmod
from eye_pre_flash.plotting.plot_io import save_figure

FEATURES = {"occupancy": "occ_ms", "occupancy_bin": "occ_bin"}
MONKEYS = ("Faure", "Nielsen")

# Mazes 2-5, matching `comp_pooled_build`. Maze 1 and 6 are the SVM
# classifier's own training axis and its cells there are artifacts.
MAZES = (2, 3, 4, 5)

# SVM labels only, at their one scope -- same as both comparison modules.
SOURCE_SCOPES_WANTED = (("svm", "top_ten"),)


@dataclass
class Panel:
    """What `figures.pair_figure` needs: a K and a `Result`."""

    k: int
    result: object = None


def select_for_monkey(monkey, *, keep_sessions, args):
    """``(k_by_maze, curves, selections, stability)`` for one monkey.

    The pool is built once and reused by both the EV sweep and the stability
    sweep, since it is the attractor scan that costs anything.
    """
    print(f"\n=== {monkey}: selecting K per maze ===")
    pools = elbow_select.maze_pools(
        monkey, keep_sessions=keep_sessions, mazes=args.maze
    )

    k_by_maze, curves, selections, stability = {}, {}, {}, {}
    for maze in args.maze:
        trial_pools = pools.get(maze, [])
        curve, reason = elbow_select.ev_curve(
            trial_pools, args.k_grid, n_splits=args.n_splits, seed=args.seed,
            with_stability=not args.no_stability,
        )
        if curve is None:
            print(f"  maze {maze}: no EV curve ({reason})")
            curves[maze] = None
            continue
        curves[maze] = curve
        selections[maze] = elbow_select.select_k(curve)
        if curve["stability"] is not None:
            stability[maze] = curve["stability"]

        sel = selections[maze]
        if sel["k"] is None:
            print(f"  maze {maze}: EV curve is flat; no knee. Maze skipped.")
            continue
        k_by_maze[maze] = sel["k"]
        edge = "  AT GRID EDGE" if sel["at_grid_edge"] else ""
        print(
            f"  maze {maze}: K={sel['k']} (gain-rule {sel['k_marginal_gain']}, "
            f"curvature {sel['k_max_curvature']}); modal-per-split "
            f"{sel['k_modal_per_split']} at {sel['modal_share']:.0%} over "
            f"{sel['n_distinct_per_split']} distinct values; "
            f"{len(trial_pools)} trials{edge}"
        )
    return k_by_maze, curves, selections, stability


def selection_rows(monkey, curves, selections, stability, *, source, scope):
    """One row per maze: the three rules, the diagnostics, the EV curve itself."""
    rows = []
    for maze in sorted(curves):
        curve, sel = curves[maze], selections.get(maze)
        base = dict(monkey=monkey, source=source, scope=scope, maze=maze)
        if curve is None or sel is None:
            rows.append({**base, "reason_skipped": "no EV curve"})
            continue
        ks = np.asarray(curve["ks"], dtype=int)
        knee = sel["k"]
        j = int(np.flatnonzero(ks == knee)[0]) if knee is not None else None
        st = stability.get(maze)
        row = {
            **base,
            "k": "" if knee is None else knee,
            "k_marginal_gain": sel["k_marginal_gain"] or "",
            "k_max_curvature": sel["k_max_curvature"] or "",
            "k_modal_per_split": sel["k_modal_per_split"] or "",
            "modal_share": f"{sel['modal_share']:.3f}",
            "n_distinct_per_split": sel["n_distinct_per_split"],
            "at_grid_edge": int(sel["at_grid_edge"]),
            "ev_at_k": "" if j is None else f"{curve['ev'][j]:.4f}",
            "ev_sd_at_k": "" if j is None else f"{curve['ev_sd'][j]:.4f}",
            "assigned_fraction_at_k": (
                "" if j is None else f"{curve['assigned_fraction'][j]:.4f}"
            ),
            "sse_per_n_at_k": "" if j is None else f"{curve['sse_per_n'][j]:.5f}",
            "stability_at_k": (
                "" if (j is None or st is None) else f"{float(st[j]):.4f}"
            ),
            "reason_skipped": "",
        }
        # The whole curve, so a reader can redo any rule without a rerun.
        for i, k in enumerate(ks.tolist()):
            row[f"ev_K{k}"] = f"{curve['ev'][i]:.4f}"
            row[f"assigned_K{k}"] = f"{curve['assigned_fraction'][i]:.4f}"
            if st is not None:
                row[f"stability_K{k}"] = f"{float(st[i]):.4f}"
        rows.append(row)
    return rows


def run_cell(data, *, maze, feature, variant, method, lookup, args):
    """One pooled estimate at this maze's selected K."""
    mask, raw = elbow_features.maze_block(data, maze, FEATURES[feature])
    if not mask.any():
        return None, "no trial in this maze", None

    sessions = np.asarray(data["session"]).astype(str)[mask]
    trials = np.asarray(data["trial_indices_all"], dtype=int)[mask]
    y_full = labels_for_rows(sessions, trials, lookup)
    labelled = np.isfinite(y_full)
    if not labelled.any():
        return None, "no labelled trial in this maze", None

    codebook = elbow_features.maze_codebook(data, maze)
    if codebook is None:
        return None, "codebook did not fit", None

    X_raw = raw[labelled]
    y = y_full[labelled].astype(int)
    session_ids = sessions[labelled]
    k = int(codebook.shape[0])

    # Per-maze by necessity: each maze has its own codebook, so `no_origin`'s
    # dropped dimension and `mean_removed`'s grand mean must come from this
    # maze's own centres and its own labelled rows. No wider fit shares these
    # dimensions, and a narrower one would reintroduce per-session averaging.
    mean_profile = variantmod.fit_grand_mean(X_raw) if variant == "mean_removed" else None
    X, _dims = variantmod.apply_variant(
        X_raw,
        variant,
        feature=FEATURES[feature],
        k=k,
        origin=variantmod.origin_state(codebook),
        mean_profile=mean_profile,
    )

    result, reason = estimator.estimate(
        X, y, session_ids,
        method=method,
        n_rounds=args.n_rounds,
        n_perm=args.n_perm,
        seed=args.seed,
        min_trials=args.min_trials,
    )
    return result, reason, k


def cell_row(result, reason, k, *, method, monkey, source, scope, maze, feature, variant):
    base = dict(
        method=method, monkey=monkey, source=source, scope=scope,
        variant=variant, feature=feature, maze=maze, k="" if k is None else k,
    )
    if result is None:
        return {**base, "reason_skipped": reason}
    r = result
    return {
        **base,
        "r_HH": r.q[estimator.H, estimator.H],
        "r_SS": r.q[estimator.S, estimator.S],
        "r_HS": r.q[estimator.H, estimator.S],
        "r_SH": r.q[estimator.S, estimator.H],
        "delta": r.delta,
        "z": r.z,
        "p": r.p,
        "p_at_floor": int(r.p_at_floor),
        "null_mean": r.null_mean,
        "null_sd": r.null_sd,
        "n_H": r.n_H,
        "n_S": r.n_S,
        "m": r.m,
        "d": r.d,
        "n_sessions": r.n_sessions,
        "n_degenerate": r.n_degenerate,
        "n_rounds": r.n_rounds,
        "n_perm": r.n_perm,
        "reason_skipped": "",
    }


def write_csv(path, rows):
    """Union of keys as the header, first-seen order, so a partial row is fine."""
    if not rows:
        return None
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")
    return path


def write_summary_md(path, rows, *, monkey, source, mazes):
    """One block per (feature, variant): maze, selected K, and the estimate."""
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Elbow-selected K - {monkey} / {source}",
        "",
        "K chosen per maze from the held-out explained-variance elbow, with",
        "the H/S labels never consulted, and fixed before this test was run --",
        "so `p` needs no correction over K. It is still one of 24 cells across",
        "maze, feature and variant: read `elbow.md` for the",
        "pre-committed primary cell and the correction family over mazes.",
        "",
        "`p` here is **not** comparable to `out/comp_pooled/`'s: that codebook",
        "is fitted label-blind over every QC-passing session, this one over the",
        "in-scope sessions only.",
        "",
    ]
    by_block = defaultdict(list)
    for row in rows:
        by_block[(row["feature"], row["variant"])].append(row)

    for (feature, variant), block in sorted(by_block.items()):
        lines += [
            f"## {feature} / {variant}",
            "",
            "| maze | K | r_HH | r_SS | r_HS | Δ | z | p | n_H | n_S |",
            "|" + "---|" * 10,
        ]
        for row in sorted(block, key=lambda r: r["maze"]):
            if row.get("reason_skipped"):
                lines.append(
                    f"| {row['maze']} | {row.get('k', '')} | "
                    + "| ".join([""] * 7)
                    + f" {row['reason_skipped']} |"
                )
                continue
            p = f"{row['p']:.4f}"
            if row["p_at_floor"]:
                p = f"~{p}"
            z = "—" if not np.isfinite(row["z"]) else f"{row['z']:+.2f}"
            lines.append(
                f"| {row['maze']} | {row['k']} | {row['r_HH']:.3f} | "
                f"{row['r_SS']:.3f} | {row['r_HS']:.3f} | {row['delta']:+.4f} | "
                f"{z} | {p} | {row['n_H']} | {row['n_S']} |"
            )
        lines.append("")

    path.write_text("\n".join(lines))
    print(f"Saved {path}")
    return path


def sweep(args):
    """Select K, build features once per monkey, then estimate every cell."""
    rows_by_tree = defaultdict(list)

    for monkey in args.monkey:
        for source, scope in SOURCE_SCOPES_WANTED:
            keep_sessions = tuple(
                source_scope_sessions(source, scope)[0].get(monkey, ())
            )
            if not keep_sessions:
                print(f"{monkey}/{source}/{scope}: no sessions in scope; skipping")
                continue
            lookup = source_lookup(source, keep_sessions)

            k_by_maze, curves, selections, stability = select_for_monkey(
                monkey, keep_sessions=keep_sessions, args=args
            )

            write_csv(
                elbowpath.selected_k_csv(args.out_root, source, monkey),
                selection_rows(
                    monkey, curves, selections, stability, source=source, scope=scope
                ),
            )
            if any(c is not None for c in curves.values()):
                fig = elbowfig.ev_figure(
                    curves, selections,
                    title=elbowfig.ev_suptitle(
                        monkey=monkey, source=source,
                        n_sessions=len(keep_sessions), n_splits=args.n_splits,
                        marginal_gain=elbow_select.MARGINAL_GAIN,
                    ),
                    stability=stability or None,
                )
                save_figure(
                    fig, elbowpath.EV_STEM,
                    out_root=args.out_root,
                    rel_dir=elbowpath.select_dir(source, monkey),
                    dpi=args.dpi,
                )
                plt.close(fig)

            if not k_by_maze:
                print(f"  {monkey}: no maze produced a usable K; nothing to estimate")
                continue

            data = elbow_features.load_features(
                monkey, k_by_maze=k_by_maze, keep_sessions=keep_sessions,
                refresh=args.refresh,
            )
            if len(data["session"]) == 0:
                print("  no usable trials; skipping")
                continue

            for method in args.methods:
                for maze in sorted(k_by_maze):
                    for feature in args.feature:
                        for variant in args.variant:
                            result, reason, k = run_cell(
                                data, maze=maze, feature=feature, variant=variant,
                                method=method, lookup=lookup, args=args,
                            )
                            rows_by_tree[(method, source, scope, monkey)].append(
                                cell_row(
                                    result, reason, k, method=method, monkey=monkey,
                                    source=source, scope=scope, maze=maze,
                                    feature=feature, variant=variant,
                                )
                            )
                rows = rows_by_tree[(method, source, scope, monkey)]
                done = sum(1 for r in rows if not r.get("reason_skipped"))
                print(f"  {method}/{source}: {done}/{len(rows)} cells estimated")

            del data

    return rows_by_tree


def emit(rows_by_tree, args):
    for (method, source, scope, monkey), rows in sorted(rows_by_tree.items()):
        write_csv(elbowpath.results_csv(args.out_root, source, monkey), rows)
        write_summary_md(
            elbowpath.summary_md(args.out_root, source, monkey),
            rows, monkey=monkey, source=source, mazes=args.maze,
        )

        for row in rows:
            if row.get("reason_skipped"):
                continue
            panel = Panel(k=int(row["k"]), result=_result_from_row(row))
            fig = figmod.pair_figure(
                [panel],
                title=elbowfig.selected_suptitle(
                    monkey=monkey, maze=row["maze"], feature=row["feature"],
                    variant=row["variant"], source=source, method=method,
                    k=row["k"],
                ),
                names=[f"{row['maze']}H", f"{row['maze']}S"],
                method=method,
                vmax=args.vmax,
            )
            save_figure(
                fig, elbowpath.stem(row["maze"], row["feature"], row["variant"]),
                out_root=args.out_root,
                rel_dir=elbowpath.rel_dir(source, monkey),
                dpi=args.dpi,
            )
            plt.close(fig)


@dataclass
class _Row:
    """The subset of `estimator.Result` the figure reads, rebuilt from a CSV row.

    `sweep` keeps rows rather than `Result` objects so the CSV stays the single
    source of truth for what was written; the figure needs five of those fields
    back. Rebuilding beats threading `Result` objects through, which would let
    the figure and the CSV disagree.
    """

    q: object
    delta: float
    z: float
    p: float
    p_at_floor: bool
    n_H: int
    n_S: int


def _result_from_row(row):
    q = np.array(
        [[row["r_HH"], row["r_HS"]], [row["r_SH"], row["r_SS"]]], dtype=float
    )
    return _Row(
        q=q, delta=row["delta"], z=row["z"], p=row["p"],
        p_at_floor=bool(row["p_at_floor"]), n_H=row["n_H"], n_S=row["n_S"],
    )


def dry_run(args):
    """Per-maze in-scope trial and fixation counts, and whether K can be chosen.

    Reports what the selection has to work with, without fitting anything: the
    pool sizes and whether each maze clears `MIN_TRIALS_FOR_SPLIT`. The label
    counts must match `comp_pooled_build --dry-run` for the same (monkey,
    source, scope, maze), since both mask on the same label logic -- the only
    difference is that the pool feeding the codebook is narrower here.
    """
    for monkey in args.monkey:
        for source, scope in SOURCE_SCOPES_WANTED:
            keep_sessions = tuple(
                source_scope_sessions(source, scope)[0].get(monkey, ())
            )
            if not keep_sessions:
                print(f"{monkey}/{source}/{scope}: no sessions in scope")
                continue
            lookup = source_lookup(source, keep_sessions)
            pools = elbow_select.maze_pools(
                monkey, keep_sessions=keep_sessions, mazes=args.maze
            )
            print(f"\n{monkey}/{source}/{scope}  ({len(keep_sessions)} sessions)")
            for maze in args.maze:
                trial_pools = pools.get(maze, [])
                n_samples = sum(t.samples.shape[0] for t in trial_pools)
                n_fix = sum(t.centroids.shape[0] for t in trial_pools)
                y = labels_for_rows(
                    np.asarray([t.session for t in trial_pools]),
                    np.asarray([t.trial_id for t in trial_pools], dtype=int),
                    lookup,
                )
                y = y[np.isfinite(y)].astype(int)
                n_h, n_s, m, reason = estimator.coverage(y, min_trials=args.min_trials)
                splittable = len(trial_pools) >= elbow_select.MIN_TRIALS_FOR_SPLIT
                note = "" if splittable else "  TOO FEW TRIALS TO SPLIT"
                cov = f"SKIPPED ({reason})" if reason else f"m={m:3d}"
                print(
                    f"  maze {maze}: trials={len(trial_pools):4d} "
                    f"samples={n_samples:6d} fixations={n_fix:6d}  "
                    f"n_H={n_h:4d} n_S={n_s:4d} {cov}{note}"
                )

    n_cells = (
        len(args.monkey) * len(args.maze) * len(args.feature)
        * len(args.variant) * len(args.methods)
    )
    n_fits = (
        len(args.monkey) * len(args.maze) * len(args.k_grid) * args.n_splits
    )
    print(
        f"\n{n_cells} estimator cells at n_rounds={args.n_rounds}, "
        f"n_perm={args.n_perm}; {n_fits} k-means fits for selection"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--monkey", nargs="*", default=list(MONKEYS))
    parser.add_argument("--maze", type=int, nargs="*", default=list(MAZES))
    parser.add_argument(
        "--feature", nargs="*", default=list(FEATURES), choices=tuple(FEATURES)
    )
    parser.add_argument(
        "--variant", nargs="*", default=list(variantmod.VARIANTS),
        choices=variantmod.VARIANTS,
    )
    parser.add_argument(
        "--k-grid", type=int, nargs="*", default=list(elbow_select.KS),
        dest="k_grid",
        help="candidate codebook sizes; must be uniformly spaced for the "
             "curvature rule to mean anything (default 2..20 step 1)",
    )
    parser.add_argument(
        "--n-splits", type=int, default=elbow_select.N_SPLITS,
        help="random 80/20 trial splits averaged into the EV curve",
    )
    parser.add_argument(
        "--no-stability", action="store_true",
        help="skip the partition-stability diagnostic. It shares the EV "
             "sweep's codebooks so it costs no extra k-means, only the "
             "pairwise ARI over n_splits label vectors",
    )
    parser.add_argument(
        "--estimator", nargs="*", default=list(estimator.METHODS),
        choices=estimator.METHODS, dest="methods",
    )
    parser.add_argument("--n-rounds", type=int, default=estimator.N_ROUNDS)
    parser.add_argument("--n-perm", type=int, default=estimator.N_PERM)
    parser.add_argument("--min-trials", type=int, default=estimator.MIN_TRIALS)
    parser.add_argument(
        "--vmax", type=float, default=None,
        help="fixed 0-to-VMAX colour scale instead of the per-figure one",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--out-root", type=Path, default=elbowpath.OUT_ROOT)
    parser.add_argument(
        "--refresh", action="store_true", help="rebuild the feature cache"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report pool sizes and coverage, fit and estimate nothing",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args)
        return

    emit(sweep(args), args)


if __name__ == "__main__":
    main()
