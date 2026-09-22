"""Maze-strategy pairs computed per maze, swept over codebook size K, pooled.

`comp_build.py` sweeps the same K grid but fits a codebook per (session,
maze) and estimates one session at a time, then averages the per-session
p-values -- which its own docstring says is not a valid combined significance
test, and which measurably washes out real effects (see module history: the
same cell that is significant at p~0.001 when pooled across all ten sessions
comes back p~0.44 once split into ten separate per-session estimates and
averaged). `build.py` avoids that by pooling every in-scope session's trials
of one maze into a single estimate -- but only ever at K=6 and K=12, because
it shares one codebook across all six mazes and was never swept over K.

This module is the missing combination: **one codebook per (monkey, maze,
K)** (`comp_pooled_features`, pooling that maze's fixation samples across
every one of the monkey's usable sessions, the same way `classifier.features`
pools across every maze) feeding **one pooled `estimator.estimate()` call per
(maze, K, feature, variant, method)** (`build.py`'s `pool_maze` pattern,
ported to a per-maze codebook instead of a whole-monkey one). Every cell here
is already the single, genuinely significance-testable number `build.py`
would have produced at that K -- there is no averaging step, and none of the
"averaging p-values is not a meta-analysis" caveat in `comp_build.py` applies.

`comp_build.py`, `comp_features.py`, `comp_figures.py`, `comp_paths.py`,
`build.py`, `estimator.py` and `variants.py` are all left completely
untouched -- this module and its siblings (`comp_pooled_features`,
`comp_pooled_paths`, `comp_pooled_figures`) are new files, writing to a new
`out/comp_pooled/` tree.

Two things follow from the per-maze codebook and are worth knowing before
reading any output:

* **`no_origin` and `mean_removed` are maze-scoped, not maze-and-session- or
  monkey-scoped.** Each maze has its own codebook, so `no_origin`'s dropped
  dimension and `mean_removed`'s grand mean must be recomputed per maze, from
  that maze's own pooled-across-sessions trials -- the same population the
  estimator itself pools over. A `build.py`-style cross-maze grand mean is
  dimensionally impossible here (every maze's occupancy vector lives in a
  different codebook's coordinate system); a narrower, per-session grand mean
  would reintroduce the exact averaging problem this module exists to avoid.
* **Coverage does not depend on K.** `n_H`/`n_S` for a (monkey, source,
  scope, maze) come from the same label mask `build.py` uses, so they must
  match `build.py --dry-run`'s numbers exactly. Only whether the *codebook*
  itself fits varies with K (a maze's pooled sample count has to clear K),
  and that essentially never fails for maze 2-5 pooled across ten sessions.

Usage:
    uv run python -m eye_pre_flash.maze_strategy_pairs.comp_pooled_build --dry-run
    uv run python -m eye_pre_flash.maze_strategy_pairs.comp_pooled_build
    uv run python -m eye_pre_flash.maze_strategy_pairs.comp_pooled_build --k 6 12 --maze 4

Writes out/comp_pooled/<method>/<source>/<monkey>/maze<M>.png, results.csv,
p_by_maze_k.csv and a summary_<monkey>.md table of p for maze x K.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eye_pre_flash.classifier.labels import labels_for_rows
from eye_pre_flash.label_sources import source_lookup, source_scope_sessions
from eye_pre_flash.maze_strategy_pairs import comp_pooled_features
from eye_pre_flash.maze_strategy_pairs import comp_pooled_figures as compfig
from eye_pre_flash.maze_strategy_pairs import comp_pooled_paths as comppath
from eye_pre_flash.maze_strategy_pairs import estimator
from eye_pre_flash.maze_strategy_pairs import variants as variantmod
from eye_pre_flash.plotting.plot_io import save_figure

FEATURES = {"occupancy": "occ_ms", "occupancy_bin": "occ_bin"}
MONKEYS = ("Faure", "Nielsen")

# Mazes 2-5 only, same as `comp_build.py`: maze 1 and maze 6 are out of scope.
MAZES = (2, 3, 4, 5)

# Same grid as `comp_build.py`, for direct comparability between the trees.
KS = (2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 25, 30)

# K=2 and K=3 occupancy is degenerate at this maze-scoped codebook -- too few
# states to resolve real structure (K=2's fractions sum to 1 and correlate
# trivially at 1.0). Both are swept and reported in results.csv like any
# other K but dropped from the figure's x axis.
PLOT_DEGENERATE_K = {2, 3}

# SVM labels only, at their one scope -- same as `comp_build.py`.
SOURCE_SCOPES_WANTED = (("svm", "top_ten"),)


def _meta(data):
    return (
        np.asarray(data["session"]).astype(str),
        np.asarray(data["trial_indices_all"], dtype=int),
        np.asarray(data["maze_id"], dtype=int),
        np.asarray(data["group_index"], dtype=int),
    )


def _maze_codebook(data, maze):
    """The (k, 2) codebook fitted for `maze`, or None if it didn't fit at this K."""
    group_maze = np.asarray(data["group_maze"], dtype=int)
    hit = np.flatnonzero(group_maze == maze)
    if hit.size == 0:
        return None
    return np.asarray(data["codebook_xy"])[int(hit[0])]


def pool_maze(data, *, maze, keep_sessions, lookup):
    """Row mask, labels and session ids for every in-scope trial of one maze.

    The direct port of `build.py`'s `pool_maze` to this module's row-per-trial
    schema: every labelled trial of this maze, from every in-scope session.
    """
    sessions, trials, mazes, _groups = _meta(data)
    y_full = labels_for_rows(sessions, trials, lookup)
    mask = (
        np.isin(sessions, list(keep_sessions))
        & np.isfinite(y_full)
        & (mazes == maze)
    )
    return mask, y_full[mask].astype(int), sessions[mask]


def run_cell(data, *, maze, k, feature, variant, method, keep_sessions, lookup, args):
    """One pooled estimate for (maze, K, feature, variant, method).

    The variant is applied using **this maze's own** codebook and this maze's
    own pooled trials: `no_origin` drops the state nearest the origin of the
    maze's k-means, `mean_removed` subtracts a grand mean fitted over every
    in-scope session's labelled trials of this maze -- exactly the population
    the estimator below pools over. Neither can be inherited from a wider or
    narrower fit; see the module docstring.
    """
    mask, y, session_ids = pool_maze(
        data, maze=maze, keep_sessions=keep_sessions, lookup=lookup
    )
    if mask.sum() == 0:
        return None, "no labelled trial in this maze"

    codebook = _maze_codebook(data, maze)
    if codebook is None:
        return None, f"maze {maze}: codebook unfittable at K={k}"

    raw = np.asarray(data[FEATURES[feature]], dtype=float)[mask]

    if variant == "mean_removed":
        mean_profile = variantmod.fit_grand_mean(raw)
    else:
        mean_profile = None

    X, _dims = variantmod.apply_variant(
        raw,
        variant,
        feature=FEATURES[feature],
        k=k,
        origin=variantmod.origin_state(codebook),
        mean_profile=mean_profile,
    )

    return estimator.estimate(
        X,
        y,
        session_ids,
        method=method,
        n_rounds=args.n_rounds,
        n_perm=args.n_perm,
        seed=args.seed,
        min_trials=args.min_trials,
    )


def cell_row(result, reason, *, method, monkey, source, scope, maze, k, feature, variant):
    base = dict(
        method=method, monkey=monkey, source=source, scope=scope,
        variant=variant, feature=feature, maze=maze, k=k,
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


def index_rows(rows):
    """``(maze, k, feature, variant) -> row``, one genuine estimate each.

    Unlike `comp_build.average_over_sessions`, there is nothing to combine:
    each cell is already the single pooled estimate `build.py` would have
    produced at that K, so this is a lookup, not a summary statistic.
    """
    methods = {row["method"] for row in rows}
    if len(methods) > 1:
        raise ValueError(
            f"rows span more than one estimator method ({sorted(methods)}); "
            "index them separately"
        )
    out = {}
    for row in rows:
        if row.get("reason_skipped"):
            continue
        key = (row["maze"], row["k"], row["feature"], row["variant"])
        out[key] = row
    return out


def table_rows(indexed, ks, mazes):
    """Long-to-wide: one row per (feature, variant, maze), one column per K.

    Also carries `best_k` and `best_p` -- the K minimising `p` for that row.
    Unlike `comp_build.table_rows`, each cell is already a real p, not a mean
    of several per-session p-values.
    """
    features = sorted({key[2] for key in indexed})
    variants = sorted({key[3] for key in indexed})
    out = []
    for feature in features:
        for variant in variants:
            for maze in mazes:
                row = {"feature": feature, "variant": variant, "maze": maze}
                best = (np.inf, None)
                for k in ks:
                    entry = indexed.get((maze, k, feature, variant))
                    if entry is None:
                        row[f"K{k}"] = ""
                        row[f"n_K{k}"] = ""
                        continue
                    row[f"K{k}"] = f"{entry['p']:.4f}"
                    row[f"n_K{k}"] = entry["n_sessions"]
                    if entry["p"] < best[0]:
                        best = (entry["p"], k)
                row["best_k"] = "" if best[1] is None else best[1]
                row["best_p"] = "" if best[1] is None else f"{best[0]:.4f}"
                out.append(row)
    return out


def write_summary_md(path, rows, ks, mazes, *, monkey, method, source):
    """The maze x K table, one block per (feature, variant), human-readable."""
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Pooled p by maze and K - {monkey} / {method} / {source}",
        "",
        "Codebook fitted per (monkey, maze), pooled across every in-scope",
        "session; the estimator pools every in-scope session's trials of one",
        "maze into a single estimate -- no per-session averaging. Each cell",
        "is `p (n)`, where `n` is the number of sessions pooled (constant",
        "across K for a given maze); a blank cell means the codebook itself",
        "could not be fit at that K.",
        "",
    ]
    by_block = defaultdict(list)
    for row in rows:
        by_block[(row["feature"], row["variant"])].append(row)

    for (feature, variant), block in sorted(by_block.items()):
        lines += [f"## {feature} / {variant}", ""]
        header = "| maze | " + " | ".join(f"K={k}" for k in ks) + " | best K | best p |"
        lines.append(header)
        lines.append("|" + "---|" * (len(ks) + 3))
        for row in sorted(block, key=lambda r: r["maze"]):
            cells = []
            for k in ks:
                p = row.get(f"K{k}", "")
                n = row.get(f"n_K{k}", "")
                cells.append(f"{p} ({n})" if p != "" else "")
            lines.append(
                f"| {row['maze']} | " + " | ".join(cells)
                + f" | {row['best_k']} | {row['best_p']} |"
            )
        lines.append("")

    path.write_text("\n".join(lines))
    print(f"Saved {path}")
    return path


def sweep(args):
    """Every (monkey, K) cache read exactly once; everything else nests inside."""
    rows_by_tree = defaultdict(list)

    for monkey in args.monkey:
        scopes = {}
        for source, scope in SOURCE_SCOPES_WANTED:
            keep = tuple(source_scope_sessions(source, scope)[0].get(monkey, ()))
            if not keep:
                print(f"{monkey}/{source}/{scope}: no sessions in scope; skipping")
                continue
            scopes[(source, scope)] = (keep, source_lookup(source, keep))
        if not scopes:
            continue

        for k in args.k:
            print(f"\n=== {monkey} K={k} ===")
            data = comp_pooled_features.load_features(monkey, k=k, refresh=args.refresh)
            if len(data["session"]) == 0:
                print("  no usable trials at this K; skipping")
                continue

            for (source, scope), (keep_sessions, lookup) in scopes.items():
                for method in args.methods:
                    for maze in args.maze:
                        for feature in args.feature:
                            for variant in args.variant:
                                result, reason = run_cell(
                                    data, maze=maze, k=k, feature=feature,
                                    variant=variant, method=method,
                                    keep_sessions=keep_sessions, lookup=lookup,
                                    args=args,
                                )
                                rows_by_tree[(method, source, scope, monkey)].append(
                                    cell_row(
                                        result, reason, method=method,
                                        monkey=monkey, source=source, scope=scope,
                                        maze=maze, k=k, feature=feature,
                                        variant=variant,
                                    )
                                )
                    done = sum(
                        1 for r in rows_by_tree[(method, source, scope, monkey)]
                        if r["k"] == k and not r.get("reason_skipped")
                    )
                    total = sum(
                        1 for r in rows_by_tree[(method, source, scope, monkey)]
                        if r["k"] == k
                    )
                    print(f"  {method}/{source}: {done}/{total} cells estimated")

            # The cache is large and nothing below needs it again.
            del data

    return rows_by_tree


def emit(rows_by_tree, args):
    for (method, source, scope, monkey), rows in sorted(rows_by_tree.items()):
        write_csv(comppath.results_csv(args.out_root, method, source, monkey), rows)

        indexed = index_rows(rows)
        if not indexed:
            print(f"  {method}/{source}/{monkey}: no cell cleared coverage; no table")
            continue

        table = table_rows(indexed, args.k, args.maze)
        write_csv(comppath.table_csv(args.out_root, method, source, monkey), table)
        write_summary_md(
            comppath.summary_md(args.out_root, method, source, monkey),
            table, args.k, args.maze,
            monkey=monkey, method=method, source=source,
        )

        for maze in args.maze:
            curves = {
                (feature, variant): [
                    (k, indexed[(maze, k, feature, variant)]["p"],
                     indexed[(maze, k, feature, variant)]["z"])
                    for k in args.k
                    if k not in PLOT_DEGENERATE_K
                    and (maze, k, feature, variant) in indexed
                ]
                for feature in args.feature
                for variant in args.variant
            }
            if not any(curves.values()):
                print(f"  {method}/{source}/{monkey}/maze{maze}: no points; no figure")
                continue

            # Coverage (n_H, n_S, n_sessions) is identical across every K,
            # feature and variant for this (maze, scope) -- any cleared cell
            # names it.
            sample = next(
                indexed[(maze, k, feature, variant)]
                for feature in args.feature
                for variant in args.variant
                for k in args.k
                if (maze, k, feature, variant) in indexed
            )
            fig = compfig.p_vs_k_figure(
                curves,
                title=compfig.suptitle(
                    monkey=monkey, maze=maze, source=source, method=method,
                    n_H=sample["n_H"], n_S=sample["n_S"],
                    n_sessions=sample["n_sessions"],
                ),
                p_floor=1.0 / (args.n_perm + 1),
            )
            save_figure(
                fig, comppath.stem(maze),
                out_root=args.out_root,
                rel_dir=comppath.rel_dir(method, source, monkey),
                dpi=args.dpi,
            )
            plt.close(fig)


def dry_run(args):
    """Per-maze pooled H/S counts, plus whether the codebook itself would fit.

    Coverage depends on the labels and the maze only, so it is the same at
    every K, feature and variant -- exactly as in `build.py --dry-run`, and
    the two must report identical n_H/n_S for the same (monkey, source,
    scope, maze), since both mask on the same label logic. The one thing
    genuinely new here is whether each maze's pooled fixation-sample count
    clears the largest K in the sweep; checked with one extraction at the
    smallest K, since the pool itself does not depend on K.
    """
    k0 = args.k[0]
    for monkey in args.monkey:
        data0 = comp_pooled_features.load_features(monkey, k=k0, refresh=args.refresh)
        group_maze = np.asarray(data0["group_maze"], dtype=int)
        group_n_pool = np.asarray(data0["group_n_pool"], dtype=int)
        pool_by_maze = dict(zip(group_maze.tolist(), group_n_pool.tolist()))

        for source, scope in SOURCE_SCOPES_WANTED:
            keep_sessions = tuple(
                source_scope_sessions(source, scope)[0].get(monkey, ())
            )
            if not keep_sessions:
                print(f"{monkey}/{source}/{scope}: no sessions in scope")
                continue
            lookup = source_lookup(source, keep_sessions)
            print(f"\n{monkey}/{source}/{scope}  ({len(keep_sessions)} sessions)")

            n_cells = 0
            for maze in args.maze:
                mask, y, session_ids = pool_maze(
                    data0, maze=maze, keep_sessions=keep_sessions, lookup=lookup
                )
                n_h, n_s, m, reason = estimator.coverage(y, min_trials=args.min_trials)
                n_pool = pool_by_maze.get(maze)
                if n_pool is None:
                    pool_flag = f"no codebook at K={k0}"
                elif n_pool < max(args.k):
                    pool_flag = f"pool={n_pool} (needs >= {max(args.k)} at the top K!)"
                else:
                    pool_flag = f"pool={n_pool}"
                if reason:
                    print(
                        f"  maze {maze}: n_H={n_h:4d} n_S={n_s:4d}  "
                        f"SKIPPED ({reason})  {pool_flag}"
                    )
                    continue
                n = len(args.k) * len(args.feature) * len(args.variant) * len(args.methods)
                n_cells += n
                print(
                    f"  maze {maze}: n_H={n_h:4d} n_S={n_s:4d} m={m:3d} "
                    f"({len(set(session_ids.tolist()))} sessions)  {pool_flag}  "
                    f"-> {n} cells"
                )
            print(f"  -> {n_cells} estimator runs across the K sweep")


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
        "--k", type=int, nargs="*", default=list(KS),
        help="codebook sizes to sweep; one x-axis point each",
    )
    parser.add_argument(
        "--estimator", nargs="*", default=list(estimator.METHODS),
        choices=estimator.METHODS, dest="methods",
    )
    parser.add_argument("--n-rounds", type=int, default=estimator.N_ROUNDS)
    parser.add_argument("--n-perm", type=int, default=estimator.N_PERM)
    parser.add_argument(
        "--min-trials", type=int, default=estimator.MIN_TRIALS,
        help="per strategy, pooled across sessions; below this the maze is skipped",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--out-root", type=Path, default=comppath.OUT_ROOT)
    parser.add_argument(
        "--refresh", action="store_true", help="rebuild the per-maze caches"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="per-maze pooled counts and coverage, compute no statistics",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.k = sorted(set(int(k) for k in args.k))

    if args.dry_run:
        dry_run(args)
        return

    emit(sweep(args), args)


if __name__ == "__main__":
    main()
