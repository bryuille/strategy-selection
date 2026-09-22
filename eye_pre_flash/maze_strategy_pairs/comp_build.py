"""Maze-strategy pairs computed per session, swept over codebook size K.

The same analysis as `build.py`, with the pooling undone at both levels. There
the codebook is fitted once per (monkey, K) and the estimator then pools every
in-scope session's trials of one maze; here the codebook is fitted per
(session, maze) (`comp_features`) and the estimator runs on **one session's
trials of one maze** at a time. The per-session p-values are then averaged
across sessions to give one number per (maze, K), which is what the figures
plot and the tables report.

`build.py`, `estimator.py`, `variants.py` and `classifier.features` are
untouched: the pooled numbers other analyses depend on are unchanged. This
module reuses `estimator.estimate` and `variants.apply_variant` verbatim and
only changes what is handed to them.

Three things follow from the narrower scope and are worth knowing before
reading any output:

* **Coverage bites much harder.** `estimator.MIN_TRIALS` is 10 per strategy
  and was written for pools spanning ten sessions. One session's trials of one
  maze will often not clear it, so most (session, maze) cells are skipped and
  each plotted mean rests on however many sessions did clear it. That count is
  carried on every row, every marker and every table cell -- read it. A point
  averaging two sessions is not comparable to one averaging nine.
* **The null is now a plain permutation test.** `permute_within_session`
  shuffles inside each session block; with one session in the pool that is a
  single unrestricted shuffle. Sound, but it no longer controls for the
  between-session structure the pooled version was built to control for,
  because there is none left to control for.
* **Averaging p-values is not a meta-analysis.** The mean of per-session
  p-values is a descriptive summary for reading the K curve, not a combined
  significance test, and it cannot be interpreted as one. It is also bounded
  below by the permutation floor ``1 / (n_perm + 1)``.

Usage:
    uv run python -m eye_pre_flash.maze_strategy_pairs.comp_build --dry-run
    uv run python -m eye_pre_flash.maze_strategy_pairs.comp_build
    uv run python -m eye_pre_flash.maze_strategy_pairs.comp_build --k 6 12 --maze 3

Writes out/comp/<method>/<source>/<monkey>/maze<M>.png, results.csv,
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
from eye_pre_flash.maze_strategy_pairs import comp_features
from eye_pre_flash.maze_strategy_pairs import comp_figures as compfig
from eye_pre_flash.maze_strategy_pairs import comp_paths as comppath
from eye_pre_flash.maze_strategy_pairs import estimator
from eye_pre_flash.maze_strategy_pairs import variants as variantmod
from eye_pre_flash.plotting.plot_io import save_figure

FEATURES = {"occupancy": "occ_ms", "occupancy_bin": "occ_bin"}
MONKEYS = ("Faure", "Nielsen")

# Mazes 2-5 only: maze 1 and maze 6 are out of scope for this comparison.
MAZES = (2, 3, 4, 5)

# Dense where the inherited K = 6 / K = 12 live and where the curve is
# expected to turn, coarser above 16 where one more prototype changes little
# and the per-session sample count starts to bind on the k-means itself.
KS = (2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 25, 30)

# SVM labels only, at their one scope.
SOURCE_SCOPES_WANTED = (("svm", "top_ten"),)


def _meta(data):
    return (
        np.asarray(data["session"]).astype(str),
        np.asarray(data["trial_indices_all"], dtype=int),
        np.asarray(data["maze_id"], dtype=int),
        np.asarray(data["group_index"], dtype=int),
    )


def _group_codebook(data, group_index):
    """The (k, 2) codebook a row was assigned against, by its group index."""
    return np.asarray(data["codebook_xy"])[int(group_index)]


def session_cell(data, *, session, maze, lookup):
    """One (session, maze) cell: row mask, labels, and its group index.

    Returns ``(mask, y, group_index)``; `group_index` is None when the cell has
    no rows, which happens when the group was unfittable at this K.
    """
    sessions, trials, mazes, groups = _meta(data)
    y_full = labels_for_rows(sessions, trials, lookup)
    mask = (sessions == session) & (mazes == maze) & np.isfinite(y_full)
    if not mask.any():
        return mask, np.empty(0, dtype=int), None
    return mask, y_full[mask].astype(int), int(groups[mask][0])


def run_cell(data, *, session, maze, k, feature, variant, method, lookup, args):
    """One per-session estimate, or the reason there isn't one.

    The variant is applied to this cell's rows using **this cell's own**
    codebook: `no_origin` drops the state nearest the origin of the group's
    k-means, and `mean_removed` subtracts a grand mean fitted over the cell's
    own labelled trials. Neither can be inherited from a wider fit, because no
    wider fit shares this cell's dimensions.
    """
    mask, y, group_index = session_cell(data, session=session, maze=maze, lookup=lookup)
    if group_index is None:
        return None, "no labelled trial in this session/maze"

    raw = np.asarray(data[FEATURES[feature]], dtype=float)[mask]
    codebook = _group_codebook(data, group_index)

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

    session_ids = np.full(X.shape[0], session)
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


def cell_row(result, reason, *, method, monkey, source, scope, maze, session, k, feature, variant):
    base = dict(
        method=method, monkey=monkey, source=source, scope=scope,
        maze=maze, session=session, k=k, feature=feature, variant=variant,
    )
    if result is None:
        return {**base, "reason_skipped": reason}
    return {
        **base,
        "r_HH": result.q[estimator.H, estimator.H],
        "r_SS": result.q[estimator.S, estimator.S],
        "r_HS": result.q[estimator.H, estimator.S],
        "r_SH": result.q[estimator.S, estimator.H],
        "delta": result.delta,
        "z": result.z,
        "p": result.p,
        "p_at_floor": int(result.p_at_floor),
        "null_mean": result.null_mean,
        "null_sd": result.null_sd,
        "n_H": result.n_H,
        "n_S": result.n_S,
        "m": result.m,
        "d": result.d,
        "n_degenerate": result.n_degenerate,
        "n_rounds": result.n_rounds,
        "n_perm": result.n_perm,
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


def average_over_sessions(rows):
    """``(maze, k, feature, variant) -> (p_mean, n_sessions_used)``.

    The plain arithmetic mean the analysis asked for, over whichever sessions
    produced an estimate. A cell with no contributing session is absent rather
    than NaN, so a caller cannot mistake "nothing cleared coverage" for
    "p happened to be missing".
    """
    # The two estimator arms put their numbers on different scales and must
    # never land in one mean -- see `estimator`'s docstring and `CAVEATS.md`.
    # `emit` already groups by method, so this only fires on a caller that
    # bypassed it, which is exactly when it is worth failing loudly.
    methods = {row["method"] for row in rows}
    if len(methods) > 1:
        raise ValueError(
            f"rows span more than one estimator method ({sorted(methods)}); "
            "average them separately"
        )

    bucket = defaultdict(list)
    for row in rows:
        if row.get("reason_skipped"):
            continue
        key = (row["maze"], row["k"], row["feature"], row["variant"])
        bucket[key].append(row["p"])
    return {
        key: (float(np.mean(ps)), len(ps)) for key, ps in bucket.items() if ps
    }


def table_rows(averages, ks, mazes):
    """Long-to-wide: one row per (feature, variant, maze), one column per K.

    Also carries `best_k` and `best_p` -- the K minimising the mean p for that
    row, which is the thing the sweep exists to find.
    """
    features = sorted({key[2] for key in averages})
    variants = sorted({key[3] for key in averages})
    out = []
    for feature in features:
        for variant in variants:
            for maze in mazes:
                row = {"feature": feature, "variant": variant, "maze": maze}
                best = (np.inf, None)
                for k in ks:
                    entry = averages.get((maze, k, feature, variant))
                    if entry is None:
                        row[f"K{k}"] = ""
                        row[f"n_K{k}"] = ""
                        continue
                    p_mean, n_used = entry
                    row[f"K{k}"] = f"{p_mean:.4f}"
                    row[f"n_K{k}"] = n_used
                    if p_mean < best[0]:
                        best = (p_mean, k)
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
        f"# Mean p over sessions - {monkey} / {method} / {source}",
        "",
        "Codebook fitted per (session, maze); the estimator runs on one session",
        "of one maze at a time and the p-values are averaged across sessions.",
        "Each cell is `p (n)`, where `n` is how many sessions contributed --",
        "a blank cell means no session cleared coverage at that K.",
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
            data = comp_features.load_features(monkey, k=k, refresh=args.refresh)
            if len(data["session"]) == 0:
                print("  no usable trials at this K; skipping")
                continue

            for (source, scope), (keep_sessions, lookup) in scopes.items():
                for method in args.methods:
                    for maze in args.maze:
                        for session in keep_sessions:
                            for feature in args.feature:
                                for variant in args.variant:
                                    result, reason = run_cell(
                                        data, session=session, maze=maze, k=k,
                                        feature=feature, variant=variant,
                                        method=method, lookup=lookup, args=args,
                                    )
                                    rows_by_tree[(method, source, scope, monkey)].append(
                                        cell_row(
                                            result, reason, method=method,
                                            monkey=monkey, source=source, scope=scope,
                                            maze=maze, session=session, k=k,
                                            feature=feature, variant=variant,
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

        averages = average_over_sessions(rows)
        if not averages:
            print(f"  {method}/{source}/{monkey}: no cell cleared coverage; no table")
            continue

        table = table_rows(averages, args.k, args.maze)
        write_csv(comppath.table_csv(args.out_root, method, source, monkey), table)
        write_summary_md(
            comppath.summary_md(args.out_root, method, source, monkey),
            table, args.k, args.maze,
            monkey=monkey, method=method, source=source,
        )

        n_scope = len(set(r["session"] for r in rows))
        for maze in args.maze:
            curves = {
                (feature, variant): [
                    (k, averages[(maze, k, feature, variant)][0],
                     averages[(maze, k, feature, variant)][1])
                    for k in args.k
                    if (maze, k, feature, variant) in averages
                ]
                for feature in args.feature
                for variant in args.variant
            }
            if not any(curves.values()):
                print(f"  {method}/{source}/{monkey}/maze{maze}: no points; no figure")
                continue
            fig = compfig.p_vs_k_figure(
                curves,
                title=compfig.suptitle(
                    monkey=monkey, maze=maze, source=source,
                    method=method, n_sessions_scope=n_scope,
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
    """Per-(session, maze) H/S counts and whether they clear coverage.

    Worth running first. Coverage depends on the labels and the maze only, so
    it is the same at every K, every feature and every variant -- one pass at
    the smallest K answers it for the whole sweep.
    """
    k = args.k[0]
    for monkey in args.monkey:
        for source, scope in SOURCE_SCOPES_WANTED:
            keep_sessions = tuple(source_scope_sessions(source, scope)[0].get(monkey, ()))
            if not keep_sessions:
                print(f"{monkey}/{source}/{scope}: no sessions in scope")
                continue
            lookup = source_lookup(source, keep_sessions)
            data = comp_features.load_features(monkey, k=k, refresh=args.refresh)
            print(f"\n{monkey}/{source}/{scope}  ({len(keep_sessions)} sessions, K={k})")

            qualifying = 0
            for maze in args.maze:
                ok = []
                for session in keep_sessions:
                    _mask, y, group_index = session_cell(
                        data, session=session, maze=maze, lookup=lookup
                    )
                    if group_index is None:
                        continue
                    n_h, n_s, m, reason = estimator.coverage(
                        y, min_trials=args.min_trials
                    )
                    flag = "SKIP" if reason else f"m={m}"
                    print(f"  maze {maze} {session:>14}: n_H={n_h:3d} n_S={n_s:3d}  {flag}")
                    if not reason:
                        ok.append(session)
                qualifying += len(ok)
                print(f"  maze {maze}: {len(ok)}/{len(keep_sessions)} sessions qualify")
            n_cells = (
                qualifying * len(args.k) * len(args.feature)
                * len(args.variant) * len(args.methods)
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
        help="per strategy, within one session and maze",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--out-root", type=Path, default=comppath.OUT_ROOT)
    parser.add_argument(
        "--refresh", action="store_true", help="rebuild the per-group caches"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="per-(session, maze) counts and coverage, compute no statistics",
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
