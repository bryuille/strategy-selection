"""Merged-K single-maze H-vs-S similarity figures.

One figure per (monkey, maze, source, feature, variant), carrying **both
K = 6 and K = 12** as side-by-side 2x2 panels with independently computed
statistics. Each panel's diagonal is a strategy's own split-half reliability
and its off-diagonal the cross-strategy correlation, at an identical half size
on both sides; the statistic is ``Δ = 0.5*(r_HH + r_SS) − r_HS`` against a
within-(session, maze) label-shuffle null, reported beside the diagnostics
that null cannot provide (`pair.amgm_floor`, `drift.delta_time`).

`METHODS.md` is the methodology record. `STATS.md` is the slide-reading guide.

Sweeps all three variants (`full`, `no_origin`, `mean_removed`) -- the variant
is a directory level, which also fixes a latent bug: the old output path
omitted it, so a three-variant sweep would have overwritten two thirds of its
own figures.

Must run where the feature caches live (the cluster) -- see cloud.md.

Usage:
    uv run python -m eye_pre_flash.maze_strategy_pairs.build
    uv run python -m eye_pre_flash.maze_strategy_pairs.build --maze 4 --n-perm 200
    uv run python -m eye_pre_flash.maze_strategy_pairs.build --dry-run

Writes out/<source>/<monkey>/<variant>/maze<M>_<feature>.png plus
results_pair.csv and results_raw_pair.csv beside them.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eye_pre_flash.classifier.features import load_features
from eye_pre_flash.classifier.labels import labels_for_rows
from eye_pre_flash.label_sources import (
    WIDEST_SCOPE,
    source_lookup,
    source_scope_sessions,
)
from eye_pre_flash.maze_strategy_pairs import drift
from eye_pre_flash.maze_strategy_pairs import figures as figmod
from eye_pre_flash.maze_strategy_pairs import pair
from eye_pre_flash.maze_strategy_pairs import paths as pathmod
from eye_pre_flash.maze_strategy_pairs import variants as variantmod
from eye_pre_flash.plotting.plot_io import save_figure

# Feature name -> the block it reads from `classifier.features`.
FEATURES = {"occupancy": "occ_ms", "occupancy_bin": "occ_bin", "bigram": "bigram"}

MONKEYS = ("Faure", "Nielsen")
MAZES = (1, 2, 3, 4, 5, 6)
KS = (6, 12)
FEATURE_LIST = ("occupancy", "occupancy_bin")
SOURCE_SCOPES_WANTED = (("dendro", "publication"), ("svm", "top_ten"))


@dataclass
class Panel:
    """One K's worth of a figure: the estimate, or the reason there isn't one."""

    k: int
    d: int
    n_in_scope: int
    result: object = None
    null: dict = None
    reason: str = None
    delta_time_el: float = np.nan
    delta_time_oe: float = np.nan
    r_pb: float = np.nan
    assoc: list = field(default_factory=list)
    dropped: dict = field(default_factory=dict)
    time_info: dict = field(default_factory=dict)


# `load_features` reads and fully materialises the npz on every call (no
# mmap), so the 288-run sweep would otherwise do 288 full reads of four
# distinct caches. Cached here rather than on `features.load_features` itself
# because that returns a mutable dict shared with every other caller in the
# repo; confining the aliasing to this script, whose use is read-only, is the
# safer placement.
@lru_cache(maxsize=8)
def _features(monkey, k):
    return load_features(monkey, k=k)


@lru_cache(maxsize=16)
def _scope_sessions(source, scope):
    by_monkey, _dropped = source_scope_sessions(source, scope)
    return by_monkey


@lru_cache(maxsize=32)
def _lookup(source, sessions):
    return source_lookup(source, sessions)


def _meta(data):
    return (
        np.asarray(data["session"]).astype(str),
        np.asarray(data["trial_indices_all"], dtype=int),
        np.asarray(data["maze_id"], dtype=int),
    )


def grand_mean_profile(monkey, feature, k, source):
    """The `mean_removed` profile, keyed on the **source** as well.

    The fit mask comes from `WIDEST_SCOPE[source]`, which is `top_four` for
    dendro and `top_ten` for svm -- two different session sets. Keying on
    (monkey, feature, k) alone would silently apply one source's grand mean to
    the other source's panels and change every `mean_removed` number.
    """
    data = _features(monkey, k)
    sessions, trials, _mazes = _meta(data)
    widest = WIDEST_SCOPE[source]
    widest_sessions = tuple(_scope_sessions(source, widest).get(monkey, ()))
    if not widest_sessions:
        return None
    y_widest = labels_for_rows(sessions, trials, _lookup(source, widest_sessions))
    fit_mask = np.isin(sessions, list(widest_sessions)) & np.isfinite(y_widest)
    if not fit_mask.any():
        return None
    raw = np.asarray(data[FEATURES[feature]], dtype=float)
    return variantmod.fit_grand_mean(raw[fit_mask])


def run_one_k(
    X_v, meta, *, monkey, maze, k, keep_sessions, lookup, args
):
    """Estimate one panel: pools, the null, and the drift controls."""
    sessions, trials, mazes = meta
    panel = Panel(k=k, d=X_v.shape[1], n_in_scope=len(keep_sessions))

    pools, dropped = pair.build_pools(
        X_v, sessions, trials, mazes,
        maze=maze, keep_sessions=keep_sessions, label_lookup=lookup,
        min_trials=args.min_trials, min_half=args.min_half,
        n_half=args.n_half, seed=args.seed,
    )
    panel.dropped = dropped
    if not pools:
        panel.reason = f"no session clears min_half {args.min_half}"
        return panel

    # Drift diagnostics are computed on the same pools, before any detrending,
    # so the association number describes the data as collected.
    panel.assoc, assoc_summary = drift.association_summary(pools)
    panel.r_pb = assoc_summary["r_pb_median_abs"]
    panel.time_info = dict(assoc_summary)

    if args.detrend:
        pools = drift.detrend_pools(pools, order=args.detrend_order)

    plans = pair.build_plans(pools, n_splits=args.n_splits)
    null, result, _draws = pair.permutation_null(
        pools, plans=plans, maze=maze, variant=args.variant_current,
        n_splits=args.n_splits, n_perm=args.n_perm, seed=args.seed,
        r_clip=args.r_clip, min_trials=args.min_trials, min_half=args.min_half,
        n_sessions_in_scope=len(keep_sessions),
    )
    if not result.n_sessions:
        panel.reason = "no usable session (all matrices degenerate)"
        return panel

    for scheme, attr in (("early_late", "delta_time_el"), ("odd_even", "delta_time_oe")):
        value, info = drift.delta_time(
            pools, scheme, maze=maze, variant=args.variant_current,
            n_splits=args.n_splits, r_clip=args.r_clip,
            min_trials=args.min_trials, min_half=args.min_half,
        )
        setattr(panel, attr, value)
        panel.time_info[f"delta_time_{scheme}"] = value
        panel.time_info[f"n_sessions_{scheme}"] = info["n_sessions"]

    result.d = panel.d
    panel.result, panel.null = result, null
    return panel


def csv_rows(panels, *, monkey, maze, feature, variant, source, scope):
    """One summary row per (maze, feature, k), and one raw row per session."""
    summary, raw = [], []
    for panel in panels:
        base = dict(
            monkey=monkey, source=source, scope=scope, variant=variant,
            feature=feature, maze=maze, k=panel.k, d=panel.d,
            n_sessions_in_scope=panel.n_in_scope,
        )
        if panel.null is None:
            summary.append({**base, "reason_skipped": panel.reason})
            continue
        summary.append(
            {**base, **panel.null, **panel.time_info, "reason_skipped": ""}
        )
        assoc_by_session = {a["session"]: a for a in panel.assoc}
        for i, name in enumerate(panel.result.session_names):
            sm = panel.result.session_mats[i]
            counts = panel.result.cell_counts[i]
            a = assoc_by_session.get(name, {})
            raw.append(
                {
                    **base,
                    "session": name,
                    "n_H": counts[pair.H_CELL],
                    "n_S": counts[pair.S_CELL],
                    "m": panel.result.half_sizes[i],
                    "r_HH": sm.r[pair.H_CELL, pair.H_CELL],
                    "r_SS": sm.r[pair.S_CELL, pair.S_CELL],
                    "r_HS_ab": sm.r[pair.H_CELL, pair.S_CELL],
                    "r_HS_ba": sm.r[pair.S_CELL, pair.H_CELL],
                    "r_HH_raw": sm.r_raw[pair.H_CELL, pair.H_CELL],
                    "r_SS_raw": sm.r_raw[pair.S_CELL, pair.S_CELL],
                    "delta_session": panel.result.session_deltas[i],
                    "n_splits_used": sm.n_splits_used,
                    "n_splits_dropped": sm.n_splits_dropped,
                    "n_clipped": int(sm.n_clipped_by_entry.sum()),
                    "mw_U": a.get("mw_U", np.nan),
                    "mw_p": a.get("mw_p", np.nan),
                    "r_pb": a.get("r_pb", np.nan),
                    "t_median_H": a.get("t_median_H", np.nan),
                    "t_median_S": a.get("t_median_S", np.nan),
                }
            )
    return summary, raw


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
        for row in rows:
            writer.writerow(row)
    print(f"Saved {path}")
    return path


def warn_stale(out_root):
    stale = pathmod.stale_dirs(out_root)
    if not stale:
        return
    width = 74
    print("\n" + "=" * width)
    print("STALE OUTPUT from the previous single-K layout (not overwritten):")
    for path, n_png in stale:
        print(f"  {path}  ({n_png} png)")
    print("\nThese are single-K `mean_removed` figures the new tree never writes to.")
    print("They are gitignored, so git cannot restore them if you are wrong.")
    print("Remove with `--prune-stale`, or by hand:")
    for path, _ in stale:
        print(f"  rm -rf {path}")
    print("=" * width + "\n")


def prune_stale(out_root):
    stale = pathmod.stale_dirs(out_root)
    if not stale:
        print("No stale single-K directories found.")
        return
    for path, n_png in stale:
        shutil.rmtree(path)
        print(f"Removed {path} ({n_png} png)")


def build_variant(
    *, monkey, feature, source, scope, variant, ks, keep_sessions, lookup, args
):
    """Every maze for one (monkey, feature, source, variant). Returns CSV rows."""
    X_v, meta = {}, {}
    for k in ks:
        data = _features(monkey, k)
        meta[k] = _meta(data)
        raw = np.asarray(data[FEATURES[feature]], dtype=float)
        mean_profile = (
            grand_mean_profile(monkey, feature, k, source)
            if variant == "mean_removed"
            else None
        )
        if variant == "mean_removed" and mean_profile is None:
            print(f"  {monkey}/{source}/{feature}/k{k}: no widest-scope trial; skipping")
            return [], []
        X_v[k], _dims = variantmod.apply_variant(
            raw, variant, feature=FEATURES[feature], k=k,
            origin=variantmod.origin_state(data["codebook_xy"]),
            mean_profile=mean_profile,
        )

    args.variant_current = variant
    summary_rows, raw_rows = [], []
    for maze in args.maze:
        panels = [
            run_one_k(
                X_v[k], meta[k], monkey=monkey, maze=maze, k=k,
                keep_sessions=keep_sessions, lookup=lookup, args=args,
            )
            for k in ks
        ]
        s_rows, r_rows = csv_rows(
            panels, monkey=monkey, maze=maze, feature=feature,
            variant=variant, source=source, scope=scope,
        )
        summary_rows += s_rows
        raw_rows += r_rows

        if not any(p.result is not None and p.result.n_sessions for p in panels):
            reasons = "; ".join(f"K{p.k}: {p.reason}" for p in panels)
            print(f"  {monkey}/{source}/{variant}/maze{maze}/{feature}: {reasons}")
            continue

        fig = figmod.pair_figure(
            panels,
            suptitle=figmod.pair_suptitle(
                monkey=monkey, maze=maze, feature=feature,
                variant=variant, source=source, scope=scope,
            ),
            footer=figmod.pair_footer(
                n_splits=args.n_splits, n_perm=args.n_perm,
                min_trials=args.min_trials, min_half=args.min_half,
                seed=args.seed, detrend=args.detrend_order if args.detrend else 0,
            ),
            names=pair.cell_labels(maze),
        )
        save_figure(
            fig, pathmod.stem(maze, feature),
            out_root=args.out_root,
            rel_dir=pathmod.rel_dir(source, monkey, variant),
            dpi=args.dpi,
        )
        plt.close(fig)
        for panel in panels:
            if panel.null:
                print(
                    f"    K={panel.k}: z={panel.null['z_perm']:+.2f} "
                    f"Δ={panel.null['delta_obs']:+.4f} "
                    f"p={panel.null['p_two_sided']:.4f} "
                    f"m={panel.null['half_size_median']} "
                    f"n={panel.result.n_sessions} sessions"
                )
    return summary_rows, raw_rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--monkey", nargs="*", default=list(MONKEYS))
    parser.add_argument("--maze", type=int, nargs="*", default=list(MAZES))
    parser.add_argument(
        "--feature", nargs="*", default=list(FEATURE_LIST), choices=tuple(FEATURES)
    )
    parser.add_argument(
        "--variant", nargs="*", default=list(variantmod.VARIANTS),
        choices=variantmod.VARIANTS,
    )
    parser.add_argument(
        "--k", type=int, nargs="*", default=list(KS),
        help="one panel per K (default both; pass a single value for one panel)",
    )
    parser.add_argument("--n-splits", type=int, default=pair.N_SPLITS)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--min-trials", type=int, default=pair.MIN_TRIALS)
    parser.add_argument(
        "--min-half", type=int, default=pair.MIN_HALF,
        help="drop a session whose m = min(n_H, n_S)//2 falls below this",
    )
    parser.add_argument(
        "--n-half", type=int, default=None,
        help="force one half size across sessions (robustness check)",
    )
    parser.add_argument("--r-clip", type=float, default=pair.R_CLIP)
    parser.add_argument(
        "--detrend", action="store_true",
        help="regress trial index out of the features within (session, maze)",
    )
    parser.add_argument("--detrend-order", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--out-root", type=Path, default=pathmod.OUT_ROOT)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="list the figures that would be written, compute nothing",
    )
    parser.add_argument("--prune-stale", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.variant_current = None

    if args.prune_stale:
        prune_stale(args.out_root)

    if args.dry_run:
        n = 0
        for monkey in args.monkey:
            for source, scope in SOURCE_SCOPES_WANTED:
                for variant in args.variant:
                    for feature in args.feature:
                        for maze in args.maze:
                            rel = pathmod.rel_dir(source, monkey, variant)
                            print(f"{args.out_root / rel / (pathmod.stem(maze, feature) + '.png')}")
                            n += 1
        print(f"\n{n} figures, {n * len(args.k)} estimator runs "
              f"at n_perm={args.n_perm}, n_splits={args.n_splits}")
        return

    # The CSVs live at <source>/<monkey>/<variant>/, which has no feature
    # level, so every feature's rows must accumulate into one file -- writing
    # per feature would have each overwrite the last.
    for monkey in args.monkey:
        for source, scope in SOURCE_SCOPES_WANTED:
            keep_sessions = tuple(_scope_sessions(source, scope).get(monkey, ()))
            if not keep_sessions:
                print(f"  {monkey}/{source}/{scope}: no sessions in scope; skipping")
                continue
            lookup = _lookup(source, keep_sessions)
            for variant in args.variant:
                summary, raw = [], []
                for feature in args.feature:
                    s_rows, r_rows = build_variant(
                        monkey=monkey, feature=feature, source=source, scope=scope,
                        variant=variant, ks=args.k, keep_sessions=keep_sessions,
                        lookup=lookup, args=args,
                    )
                    summary += s_rows
                    raw += r_rows
                if summary:
                    write_csv(
                        pathmod.results_csv(args.out_root, source, monkey, variant),
                        summary,
                    )
                    write_csv(
                        pathmod.results_raw_csv(args.out_root, source, monkey, variant),
                        raw,
                    )

    warn_stale(args.out_root)


if __name__ == "__main__":
    main()
