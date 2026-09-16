"""Single-maze H-vs-S gaze similarity figures, pooled across sessions.

One figure per (method, monkey, maze, source, feature, variant), carrying
both K = 6 and K = 12 as side-by-side 2x2 panels with independently computed
statistics. Each panel's diagonal is within-strategy similarity and its
off-diagonal cross-strategy, against a within-session label-shuffle null.

Both scoring methods are swept by default into parallel output trees:
`trial_by_trial` averages similarities between individual trials,
`block_means` correlates the groups' mean vectors. Their cells are on
different scales and must not be compared directly -- see `CAVEATS.md`.

Runs on a laptop -- the estimator is a few seconds per panel and needs no
neural data, only the cached labels and feature blocks under
``$STRATEGY_DATA_ROOT`` (default ``./data``).

Usage:
    uv run python -m eye_pre_flash.maze_strategy_pairs.build
    uv run python -m eye_pre_flash.maze_strategy_pairs.build --dry-run
    uv run python -m eye_pre_flash.maze_strategy_pairs.build --maze 3 --n-perm 200

Writes out/<method>/<source>/<monkey>/<variant>/maze<M>_<feature>.png plus
results.csv.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
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
from eye_pre_flash.maze_strategy_pairs import estimator
from eye_pre_flash.maze_strategy_pairs import figures as figmod
from eye_pre_flash.maze_strategy_pairs import paths as pathmod
from eye_pre_flash.maze_strategy_pairs import variants as variantmod
from eye_pre_flash.plotting.plot_io import save_figure

# Feature name -> the block it reads from `classifier.features`.
FEATURES = {"occupancy": "occ_ms", "occupancy_bin": "occ_bin"}

MONKEYS = ("Faure", "Nielsen")
MAZES = (1, 2, 3, 4, 5, 6)
KS = (6, 12)
SOURCE_SCOPES_WANTED = (("dendro", "publication"), ("svm", "top_ten"))


@dataclass
class Panel:
    k: int
    result: object = None
    reason: str = ""


# `load_features` reads and fully materialises the npz on every call (no
# mmap), so the sweep would otherwise do hundreds of full reads of four
# caches. Cached here rather than on `features.load_features` itself, because
# that returns a mutable dict shared with every other caller in the repo;
# confining the aliasing to this script, whose use is read-only, is safer.
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
    widest_sessions = tuple(_scope_sessions(source, WIDEST_SCOPE[source]).get(monkey, ()))
    if not widest_sessions:
        return None
    y_widest = labels_for_rows(sessions, trials, _lookup(source, widest_sessions))
    fit_mask = np.isin(sessions, list(widest_sessions)) & np.isfinite(y_widest)
    if not fit_mask.any():
        return None
    raw = np.asarray(data[FEATURES[feature]], dtype=float)
    return variantmod.fit_grand_mean(raw[fit_mask])


def pool_maze(X, meta, *, maze, keep_sessions, lookup):
    """Every labelled trial of one maze, from every in-scope session, stacked.

    Pooling is the point: a single session rarely has enough trials of both
    strategies in one maze to estimate from. The session each trial came from
    is carried out alongside, because the null has to shuffle within it.
    """
    sessions, trials, mazes = meta
    y_full = labels_for_rows(sessions, trials, lookup)
    mask = (
        np.isin(sessions, list(keep_sessions))
        & np.isfinite(y_full)
        & (mazes == maze)
    )
    return X[mask], y_full[mask].astype(int), sessions[mask]


def run_panel(X_v, meta, *, maze, k, method, keep_sessions, lookup, args):
    """One K's worth of a figure: the estimate, or the reason there isn't one."""
    X, y, session_ids = pool_maze(
        X_v, meta, maze=maze, keep_sessions=keep_sessions, lookup=lookup
    )
    if X.size == 0:
        return Panel(k=k, reason="no labelled trial in this maze")
    result, reason = estimator.estimate(
        X, y, session_ids, method=method,
        n_rounds=args.n_rounds, n_perm=args.n_perm,
        seed=args.seed, min_trials=args.min_trials,
    )
    return Panel(k=k, result=result, reason=reason)


def csv_row(panel, *, method, monkey, maze, feature, variant, source, scope):
    base = dict(
        method=method, monkey=monkey, source=source, scope=scope,
        variant=variant, feature=feature, maze=maze, k=panel.k,
    )
    if panel.result is None:
        return {**base, "reason_skipped": panel.reason}
    r = panel.result
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


def build_variant(
    *, method, monkey, feature, source, scope, variant, keep_sessions, lookup, args
):
    """Every maze for one (method, monkey, feature, source, variant)."""
    X_v, meta = {}, {}
    for k in args.k:
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
            return []
        X_v[k], _dims = variantmod.apply_variant(
            raw, variant, feature=FEATURES[feature], k=k,
            origin=variantmod.origin_state(data["codebook_xy"]),
            mean_profile=mean_profile,
        )

    rows = []
    for maze in args.maze:
        panels = [
            run_panel(
                X_v[k], meta[k], maze=maze, k=k, method=method,
                keep_sessions=keep_sessions, lookup=lookup, args=args,
            )
            for k in args.k
        ]
        rows += [
            csv_row(p, method=method, monkey=monkey, maze=maze, feature=feature,
                    variant=variant, source=source, scope=scope)
            for p in panels
        ]

        # Coverage is decided on the raw labelled counts, before any
        # variant, so it is identical across K and across variants and both
        # panels stand or fall together. The partial case is still handled
        # rather than asserted away -- rendering whatever qualifies costs
        # nothing and never silently discards a good panel.
        target = (
            args.out_root
            / pathmod.rel_dir(method, source, monkey, variant)
            / f"{pathmod.stem(maze, feature)}.png"
        )
        usable = [p for p in panels if p.result is not None]
        if len(usable) < len(panels):
            why = "; ".join(f"K{p.k}: {p.reason}" for p in panels if p.result is None)
            print(f"  {method}/{monkey}/{source}/{variant}/maze{maze}/{feature}: {why}")
        if not usable:
            # A previous run may have left a figure here. Leaving it would put
            # stale numbers under a current-looking filename, which is worse
            # than an absent panel.
            if target.exists():
                target.unlink()
                print(f"    removed stale {target}")
            continue

        fig = figmod.pair_figure(
            usable,
            title=figmod.suptitle(
                monkey=monkey, maze=maze, feature=feature,
                variant=variant, source=source, method=method,
            ),
            names=[f"{maze}H", f"{maze}S"],
            method=method,
            vmax=args.vmax,
        )
        save_figure(
            fig, pathmod.stem(maze, feature),
            out_root=args.out_root,
            rel_dir=pathmod.rel_dir(method, source, monkey, variant),
            dpi=args.dpi,
        )
        plt.close(fig)
        for p in usable:
            r = p.result
            print(
                f"    K={p.k}: Δ={r.delta:+.4f} z={r.z:+.2f} p={r.p:.4f} "
                f"n_H={r.n_H} n_S={r.n_S} m={r.m} ({r.n_sessions} sessions)"
            )
    return rows


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
        help="one panel per K (default both)",
    )
    parser.add_argument(
        "--estimator", nargs="*", default=list(estimator.METHODS),
        choices=estimator.METHODS, dest="methods",
        help="how a group pairing is scored; both arms by default",
    )
    parser.add_argument("--n-rounds", type=int, default=estimator.N_ROUNDS)
    parser.add_argument("--n-perm", type=int, default=estimator.N_PERM)
    parser.add_argument(
        "--min-trials", type=int, default=estimator.MIN_TRIALS,
        help="per strategy, pooled across sessions; below this the maze is skipped",
    )
    parser.add_argument(
        "--vmax", type=float, default=None,
        help="fixed 0-to-VMAX colour scale instead of the per-figure symmetric one",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--out-root", type=Path, default=pathmod.OUT_ROOT)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report target paths and pooled trial counts, compute no statistics",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args)
        return

    for method in args.methods:
        print(f"\n=== {method} ===")
        for monkey in args.monkey:
            for source, scope in SOURCE_SCOPES_WANTED:
                keep_sessions = tuple(_scope_sessions(source, scope).get(monkey, ()))
                if not keep_sessions:
                    print(f"  {monkey}/{source}/{scope}: no sessions in scope; skipping")
                    continue
                lookup = _lookup(source, keep_sessions)
                for variant in args.variant:
                    # The CSV lives at <method>/<source>/<monkey>/<variant>/,
                    # which has no feature level, so every feature's rows
                    # accumulate into one file -- writing per feature would
                    # have each overwrite the last.
                    rows = []
                    for feature in args.feature:
                        rows += build_variant(
                            method=method, monkey=monkey, feature=feature,
                            source=source, scope=scope, variant=variant,
                            keep_sessions=keep_sessions, lookup=lookup, args=args,
                        )
                    if rows:
                        write_csv(
                            pathmod.results_csv(
                                args.out_root, method, source, monkey, variant
                            ),
                            rows,
                        )


def dry_run(args):
    """Target paths plus the pooled counts that decide coverage.

    Coverage does not depend on K, variant or feature -- the labels and the
    maze decide it -- so the counts are reported once per (monkey, source,
    maze) rather than once per figure.
    """
    n_figures = 0
    for monkey in args.monkey:
        for source, scope in SOURCE_SCOPES_WANTED:
            keep_sessions = tuple(_scope_sessions(source, scope).get(monkey, ()))
            if not keep_sessions:
                print(f"{monkey}/{source}/{scope}: no sessions in scope")
                continue
            lookup = _lookup(source, keep_sessions)
            data = _features(monkey, args.k[0])
            sessions, trials, mazes = _meta(data)
            y_full = labels_for_rows(sessions, trials, lookup)
            in_scope = np.isin(sessions, list(keep_sessions)) & np.isfinite(y_full)

            print(f"\n{monkey}/{source}/{scope}  ({len(keep_sessions)} sessions)")
            for maze in args.maze:
                y = y_full[in_scope & (mazes == maze)].astype(int)
                n_h, n_s, m, reason = estimator.coverage(y, min_trials=args.min_trials)
                if reason:
                    print(f"  maze {maze}: n_H={n_h:4d} n_S={n_s:4d}  SKIPPED ({reason})")
                    continue
                n = len(args.variant) * len(args.feature) * len(args.methods)
                n_figures += n
                print(f"  maze {maze}: n_H={n_h:4d} n_S={n_s:4d} m={m:3d}  -> {n} figures")

    print(
        f"\n{n_figures} figures, {n_figures * len(args.k)} estimator runs "
        f"at n_rounds={args.n_rounds}, n_perm={args.n_perm}"
    )


if __name__ == "__main__":
    main()
