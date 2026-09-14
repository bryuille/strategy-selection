"""Pooled-across-sessions comparison figures, `svm` / `k=12` only.

`eye_pre_flash.maze_strategy_pairs.build` computes one split-half 2x2 per
session and averages the matrices unweighted. This script computes the
alternative reading someone might expect instead -- concatenate every kept
session's trials for a maze into one bag, then run the identical split-half CV
once on the pool (`pooled.build_pooled_pool` /
`pooled.pooled_permutation_null`). It shares `variants.apply_variant`, the
`mean_removed` mean-profile fit and the whole rewritten estimator with
`build.py`, so the two are otherwise directly comparable; only the
per-session-vs-pooled step differs.

Restricted to the SVM label source, K=12 and `mean_removed` per request -- not
a general sweep like `build.py`. It shares `figures.pair_figure` at
``len(panels) == 1`` rather than keeping its own copy of the panel-drawing
code, so the colour convention, the fixed 0-1 scale, the under-range treatment
and the permutation-floor tilde cannot drift between the two trees.

The time-based drift controls in `drift.py` are deliberately **not** run here:
pooling concatenates sessions, so `trial_index` no longer orders anything.

Usage:
    uv run python -m eye_pre_flash.maze_strategy_pairs.build_pooled

Writes out/pooled/svm/<monkey>/mean_removed/maze<M>_<feature>.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from eye_pre_flash.maze_strategy_pairs import figures as figmod
from eye_pre_flash.maze_strategy_pairs import pair
from eye_pre_flash.maze_strategy_pairs import paths as pathmod
from eye_pre_flash.maze_strategy_pairs import pooled
from eye_pre_flash.maze_strategy_pairs import variants as variantmod
from eye_pre_flash.plotting.plot_io import save_figure

# Feature name -> the block it reads from `classifier.features`.
FEATURES = {"occupancy": "occ_ms", "occupancy_bin": "occ_bin"}

VARIANT = "mean_removed"
K = 12
MONKEYS = ("Faure", "Nielsen")
MAZES = (1, 2, 3, 4, 5, 6)
FEATURE_LIST = ("occupancy", "occupancy_bin")
SOURCE_SCOPES_WANTED = (("svm", "top_ten"),)  # svm only, per request


@dataclass
class PooledPanel:
    """`figures.pair_figure`'s panel protocol, backed by one pooled matrix.

    `figures.panel_stats` reads `result.n_sessions` / `result.census`, so this
    adapter presents the pooled pool as a single "session" whose census is the
    pooled trial counts.
    """

    k: int
    d: int
    n_in_scope: int
    result: object = None
    null: dict = None
    reason: str = None
    delta_time_el: float = np.nan
    delta_time_oe: float = np.nan
    r_pb: float = np.nan


@dataclass
class _PooledResult:
    mean: np.ndarray
    census: np.ndarray
    n_sessions: int = 1


def build_one(monkey, maze, source, scope, feature, *, args):
    data = load_features(monkey, k=args.k)
    raw_X = np.asarray(data[FEATURES[feature]], dtype=float)
    sessions = np.asarray(data["session"]).astype(str)
    trials = np.asarray(data["trial_indices_all"], dtype=int)
    mazes_all = np.asarray(data["maze_id"], dtype=int)
    origin = variantmod.origin_state(data["codebook_xy"])

    widest = WIDEST_SCOPE[source]
    by_monkey_w, _dropped = source_scope_sessions(source, widest)
    widest_sessions = by_monkey_w.get(monkey, ())
    y_widest = labels_for_rows(sessions, trials, source_lookup(source, widest_sessions))
    fit_mask = np.isin(sessions, list(widest_sessions)) & np.isfinite(y_widest)
    if not fit_mask.any():
        print(f"  {monkey}/{source}/{feature}: no labelled trial in {widest}; skipping")
        return None
    mean_profile = variantmod.fit_grand_mean(raw_X[fit_mask])

    X_v, _dims = variantmod.apply_variant(
        raw_X, VARIANT, feature=FEATURES[feature], k=args.k,
        origin=origin, mean_profile=mean_profile,
    )

    by_monkey_s, _dropped = source_scope_sessions(source, scope)
    keep_sessions = by_monkey_s.get(monkey, ())
    if not keep_sessions:
        print(f"  {monkey}/{source}/{scope}: no sessions in scope; skipping")
        return None
    lookup = source_lookup(source, keep_sessions)

    pool = pooled.build_pooled_pool(
        X_v, sessions, trials, mazes_all, maze=maze, keep_sessions=keep_sessions,
        label_lookup=lookup, min_trials=args.min_trials, min_half=args.min_half,
        n_half=args.n_half, seed=args.seed,
    )
    panel = PooledPanel(k=args.k, d=X_v.shape[1], n_in_scope=len(keep_sessions))
    if pool is None:
        print(
            f"  {monkey}/{source}/{scope}/maze{maze}/{feature}: "
            f"no pooled trials clearing min_trials/min_half"
        )
        return None

    null, obs_mat, _draws = pooled.pooled_permutation_null(
        pool, n_splits=args.n_splits, n_perm=args.n_perm, seed=args.seed,
        r_clip=args.r_clip,
    )
    if not np.isfinite(obs_mat.r).all():
        print(f"  {monkey}/{source}/{scope}/maze{maze}/{feature}: degenerate pooled matrix")
        return None

    panel.result = _PooledResult(
        mean=obs_mat.r,
        census=np.array([pool.counts[pair.H_CELL], pool.counts[pair.S_CELL]]),
    )
    panel.null = null

    fig = figmod.pair_figure(
        [panel],
        suptitle=figmod.pair_suptitle(
            monkey=monkey, maze=maze, feature=feature,
            variant=f"{VARIANT} (POOLED)", source=source, scope=scope,
        ),
        footer=figmod.pair_footer(
            n_splits=args.n_splits, n_perm=args.n_perm,
            min_trials=args.min_trials, min_half=args.min_half, seed=args.seed,
        ),
        names=pair.cell_labels(maze),
    )
    path = save_figure(
        fig, pathmod.stem(maze, feature),
        out_root=args.out_root,
        rel_dir=pathmod.rel_dir(source, monkey, VARIANT),
        dpi=args.dpi,
    )
    plt.close(fig)
    print(
        f"    z={null['z_perm']:+.2f} Δ={null['delta_obs']:+.4f} "
        f"p={null['p_two_sided']:.4f} m={pool.m} "
        f"trials H={pool.counts[0]} S={pool.counts[1]}"
    )
    return path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--monkey", nargs="*", default=list(MONKEYS))
    parser.add_argument("--maze", type=int, nargs="*", default=list(MAZES))
    parser.add_argument(
        "--feature", nargs="*", default=list(FEATURE_LIST), choices=tuple(FEATURES)
    )
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--n-splits", type=int, default=pair.N_SPLITS)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--min-trials", type=int, default=pair.MIN_TRIALS)
    parser.add_argument("--min-half", type=int, default=pair.MIN_HALF)
    parser.add_argument("--n-half", type=int, default=None)
    parser.add_argument("--r-clip", type=float, default=pair.R_CLIP)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--out-root", type=Path, default=pathmod.POOLED_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for monkey in args.monkey:
        for maze in args.maze:
            for feature in args.feature:
                for source, scope in SOURCE_SCOPES_WANTED:
                    build_one(monkey, maze, source, scope, feature, args=args)


if __name__ == "__main__":
    main()
