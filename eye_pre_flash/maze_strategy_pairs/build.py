"""Presentation-ready single-panel `mazestratpair` figures.

`eye_pre_flash.maze_strategy_pairs.build` renders one 1x3 figure (full / no_origin /
mean_removed side by side) per (monkey, maze, source, scope, feature, k), full
of diagnostic text a slide doesn't want: half sizes, split counts, the null's
mean/sd, degenerate-split counts, an explanatory xlabel. This script computes
the exact same statistic -- it calls `pair.build_pools` and
`pair.permutation_null` unchanged, so the numbers cannot drift from the
canonical run -- and draws only one variant (`mean_removed`, the panel this
project has been discussing) as its own compact figure: the 2x2 matrix, its
trial counts, and the three statistics that matter (z, delta, p).

One deliberate deviation from `mazestratpair.figures`: that module gives
`mean_removed` a zero-centred diverging colormap because the variant can go
negative by construction (see `mazestratpair/figures.py`'s `_panel_scale`).
This script forces the same parula (`BLUE_YELLOW`) ramp every other panel
uses, per request, so a viewer scans one colour language across every slide.

Must run where the feature caches live (the cluster) -- see cloud.md.

Usage:
    uv run python -m eye_pre_flash.maze_strategy_pairs.build
    uv run python -m eye_pre_flash.maze_strategy_pairs.build --maze 2 --n-perm 2000

Writes eye_pre_flash/pres/mazestratpair/<monkey>_maze<M>_<feature>_<source>_<scope>_k<K>.png
"""

from __future__ import annotations

import argparse
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
from eye_pre_flash.maze_strategy_pairs import matrix as corrmatrix
from eye_pre_flash.maze_strategy_pairs import pair
from eye_pre_flash.maze_strategy_pairs import variants as variantmod
from eye_pre_flash.plotting.similarities.common import BLUE_YELLOW, N_SPLITS

OUT_ROOT = Path(__file__).resolve().parent / "out"

FEATURE_LABEL = {
    "occupancy": "occupancy",
    "occupancy_bin": "binary occupancy",
    "bigram": "state-bigram",
}
SOURCE_LABEL = {"dendro": "dendrogram", "svm": "SVM"}
# Feature name -> the block it reads from `classifier.features`.
FEATURES = {"occupancy": "occ_ms", "occupancy_bin": "occ_bin", "bigram": "bigram"}


def pair_labels(sessions, trials, source, keep_sessions):
    return labels_for_rows(sessions, trials, source_lookup(source, keep_sessions))

VARIANT = "mean_removed"
K = 12
MONKEYS = ("Faure", "Nielsen")
MAZES = (1, 2, 3, 4, 5, 6)
FEATURE_LIST = ("occupancy", "occupancy_bin")
SOURCE_SCOPES_WANTED = (("dendro", "publication"), ("svm", "top_ten"))


def _fmt(value, digits=3):
    return "—" if not np.isfinite(value) else f"{value:.{digits}f}"


def plot_single(result, null, *, monkey, maze, source, scope, feature, k):
    names = pair.cell_labels(maze)
    thin = pair.thin_mask(result)
    corr = result.mean

    finite = corr[np.isfinite(corr)]
    vmin, vmax = (0.0, 1.0) if not finite.size else (float(finite.min()), float(finite.max()))
    if vmin == vmax:
        vmax = vmin + 1e-12

    fig, ax = plt.subplots(figsize=(4.6, 5.0), layout="constrained")
    im = ax.imshow(corr, origin="upper", cmap=BLUE_YELLOW, vmin=vmin, vmax=vmax, aspect="equal")

    tick_names = [f"{name}*" if thin[i] else name for i, name in enumerate(names)]
    ax.set_xticks(range(pair.N_CELLS), labels=tick_names, fontsize=13)
    ax.set_yticks(range(pair.N_CELLS), labels=tick_names, fontsize=13)

    for i in range(pair.N_CELLS):
        for j in range(pair.N_CELLS):
            val = corr[i, j]
            if not np.isfinite(val):
                ax.text(j, i, "—", ha="center", va="center", color="0.4", fontsize=13)
                continue
            r, g, b, _ = im.cmap(im.norm(val))
            colour = "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=colour, fontsize=15)
            if i == j:
                ax.text(
                    j, i + 0.28, f"n = {int(result.census[i])}",
                    ha="center", va="center", color=colour, fontsize=10,
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([vmin, 0.5 * (vmin + vmax), vmax])
    cbar.ax.set_yticklabels([f"{t:.2f}" for t in [vmin, 0.5 * (vmin + vmax), vmax]], fontsize=8)
    cbar.set_label("r", fontsize=9)

    p = null.get("p_two_sided", np.nan)
    p_txt = _fmt(p, 4)
    # A tilde marks the permutation floor: p cannot go below 1/(n_perm+1), so
    # the printed value is an upper bound rather than an estimate.
    if np.isfinite(p) and null.get("n_perm") and abs(p - 1.0 / (null["n_perm"] + 1)) < 1e-12:
        p_txt = f"~{p_txt}"

    feature_label = FEATURE_LABEL.get(feature, feature)
    source_label = SOURCE_LABEL.get(source, source)
    ax.set_title(
        f"{monkey} maze {maze} — {feature_label}\n{source_label}, {scope}, K={k}\n"
        f"z = {_fmt(null.get('z_perm', np.nan), 2)}   Δ = {_fmt(null.get('delta_obs', np.nan))}   "
        f"p = {p_txt}   (n = {result.n_sessions} sessions)",
        fontsize=11,
    )
    return fig


def build_one(monkey, maze, source, scope, feature, *, k, n_perm, min_trials, min_stable, half_rule, seed):
    data = load_features(monkey, k=k)
    raw_X = np.asarray(data[FEATURES[feature]], dtype=float)
    sessions = np.asarray(data["session"]).astype(str)
    trials = np.asarray(data["trial_indices_all"], dtype=int)
    mazes_all = np.asarray(data["maze_id"], dtype=int)
    origin = variantmod.origin_state(data["codebook_xy"])

    widest = WIDEST_SCOPE[source]
    by_monkey_w, _dropped = source_scope_sessions(source, widest)
    widest_sessions = by_monkey_w.get(monkey, ())
    y_widest = pair_labels(sessions, trials, source, widest_sessions)
    fit_mask = np.isin(sessions, list(widest_sessions)) & np.isfinite(y_widest)
    if not fit_mask.any():
        print(f"  {monkey}/{source}/{feature}: no labelled trial in {widest}; skipping")
        return None
    mean_profile = variantmod.fit_grand_mean(raw_X[fit_mask])

    X_v, _dim_labels = variantmod.apply_variant(
        raw_X, VARIANT, feature=FEATURES[feature], k=k, origin=origin, mean_profile=mean_profile,
    )

    by_monkey_s, _dropped = source_scope_sessions(source, scope)
    keep_sessions = by_monkey_s.get(monkey, ())
    if not keep_sessions:
        print(f"  {monkey}/{source}/{scope}: no sessions in scope; skipping")
        return None
    lookup = source_lookup(source, keep_sessions)

    pools = pair.build_pools(
        X_v, sessions, trials, mazes_all, maze=maze, keep_sessions=keep_sessions,
        label_lookup=lookup, min_trials=min_trials, min_stable=min_stable,
        half_rule=half_rule, seed=seed,
    )
    null, result, _draws = pair.permutation_null(
        pools, maze=maze, variant=VARIANT, n_splits=N_SPLITS, n_perm=n_perm, seed=seed,
        half_rule=half_rule, min_trials=min_trials, min_stable=min_stable,
        n_sessions_in_scope=len(keep_sessions),
    )
    if not result.n_sessions:
        print(f"  {monkey}/{source}/{scope}/maze{maze}/{feature}: no usable session")
        return None

    fig = plot_single(result, null, monkey=monkey, maze=maze, source=source, scope=scope, feature=feature, k=k)
    # out/<source>/<monkey>/ -- the two axes you compare across, then the
    # 6 mazes x 2 features that vary within one comparison.
    out_dir = OUT_ROOT / source / monkey
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"maze{maze}_{feature}_k{k}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(
        f"Saved {path}  (z={null['z_perm']:+.2f}, delta={null['delta_obs']:+.4f}, "
        f"p={null['p_two_sided']:.4f}, n={result.n_sessions} sessions)"
    )
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--monkey", nargs="*", default=list(MONKEYS))
    parser.add_argument("--maze", type=int, nargs="*", default=list(MAZES))
    parser.add_argument("--feature", nargs="*", default=list(FEATURE_LIST), choices=tuple(FEATURES))
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--min-trials", type=int, default=corrmatrix.MIN_TRIALS)
    parser.add_argument("--min-stable", type=int, default=corrmatrix.MIN_STABLE)
    parser.add_argument("--half-rule", default="stable_shared", choices=pair.HALF_RULES)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for monkey in args.monkey:
      for maze in args.maze:
        for feature in args.feature:
            for source, scope in SOURCE_SCOPES_WANTED:
                build_one(
                    monkey, maze, source, scope, feature, k=args.k, n_perm=args.n_perm,
                    min_trials=args.min_trials, min_stable=args.min_stable,
                    half_rule=args.half_rule, seed=args.seed,
                )


if __name__ == "__main__":
    main()
