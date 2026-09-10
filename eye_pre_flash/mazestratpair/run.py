"""2x2 (H, S) split-half occupancy similarity for a single maze.

`eye_pre_flash.corr` builds a 12x12 (maze x decoded strategy) matrix and
argues its decisive read is the same-maze H-vs-S comparison: geometry is
identical on both sides, so a cross-strategy correlation below that maze's own
split-half reliability is strategy-dependent gaze sampling, not a visual
confound. This package is that read on its own terms -- one maze, two cells --
and defaults to **Nielsen maze 4**, the one cell where both label sources
independently agree the strategy split is genuinely mixed (dendro 332 H /
410 S, sequential share 0.553; SVM 352 H / 390 S, share 0.526) and which
`classifier.labels.ANCHOR_MAJORITY` deliberately leaves unscored, so nothing
about its labels is circular.

Restricting to one maze is not just a crop. `corr.matrix.session_half_sizes`
picks one shared half size per session over every cell that clears
`min_stable`, so maze 4's comfortable counts get halved at the size set by
maze 1's near-empty sequential cell -- Nielsen `Nov_1_g0` has 45 H and 36 S
maze-4 trials and corr splits it into halves of 5, where these two cells alone
give 18. Same estimator, 3.6x the trials per half, and therefore different
(better) numbers for the same cells. `--n-half` forces corr's size back for a
like-for-like check, and the report prints both rules side by side.

Each figure holds all three feature variants as three 2x2 panels, so the
variant is a panel and a CSV column rather than a directory. Significance is
the within-(session, maze) label-shuffle permutation null on
``Δ = 0.5*(r_HH + r_SS) − r_HS``: the shuffle preserves every cell's trial
count exactly, hence both half sizes and both thin flags, so it tests the
labels rather than the coverage.

Usage:
    uv run python -m eye_pre_flash.mazestratpair.run
    uv run python -m eye_pre_flash.mazestratpair.run --source svm --k 6 --feature bigram
    uv run python -m eye_pre_flash.mazestratpair.run --monkey Faure --maze 2
    uv run python -m eye_pre_flash.mazestratpair.run --all-monkeys --all-mazes --n-perm 200

Defaults are Nielsen and maze 4; every axis takes a list, so the same pipeline
sweeps other mazes and Faure without editing anything. Scopes are per source
(`corr.labels.SOURCE_SCOPES`): `dendro` runs `publication` and `top_four`,
`svm` runs `top_ten`.

Output: `eye_pre_flash/mazestratpair/out/<source>/<feature>/<scope>/`.
"""

from __future__ import annotations

import argparse

import numpy as np

from data.config import MONKEYS
from eye_pre_flash.classifier.features import KS, load_features
from eye_pre_flash.corr import matrix as corrmatrix
from eye_pre_flash.corr import variants as variantmod
from eye_pre_flash.corr.cells import MAZES
from eye_pre_flash.corr.labels import (
    SCOPES,
    SOURCE_SCOPES,
    SOURCE_SKIP_CELLS,
    SOURCES,
    WIDEST_SCOPE,
    source_lookup,
    source_scope_sessions,
)
from eye_pre_flash.corr.run import FEATURES, SPACE, _session_diagnostics
from eye_pre_flash.corr.variants import VARIANTS
from eye_pre_flash.mazestratpair import figures, pair, pair_io, report
from eye_pre_flash.plotting.similarities.common import N_SPLITS


def _stem(kind, monkey, maze, k, variants):
    """`kind_<Monkey>_maze<M>_k<K>`, plus a variant tag for a narrowed rerun.

    Monkey, maze and K are always in the stem because `pair_io.write_rows`
    truncates -- a stem shared between two mazes would silently overwrite. The
    variant tag is appended only when fewer than all three variants ran, so a
    debugging rerun of one panel cannot clobber the canonical 3-panel outputs.
    """
    stem = f"{kind}_{monkey}_maze{maze}_k{k}"
    if tuple(variants) != tuple(VARIANTS):
        stem += "_" + "-".join(variants)
    return stem


def build_raw_rows(
    result, *, monkey, maze, scope, source, feature, variant, k, space, dim,
    n_splits, min_trials, min_stable, half_rule, n_half, seed,
):
    """One row per session, for this (variant) panel."""
    names = pair.cell_labels(maze)
    rows = []
    for i, session in enumerate(result.session_names):
        counts = result.cell_counts[i]
        halves = result.half_sizes[i]
        corr_halves = result.half_corr_rule[i]
        stable = result.stable[i]
        mat = result.session_mats[i]
        rows.append(
            dict(
                scope=scope, label_source=source, feature=feature, variant=variant,
                k=k, space=space, dim=dim, monkey=monkey, maze=maze, session=session,
                **_session_diagnostics(source, session),
                name_H=names[0], name_S=names[1],
                n_trials_H=counts.get(0, 0), n_trials_S=counts.get(1, 0),
                half_rule=half_rule,
                half_H=halves.get(0, 0), half_S=halves.get(1, 0),
                half_corr_rule_H=corr_halves.get(0, 0),
                half_corr_rule_S=corr_halves.get(1, 0),
                equal_n=halves.get(0) == halves.get(1),
                stable_H=stable.get(0, False), stable_S=stable.get(1, False),
                r_HH=_num(mat[0, 0]), r_SS=_num(mat[1, 1]), r_HS=_num(mat[0, 1]),
                delta=_num(result.session_deltas[i]),
                n_degenerate=result.n_degenerate,
                n_splits=n_splits, min_trials=min_trials, min_stable=min_stable,
                n_half="" if n_half is None else n_half, seed=seed,
            )
        )
    return rows


def _num(value):
    return float(value) if np.isfinite(value) else ""


def run_leaf(
    source, feature, k, *, mazes, monkeys_wanted, scopes_wanted, variants_wanted,
    n_splits, min_trials, min_stable, half_rule, n_half, n_perm, write_draws, seed,
):
    scopes = tuple(s for s in scopes_wanted if s in SOURCE_SCOPES[source])
    if not scopes:
        return
    print(f"\n=== source={source} feature={feature} k={k} space={SPACE} ===")

    widest = WIDEST_SCOPE[source]
    by_monkey_w, _missing, _dropped = source_scope_sessions(source, widest)

    for monkey in monkeys_wanted:
        data = load_features(monkey, k=k, space=SPACE)
        raw_X = np.asarray(data[FEATURES[feature]], dtype=float)
        sessions = np.asarray(data["session"]).astype(str)
        trials = np.asarray(data["trial_indices_all"], dtype=int)
        mazes_all = np.asarray(data["maze_id"], dtype=int)
        origin = variantmod.origin_state(data["codebook_xy"])

        # The grand-mean profile stays exactly where corr fits it: once per
        # (monkey, feature, k) over the widest scope's labelled trials across
        # ALL mazes. Refitting it on this maze alone would be a different
        # transform -- it would let the profile absorb this maze's own
        # geometry, which is the confound the analysis exists to exclude --
        # and would make the mean_removed panel incomparable with corr's.
        widest_sessions = by_monkey_w.get(monkey, ())
        y_widest = pair_labels(sessions, trials, source, widest_sessions)
        fit_mask = np.isin(sessions, list(widest_sessions)) & np.isfinite(y_widest)
        if not fit_mask.any():
            print(f"  {monkey}: no labelled trial in {widest}; skipping")
            continue
        mean_profile = variantmod.fit_grand_mean(raw_X[fit_mask])

        variant_X = {}
        for variant in variants_wanted:
            variant_X[variant] = variantmod.apply_variant(
                raw_X, variant, feature=FEATURES[feature], k=k,
                origin=origin, mean_profile=mean_profile,
            )

        for scope in scopes:
            by_monkey_s, missing, dropped = source_scope_sessions(source, scope)
            keep_sessions = by_monkey_s.get(monkey, ())
            if missing:
                print(f"  {monkey}/{scope}: {len(missing)} session(s) not labelled, skipped")
            if dropped:
                print(f"  {monkey}/{scope}: {len(dropped)} session(s) dropped on vetting")
            if not keep_sessions:
                print(f"  {monkey}/{scope}: no sessions in scope; skipping")
                continue
            lookup = source_lookup(source, keep_sessions)

            for maze in mazes:
                skipped = pair.skip_pair_cells(SOURCE_SKIP_CELLS[source], maze)
                if skipped:
                    blank = ", ".join(pair.cell_labels(maze)[s] for s in skipped)
                    print(
                        f"  {monkey}/{scope}/maze {maze}: {blank} is skipped by the "
                        f"{source} source policy (corr.labels.SOURCE_SKIP_CELLS) as an "
                        f"anchor-minority cell, so this maze has no H/S pair; skipping"
                    )
                    continue

                results, nulls, dims, draw_rows = {}, {}, {}, []
                for variant in variants_wanted:
                    X_v, dim_labels = variant_X[variant]
                    dims[variant] = X_v.shape[1]
                    pools = pair.build_pools(
                        X_v, sessions, trials, mazes_all, maze=maze,
                        keep_sessions=keep_sessions, label_lookup=lookup,
                        min_trials=min_trials, min_stable=min_stable,
                        n_half=n_half, half_rule=half_rule, seed=seed,
                    )
                    row, result, draws = pair.permutation_null(
                        pools, maze=maze, variant=variant, n_splits=n_splits,
                        n_perm=n_perm, seed=seed, half_rule=half_rule,
                        min_trials=min_trials, min_stable=min_stable,
                        n_sessions_in_scope=len(keep_sessions),
                        skipped_cells=skipped,
                    )
                    results[variant] = result
                    nulls[variant] = row
                    if write_draws:
                        for p_i, value in enumerate(draws):
                            draw_rows.append(
                                dict(
                                    label_source=source, feature=feature,
                                    variant=variant, scope=scope, monkey=monkey,
                                    maze=maze, k=k, perm_index=p_i,
                                    delta_null=_num(value),
                                )
                            )
                    tag = f"  {monkey}/{scope}/maze {maze}/{variant}:"
                    if np.isfinite(row["delta_obs"]):
                        print(
                            f"{tag} {result.n_sessions} sessions, "
                            f"Δ = {row['delta_obs']:+.4f}, "
                            f"null {row['null_mean']:+.4f} ± {row['null_sd']:.4f}, "
                            f"z = {row['z_perm']:+.2f}, p = {row['p_two_sided']:.4f}"
                        )
                    else:
                        print(f"{tag} no usable session")

                rel_dir = f"{source}/{feature}/{scope}"
                figures.plot_pair_panels(
                    results, nulls, dims, monkey=monkey, maze=maze, scope=scope,
                    source=source, feature=feature, k=k, space=SPACE,
                    n_splits=n_splits, min_stable=min_stable, n_perm=n_perm,
                    half_rule=half_rule, variants=variants_wanted,
                    stem=_stem("pair", monkey, maze, k, variants_wanted),
                    rel_dir=rel_dir,
                )
                report.write_report(
                    results, nulls, dims, monkey=monkey, maze=maze, source=source,
                    feature=feature, scope=scope, k=k, space=SPACE,
                    n_splits=n_splits, min_trials=min_trials, min_stable=min_stable,
                    half_rule=half_rule, n_perm=n_perm, variants=variants_wanted,
                )

                raw_rows, null_rows = [], []
                for variant in variants_wanted:
                    result, row = results[variant], nulls[variant]
                    raw_rows += build_raw_rows(
                        result, monkey=monkey, maze=maze, scope=scope,
                        source=source, feature=feature, variant=variant, k=k,
                        space=SPACE, dim=dims[variant], n_splits=n_splits,
                        min_trials=min_trials, min_stable=min_stable,
                        half_rule=half_rule, n_half=n_half, seed=seed,
                    )
                    thin = pair.thin_mask(result)
                    null_rows.append(
                        dict(
                            label_source=source, feature=feature, variant=variant,
                            scope=scope, monkey=monkey, maze=maze, k=k, space=SPACE,
                            dim=dims[variant],
                            n_sessions_in_scope=len(keep_sessions),
                            n_sessions_used=result.n_sessions,
                            session_names=";".join(result.session_names),
                            census_H=int(result.census[0]), census_S=int(result.census[1]),
                            half_rule=half_rule,
                            half_H_median=_median(result, 0),
                            half_S_median=_median(result, 1),
                            half_corr_rule_H_median=_median(result, 0, corr_rule=True),
                            half_corr_rule_S_median=_median(result, 1, corr_rule=True),
                            **{key: _num(value) if isinstance(value, float) else value
                               for key, value in row.items()},
                            thin_H=bool(thin[0]), thin_S=bool(thin[1]),
                            skipped_H=0 in skipped, skipped_S=1 in skipped,
                            n_degenerate=result.n_degenerate,
                            n_splits=n_splits, min_trials=min_trials,
                            min_stable=min_stable,
                            n_half="" if n_half is None else n_half, seed=seed,
                        )
                    )
                pair_io.write_rows(
                    raw_rows, _stem("results_raw", monkey, maze, k, variants_wanted),
                    rel_dir=rel_dir,
                )
                pair_io.write_rows(
                    null_rows, _stem("nulls", monkey, maze, k, variants_wanted),
                    rel_dir=rel_dir,
                )
                if write_draws:
                    pair_io.write_rows(
                        draw_rows,
                        _stem("null_draws", monkey, maze, k, variants_wanted),
                        rel_dir=rel_dir,
                    )


def _median(result, cell, *, corr_rule=False):
    source = result.half_corr_rule if corr_rule else result.half_sizes
    values = [h[cell] for h in source if cell in h]
    return int(np.median(values)) if values else 0


def pair_labels(sessions, trials, source, keep_sessions):
    from eye_pre_flash.classifier.labels import labels_for_rows

    return labels_for_rows(sessions, trials, source_lookup(source, keep_sessions))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--monkey", choices=MONKEYS, nargs="*", default=["Nielsen"])
    parser.add_argument("--maze", type=int, choices=MAZES, nargs="*", default=[pair.DEFAULT_MAZE])
    parser.add_argument("--all-monkeys", action="store_true", help="both monkeys")
    parser.add_argument("--all-mazes", action="store_true", help="mazes 1-6")
    parser.add_argument("--source", choices=SOURCES, nargs="*", default=None, help="default: both")
    parser.add_argument("--feature", choices=tuple(FEATURES), nargs="*", default=None,
                        help="default: all three")
    parser.add_argument("--variant", choices=VARIANTS, nargs="*", default=None,
                        help="default: all three, one panel each; a subset gets its own file stem")
    parser.add_argument("--scope", choices=SCOPES, nargs="*", default=None,
                        help="default: every scope defined for each source (corr.labels.SOURCE_SCOPES)")
    parser.add_argument("--k", type=int, choices=KS, nargs="*", default=None, help="default: both")
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--min-trials", type=int, default=corrmatrix.MIN_TRIALS)
    parser.add_argument("--min-stable", type=int, default=corrmatrix.MIN_STABLE)
    parser.add_argument(
        "--half-rule", choices=pair.HALF_RULES, default="stable_shared",
        help="stable_shared is corr's rule verbatim (default); min_pair always balances "
             "the two halves. They coincide unless one strategy is thin.",
    )
    parser.add_argument("--n-half", type=int, default=None,
                        help="fix the shared half size across sessions (use corr's to compare)")
    parser.add_argument("--n-perm", type=int, default=1000,
                        help="permutation-null resamples per (variant, maze)")
    parser.add_argument("--no-null-draws", dest="write_draws", action="store_false",
                        default=True, help="skip null_draws_*.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sources = tuple(args.source) if args.source else SOURCES
    scopes_wanted = tuple(args.scope) if args.scope else SCOPES
    for source in sources:
        if args.scope and not set(scopes_wanted) & set(SOURCE_SCOPES[source]):
            parser.error(
                f"none of --scope {' '.join(scopes_wanted)} is defined for --source "
                f"{source}; choose from {', '.join(SOURCE_SCOPES[source])} "
                f"(scopes are per source, see corr.labels.SOURCE_SCOPES)"
            )

    monkeys = MONKEYS if args.all_monkeys else tuple(args.monkey)
    mazes = MAZES if args.all_mazes else tuple(args.maze)
    features = tuple(args.feature) if args.feature else tuple(FEATURES)
    # Panel order follows VARIANTS, not the order the flag was typed, so
    # `--variant mean_removed full` still reads left-to-right as full,
    # mean_removed -- the same order as every other figure in the repo.
    variants_wanted = (
        tuple(v for v in VARIANTS if v in set(args.variant))
        if args.variant
        else VARIANTS
    )
    ks = tuple(args.k) if args.k else KS

    for source in sources:
        for feature in features:
            for k in ks:
                run_leaf(
                    source, feature, k, mazes=mazes, monkeys_wanted=monkeys,
                    scopes_wanted=scopes_wanted, variants_wanted=variants_wanted,
                    n_splits=args.n_splits, min_trials=args.min_trials,
                    min_stable=args.min_stable, half_rule=args.half_rule,
                    n_half=args.n_half, n_perm=args.n_perm,
                    write_draws=args.write_draws, seed=args.seed,
                )


if __name__ == "__main__":
    main()
