"""12x12 (maze x decoded strategy) occupancy similarity matrices.

Splits each maze's trials by a decoded neural strategy label and correlates
the resulting occupancy vectors, in the `similarities.heatmap_eq` split-half
CV style, extended to the label axis: cells are ``(maze, strategy)``, mazes
1-6 hierarchical then mazes 1-6 sequential (`eye_pre_flash.corr.cells`). The
6x6 maze-only matrices cannot separate strategy from geometry -- strategy is
close to a deterministic function of (session, maze) -- so holding maze type
fixed and reading the boxed cells (same maze, H vs S) is the decisive test:
geometry is identical on both sides, so anything below that maze's own
split-half reliability is strategy-dependent gaze sampling, not a visual
confound.

This revives the deleted `eye_pre_flash.plotting.similarities.heatmap_labels`
(`git show HEAD:...`) against low-dimensional codebook-occupancy features
rather than 102,400-dim spatial gaze maps, adds a second label source (a
maze-1-vs-6 linear SVM, `data.labeler.build_svm_choices`, alongside the
existing Ward-clustering labels), adds two variants that strip the dominant
central-fixation profile, and replaces the assumed-good-estimator bootstrap
with a within-(session, maze) label-shuffle permutation null -- because at
d as low as 6 or 12 the estimator's own noise floor should not be assumed,
it should be measured.

Everything here is unit-H (`SPACE`); `deg` was dropped as a swept axis
because a degree-space codebook re-encodes maze geometry (see the comment on
`SPACE` and `corr.md` section 5).

For each (label_source, feature, variant, k) this loads
`eye_pre_flash.classifier.features.load_features` once per (monkey, k) and
reuses it across every source, variant and scope; writes
`examples_k<k>.csv` / `examples_dims_k<k>.csv` once per (source, feature,
variant) at the widest scope; and per (scope, monkey) writes a figure plus
rows into a shared `results_raw_k<k>.csv` / `nulls_k<k>.csv` for that leaf.
The CSV stems carry `k` because `write_rows` truncates: with `k` only in the
sweep and not in the path, each k overwrote the previous one's rows.

Usage:
    uv run python -m eye_pre_flash.corr.run
    uv run python -m eye_pre_flash.corr.run --source svm --scope top_ten --monkey Nielsen
    uv run python -m eye_pre_flash.corr.run --feature occupancy --variant mean_removed \\
        --k 12 --n-perm 200

Scopes are per source (`corr.labels.SOURCE_SCOPES`): `dendro` runs
`publication` and `top_four`, `svm` runs `top_ten`.

Output: `eye_pre_flash/corr/out/<source>/<feature>/<variant>/[examples*.csv, <scope>/...]`.
"""

from __future__ import annotations

import argparse

import numpy as np

from data.config import MONKEYS
from data.labeler import SNR_AUC
from data.loader import load_svm_choices
from eye_pre_flash.classifier.features import KS, load_features
from eye_pre_flash.classifier.labels import labels_for_rows, monkey_for_session
from eye_pre_flash.corr import cells as cellmod
from eye_pre_flash.corr import corr_io, examples, figures, matrix, report
from eye_pre_flash.corr import variants as variantmod
from eye_pre_flash.corr.labels import (
    SCOPES,
    SOURCE_SCOPES,
    SOURCE_SKIP_CELLS,
    SOURCES,
    WIDEST_SCOPE,
    source_lookup,
    source_scope_sessions,
)
from eye_pre_flash.plotting.similarities.common import N_SPLITS

FEATURES = {"occupancy": "occ_ms", "occupancy_bin": "occ_bin", "bigram": "bigram"}
# Unit-H only. A `deg` codebook is fit on pooled screen degrees, so its states
# sit where *some* mazes' arms are and a given state is on-arm for one geometry
# and off-arm for another -- `deg` occupancy therefore encodes which maze was on
# screen, which is the one confound this analysis exists to exclude. `deg`
# remains correct for `classifier.decoding`'s per-maze regime, where geometry is
# constant; it is never right for these cross-maze cells. Still carried as a
# column for provenance, but no longer a swept axis or a filename component.
SPACE = "unith"
_AGREEMENT_CACHE: dict = {}


def _origin_index(monkey, k, space, data):
    """Codebook index nearest the screen origin, in whichever space `data` is in.

    Read off the feature block's own `codebook_xy`, which
    `classifier.features` fits on in-window samples. Not off
    `data.attractor`'s cached `codebook_xy` -- that one is fit on whole-trial
    gaze and is no longer the book these features were assigned against, so
    using it would name the wrong dimension as the origin state.
    """
    return variantmod.origin_state(data["codebook_xy"])


def _dendro_svm_agreement_2345(session):
    """Agreement of the dendro and SVM labels on mazes 2, 3, 5 for `session`.

    The only external handle on the SVM labels' quality where the analysis
    actually depends on them: mazes 1 and 6 are the SVM's own training axis
    (validated out-of-fold, see `data.labeler.build_svm_choices`), but mazes
    2-5 are never in any fold's train or test split. Cached per session --
    called once per raw-CSV row that needs it, many times over a full sweep.
    """
    if session in _AGREEMENT_CACHE:
        return _AGREEMENT_CACHE[session]
    try:
        monkey = monkey_for_session(session)
        data = load_features(monkey, k=6, space="unith")
        rows = np.asarray(data["session"]).astype(str)
        mazes = np.asarray(data["maze_id"], dtype=int)
        y_d = labels_for_rows(
            rows, data["trial_indices_all"], source_lookup("dendro", (session,))
        )
        y_s = labels_for_rows(
            rows, data["trial_indices_all"], source_lookup("svm", (session,))
        )
        keep = (
            (rows == session)
            & np.isfinite(y_d)
            & np.isfinite(y_s)
            & np.isin(mazes, (2, 3, 5))
        )
        out = float((y_d[keep].astype(int) == y_s[keep].astype(int)).mean()) if keep.any() else np.nan
    except Exception:  # noqa: BLE001 - a diagnostic column must not kill the sweep
        out = np.nan
    _AGREEMENT_CACHE[session] = out
    return out


def _session_diagnostics(source, session):
    diag = {
        "snr_auc": SNR_AUC.get(session, ""),
        "svm_auc": "",
        "oof_balanced_acc": "",
        "dendro_agreement_2345": "",
    }
    if source == "svm":
        try:
            d = load_svm_choices(session)
            diag["svm_auc"] = float(d["svm_auc"])
            diag["oof_balanced_acc"] = float(d["oof_balanced_acc"])
        except Exception:  # noqa: BLE001
            pass
        diag["dendro_agreement_2345"] = _dendro_svm_agreement_2345(session)
    return diag


def build_raw_rows(
    result, *, monkey, scope, source, feature, variant, k, space, dim,
    n_splits, min_trials, min_stable, seed,
):
    names = cellmod.cell_labels()
    n = cellmod.N_CELLS
    rows = []
    for si, session in enumerate(result.session_names):
        mat = result.session_mats[si]
        counts = result.cell_counts[si]
        stable = result.stable[si]
        halves = result.half_sizes[si]
        diag = _session_diagnostics(source, session)
        for i in range(n):
            maze_i, strat_i = cellmod.cell_maze_strategy(i)
            for j in range(n):
                maze_j, strat_j = cellmod.cell_maze_strategy(j)
                r_val = mat[i, j]
                rows.append(
                    dict(
                        scope=scope, label_source=source, feature=feature,
                        variant=variant, k=k, space=space, dim=dim, monkey=monkey,
                        session=session, **diag,
                        half_i=halves.get(i, ""), half_j=halves.get(j, ""),
                        n_splits=n_splits, min_trials=min_trials, min_stable=min_stable,
                        seed=seed, cell_i=i, cell_j=j, maze_i=maze_i, strategy_i=strat_i,
                        maze_j=maze_j, strategy_j=strat_j, name_i=names[i], name_j=names[j],
                        r=float(r_val) if np.isfinite(r_val) else "",
                        n_trials_i=counts.get(i, 0), n_trials_j=counts.get(j, 0),
                        present_i=i in counts, present_j=j in counts,
                        stable_i=stable.get(i, False), stable_j=stable.get(j, False),
                        n_degenerate=result.n_degenerate,
                    )
                )
    return rows


def run_leaf(
    source, feature, k, space, *, variants_wanted, monkeys_wanted, scopes_wanted,
    n_splits, min_trials, min_stable, n_half, n_perm, n_examples, seed,
):
    scopes = tuple(s for s in scopes_wanted if s in SOURCE_SCOPES[source])
    if not scopes:
        return
    print(f"\n=== source={source} feature={feature} k={k} space={space} ===")
    # The `mean_removed` grand-mean profile (and the `pc1` kept to audit the
    # retired `pc1_removed`) are fitted here, once per monkey, over the widest
    # scope's labelled trials -- not per session, per cell or per scope. Fitting
    # per session would put each session's matrix in a different subspace, the
    # cross-session comparability failure `variants` documents; fitting per
    # scope would make `publication` incomparable to the widest scope.
    widest = WIDEST_SCOPE[source]
    by_monkey_w, _missing, _dropped = source_scope_sessions(source, widest)
    per_monkey = {}
    for monkey in monkeys_wanted:
        data = load_features(monkey, k=k, space=space)
        raw_X = np.asarray(data[FEATURES[feature]], dtype=float)
        sessions = np.asarray(data["session"]).astype(str)
        trials = np.asarray(data["trial_indices_all"], dtype=int)
        mazes = np.asarray(data["maze_id"], dtype=int)
        origin = _origin_index(monkey, k, space, data)

        widest_sessions = by_monkey_w.get(monkey, ())
        y_full = labels_for_rows(sessions, trials, source_lookup(source, widest_sessions))
        fit_mask = np.isin(sessions, list(widest_sessions)) & np.isfinite(y_full)
        if not fit_mask.any():
            print(f"  {monkey}: no labelled trial in {widest}; skipping")
            continue

        mean_profile = variantmod.fit_grand_mean(raw_X[fit_mask])
        pc1, explained = variantmod.fit_pc1(raw_X[fit_mask])
        fit_stats = variantmod.pc1_diagnostics(
            raw_X[fit_mask], pc1=pc1, explained=explained, mean_profile=mean_profile,
            sessions=sessions[fit_mask], mazes=mazes[fit_mask],
            y=y_full[fit_mask].astype(int),
        )
        print(
            f"  {monkey}: PC1 explains {explained:.3f}; |cos(PC1, grand mean)| = "
            f"{fit_stats['pc1_mean_cos_abs']:.3f}; PC1-label r = "
            f"{fit_stats['pc1_label_pointbiserial']:.3f} pooled, "
            f"{fit_stats['pc1_label_pointbiserial_within']:.3f} within (session, maze)"
        )
        per_monkey[monkey] = dict(
            data=data, sessions=sessions, trials=trials, mazes=mazes, raw_X=raw_X,
            origin=origin, mean_profile=mean_profile, y_full=y_full, fit_mask=fit_mask,
            fit_info=dict(pc1=pc1, mean_profile=mean_profile, stats=fit_stats),
        )

    for variant in variants_wanted:
        variant_X = {}
        for monkey, d in per_monkey.items():
            X_v, dims = variantmod.apply_variant(
                d["raw_X"], variant, feature=FEATURES[feature], k=k,
                origin=d["origin"], mean_profile=d["mean_profile"],
            )
            variant_X[monkey] = (X_v, dims)

        variant_rel = f"{source}/{feature}/{variant}"

        ex_rows, ex_dim_rows = [], []
        for monkey, (X_v, dims) in variant_X.items():
            d = per_monkey[monkey]
            mask, y_full = d["fit_mask"], d["y_full"]
            meta = dict(
                label_source=source, feature=feature, variant=variant, k=k, space=space,
                monkey=monkey, scope_computed_on=widest,
            )
            # Both fits are properties of the untransformed block, so recording
            # them under all three variants would triple identical rows.
            fit_info = d["fit_info"] if variant == "mean_removed" else None
            ex_rows += examples.trial_example_rows(
                X_v[mask], d["sessions"][mask], d["trials"][mask], d["mazes"][mask],
                y_full[mask].astype(int), dims=dims, meta=meta, n_per_cell=n_examples, seed=seed,
            )
            ex_rows += examples.half_mean_rows(
                X_v[mask], d["mazes"][mask], y_full[mask].astype(int), dims=dims, meta=meta, seed=seed,
            )
            ex_dim_rows += examples.dimension_stat_rows(
                X_v[mask], d["mazes"][mask], y_full[mask].astype(int), dims=dims, meta=meta,
                fit_info=fit_info,
            )
        corr_io.write_rows(ex_rows, f"examples_k{k}", rel_dir=variant_rel)
        corr_io.write_rows(ex_dim_rows, f"examples_dims_k{k}", rel_dir=variant_rel)

        for scope in scopes:
            raw_rows, null_rows_all = [], []
            for monkey, (X_v, dims) in variant_X.items():
                d = per_monkey[monkey]
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
                null_rows, result = matrix.permutation_null(
                    X_v, d["sessions"], d["trials"], d["mazes"],
                    keep_sessions=keep_sessions, label_lookup=lookup, n_perm=n_perm,
                    seed=seed, n_splits=n_splits, min_trials=min_trials,
                    min_stable=min_stable, skip_cells=SOURCE_SKIP_CELLS[source], n_half=n_half,
                )
                if result.n_sessions == 0:
                    print(f"  {monkey}/{scope}: no session produced a usable matrix; skipping")
                    continue

                stem = f"matrix_{monkey}_k{k}"
                rel_dir = f"{variant_rel}/{scope}"
                figures.plot_label_matrix(
                    result, monkey=monkey, scope=scope, source=source, feature=feature,
                    variant=variant, k=k, space=space, dim=X_v.shape[1], n_splits=n_splits,
                    min_stable=min_stable, stem=stem, rel_dir=rel_dir,
                )
                report.write_report(
                    result, null_rows, monkey=monkey, source=source, feature=feature,
                    variant=variant, scope=scope, k=k, space=space, dim=X_v.shape[1],
                    n_splits=n_splits, min_trials=min_trials, min_stable=min_stable,
                )
                raw_rows += build_raw_rows(
                    result, monkey=monkey, scope=scope, source=source, feature=feature,
                    variant=variant, k=k, space=space, dim=X_v.shape[1], n_splits=n_splits,
                    min_trials=min_trials, min_stable=min_stable, seed=seed,
                )
                for nrow in null_rows:
                    null_rows_all.append(
                        {
                            "scope": scope, "label_source": source, "feature": feature,
                            "variant": variant, "k": k, "space": space, "monkey": monkey,
                            **nrow,
                        }
                    )

            if raw_rows:
                corr_io.write_rows(raw_rows, f"results_raw_k{k}", rel_dir=f"{variant_rel}/{scope}")
                corr_io.write_rows(null_rows_all, f"nulls_k{k}", rel_dir=f"{variant_rel}/{scope}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=SOURCES, default=None, help="default: both")
    parser.add_argument("--feature", choices=tuple(FEATURES), default=None, help="default: all three")
    parser.add_argument("--variant", choices=variantmod.VARIANTS, default=None, help="default: all three")
    parser.add_argument(
        "--scope", choices=SCOPES, default=None,
        help="default: every scope defined for each source (see corr.labels.SOURCE_SCOPES)",
    )
    parser.add_argument("--monkey", choices=MONKEYS, default=None, help="default: both")
    parser.add_argument("--k", type=int, choices=KS, default=None, help="default: both")
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--min-trials", type=int, default=matrix.MIN_TRIALS)
    parser.add_argument("--min-stable", type=int, default=matrix.MIN_STABLE)
    parser.add_argument("--n-half", type=int, default=None, help="fix the shared half size across sessions")
    parser.add_argument("--n-perm", type=int, default=1000, help="permutation-null resamples per maze")
    parser.add_argument("--n-examples", type=int, default=3, help="sampled trials per cell in examples.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.source and args.scope and args.scope not in SOURCE_SCOPES[args.source]:
        parser.error(
            f"--scope {args.scope} is not defined for --source {args.source}; "
            f"choose from {', '.join(SOURCE_SCOPES[args.source])} "
            f"(scopes are per source, see corr.labels.SOURCE_SCOPES)"
        )

    sources = (args.source,) if args.source else SOURCES
    features = (args.feature,) if args.feature else tuple(FEATURES)
    variants_wanted = (args.variant,) if args.variant else variantmod.VARIANTS
    scopes_wanted = (args.scope,) if args.scope else SCOPES
    monkeys_wanted = (args.monkey,) if args.monkey else MONKEYS
    ks = (args.k,) if args.k else KS

    for source in sources:
        for feature in features:
            for k in ks:
                run_leaf(
                    source, feature, k, SPACE,
                    variants_wanted=variants_wanted, monkeys_wanted=monkeys_wanted,
                    scopes_wanted=scopes_wanted, n_splits=args.n_splits,
                    min_trials=args.min_trials, min_stable=args.min_stable,
                    n_half=args.n_half, n_perm=args.n_perm, n_examples=args.n_examples,
                    seed=args.seed,
                )


if __name__ == "__main__":
    main()
