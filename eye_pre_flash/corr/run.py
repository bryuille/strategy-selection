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

For each (label_source, feature, variant, k, space) this loads
`eye_pre_flash.classifier.features.load_features` once per (monkey, k,
space) and reuses it across every source, variant and scope; writes
`examples.csv` / `examples_dims.csv` once per (source, feature, variant) at
the widest scope; and per (scope, monkey) writes a figure plus rows into a
shared `results_raw.csv` / `nulls.csv` for that leaf.

Usage:
    uv run python -m eye_pre_flash.corr.run
    uv run python -m eye_pre_flash.corr.run --source svm --scope allplus --monkey Nielsen
    uv run python -m eye_pre_flash.corr.run --feature occupancy --variant pc1_removed \\
        --k 12 --space unith --n-perm 200

Output: `eye_pre_flash/corr/out/<source>/<feature>/<variant>/[examples*.csv, <scope>/...]`.
"""

from __future__ import annotations

import argparse

import numpy as np

from data.config import MONKEYS
from data.labeler import SNR_AUC
from data.loader import load_attractor_eye_data, load_svm_choices
from eye_pre_flash.classifier.features import KS, SPACES, load_features
from eye_pre_flash.classifier.labels import labels_for_rows, monkey_for_session
from eye_pre_flash.corr import cells as cellmod
from eye_pre_flash.corr import corr_io, examples, figures, matrix, report
from eye_pre_flash.corr import variants as variantmod
from eye_pre_flash.corr.labels import (
    SOURCE_SKIP_CELLS,
    SOURCES,
    source_lookup,
    source_scope_sessions,
    top_sessions,
)
from eye_pre_flash.plotting.similarities.common import N_SPLITS

FEATURES = {"occupancy": "occ_ms", "occupancy_bin": "occ_bin", "bigram": "bigram"}
SCOPES = ("publication", "all", "allplus")
WIDEST_SCOPE = "allplus"

_AGREEMENT_CACHE: dict = {}


def _origin_index(monkey, k, space, data):
    """Codebook index nearest the screen origin, in whichever space `data` is in."""
    if space == "unith":
        codebook = load_attractor_eye_data(monkey, k=k)["codebook_xy"]
    else:
        codebook = data["codebook_deg_xy"]
    return variantmod.origin_state(codebook)


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
    n_splits, min_trials, min_stable, n_half, n_perm, n_examples, top_n, seed,
):
    print(f"\n=== source={source} feature={feature} k={k} space={space} ===")
    per_monkey = {}
    for monkey in monkeys_wanted:
        data = load_features(monkey, k=k, space=space)
        raw_X = np.asarray(data[FEATURES[feature]], dtype=float)
        origin = _origin_index(monkey, k, space, data)
        pc1 = variantmod.fit_pc1(raw_X)
        per_monkey[monkey] = dict(
            data=data,
            sessions=np.asarray(data["session"]).astype(str),
            trials=np.asarray(data["trial_indices_all"], dtype=int),
            mazes=np.asarray(data["maze_id"], dtype=int),
            raw_X=raw_X,
            origin=origin,
            pc1=pc1,
        )

    for variant in variants_wanted:
        variant_X = {}
        for monkey, d in per_monkey.items():
            X_v, dims = variantmod.apply_variant(
                d["raw_X"], variant, feature=FEATURES[feature], k=k,
                origin=d["origin"], pc1=d["pc1"][0],
            )
            variant_X[monkey] = (X_v, dims)

        variant_rel = f"{source}/{feature}/{variant}"

        ex_rows, ex_dim_rows = [], []
        by_monkey_w, _missing, _dropped = source_scope_sessions(source, WIDEST_SCOPE)
        for monkey, (X_v, dims) in variant_X.items():
            widest_sessions = by_monkey_w.get(monkey, ())
            if not widest_sessions:
                continue
            d = per_monkey[monkey]
            lookup = source_lookup(source, widest_sessions)
            y_full = labels_for_rows(d["sessions"], d["trials"], lookup)
            mask = np.isin(d["sessions"], list(widest_sessions)) & np.isfinite(y_full)
            if not mask.any():
                continue
            meta = dict(
                label_source=source, feature=feature, variant=variant, k=k, space=space,
                monkey=monkey, scope_computed_on=WIDEST_SCOPE,
            )
            pc1_info = d["pc1"] if variant == "pc1_removed" else None
            ex_rows += examples.trial_example_rows(
                X_v[mask], d["sessions"][mask], d["trials"][mask], d["mazes"][mask],
                y_full[mask].astype(int), dims=dims, meta=meta, n_per_cell=n_examples, seed=seed,
            )
            ex_rows += examples.half_mean_rows(
                X_v[mask], d["mazes"][mask], y_full[mask].astype(int), dims=dims, meta=meta, seed=seed,
            )
            ex_dim_rows += examples.dimension_stat_rows(
                X_v[mask], d["mazes"][mask], y_full[mask].astype(int), dims=dims, meta=meta,
                pc1_info=pc1_info,
            )
        corr_io.write_rows(ex_rows, "examples", rel_dir=variant_rel)
        corr_io.write_rows(ex_dim_rows, "examples_dims", rel_dir=variant_rel)

        for scope in scopes_wanted:
            raw_rows, null_rows_all = [], []
            for monkey, (X_v, dims) in variant_X.items():
                d = per_monkey[monkey]
                by_monkey_s, missing, dropped = source_scope_sessions(source, scope)
                keep_sessions = top_sessions(by_monkey_s.get(monkey, ()), top_n)
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

                stem = f"matrix_{monkey}_k{k}_{space}"
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
                corr_io.write_rows(raw_rows, "results_raw", rel_dir=f"{variant_rel}/{scope}")
                corr_io.write_rows(null_rows_all, "nulls", rel_dir=f"{variant_rel}/{scope}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=SOURCES, default=None, help="default: both")
    parser.add_argument("--feature", choices=tuple(FEATURES), default=None, help="default: all three")
    parser.add_argument("--variant", choices=variantmod.VARIANTS, default=None, help="default: all three")
    parser.add_argument("--scope", choices=SCOPES, default=None, help="default: all three")
    parser.add_argument("--monkey", choices=MONKEYS, default=None, help="default: both")
    parser.add_argument("--k", type=int, choices=KS, default=None, help="default: both")
    parser.add_argument("--space", choices=SPACES, default=None, help="default: both")
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--min-trials", type=int, default=matrix.MIN_TRIALS)
    parser.add_argument("--min-stable", type=int, default=matrix.MIN_STABLE)
    parser.add_argument("--n-half", type=int, default=None, help="fix the shared half size across sessions")
    parser.add_argument("--n-perm", type=int, default=1000, help="permutation-null resamples per maze")
    parser.add_argument("--n-examples", type=int, default=3, help="sampled trials per cell in examples.csv")
    parser.add_argument("--top-n", type=int, default=0, help="truncate each monkey's scope to its top-N sessions by snr_auc; 0 = off")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sources = (args.source,) if args.source else SOURCES
    features = (args.feature,) if args.feature else tuple(FEATURES)
    variants_wanted = (args.variant,) if args.variant else variantmod.VARIANTS
    scopes_wanted = (args.scope,) if args.scope else SCOPES
    monkeys_wanted = (args.monkey,) if args.monkey else MONKEYS
    ks = (args.k,) if args.k else KS
    spaces = (args.space,) if args.space else SPACES

    for source in sources:
        for feature in features:
            for k in ks:
                for space in spaces:
                    run_leaf(
                        source, feature, k, space,
                        variants_wanted=variants_wanted, monkeys_wanted=monkeys_wanted,
                        scopes_wanted=scopes_wanted, n_splits=args.n_splits,
                        min_trials=args.min_trials, min_stable=args.min_stable,
                        n_half=args.n_half, n_perm=args.n_perm, n_examples=args.n_examples,
                        top_n=args.top_n, seed=args.seed,
                    )


if __name__ == "__main__":
    main()
