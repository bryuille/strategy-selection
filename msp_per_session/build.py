"""MSP 2x2 / strategy / codebook panels, one recording session at a time.

Reuses ``msp`` for features, labels, estimator, and figures.
Does not edit that package. Each (monkey, session, maze, variant) cell uses
only that session's labelled trials of the maze; the within-session label
shuffle null is then a shuffle of the sole session.

Usage:
    uv run python -m msp_per_session.build --dry-run
    uv run python -m msp_per_session.build --radius 0.5
    uv run python -m msp_per_session.build --radius 0.5 --window pre1466   # legacy window
    uv run python -m msp_per_session.build \\
        --monkey Faure --session june_24_g0 --maze 2 --variant mean_removed

Writes under ``msp_per_session/out/<tag>/<Monkey>/``, grouped by plot type:

  ``<variant>/maze<M>_<session>.png``, ``<variant>/results.csv``
  ``codebook/maze<M>_<session>.png``, ``codebook/all_<session>.png``
  ``strategy/maze<M>_<session>.png``
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from msp import config as cfg
from msp import estimator
from msp import features
from msp import figures as figmod
from msp import labels as labelmod

MONKEYS = ("Faure", "Nielsen")
MAZES = (2, 3, 4, 5)
FEATURE_BLOCK = "occ_bin"
OUT_ROOT = Path(__file__).resolve().parent / "out"


def monkey_rel_dir(tag, monkey):
    return f"{cfg.radius_dir(tag)}/{monkey}"


def type_rel_dir(tag, monkey, plot_type):
    return f"{monkey_rel_dir(tag, monkey)}/{plot_type}"


def maze_session_stem(maze, session):
    return f"maze{int(maze)}_{session}"


def all_session_stem(session):
    return f"all_{session}"


def figure_png(out_root, tag, monkey, variant, maze, session):
    return (
        out_root
        / type_rel_dir(tag, monkey, variant)
        / f"{maze_session_stem(maze, session)}.png"
    )


def results_csv(out_root, tag, monkey, variant):
    return out_root / type_rel_dir(tag, monkey, variant) / "results.csv"


def _meta(data):
    return (
        np.asarray(data["session"]).astype(str),
        np.asarray(data["trial_indices_all"], dtype=int),
        np.asarray(data["maze_id"], dtype=int),
    )


def scope(monkey, session_filter=None):
    """``(sessions, lookup)`` for the monkey's svm/top_ten scope, optionally filtered."""
    keep = list(labelmod.top_ten_sessions(monkey))
    if session_filter is not None:
        wanted = set(session_filter)
        missing = sorted(wanted - set(keep))
        if missing:
            raise SystemExit(
                f"{monkey}: session(s) not in top-ten SVM scope: {', '.join(missing)}; "
                f"in scope: {', '.join(keep)}"
            )
        keep = [s for s in keep if s in wanted]
    return tuple(keep), labelmod.svm_lookup(keep)


def pool_maze(data, *, maze, session, lookup):
    """Row mask, labels and session ids for one session's labelled trials of one maze."""
    sessions, trials, mazes = _meta(data)
    y_full = labelmod.labels_for_rows(sessions, trials, lookup)
    mask = (sessions == session) & np.isfinite(y_full) & (mazes == maze)
    return mask, y_full[mask].astype(int), sessions[mask]


def maze_diagnostics(data, mask):
    occ = np.asarray(data[FEATURE_BLOCK], dtype=float)[mask]
    n_fix = int(np.asarray(data["n_fix"], dtype=int)[mask].sum())
    n_ok = int(np.asarray(data["n_fix_assigned"], dtype=int)[mask].sum())
    if occ.shape[0] == 0:
        return dict(
            assigned=np.nan,
            visit=np.full(cfg.K, np.nan),
            zero_full=np.nan,
            zero_no_origin=np.nan,
            n_fix=n_fix,
        )
    return dict(
        assigned=n_ok / max(n_fix, 1),
        visit=occ.mean(axis=0),
        zero_full=float((occ.sum(axis=1) == 0).mean()),
        zero_no_origin=float((occ[:, features.no_origin_dims()].sum(axis=1) == 0).mean()),
        n_fix=n_fix,
    )


def format_diagnostics(diag):
    visit = "  ".join(
        f"{name} {v:.2f}" for name, v in zip(cfg.STATE_NAMES, diag["visit"])
    )
    return (
        f"fixations assigned {diag['assigned']:.1%} of {diag['n_fix']}; "
        f"visit rate: {visit}; all-zero rows: full {diag['zero_full']:.1%}, "
        f"no_origin {diag['zero_no_origin']:.1%}"
    )


def run_cell(data, *, maze, session, variant, lookup, args):
    mask, y, session_ids = pool_maze(
        data, maze=maze, session=session, lookup=lookup
    )
    if mask.sum() == 0:
        return None, "no labelled trial in this session/maze"
    raw = np.asarray(data[FEATURE_BLOCK], dtype=float)[mask]
    X, _dims = features.apply_variant(raw, variant)
    return estimator.estimate(
        X, y, session_ids,
        n_rounds=args.n_rounds, n_perm=args.n_perm,
        seed=args.seed, min_trials=args.min_trials,
    )


def cell_row(result, reason, *, monkey, session, variant, maze, radius):
    base = dict(
        monkey=monkey, session=session, variant=variant, maze=maze, radius=radius
    )
    if result is None:
        return {**base, "reason_skipped": reason}
    r = result
    hh, ss, hs, sh = (
        (estimator.H, estimator.H), (estimator.S, estimator.S),
        (estimator.H, estimator.S), (estimator.S, estimator.H),
    )
    return {
        **base,
        "r_HH": r.q[hh],
        "r_SS": r.q[ss],
        "r_HS": r.q[hs],
        "r_SH": r.q[sh],
        "delta": r.delta,
        "z": r.z,
        "p": r.p,
        "p_at_floor": int(r.p_at_floor),
        "null_mean": r.null_mean,
        "null_sd": r.null_sd,
        "n_H": r.n_H,
        "n_S": r.n_S,
        "m_H": r.m_H,
        "m_S": r.m_S,
        "d": r.d,
        "n_sessions": r.n_sessions,
        "n_degenerate": r.n_degenerate,
        "n_rounds": r.n_rounds,
        "n_perm": r.n_perm,
        "reason_skipped": "",
    }


def write_csv(path, rows):
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


def _session_trial_mask(data, session):
    sessions = np.asarray(data["session"]).astype(str)
    return sessions == session


def codebook_figure(data, *, monkey, session, args, maze=None):
    """Fixation centroids for one session (all mazes, or one maze)."""
    trial_mask = _session_trial_mask(data, session)
    fix_row = np.asarray(data["fix_row"], dtype=int)
    fix_sel = trial_mask[fix_row]
    if maze is None:
        fix_xy = np.asarray(data["fix_xy"])[fix_sel]
        fix_state = np.asarray(data["fix_state"])[fix_sel]
        scope_txt = f"session {session}, mazes 1–6"
        stem = all_session_stem(session)
    else:
        fix_maze = np.asarray(data["maze_id"], dtype=int)[fix_row]
        sel = fix_sel & (fix_maze == maze)
        fix_xy = np.asarray(data["fix_xy"])[sel]
        fix_state = np.asarray(data["fix_state"])[sel]
        scope_txt = f"session {session}, maze {maze}"
        stem = maze_session_stem(maze, session)
    fig = figmod.codebook_figure(
        fix_xy, fix_state,
        codebook=data["codebook_xy"], names=cfg.STATE_NAMES,
        assignment=args.assignment, lim=cfg.MAZE_SCREEN_LIM,
        title=(
            f"{monkey} — {session} — fixed codebook, {args.assignment.label()}\n"
            f"every in-window fixation centroid, {scope_txt} ({fix_state.size} fixations)"
        ),
    )
    figmod.save_figure(
        fig, stem, out_root=args.out_root,
        rel_dir=type_rel_dir(args.assignment.tag, monkey, "codebook"), dpi=args.dpi,
    )
    plt.close(fig)


def strategy_figure(data, *, monkey, session, maze, lookup, args):
    mask, y, _session_ids = pool_maze(
        data, maze=maze, session=session, lookup=lookup
    )
    if mask.sum() == 0:
        print(f"  {monkey}/strategy/{maze_session_stem(maze, session)}: no labelled trial; no figure")
        return
    row_label = np.full(mask.size, np.nan)
    row_label[mask] = y
    fix_row = np.asarray(data["fix_row"], dtype=int)
    fix_label = row_label[fix_row]
    occ_all = np.asarray(data[FEATURE_BLOCK], dtype=float)

    fix, occ = {}, {}
    for tag, code in (("H", estimator.H), ("S", estimator.S)):
        sel = fix_label == code
        fix[tag] = (np.asarray(data["fix_xy"])[sel], np.asarray(data["fix_state"])[sel])
        occ[tag] = occ_all[mask & (row_label == code)]

    fig = figmod.strategy_figure(
        fix, occ,
        codebook=data["codebook_xy"], names=cfg.STATE_NAMES,
        assignment=args.assignment, lim=cfg.MAZE_SCREEN_LIM,
        title=(
            f"{monkey} — {session} — maze {maze} — fixation concentration by strategy\n"
            f"SVM labels, single session; every panel normalised within strategy"
        ),
    )
    figmod.save_figure(
        fig, maze_session_stem(maze, session), out_root=args.out_root,
        rel_dir=type_rel_dir(args.assignment.tag, monkey, "strategy"), dpi=args.dpi,
    )
    plt.close(fig)


def panel_suptitle(*, monkey, session, maze, variant):
    return (
        f"{monkey} — {session} — maze {maze} — binary occupancy\n"
        f"{variant}  ·  SVM labels  ·  similarity of group means"
    )


def build_session(monkey, session, data, lookup, args, variant_rows):
    print(f"\n=== {monkey} / {session} ===")
    codebook_figure(data, monkey=monkey, session=session, args=args)
    for maze in args.maze:
        codebook_figure(data, monkey=monkey, session=session, args=args, maze=maze)
        strategy_figure(
            data, monkey=monkey, session=session, maze=maze, lookup=lookup, args=args
        )

    tag = args.assignment.tag
    for variant in args.variant:
        for maze in args.maze:
            result, reason = run_cell(
                data, maze=maze, session=session, variant=variant,
                lookup=lookup, args=args,
            )
            variant_rows[variant].append(
                cell_row(
                    result, reason,
                    monkey=monkey, session=session, variant=variant,
                    maze=maze, radius=tag,
                )
            )
            target = figure_png(args.out_root, tag, monkey, variant, maze, session)
            if result is None:
                print(f"  {variant}/{maze_session_stem(maze, session)}: {reason}")
                if target.exists():
                    target.unlink()
                    print(f"    removed stale {target}")
                continue
            fig = figmod.panel_figure(
                result, k=cfg.K,
                title=panel_suptitle(
                    monkey=monkey, session=session, maze=maze, variant=variant
                ),
                names=[f"{maze}H", f"{maze}S"], vmax=args.vmax,
            )
            figmod.save_figure(
                fig, maze_session_stem(maze, session), out_root=args.out_root,
                rel_dir=type_rel_dir(tag, monkey, variant), dpi=args.dpi,
                bbox_inches="tight", pad_inches=0.08,
            )
            plt.close(fig)
            print(
                f"    {variant}/{maze_session_stem(maze, session)}: "
                f"Δ={result.delta:+.4f} z={result.z:+.2f} p={result.p:.4f} "
                f"n_H={result.n_H} n_S={result.n_S} "
                f"m_H={result.m_H} m_S={result.m_S} "
                f"degenerate={result.n_degenerate}"
            )


def build_monkey(monkey, args):
    keep_sessions, lookup = scope(monkey, args.session)
    print(f"\n### {monkey}  ({len(keep_sessions)} sessions) ###")
    data = features.load_features(
        monkey, assignment=args.assignment, refresh=args.refresh
    )
    variant_rows = {variant: [] for variant in args.variant}
    for session in keep_sessions:
        build_session(monkey, session, data, lookup, args, variant_rows)
    tag = args.assignment.tag
    for variant, rows in variant_rows.items():
        write_csv(results_csv(args.out_root, tag, monkey, variant), rows)


def dry_run(args):
    n_cells = 0
    for monkey in args.monkey:
        keep_sessions, lookup = scope(monkey, args.session)
        data = features.load_features(
            monkey, assignment=args.assignment, refresh=args.refresh
        )
        print(f"\n{monkey}  ({len(keep_sessions)} sessions: {', '.join(keep_sessions)})")
        print(
            f"  cache rows {len(data['session'])}, tag={args.assignment.tag}, "
            f"{args.assignment.label()}"
        )
        for session in keep_sessions:
            print(f"  -- {session} --")
            for maze in args.maze:
                mask, y, _session_ids = pool_maze(
                    data, maze=maze, session=session, lookup=lookup
                )
                n_h, n_s, m_h, m_s, reason = estimator.coverage(
                    y, min_trials=args.min_trials
                )
                diag = maze_diagnostics(data, mask)
                if reason:
                    print(
                        f"    maze {maze}: n_H={n_h:4d} n_S={n_s:4d}  SKIPPED ({reason})"
                    )
                else:
                    n_cells += len(args.variant)
                    print(
                        f"    maze {maze}: n_H={n_h:4d} n_S={n_s:4d} "
                        f"m_H={m_h:3d} m_S={m_s:3d} -> {len(args.variant)} figures"
                    )
                print(f"            {format_diagnostics(diag)}")
    print(f"\n{n_cells} estimator runs at n_rounds={args.n_rounds}, n_perm={args.n_perm}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--monkey", nargs="*", default=list(MONKEYS), choices=MONKEYS)
    parser.add_argument(
        "--session", nargs="*", default=None,
        help="limit to these session id(s); default is the monkey's top-ten SVM scope",
    )
    parser.add_argument("--maze", type=int, nargs="*", default=list(MAZES), choices=MAZES)
    parser.add_argument(
        "--variant", nargs="*", default=list(features.VARIANTS),
        choices=features.VARIANTS,
    )
    parser.add_argument(
        "--radius", type=float, default=None,
        help="uniform assignment radius (unit H); same feature caches as msp",
    )
    parser.add_argument(
        "--window", default=cfg.DEFAULT_WINDOW, choices=list(cfg.WINDOW_NAMES),
        help="analysis clip: geofix (default, tag r<radius>) or pre1466 "
             "(tag r<radius>_pre1466); msp's feature caches either way",
    )
    parser.add_argument("--n-rounds", type=int, default=estimator.N_ROUNDS)
    parser.add_argument("--n-perm", type=int, default=estimator.N_PERM)
    parser.add_argument(
        "--min-trials", type=int, default=estimator.MIN_TRIALS,
        help="per strategy within the session; below this the cell is skipped",
    )
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--refresh", action="store_true", help="rebuild the msp feature cache")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.maze = sorted(set(args.maze))
    args.assignment = cfg.resolve_assignment(
        radius=args.radius,
        window=args.window,
    )

    if args.dry_run:
        dry_run(args)
        return
    for monkey in args.monkey:
        build_monkey(monkey, args)


if __name__ == "__main__":
    main()
