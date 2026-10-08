"""Single-maze H-vs-S gaze similarity with a fixed five-state codebook.

One 2x2 per (monkey, variant, maze). Trials of one maze are pooled across
every in-scope session; each strategy is cut into two disjoint halves of
size `n_strategy // 2` (H and S may differ). Each half is averaged into one
binary-occupancy profile over the five fixed states (origin + four exits)
and the pairings are scored by the correlation of those means; the diagonal
is within-strategy similarity, the off-diagonal cross-strategy, against a
within-session label-shuffle null.

Fixed here, and so not a CLI axis: SVM labels at the top-ten scope, binary
occupancy, mazes 2-5 only. Assignment is uniform balls around the five fixed
prototypes by default (`--radius`, e.g. ``r0.5`` / ``r1``); ``--codebook``
swaps in a region codebook (`config.CODEBOOKS`): ``quads`` (origin ball +
maze quadrants, K = 5), ``halves`` (origin ball + maze halves, K = 3) or
``capsules`` (origin-to-exit capsules, no origin state, K = 4). The tag
carries the codebook (``r0.5capsules``). ``no_origin`` runs only where it
leaves >= 3 states with an origin to drop (balls, quads).
Each trial runs from maze onset to fixation onset, re-detected per trial over
the top-ten sessions.

Usage:
    uv run python -m msp.build --dry-run
    uv run python -m msp.build
    uv run python -m msp.build --radius 0.5
    uv run python -m msp.build --radius 0.5 --codebook capsules   # also: quads, halves
    uv run python -m msp.build --monkey Faure --maze 4 --n-perm 200

Writes, under out/<tag>/<Monkey>/: <variant>/maze<M>.png, <variant>/results.csv,
codebook.png, codebook_maze<M>.png and strategy_maze<M>.png.
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


def _meta(data):
    return (
        np.asarray(data["session"]).astype(str),
        np.asarray(data["trial_indices_all"], dtype=int),
        np.asarray(data["maze_id"], dtype=int),
    )


def scope(monkey):
    """``(sessions, lookup)`` for the monkey's svm/top_ten scope."""
    keep = labelmod.top_ten_sessions(monkey)
    return keep, labelmod.svm_lookup(keep)


def pool_maze(data, *, maze, keep_sessions, lookup):
    """Row mask, labels and session ids for every labelled in-scope trial of one maze."""
    sessions, trials, mazes = _meta(data)
    y_full = labelmod.labels_for_rows(sessions, trials, lookup)
    mask = np.isin(sessions, list(keep_sessions)) & np.isfinite(y_full) & (mazes == maze)
    return mask, y_full[mask].astype(int), sessions[mask]


def maze_diagnostics(data, mask, codebook):
    """Codebook coverage for the pooled rows of one maze."""
    occ = features.codebook_block(data, codebook, FEATURE_BLOCK)[mask]
    n_fix = int(np.asarray(data["n_fix"], dtype=int)[mask].sum())
    n_ok = int(np.asarray(data["n_fix_assigned"], dtype=int)[mask].sum())
    if occ.shape[0] == 0:
        return dict(assigned=np.nan, visit=np.full(codebook.k, np.nan),
                    zero_full=np.nan, zero_no_origin=np.nan, n_fix=n_fix)
    zero_no = (float((occ[:, features.no_origin_dims(codebook.k)].sum(axis=1) == 0).mean())
               if codebook.has_origin else np.nan)
    return dict(
        assigned=n_ok / max(n_fix, 1),
        visit=occ.mean(axis=0),
        zero_full=float((occ.sum(axis=1) == 0).mean()),
        zero_no_origin=zero_no,
        n_fix=n_fix,
    )


def format_diagnostics(diag, codebook):
    visit = "  ".join(
        f"{name} {v:.2f}" for name, v in zip(codebook.names, diag["visit"])
    )
    no = (f", no_origin {diag['zero_no_origin']:.1%}"
          if np.isfinite(diag["zero_no_origin"]) else "")
    return (
        f"fixations assigned {diag['assigned']:.1%} of {diag['n_fix']}; "
        f"visit rate: {visit}; all-zero rows: full {diag['zero_full']:.1%}{no}"
    )


def run_cell(data, *, maze, variant, keep_sessions, lookup, args):
    """One pooled estimate. `mean_removed` subtracts the maze grand mean
    (average over every pooled labelled trial of this maze, each trial once).
    That mean is label-free, so it is applied once up front with the other
    variants; the null shuffles labels on the centred matrix."""
    mask, y, session_ids = pool_maze(
        data, maze=maze, keep_sessions=keep_sessions, lookup=lookup
    )
    if mask.sum() == 0:
        return None, "no labelled trial in this maze"
    raw = features.codebook_block(data, args.assignment.codebook, FEATURE_BLOCK)[mask]
    X, _dims = features.apply_variant(raw, variant)
    return estimator.estimate(
        X, y, session_ids,
        n_rounds=args.n_rounds, n_perm=args.n_perm,
        seed=args.seed, min_trials=args.min_trials,
    )


def cell_row(result, reason, *, monkey, variant, maze, radius):
    base = dict(monkey=monkey, variant=variant, maze=maze, radius=radius)
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


def codebook_figure(data, *, monkey, args, maze=None):
    """Regions over every fixation centroid: all mazes, or one maze (all sessions)."""
    fix_xy = np.asarray(data["fix_xy"])
    assignment = args.assignment
    fix_state = features.codebook_fix_state(data, assignment.codebook)
    if maze is None:
        scope_txt, stem = "all sessions, mazes 1–6", cfg.codebook_stem()
    else:
        fix_maze = np.asarray(data["maze_id"], dtype=int)[np.asarray(data["fix_row"], dtype=int)]
        sel = fix_maze == maze
        fix_xy, fix_state = fix_xy[sel], fix_state[sel]
        scope_txt, stem = f"all sessions, maze {maze}", cfg.codebook_maze_stem(maze)
    fig = figmod.codebook_figure(
        fix_xy, fix_state,
        codebook=data["codebook_xy"], names=assignment.codebook.names,
        assignment=assignment, lim=cfg.MAZE_SCREEN_LIM,
        title=(
            f"{monkey} — {assignment.codebook.label}, radius {assignment.label()}\n"
            f"every in-window fixation centroid, {scope_txt} ({fix_state.size} fixations)"
        ),
    )
    figmod.save_figure(
        fig, stem, out_root=args.out_root,
        rel_dir=cfg.monkey_dir(args.assignment.tag, monkey), dpi=args.dpi,
    )
    plt.close(fig)


def strategy_figure(data, *, monkey, maze, keep_sessions, lookup, args):
    """Fixation concentration per state, H vs S, for one maze's pooled trials.

    Normalised within strategy throughout (see `figures.strategy_figure`),
    because n_H and n_S differ by up to an order of magnitude in some mazes.
    Uses every in-scope labelled trial of the maze, the same rows the
    estimator pools, regardless of whether the maze clears coverage.
    """
    mask, y, _session_ids = pool_maze(
        data, maze=maze, keep_sessions=keep_sessions, lookup=lookup
    )
    if mask.sum() == 0:
        print(f"  {monkey}/strategy_maze{maze}: no labelled trial; no figure")
        return
    row_label = np.full(mask.size, np.nan)
    row_label[mask] = y
    fix_row = np.asarray(data["fix_row"], dtype=int)
    fix_label = row_label[fix_row]
    cb = args.assignment.codebook
    occ_all = features.codebook_block(data, cb, FEATURE_BLOCK)
    fix_state = features.codebook_fix_state(data, cb)

    fix, occ = {}, {}
    for tag, code in (("H", estimator.H), ("S", estimator.S)):
        sel = fix_label == code
        fix[tag] = (np.asarray(data["fix_xy"])[sel], fix_state[sel])
        occ[tag] = occ_all[mask & (row_label == code)]

    fig = figmod.strategy_figure(
        fix, occ,
        codebook=data["codebook_xy"], names=cb.names,
        assignment=args.assignment, lim=cfg.MAZE_SCREEN_LIM,
        title=(
            f"{monkey} — maze {maze} — fixation concentration by strategy\n"
            f"SVM labels, {len(keep_sessions)} sessions pooled; every panel normalised within strategy"
        ),
    )
    figmod.save_figure(
        fig, cfg.strategy_stem(maze), out_root=args.out_root,
        rel_dir=cfg.monkey_dir(args.assignment.tag, monkey), dpi=args.dpi,
    )
    plt.close(fig)


def build_monkey(monkey, args):
    keep_sessions, lookup = scope(monkey)
    print(f"\n=== {monkey}  ({len(keep_sessions)} sessions in scope) ===")
    data = features.load_features(
        monkey, assignment=args.assignment, refresh=args.refresh
    )
    codebook_figure(data, monkey=monkey, args=args)
    for maze in args.maze:
        codebook_figure(data, monkey=monkey, args=args, maze=maze)
        strategy_figure(
            data, monkey=monkey, maze=maze,
            keep_sessions=keep_sessions, lookup=lookup, args=args,
        )

    tag = args.assignment.tag
    cb = args.assignment.codebook
    for variant in args.variant:
        rows = []
        for maze in args.maze:
            result, reason = run_cell(
                data, maze=maze, variant=variant,
                keep_sessions=keep_sessions, lookup=lookup, args=args,
            )
            rows.append(cell_row(result, reason, monkey=monkey, variant=variant,
                                 maze=maze, radius=tag))
            target = cfg.figure_png(args.out_root, tag, monkey, variant, maze)
            if result is None:
                print(f"  {monkey}/{variant}/maze{maze}: {reason}")
                if target.exists():
                    # Stale numbers under a current-looking filename are
                    # worse than an absent panel.
                    target.unlink()
                    print(f"    removed stale {target}")
                continue
            fig = figmod.panel_figure(
                result, k=result.d,
                title=figmod.suptitle(monkey=monkey, maze=maze, variant=variant, codebook=cb),
                names=[f"{maze}H", f"{maze}S"], vmax=args.vmax,
                states=("fixed: origin + 4 exits" if cb.key == cfg.DEFAULT_CODEBOOK
                        else ", ".join(cb.names[1:] if variant == "no_origin" else cb.names)),
            )
            # Tight bbox: a single square matrix cannot fill the canvas width,
            # so crop to the content instead of shipping the blank margins.
            figmod.save_figure(
                fig, cfg.stem(maze), out_root=args.out_root,
                rel_dir=cfg.rel_dir(tag, monkey, variant), dpi=args.dpi,
                bbox_inches="tight", pad_inches=0.08,
            )
            plt.close(fig)
            print(
                f"    {variant}/maze{maze}: Δ={result.delta:+.4f} z={result.z:+.2f} "
                f"p={result.p:.4f} n_H={result.n_H} n_S={result.n_S} "
                f"m_H={result.m_H} m_S={result.m_S} "
                f"d={result.d} degenerate={result.n_degenerate} "
                f"({result.n_sessions} sessions)"
            )
        write_csv(cfg.results_csv(args.out_root, tag, monkey, variant), rows)


def dry_run(args):
    """Per-maze pooled counts and fixed-codebook coverage; no statistics.

    Needs the feature cache (one extraction pass per monkey on a cold run).
    `n_H`/`n_S` depend on the labels and the maze only, so they must equal
    the svm/top_ten numbers of every other maze-strategy-pairs pipeline.
    """
    n_cells = 0
    for monkey in args.monkey:
        keep_sessions, lookup = scope(monkey)
        data = features.load_features(
            monkey, assignment=args.assignment, refresh=args.refresh
        )
        print(f"\n{monkey}  ({len(keep_sessions)} sessions: {', '.join(keep_sessions)})")
        print(
            f"  cache rows {len(data['session'])}, tag={args.assignment.tag}, "
            f"{args.assignment.label()}"
        )
        for maze in args.maze:
            mask, y, session_ids = pool_maze(
                data, maze=maze, keep_sessions=keep_sessions, lookup=lookup
            )
            n_h, n_s, m_h, m_s, reason = estimator.coverage(
                y, min_trials=args.min_trials
            )
            diag = maze_diagnostics(data, mask, args.assignment.codebook)
            if reason:
                print(f"  maze {maze}: n_H={n_h:4d} n_S={n_s:4d}  SKIPPED ({reason})")
            else:
                n_cells += len(args.variant)
                print(
                    f"  maze {maze}: n_H={n_h:4d} n_S={n_s:4d} "
                    f"m_H={m_h:3d} m_S={m_s:3d} "
                    f"({len(set(session_ids.tolist()))} sessions) -> {len(args.variant)} figures"
                )
            print(f"          {format_diagnostics(diag, args.assignment.codebook)}")
    print(f"\n{n_cells} estimator runs at n_rounds={args.n_rounds}, n_perm={args.n_perm}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--monkey", nargs="*", default=list(MONKEYS), choices=MONKEYS)
    parser.add_argument("--maze", type=int, nargs="*", default=list(MAZES), choices=MAZES)
    parser.add_argument(
        "--variant", nargs="*", default=list(features.VARIANTS),
        choices=features.VARIANTS,
    )
    parser.add_argument(
        "--radius", type=float, default=None,
        help="uniform assignment radius around each fixed prototype (unit H); "
             "carried in the cache stem, so a new value triggers one extraction pass",
    )
    parser.add_argument(
        "--codebook", default=cfg.DEFAULT_CODEBOOK, choices=list(cfg.CODEBOOKS),
        help="balls: origin + four exit balls; quads / halves: origin ball + maze "
             "quadrants / halves; capsules: origin-to-exit capsules, no origin state "
             "(tag suffix quads / halves / capsules)",
    )
    parser.add_argument("--n-rounds", type=int, default=estimator.N_ROUNDS)
    parser.add_argument("--n-perm", type=int, default=estimator.N_PERM)
    parser.add_argument(
        "--min-trials", type=int, default=estimator.MIN_TRIALS,
        help="per strategy, pooled across sessions; below this the maze is skipped",
    )
    parser.add_argument(
        "--vmax", type=float, default=None,
        help="fixed 0-to-VMAX colour scale instead of the per-figure min-to-max",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--out-root", type=Path, default=cfg.OUT_ROOT)
    parser.add_argument("--refresh", action="store_true", help="rebuild the feature cache")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="pooled counts and codebook coverage per maze; compute no statistics",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.maze = sorted(set(args.maze))
    args.assignment = cfg.resolve_assignment(radius=args.radius, codebook=args.codebook)
    kept = features.variants_for(args.assignment.codebook, args.variant)
    if kept != tuple(args.variant):
        print(f"{args.codebook}: skipping no_origin (needs an origin and >= 3 other states)")
    args.variant = list(kept)

    if args.dry_run:
        dry_run(args)
        return
    for monkey in args.monkey:
        build_monkey(monkey, args)


if __name__ == "__main__":
    main()
