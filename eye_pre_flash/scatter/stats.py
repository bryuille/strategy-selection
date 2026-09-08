"""The numbers behind the fix_start scatter: group means and the two tests.

The figures show that the cloud shifts; this says by how much and whether the
shift survives the confound. Maze and strategy are not independent -- maze 1-3
are mostly decoded hierarchical and 5-6 mostly sequential -- so a raw
hierarchical-vs-sequential difference in gaze is partly just a maze
difference. Two things are reported for each:

    maze trend       Spearman ρ of maze index against x and y, over trials and
                     again per session (so a session's calibration offset
                     cannot create the trend)
    label effect     the hierarchical-vs-sequential mean difference, first
                     pooled, then *stratified by maze* -- the within-maze
                     differences averaged with maze weights. The stratified
                     number is the one that is not a restatement of the maze
                     trend.

Usage:
    uv run python -m eye_pre_flash.scatter.stats --monkey Faure
    uv run python -m eye_pre_flash.scatter.stats --all
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, ttest_ind

from eye_pre_flash.classifier.labels import SCOPES, strategy_label_lookup
from eye_pre_flash.scatter.core import (
    LABELS,
    MAZES,
    OUT_ROOT,
    SPACES,
    STRATEGY_NAMES,
    center_by_session,
    centroid_ci,
    collect_fix_start,
    output_rel_dir,
)
from eye_pre_flash.scatter.plots import resolve_scope

MIN_CELL = 10


def _row(group, name, values_x, values_y):
    mx, ex = centroid_ci(values_x)
    my, ey = centroid_ci(values_y)
    return {
        "group": group,
        "level": name,
        "n": int(values_x.size),
        "mean_x": mx,
        "ci95_x": ex,
        "mean_y": my,
        "ci95_y": ey,
    }


def group_rows(trials):
    """One row per maze, per label, and per maze × label cell."""
    rows = [_row("all", "all", trials["x"], trials["y"])]
    for m in MAZES:
        k = trials["maze"] == m
        rows.append(_row("maze", str(m), trials["x"][k], trials["y"][k]))
    labelled = np.isfinite(trials["label"])
    for v in LABELS:
        k = labelled & (trials["label"] == v)
        rows.append(_row("strategy", STRATEGY_NAMES[v], trials["x"][k], trials["y"][k]))
    for m in MAZES:
        for v in LABELS:
            k = labelled & (trials["maze"] == m) & (trials["label"] == v)
            rows.append(
                _row("maze×strategy", f"{m}/{STRATEGY_NAMES[v]}", trials["x"][k], trials["y"][k])
            )
    for session in np.unique(trials["session"]):
        k = trials["session"] == session
        rows.append(_row("session", str(session), trials["x"][k], trials["y"][k]))
    return rows


def maze_trend(trials, axis):
    """Spearman ρ of maze index against one gaze axis, pooled and per session."""
    v = trials[axis]
    maze = trials["maze"].astype(float)
    ok = np.isfinite(v)
    rho, p = spearmanr(maze[ok], v[ok])
    per_session = []
    for session in np.unique(trials["session"]):
        k = ok & (trials["session"] == session)
        if k.sum() < 30 or np.unique(trials["maze"][k]).size < 3:
            continue
        r, _ = spearmanr(trials["maze"][k].astype(float), v[k])
        if np.isfinite(r):
            per_session.append(float(r))
    return dict(
        axis=axis,
        rho=float(rho),
        p=float(p),
        n=int(ok.sum()),
        session_rho_mean=float(np.mean(per_session)) if per_session else np.nan,
        n_sessions=len(per_session),
    )


def label_effect(trials, axis):
    """Sequential − hierarchical mean difference, pooled and maze-stratified."""
    v = trials[axis]
    labelled = np.isfinite(trials["label"]) & np.isfinite(v)
    a = v[labelled & (trials["label"] == 0)]
    b = v[labelled & (trials["label"] == 1)]
    out = dict(axis=axis, n_hier=int(a.size), n_seq=int(b.size))
    if a.size < 2 or b.size < 2:
        return out | dict(diff=np.nan, p_t=np.nan, p_mw=np.nan, d=np.nan, diff_strat=np.nan)

    pooled_sd = np.sqrt(
        ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
        / (a.size + b.size - 2)
    )
    t = ttest_ind(b, a, equal_var=False)
    mw = mannwhitneyu(b, a, alternative="two-sided")

    # Maze-stratified: the within-maze difference, weighted by how many
    # labelled trials the maze contributes. A maze needs MIN_CELL trials on
    # both sides to enter, or its "difference" is one or two trials.
    diffs, weights = [], []
    for m in MAZES:
        k = labelled & (trials["maze"] == m)
        am = v[k & (trials["label"] == 0)]
        bm = v[k & (trials["label"] == 1)]
        if am.size < MIN_CELL or bm.size < MIN_CELL:
            continue
        diffs.append(bm.mean() - am.mean())
        weights.append(am.size + bm.size)
    strat = (
        float(np.average(diffs, weights=weights)) if diffs else np.nan
    )
    return out | dict(
        diff=float(b.mean() - a.mean()),
        p_t=float(t.pvalue),
        p_mw=float(mw.pvalue),
        d=float((b.mean() - a.mean()) / pooled_sd) if pooled_sd > 0 else np.nan,
        diff_strat=strat,
        n_mazes_strat=len(diffs),
    )


def report(monkey="Faure", *, scope="all", space="deg", center="none", all_sessions=False):
    by_monkey = resolve_scope(scope, verbose=False)
    sessions = by_monkey.get(monkey)
    if not sessions:
        raise SystemExit(f"scope {scope!r} has no labelled {monkey} sessions")
    trials = collect_fix_start(
        monkey,
        sessions=None if all_sessions else sessions,
        label_lookup=strategy_label_lookup(sessions),
        space=space,
    )
    if center == "session":
        trials = center_by_session(trials)

    out_dir = OUT_ROOT / output_rel_dir(space, monkey, scope, centered=center == "session")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = group_rows(trials)
    path = out_dir / "group_means.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")

    unit = SPACES[space]["axis"]
    lines = [
        f"{monkey} | scope={scope} | space={space} | "
        f"center={center} | n={trials['x'].size} trials "
        f"({int(np.isfinite(trials['label']).sum())} labelled)",
        "",
        "maze trend (Spearman ρ, maze index vs gaze):",
    ]
    for axis in ("x", "y"):
        t = maze_trend(trials, axis)
        lines.append(
            f"  {axis}: ρ={t['rho']:+.3f} p={t['p']:.2e} n={t['n']} | "
            f"within-session mean ρ={t['session_rho_mean']:+.3f} "
            f"over {t['n_sessions']} session(s)"
        )
    lines += ["", f"decoded-strategy effect (sequential − hierarchical, {unit}):"]
    for axis in ("x", "y"):
        e = label_effect(trials, axis)
        lines.append(
            f"  {axis}: Δ={e['diff']:+.3f} d={e['d']:+.3f} "
            f"t-p={e['p_t']:.2e} MW-p={e['p_mw']:.2e} "
            f"(n={e['n_hier']} vs {e['n_seq']})"
        )
        lines.append(
            f"     maze-stratified Δ={e['diff_strat']:+.3f} over "
            f"{e.get('n_mazes_strat', 0)} maze(s) with ≥{MIN_CELL} trials per label"
        )
    text = "\n".join(lines)
    print(text)
    txt_path = out_dir / "tests.txt"
    txt_path.write_text(text + "\n")
    print(f"Saved {txt_path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monkey", default="Faure", choices=("Faure", "Nielsen"))
    parser.add_argument("--scope", default="all", choices=sorted(SCOPES))
    parser.add_argument("--space", default="deg", choices=(*sorted(SPACES), "both"))
    parser.add_argument("--center", default="none", choices=("none", "session", "both"))
    parser.add_argument("--all-sessions", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    monkeys = ("Faure", "Nielsen") if args.all else (args.monkey,)
    spaces = tuple(sorted(SPACES)) if args.all or args.space == "both" else (args.space,)
    centers = ("none", "session") if args.all or args.center == "both" else (args.center,)
    for monkey in monkeys:
        for space in spaces:
            for center in centers:
                report(
                    monkey,
                    scope=args.scope,
                    space=space,
                    center=center,
                    all_sessions=args.all_sessions,
                )
                print()


if __name__ == "__main__":
    main()
