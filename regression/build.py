"""Entry point: fixation caches -> one regression per monkey -> CSVs and one figure.

    python -m regression.build                      # both monkeys
    python -m regression.build --dry-run            # counts and coverage, no fit
    python -m regression.build --monkey Faure --sessions june_24_g0
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from regression.config import (
    SESSIONS,
    STATE_NAMES,
    OUT_DIR,
)
from regression.features import load_fixations, trial_measures
from regression.labels import label_lookup, labels_for_rows
from regression.model import analyse, select_mazes, select_states


def analysed_rows(data, lookup):
    """Row indices of `data` that are labelled, with their labels and maze ids."""
    sess = np.asarray(data["session"]).astype(str)
    tri = np.asarray(data["trial_indices_all"]).astype(int)
    y = labels_for_rows(sess, tri, lookup)
    idx = np.flatnonzero(np.isfinite(y))
    return idx, y[idx], np.asarray(data["maze_id"]).astype(int)[idx]


def run_monkey(monkey, sessions, *, dry_run, refresh):
    data = load_fixations(monkey, sessions=sessions, refresh=refresh)
    print(f"  {monkey} drops: {json.loads(str(data['drops_json']))}")
    meas_all = trial_measures(data)
    idx, y, mazes = analysed_rows(data, label_lookup(sessions))
    kept, maze_table = select_mazes(mazes, y)
    keep = np.isin(mazes, kept)
    idx, y, mazes = idx[keep], y[keep], mazes[keep]
    meas = {k: np.asarray(v)[idx] for k, v in meas_all.items()}
    states, rate = select_states(meas["occ"]) if idx.size else ([], np.zeros(len(STATE_NAMES)))

    print(f"\n== {monkey} ==")
    print(f"  analysed rows: {idx.size} (H {int((y == 0).sum())}, S {int((y == 1).sum())})")
    for r in maze_table:
        print(f"    maze {r['maze']}: H {r['n_H']} / S {r['n_S']}" + ("" if r["kept"] else "  (dropped)"))
    print("  visit rate: " + ", ".join(f"{STATE_NAMES[s]} {rate[s]:.2f}" for s in range(len(STATE_NAMES))))
    print(f"  states kept: {', '.join(STATE_NAMES[s] for s in states) or 'none'}")
    if dry_run or not kept or not states:
        if not dry_run:
            print("  nothing to fit (no kept mazes or states)")
        return None

    coef, summary = analyse(meas, mazes, y, kept, states)
    summary.update(
        monkey=monkey, n_mazes=len(kept),
        states_kept=";".join(STATE_NAMES[s] for s in states),
        states_dropped=";".join(
            f"{STATE_NAMES[s]}({rate[s]:.2f})" for s in range(len(STATE_NAMES)) if s not in states
        ),
    )
    for r in coef:
        r["monkey"] = monkey
    for r in maze_table:
        r["monkey"] = monkey
    for r in coef:
        if r["kind"] == "gaze":
            tag = f"p={r['p']:.3g}" if r["status"] == "fit" else r["status"]
            beta = f"{r['beta']:+.2f}" if r["status"] == "fit" else "  n/a"
            print(f"  {r['predictor']:9s} {r['state']:7s} beta {beta}  {tag}")
    print(f"  converged: {summary['converged']}")
    return {"coef": coef, "mazes": maze_table, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description="Per-maze-offset logistic regression on codebook gaze.")
    parser.add_argument("--monkey", nargs="*", default=list(SESSIONS), choices=list(SESSIONS))
    parser.add_argument("--sessions", nargs="*", default=None, help="subset of the publication sessions")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="rebuild the fixation caches")
    args = parser.parse_args()

    results = {}
    for monkey in args.monkey:
        sessions = SESSIONS[monkey]
        if args.sessions:
            sessions = tuple(s for s in sessions if s in args.sessions)
            if not sessions:
                continue
        res = run_monkey(monkey, sessions, dry_run=args.dry_run, refresh=args.refresh)
        if res is not None:
            results[monkey] = res

    if results:
        import pandas as pd

        from regression.figures import coefficient_figure

        dest = OUT_DIR
        dest.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([r for v in results.values() for r in v["coef"]]).to_csv(dest / "coefficients.csv", index=False)
        pd.DataFrame([r for v in results.values() for r in v["mazes"]]).to_csv(dest / "mazes.csv", index=False)
        pd.DataFrame([v["summary"] for v in results.values()]).to_csv(dest / "summary.csv", index=False)
        coefficient_figure(
            {m: v["coef"] for m, v in results.items()}, dest / "coefficients.png",
        )
        print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
