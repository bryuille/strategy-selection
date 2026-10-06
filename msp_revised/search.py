"""Screen every (codebook, kind, pupil, variant) and rank by cross-monkey,
cross-maze consistency of the pooled estimate.

    uv run python -m msp_revised.search                    # full grid, n_perm 250
    uv run python -m msp_revised.search --rank-only        # re-rank an existing grid
    uv run python -m msp_revised.search --codebook balls_u0.5 segments_d --kind dwell

Writes out/search/grid.csv (one row per estimate: maze 1-6 and pooled, per
monkey) and out/search/ranking.csv (one row per configuration).

Ranking. A configuration is **consistent** when, in both monkeys, the
pooled z is positive and every *solid* maze (>= SOLID_N trials of each
strategy) has Δ > 0. ``score`` is the smaller of the two monkeys' pooled z
(a result carried by one monkey scores low), penalised by any solid maze
with Δ < 0. Screening uses a reduced null; `build` the finalists at the
full n_perm. With ~900 configurations screened the best few z values are
inflated by selection: treat the ranking as a shortlist, not a p-value,
and read the per-session replication (`--replicate`) alongside.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path

import numpy as np

from msp_revised import codebooks as cbmod
from msp_revised import config as cfg
from msp_revised import estimator
from msp_revised import features
from msp_revised.build import cell_row, estimates, write_csv

SCOPES = [str(m) for m in cfg.MAZES] + ["pooled"]


def grid_rows(args):
    rows = []
    t0 = time.time()
    n_cfg = 0
    for monkey in args.monkey:
        data = features.load(monkey)
        lookup = features.label_lookup(cfg.SESSIONS[monkey])
        y_full = features.labels_for(data, lookup)
        for key in args.codebook:
            cb = cbmod.get(key)
            states = features.fixation_states(data, cb)
            for kind, pupil in itertools.product(args.kind, args.pupil):
                X_all = features.trial_features(data, cb, kind, pupil=pupil, states=states)
                for variant in args.variant:
                    n_cfg += 1
                    runs, reasons, pooled = estimates(
                        data, y_full, X_all, variant, mazes=cfg.MAZES, seed=args.seed,
                        n_rounds=args.n_rounds, n_perm=args.n_perm, min_trials=cfg.MIN_TRIALS)
                    base = dict(codebook=key, kind=kind, pupil=int(pupil), variant=variant, monkey=monkey)
                    for maze in cfg.MAZES:
                        r = runs[maze]
                        rows.append(cell_row(None if r is None else r.result, reasons[maze], **base, scope=str(maze)))
                    rows.append(cell_row(pooled, "no maze qualified", **base, scope="pooled"))
                    if n_cfg % 20 == 0:
                        el = time.time() - t0
                        print(f"  {n_cfg} configurations in {el / 60:.1f} min ({monkey} {key} {kind} "
                              f"{'pupil' if pupil else ''} {variant}: pooled z="
                              f"{'—' if pooled is None else f'{pooled.z:+.2f}'})", flush=True)
    return rows


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def rank(rows, *, solid_n=cfg.SOLID_N):
    """One row per configuration with per-monkey across / pooled z and the
    solid-maze sign tally; sorted by `score`."""
    by_cfg = {}
    for r in rows:
        k = (r["codebook"], r["kind"], int(r["pupil"]), r["variant"])
        by_cfg.setdefault(k, []).append(r)
    out = []
    for (codebook, kind, pupil, variant), rs in by_cfg.items():
        rec = dict(codebook=codebook, kind=kind, pupil=pupil, variant=variant)
        pooled_z, solid_pos, solid_tot, solid_neg_z = [], 0, 0, 0.0
        for monkey in cfg.MONKEYS:
            mine = [r for r in rs if r["monkey"] == monkey]
            p = next((r for r in mine if r["scope"] == "pooled"), None)
            zp = _f(p and p.get("z"))
            rec[f"{monkey}_pooled_z"] = zp
            rec[f"{monkey}_pooled_delta"] = _f(p and p.get("delta"))
            rec[f"{monkey}_pooled_p"] = _f(p and p.get("p"))
            pooled_z.append(zp)
            signs = []
            for r in mine:
                if r["scope"] == "pooled" or not r.get("z"):
                    continue
                n_min = min(_f(r.get("n_H")), _f(r.get("n_S")))
                d = _f(r.get("delta"))
                if n_min >= solid_n and np.isfinite(d):
                    solid_tot += 1
                    solid_pos += int(d > 0)
                    if d < 0:
                        solid_neg_z += abs(_f(r.get("z"))) if np.isfinite(_f(r.get("z"))) else 0.0
                    signs.append(f"{r['scope']}:{_f(r.get('z')):+.1f}")
            rec[f"{monkey}_solid_mazes"] = " ".join(signs)
        rec["solid_pos"] = solid_pos
        rec["solid_total"] = solid_tot
        rec["min_pooled_z"] = float(np.nanmin(pooled_z)) if np.isfinite(pooled_z).any() else np.nan
        rec["consistent"] = int(np.all(np.asarray(pooled_z) > 0) and solid_pos == solid_tot and solid_tot > 0)
        score = rec["min_pooled_z"] - solid_neg_z
        rec["score"] = score if np.isfinite(score) else -np.inf
        out.append(rec)
    out.sort(key=lambda r: -r["score"])
    return out


def replicate(top, args):
    """Pooled z computed on each session alone for the top configurations:
    does the sign hold in all four sessions?"""
    cache = {}
    rows = []
    for rec in top:
        cb = cbmod.get(rec["codebook"])
        for monkey in cfg.MONKEYS:
            if monkey not in cache:
                data = features.load(monkey)
                cache[monkey] = (data, features.labels_for(data, features.label_lookup(cfg.SESSIONS[monkey])))
            data, y_full = cache[monkey]
            states = features.fixation_states(data, cb)
            X_all = features.trial_features(data, cb, rec["kind"], pupil=bool(rec["pupil"]), states=states)
            sess = np.asarray(data["session"]).astype(str)
            for s in cfg.SESSIONS[monkey]:
                y_s = np.where(sess == s, y_full, np.nan)
                _runs, _reasons, pooled = estimates(
                    data, y_s, X_all, rec["variant"], mazes=cfg.MAZES, seed=args.seed,
                    n_rounds=args.n_rounds, n_perm=args.n_perm, min_trials=cfg.MIN_TRIALS)
                rows.append(cell_row(pooled, "no maze qualified", codebook=rec["codebook"], kind=rec["kind"],
                                     pupil=rec["pupil"], variant=rec["variant"], monkey=monkey, session=s))
                rec[f"{s}_z"] = np.nan if pooled is None else pooled.z
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--codebook", nargs="*", default=list(cbmod.SEARCH_KEYS), choices=list(cbmod.REGISTRY))
    parser.add_argument("--kind", nargs="*", default=list(features.KINDS), choices=features.KINDS)
    parser.add_argument("--pupil", nargs="*", type=int, default=[0, 1], choices=[0, 1])
    parser.add_argument("--variant", nargs="*", default=list(cfg.VARIANTS), choices=cfg.VARIANTS)
    parser.add_argument("--monkey", nargs="*", default=list(cfg.MONKEYS), choices=cfg.MONKEYS)
    parser.add_argument("--n-rounds", type=int, default=estimator.N_ROUNDS)
    parser.add_argument("--n-perm", type=int, default=250, help="screening null size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=cfg.SEARCH_DIR)
    parser.add_argument("--grid-name", default="grid.csv")
    parser.add_argument("--rank-only", action="store_true", help="re-rank the existing grid csv")
    parser.add_argument("--replicate", type=int, default=15, help="per-session check on the top N (0 = skip)")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    args.pupil = [bool(p) for p in args.pupil]

    grid_path = args.out_dir / args.grid_name
    if args.rank_only:
        with grid_path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
    else:
        rows = grid_rows(args)
        write_csv(grid_path, rows)
        print(f"wrote {grid_path} ({len(rows)} rows)")
    ranking = rank(rows)
    if args.replicate:
        rep = replicate(ranking[: args.replicate], args)
        write_csv(args.out_dir / "replication.csv", rep)
    write_csv(args.out_dir / "ranking.csv", ranking)
    print(f"\n{len(ranking)} configurations; top {args.top} by score "
          f"(min pooled z over monkeys, minus |z| of any solid maze with Δ < 0):")
    for rec in ranking[: args.top]:
        sess_z = " ".join(f"{s.split('_g0')[0]}:{rec.get(f'{s}_z', np.nan):+.1f}"
                          for m in cfg.MONKEYS for s in cfg.SESSIONS[m] if f"{s}_z" in rec)
        print(f"  {rec['score']:+5.2f} {'C' if rec['consistent'] else ' '} "
              f"{rec['codebook']:16s} {rec['kind']:6s} {'pupil' if rec['pupil'] else '     '} "
              f"{rec['variant']:12s} F pooled z={rec['Faure_pooled_z']:+.2f} N pooled z={rec['Nielsen_pooled_z']:+.2f} "
              f"solid {rec['solid_pos']}/{rec['solid_total']} [{rec['Faure_solid_mazes']} | {rec['Nielsen_solid_mazes']}]"
              + (f"  sessions {sess_z}" if sess_z else ""))


if __name__ == "__main__":
    main()
