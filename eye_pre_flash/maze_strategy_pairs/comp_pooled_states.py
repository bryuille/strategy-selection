"""Which codebook states carry the H/S difference, and where they sit.

`comp_pooled_build` reports one p per (maze, K) but says nothing about *which*
state drove it. This module opens that up for a pair of adjacent K, printing
the codebook geometry and the per-state occupancy split by strategy, so a step
in p between two K can be read against what actually changed in the codebook.

It computes no statistics the sweep does not already report and writes no
files -- it is a read-only diagnostic over caches `comp_pooled_build` has
already built, and its whole output is stdout.

Motivating case: Nielsen maze 4 under `svm/top_ten`, where p sits at the
permutation floor at every K from 5 up but not at K=4, across nearly every
feature and variant. The question that step raises is whether K=5's extra
prototype covers gaze that only one strategy produces.

**Read the cross-K matching as geometry, not identity.** `elbow.md` is
explicit that prototype indices are not comparable between two k-means fits:
"the same prototype" has no meaning between K and K+1. Going 4 -> 5 may append
one centre or may re-partition all of them. So this prints nearest-centre
distances in both directions and leaves the reading to you, rather than
declaring some state "the new one".

Two further cautions on the per-state tables:

* **The visit-rate comparisons ignore session clustering.** The estimator
  permutes labels *within session* precisely because trials from one day are
  alike for reasons unrelated to strategy. The per-state columns here are
  plain pooled proportions with no such control, so they describe the pool
  and do not test anything. The per-session breakdown is printed for the same
  reason -- to show whether a split is general or one session's doing.
* **`occ_ms` only counts assigned samples.** A fixation centroid further than
  `ASSIGN_RADIUS` from every prototype is dropped, so total dwell per trial
  rises with K. The `total assigned dwell` block reports that directly,
  because "K=5 captures gaze K=4 threw away" and "K=5 splits gaze K=4 already
  had" are different claims with different meanings.

Usage:
    uv run python -m eye_pre_flash.maze_strategy_pairs.comp_pooled_states
    uv run python -m eye_pre_flash.maze_strategy_pairs.comp_pooled_states \
        --monkey Nielsen --maze 4 --k 4 5
"""

from __future__ import annotations

import argparse

import numpy as np

from eye_pre_flash.label_sources import source_lookup, source_scope_sessions
from eye_pre_flash.maze_strategy_pairs import comp_pooled_features
from eye_pre_flash.maze_strategy_pairs.comp_pooled_build import (
    SOURCE_SCOPES_WANTED,
    _maze_codebook,
    pool_maze,
)
from eye_pre_flash.maze_strategy_pairs.estimator import H, S


def load_pack(monkey, maze, k, keep_sessions, lookup):
    """Codebook, labels and occupancy for one (monkey, maze, K), in-scope rows only."""
    data = comp_pooled_features.load_features(monkey, k=k)
    codebook = _maze_codebook(data, maze)
    if codebook is None:
        raise SystemExit(f"maze {maze}: no codebook at K={k} in the cache")
    mask, y, session_ids = pool_maze(
        data, maze=maze, keep_sessions=keep_sessions, lookup=lookup
    )
    if mask.sum() == 0:
        raise SystemExit(f"maze {maze}: no labelled in-scope trial at K={k}")
    pack = dict(
        k=k,
        codebook=np.asarray(codebook, dtype=float),
        y=y,
        sessions=session_ids,
        trials=np.asarray(data["trial_indices_all"], dtype=int)[mask],
        occ_ms=np.asarray(data["occ_ms"], dtype=float)[mask],
        occ_bin=np.asarray(data["occ_bin"], dtype=float)[mask],
    )
    del data
    return pack


def print_codebook(pack, maze):
    k = pack["k"]
    print(f"\n=== K={k}: maze {maze} codebook (unit-H warped screen coords) ===")
    print(f"{'state':>5} {'x':>9} {'y':>9} {'radius':>9} {'angle_deg':>10}")
    for j, (x, y) in enumerate(pack["codebook"]):
        print(
            f"{j:>5} {x:>9.3f} {y:>9.3f} {np.hypot(x, y):>9.3f} "
            f"{np.degrees(np.arctan2(y, x)):>10.1f}"
        )


def print_matching(pack_a, pack_b):
    """Nearest-centre distances both ways between two codebooks."""
    ca, cb = pack_a["codebook"], pack_b["codebook"]
    ka, kb = pack_a["k"], pack_b["k"]
    d = np.linalg.norm(cb[:, None, :] - ca[None, :, :], axis=2)

    print(f"\n=== codebook K={ka} vs K={kb}: nearest-centre geometry ===")
    print("Descriptive only -- prototype identity does not carry across fits.")
    print(f"\n  each K={kb} state -> nearest K={ka} state")
    for j in range(kb):
        i = int(d[j].argmin())
        second = float(np.sort(d[j])[1]) if ka > 1 else float("nan")
        print(
            f"    K{kb} state {j} -> K{ka} state {i}   dist {d[j, i]:8.3f}"
            f"   (next nearest {second:8.3f})"
        )
    print(f"\n  each K={ka} state -> nearest K={kb} state")
    for i in range(ka):
        j = int(d[:, i].argmin())
        print(f"    K{ka} state {i} -> K{kb} state {j}   dist {d[j, i]:8.3f}")

    claimed = {}
    for j in range(kb):
        claimed.setdefault(int(d[j].argmin()), []).append(j)
    unclaimed = sorted(set(range(ka)) - set(claimed))
    print(f"\n  K={ka} states claimed by >1: "
          f"{ {i: v for i, v in claimed.items() if len(v) > 1} }")
    print(f"  K={ka} states claimed by none: {unclaimed}")
    mean_shift = float(d.min(axis=1).mean())
    print(f"  mean K{kb}->K{ka} nearest distance: {mean_shift:.3f}  "
          f"(large => re-partition, not an append)")


def state_rows(pack):
    rows = []
    occ_ms, occ_bin, y = pack["occ_ms"], pack["occ_bin"], pack["y"]
    for j in range(pack["k"]):
        vis = occ_bin[:, j] > 0
        row = {"state": j}
        for tag, sel in (("H", y == H), ("S", y == S)):
            n = int(sel.sum())
            nv = int((vis & sel).sum())
            row[f"n_{tag}"] = n
            row[f"vis_{tag}"] = nv
            row[f"pct_{tag}"] = 100.0 * nv / n if n else np.nan
            row[f"ms_{tag}"] = float(occ_ms[sel, j].mean()) if n else np.nan
            row[f"msv_{tag}"] = (
                float(occ_ms[sel & vis, j].mean()) if (sel & vis).any() else 0.0
            )
        row["share_S_of_visitors"] = (
            100.0 * row["vis_S"] / (row["vis_S"] + row["vis_H"])
            if (row["vis_S"] + row["vis_H"])
            else np.nan
        )
        rows.append(row)
    return rows


def print_states(pack, maze):
    y = pack["y"]
    n_h, n_s = int((y == H).sum()), int((y == S).sum())
    base_s = 100.0 * n_s / (n_h + n_s)
    print(f"\n=== K={pack['k']}: maze {maze} per-state occupancy, H vs S ===")
    print(f"n_H={n_h}  n_S={n_s}  (base rate S = {base_s:.1f}% of trials)")
    print("Pooled proportions, no session control -- descriptive.\n")
    print(
        f"{'st':>3} | {'H visit':>14} {'S visit':>14} | "
        f"{'%S of vis':>9} | {'H ms':>8} {'S ms':>8} | {'H ms|vis':>9} {'S ms|vis':>9}"
    )
    print("-" * 96)
    for r in state_rows(pack):
        print(
            f"{r['state']:>3} | "
            f"{r['vis_H']:>4}/{r['n_H']:<4}{r['pct_H']:>5.1f}% "
            f"{r['vis_S']:>4}/{r['n_S']:<4}{r['pct_S']:>5.1f}% | "
            f"{r['share_S_of_visitors']:>8.1f}% | "
            f"{r['ms_H']:>8.1f} {r['ms_S']:>8.1f} | "
            f"{r['msv_H']:>9.1f} {r['msv_S']:>9.1f}"
        )
    print(f"\n  (a state with %S of visitors far from {base_s:.1f}% is "
          f"strategy-biased in who reaches it)")


def print_totals(packs):
    """Total assigned dwell per trial -- does a higher K capture more gaze, and for whom?"""
    print("\n=== total assigned dwell per trial (ms), by K and strategy ===")
    print("occ_ms counts assigned samples only, so a rise means the codebook")
    print("brought previously-unassigned gaze into the feature.\n")
    print(f"{'K':>4} | {'H mean':>10} {'S mean':>10} | {'S - H':>9}")
    print("-" * 42)
    for pack in packs:
        tot = pack["occ_ms"].sum(axis=1)
        y = pack["y"]
        mh = float(tot[y == H].mean())
        ms = float(tot[y == S].mean())
        print(f"{pack['k']:>4} | {mh:>10.1f} {ms:>10.1f} | {ms - mh:>+9.1f}")


def print_profiles(pack):
    y = pack["y"]
    print(f"\n=== K={pack['k']}: mean occupancy profile (ms) by strategy ===")
    ph = pack["occ_ms"][y == H].mean(axis=0)
    ps = pack["occ_ms"][y == S].mean(axis=0)
    width = 9
    print("  state:  " + " ".join(f"{j:>{width}}" for j in range(pack["k"])))
    print("  H    :  " + " ".join(f"{v:>{width}.1f}" for v in ph))
    print("  S    :  " + " ".join(f"{v:>{width}.1f}" for v in ps))
    print("  S - H:  " + " ".join(f"{v:>+{width}.1f}" for v in ps - ph))


def print_per_session(pack, maze):
    """Per-session visit counts, so a pooled split can be checked for one-session drive."""
    print(f"\n=== K={pack['k']}: maze {maze} per-session visit counts (H/S) ===")
    header = "  " + f"{'session':>14} " + " ".join(
        f"{'st' + str(j):>13}" for j in range(pack["k"])
    )
    print(header)
    print("-" * len(header))
    for sess in sorted(set(pack["sessions"].tolist())):
        in_sess = pack["sessions"] == sess
        cells = []
        for j in range(pack["k"]):
            vis = (pack["occ_bin"][:, j] > 0) & in_sess
            nh = int((vis & (pack["y"] == H)).sum())
            ns = int((vis & (pack["y"] == S)).sum())
            th = int((in_sess & (pack["y"] == H)).sum())
            ts = int((in_sess & (pack["y"] == S)).sum())
            cells.append(f"{nh:>3}/{th:<3}{ns:>3}/{ts:<3}")
        print(f"  {sess:>14} " + " ".join(f"{c:>13}" for c in cells))
    print("  cells are  Hvisit/Htotal  Svisit/Stotal")


def print_examples(pack, n_examples):
    print(f"\n=== K={pack['k']}: example trials per state (top {n_examples} by dwell) ===")
    for j in range(pack["k"]):
        order = np.argsort(-pack["occ_ms"][:, j])[:n_examples]
        print(f"\n  state {j}:")
        any_shown = False
        for i in order:
            if pack["occ_ms"][i, j] <= 0:
                continue
            any_shown = True
            tag = "H" if pack["y"][i] == H else "S"
            prof = " ".join(f"{v:7.0f}" for v in pack["occ_ms"][i])
            print(
                f"    {str(pack['sessions'][i]):>12} trial {int(pack['trials'][i]):>4} "
                f"[{tag}]  here={pack['occ_ms'][i, j]:8.1f} ms   profile [{prof}]"
            )
        if not any_shown:
            print("    (never visited by any in-scope trial)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--monkey", default="Nielsen")
    parser.add_argument("--maze", type=int, default=4)
    parser.add_argument(
        "--k", type=int, nargs="+", default=[4, 5],
        help="codebook sizes to compare; matching is printed for each adjacent pair",
    )
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument(
        "--per-session", action="store_true",
        help="print the per-session visit breakdown (wide output)",
    )
    args = parser.parse_args()

    source, scope = SOURCE_SCOPES_WANTED[0]
    keep = tuple(source_scope_sessions(source, scope)[0].get(args.monkey, ()))
    if not keep:
        raise SystemExit(f"{args.monkey}/{source}/{scope}: no sessions in scope")
    lookup = source_lookup(source, keep)

    print(f"{args.monkey} / maze {args.maze} / {source}/{scope}")
    print(f"{len(keep)} sessions: {list(keep)}")

    packs = [load_pack(args.monkey, args.maze, k, keep, lookup) for k in args.k]

    for pack in packs:
        print_codebook(pack, args.maze)
    for a, b in zip(packs, packs[1:]):
        print_matching(a, b)

    print_totals(packs)

    for pack in packs:
        print_states(pack, args.maze)
        print_profiles(pack)
        if args.per_session:
            print_per_session(pack, args.maze)

    print_examples(packs[-1], args.examples)


if __name__ == "__main__":
    main()
