"""Trial table and window looks, one cache per monkey; counterfactual items
built from it.

Per trial of `config.SESSIONS` that passes `data.builder.trial_qc_ok`, has
exactly one `trial_answer*` flag and a usable look record
(`data.looks.usable_rows`), the cache keeps the trial's geometry, choice,
outcome, neural label and every look (from `data.looks`, epoch ``trial``)
overlapping ``[flash_three, trial_end]``: raw centroid (deg), onset, offset.
Exit assignment is done at analysis time (`counterfactual_items`) so the radius can
vary without re-extracting.

Cache: ``<PROCESSED_ROOT>/<Monkey>_cfv2_trials.npz``.

    uv run python -m counterfactual.extract [--monkey ...] [--refresh]
"""

from __future__ import annotations

import argparse

import numpy as np

from data.builder import behavioral_lookup, trial_qc_ok
from data.loader import load_eye_behavioral_data, load_strategy_choices, savez_atomic
from data.looks import load_looks, looks_of, source_sha256, trial_index, usable_rows
from data.mat import load_npz
from counterfactual.config import (
    CHOICE_FIELD,
    EXIT_RADIUS_DEG,
    LOOKS_EPOCH,
    MONKEYS,
    SCHEMA,
    SESSIONS,
    WINDOW_START_FIELD,
    trials_cache,
)
from counterfactual.task import assign_exit, chosen_exit, correct_exit, true_exit

DROP_REASONS = ("qc", "no_chosen_exit", "not_in_looks", "unusable_looks", "no_window")


def _rel_ms(behavioral, field, i):
    v, g = float(behavioral[field][i]), float(behavioral["geo_present"][i])
    return (v - g) * 1000.0 if np.isfinite(v) and np.isfinite(g) else np.nan


def _labels(session):
    """``trial_id -> 0 (H) / 1 (S)`` from the dendrogram label cache."""
    lab = np.asarray(load_strategy_choices(session), dtype=float)
    return {i + 1: v for i, v in enumerate(lab) if np.isfinite(v)}


def extract(monkey, sessions):
    looks = load_looks(monkey, LOOKS_EPOCH, sessions)
    rows, usable = trial_index(looks), usable_rows(looks)
    behavioral = load_eye_behavioral_data(monkey)
    beh = behavioral_lookup(behavioral)
    tr = {k: [] for k in ("session", "trial", "maze", "h", "vel", "lr", "lr2", "chosen", "true",
                          "correct", "label", "t_lo", "t_choice", "t_hi", "look_start", "look_stop")}
    lk = {k: [] for k in ("xy", "on", "off")}
    dropped = {s: dict.fromkeys(DROP_REASONS, 0) for s in sessions}
    for session in sessions:
        labels = _labels(session)
        beh_rows = sorted((i for (s, _), i in beh.items() if s == session),
                          key=lambda i: int(behavioral["trial_indices_all"][i]))
        for i in beh_rows:
            tid = int(behavioral["trial_indices_all"][i])
            if not trial_qc_ok(behavioral, i):
                dropped[session]["qc"] += 1
                continue
            lr, lr2 = float(behavioral["LR"][i]), float(behavioral["LR2"][i])
            c = chosen_exit([float(behavioral[f"trial_answer{k}"][i]) for k in range(1, 5)], lr, lr2)
            if c is None:
                dropped[session]["no_chosen_exit"] += 1
                continue
            r = rows.get((session, tid))
            if r is None:
                dropped[session]["not_in_looks"] += 1
                continue
            if not usable[r]:
                dropped[session]["unusable_looks"] += 1
                continue
            lo, choice = _rel_ms(behavioral, WINDOW_START_FIELD, i), _rel_ms(behavioral, CHOICE_FIELD, i)
            hi = float(looks["epoch_end_ms"][r])
            if not (np.isfinite(lo) and np.isfinite(choice) and lo < choice < hi):
                dropped[session]["no_window"] += 1
                continue
            sl = looks_of(looks, r)
            on, off = looks["look_onset_ms"][sl], looks["look_offset_ms"][sl]
            keep = (off >= lo) & (on <= hi)
            start = len(lk["on"])
            lk["xy"].extend(looks["look_xy_d"][sl][keep].astype(float).reshape(-1, 2))
            lk["on"].extend(on[keep])
            lk["off"].extend(off[keep])
            tr["session"].append(session)
            tr["trial"].append(tid)
            tr["maze"].append(int(behavioral["geo_type"][i]))
            tr["h"].append([float(behavioral[f"h{k}"][i]) for k in range(1, 7)])
            tr["vel"].append(float(behavioral["vel"][i]))
            tr["lr"].append(int(lr))
            tr["lr2"].append(int(lr2))
            tr["chosen"].append(c)
            tr["true"].append(true_exit(lr, lr2))
            tr["correct"].append(correct_exit(float(behavioral["path_type"][i])))
            tr["label"].append(labels.get(tid, np.nan))
            tr["t_lo"].append(lo)
            tr["t_choice"].append(choice)
            tr["t_hi"].append(hi)
            tr["look_start"].append(start)
            tr["look_stop"].append(len(lk["on"]))
    for s in sessions:
        print(f"  {monkey} {s}: kept {sum(x == s for x in tr['session'])}, dropped {dropped[s]}")
    out = {k: np.asarray(v) for k, v in tr.items()}
    out["h"] = out["h"].reshape(-1, 6).astype(float)
    out.update(
        look_xy_d=np.asarray(lk["xy"], dtype=float).reshape(-1, 2),
        look_onset_ms=np.asarray(lk["on"], dtype=float),
        look_offset_ms=np.asarray(lk["off"], dtype=float),
        dropped=np.asarray([[dropped[s][k] for k in DROP_REASONS] for s in sessions]),
        drop_reasons=np.asarray(DROP_REASONS),
        sessions=np.asarray(sessions),
        looks_source_sha256=np.asarray(source_sha256()),
        schema=np.int64(SCHEMA),
    )
    if not (out["correct"] == out["true"]).all():
        raise AssertionError("path_type exit and (LR, LR2) exit disagree")
    return out


def load(monkey, *, refresh=False):
    sessions = SESSIONS[monkey]
    path = trials_cache(monkey)
    if path.exists() and not refresh:
        d = load_npz(path)
        try:
            ok = (int(d["schema"]) == SCHEMA and tuple(d["sessions"].astype(str)) == sessions
                  and str(d["looks_source_sha256"]) == source_sha256())
        except KeyError:
            ok = False
        if ok:
            return d
        print(f"  {path.name}: stale; rebuilding")
    d = extract(monkey, sessions)
    savez_atomic(path, **d)
    print(f"Saved {path} ({len(d['session'])} trials, {len(d['look_onset_ms'])} looks)")
    return d


# ---- counterfactual items ---------------------------------------------------------


def counterfactual_items(tr, r, radius=EXIT_RADIUS_DEG):
    """Unchosen exits looked at on trial `r`, in time order, repeats kept:
    each window look is assigned to a state and the exit states other than
    the chosen exit are kept."""
    sl = slice(int(tr["look_start"][r]), int(tr["look_stop"][r]))
    states = assign_exit(tr["look_xy_d"][sl], tr["h"][r], radius)
    c = int(tr["chosen"][r])
    return [int(s) for s in states if s >= 1 and s != c]


def main():
    p = argparse.ArgumentParser(description="Build the cf trial / look cache.")
    p.add_argument("--monkey", nargs="*", default=list(MONKEYS), choices=MONKEYS)
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()
    for m in args.monkey:
        tr = load(m, refresh=args.refresh)
        n = sum(bool(counterfactual_items(tr, r)) for r in range(len(tr["session"])))
        print(f"{m}: {len(tr['session'])} trials, {n} with a counterfactual look")


if __name__ == "__main__":
    main()
