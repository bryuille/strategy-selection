"""Post-choice look table and exit sequences, one cache per monkey.

Looks come from `data.looks` (the `timestamp_preprocessing` package, whole
trial: epoch ``trial`` = maze onset -> `trial_end`). Per trial of the
publication sessions (`config.SESSIONS`) that passes `data.builder.trial_qc_ok`
and has a usable saccade threshold:

1. **Window** starts at `saccade_init` (first finite of `WINDOW_START_FIELDS`)
   and ends at the epoch end. Looks overlapping the window are kept (the look
   on the chosen exit usually begins before `saccade_init`; those straddling
   looks are tallied as `n_straddle`).
2. **States**: each look's raw centroid (degrees) is assigned to the nearest
   exit within `EXIT_RADIUS_DEG` (1-4), the origin (0) or elsewhere (-1), with
   the exits placed from the trial's arm lengths.
3. **Sequence**: the decoded chosen exit, then the exit-state looks in order
   (`counterfactual.exits.exit_sequence`). The first look in the window is
   where the choice saccade landed; when it is on the chosen exit
   (`landing_on_chosen`, the sanity check of the decoding) item 0 already
   stands for it and the post-choice items start with the next look.

The dendrogram strategy label (0 = H, 1 = S) is attached when the session's
``<session>_strategy_choices.npz`` cache exists; it is read, never built here.

Cache: ``<PROCESSED_ROOT>/<Monkey>_cf_exits.npz``, rebuilt when the schema,
session list or looks source differs.

    uv run python -m counterfactual.extract [--monkey ...] [--refresh]
"""

from __future__ import annotations

import argparse

import numpy as np

from data.builder import behavioral_lookup, trial_qc_ok
from data.config import processed_npz
from data.loader import load_eye_behavioral_data, savez_atomic
from data.looks import load_looks, looks_of, saccades_of, trial_index, usable_rows
from data.mat import load_npz
from counterfactual.config import (
    MONKEYS,
    SCHEMA,
    SESSIONS,
    WINDOW_START_FIELDS,
    cache_path,
)
from counterfactual.exits import assign_states, chosen_exit, correct_exit, exit_sequence

DROP_REASONS = ("qc", "no_chosen_exit", "no_window", "not_in_looks", "saccade_status")
LOOKS_EPOCH = "trial"


# ---- helpers ------------------------------------------------------------------


def rel_ms(behavioral, field, i):
    """`field` of row `i` in ms from `geo_present`; NaN when absent."""
    if field not in behavioral:
        return np.nan
    v = float(behavioral[field][i])
    g = float(behavioral["geo_present"][i])
    return (v - g) * 1000.0 if np.isfinite(v) and np.isfinite(g) else np.nan


def window_start_ms(behavioral, i):
    return next((v for v in (rel_ms(behavioral, f, i) for f in WINDOW_START_FIELDS) if np.isfinite(v)), np.nan)


def strategy_labels(session):
    """``trial_id -> 0/1`` from the session's dendrogram label cache, or {}."""
    path = processed_npz(f"{session}_strategy_choices")
    if not path.exists():
        return {}
    lab = np.asarray(load_npz(path)["strategy_choices"], dtype=float)
    return {i + 1: float(v) for i, v in enumerate(lab) if np.isfinite(v)}


# ---- extraction ----------------------------------------------------------------


def extract(monkey, sessions):
    sessions = tuple(sessions)
    looks = load_looks(monkey, LOOKS_EPOCH, sessions)
    look_rows = trial_index(looks)
    usable = usable_rows(looks)
    behavioral = load_eye_behavioral_data(monkey)
    beh = behavioral_lookup(behavioral)
    labels = {s: strategy_labels(s) for s in sessions}

    tr = {k: [] for k in ("session", "trial", "maze", "h", "chosen", "correct", "strategy",
                          "lo", "hi", "n_fix", "n_straddle", "landing", "exit_seq", "state_seq")}
    fx = {k: [] for k in ("row", "xy", "on", "off", "stable", "roi_adv", "state")}
    sc = {k: [] for k in ("row", "on", "off")}
    dropped = {s: dict.fromkeys(DROP_REASONS, 0) for s in sessions}
    kept = dict.fromkeys(sessions, 0)
    for session in sessions:
        beh_rows = sorted((i for (s, t), i in beh.items() if s == session),
                          key=lambda i: int(behavioral["trial_indices_all"][i]))
        for beh_i in beh_rows:
            trial_id = int(behavioral["trial_indices_all"][beh_i])
            if not trial_qc_ok(behavioral, beh_i):
                dropped[session]["qc"] += 1
                continue
            flags = tuple(float(behavioral[f"trial_answer{k}"][beh_i]) for k in range(1, 5))
            chosen = chosen_exit(flags, float(behavioral["LR"][beh_i]), float(behavioral["LR2"][beh_i]))
            if chosen is None:
                dropped[session]["no_chosen_exit"] += 1
                continue
            r = look_rows.get((session, trial_id))
            if r is None:
                dropped[session]["not_in_looks"] += 1
                continue
            if not usable[r]:
                dropped[session]["saccade_status"] += 1
                continue
            lo = window_start_ms(behavioral, beh_i)
            hi = float(looks["epoch_end_ms"][r])
            if not np.isfinite(lo) or not hi > lo:
                dropped[session]["no_window"] += 1
                continue
            h = tuple(float(v) for v in looks["h"][r])
            sl = looks_of(looks, r)
            on = looks["look_onset_ms"][sl]
            off = looks["look_offset_ms"][sl]
            # Looks overlapping the window count: the look on the chosen exit
            # usually begins before saccade_init (gaze is already there), so
            # requiring onset >= lo would drop the landing look itself.
            inside = off >= lo
            straddle = int(np.sum((on < lo) & (off >= lo)))
            xy = looks["look_xy_d"][sl][inside].astype(float).reshape(-1, 2)
            states = assign_states(xy, h)
            landing = bool(states.size) and int(states[0]) == chosen
            row = len(tr["session"])
            tr["session"].append(session)
            tr["trial"].append(trial_id)
            tr["maze"].append(int(behavioral["geo_type"][beh_i]))
            tr["h"].append(h)
            tr["chosen"].append(chosen)
            tr["correct"].append(correct_exit(float(behavioral["path_type"][beh_i])) or -1)
            tr["strategy"].append(labels[session].get(trial_id, np.nan))
            tr["lo"].append(lo)
            tr["hi"].append(hi)
            tr["n_fix"].append(int(inside.sum()))
            tr["n_straddle"].append(straddle)
            tr["landing"].append(landing)
            tr["exit_seq"].append(exit_sequence(chosen, states[1:] if landing else states))
            tr["state_seq"].append([int(s) for s in states])
            for k_in, (p, s) in enumerate(zip(xy, states)):
                k = np.flatnonzero(inside)[k_in]
                fx["row"].append(row)
                fx["xy"].append(p)
                fx["on"].append(float(on[k]))
                fx["off"].append(float(off[k]))
                fx["stable"].append(float(looks["look_stable_ms"][sl][k]))
                fx["roi_adv"].append(str(looks["look_roi_adv"][sl][k]))
                fx["state"].append(int(s))
            ss = saccades_of(looks, r)
            for s_on, s_off in zip(looks["sac_onset_ms"][ss], looks["sac_offset_ms"][ss]):
                if s_off >= lo:
                    sc["row"].append(row)
                    sc["on"].append(float(s_on))
                    sc["off"].append(float(s_off))
            kept[session] += 1

    for s in sessions:
        print(f"  {monkey} {s}: kept={kept[s]} dropped={dropped[s]} labels={len(labels[s])}")
    return {
        "session": np.asarray(tr["session"], dtype=str),
        "trial_indices_all": np.asarray(tr["trial"], dtype=int),
        "maze_id": np.asarray(tr["maze"], dtype=int),
        "h": np.asarray(tr["h"], dtype=np.float32).reshape(-1, 6),
        "chosen_exit": np.asarray(tr["chosen"], dtype=int),
        "correct_exit": np.asarray(tr["correct"], dtype=int),
        "strategy": np.asarray(tr["strategy"], dtype=float),
        "window_lo_ms": np.asarray(tr["lo"], dtype=np.float32),
        "window_hi_ms": np.asarray(tr["hi"], dtype=np.float32),
        "n_fix": np.asarray(tr["n_fix"], dtype=int),
        "n_straddle": np.asarray(tr["n_straddle"], dtype=int),
        "landing_on_chosen": np.asarray(tr["landing"], dtype=bool),
        "exit_seq": np.asarray(tr["exit_seq"] + [None], dtype=object)[:-1],
        "state_seq": np.asarray(tr["state_seq"] + [None], dtype=object)[:-1],
        "fix_row": np.asarray(fx["row"], dtype=np.int64),
        "fix_xy_d": np.asarray(fx["xy"], dtype=np.float32).reshape(-1, 2),
        "fix_onset_ms": np.asarray(fx["on"], dtype=np.float32),
        "fix_offset_ms": np.asarray(fx["off"], dtype=np.float32),
        "fix_stable_ms": np.asarray(fx["stable"], dtype=np.float32),
        "fix_roi_adv": np.asarray(fx["roi_adv"], dtype=str),
        "fix_state": np.asarray(fx["state"], dtype=np.int16),
        "sac_row": np.asarray(sc["row"], dtype=np.int64),
        "sac_onset_ms": np.asarray(sc["on"], dtype=np.float32),
        "sac_offset_ms": np.asarray(sc["off"], dtype=np.float32),
        "dropped": np.asarray([[dropped[s][r] for r in DROP_REASONS] for s in sessions], dtype=int),
        "drop_reasons": np.asarray(DROP_REASONS, dtype=str),
        "sessions": np.asarray(sessions, dtype=str),
        "looks_epoch": np.asarray(LOOKS_EPOCH),
        "looks_source_sha256": looks["source_sha256"],
        "schema": np.int64(SCHEMA),
    }


# ---- cache -----------------------------------------------------------------------


def cache_ok(data, sessions):
    from data.looks import source_sha256

    try:
        return (
            int(data["schema"]) == SCHEMA
            and tuple(np.asarray(data["sessions"]).astype(str)) == tuple(sessions)
            and str(np.asarray(data["looks_source_sha256"]).reshape(-1)[0]) == source_sha256()
        )
    except KeyError:
        return False


def load(monkey, *, sessions=None, refresh=False):
    """The post-choice table for `monkey`; built from the looks cache when
    missing, stale or `refresh`."""
    sessions = tuple(SESSIONS[monkey] if sessions is None else sessions)
    path = cache_path(monkey)
    if path.exists() and not refresh:
        data = load_npz(path)
        if cache_ok(data, sessions):
            return data
        print(f"  {path.name}: schema / sessions / looks source differ; rebuilding")
    print(f"Extracting {monkey} post-choice exits ({', '.join(sessions)}) ...", flush=True)
    data = extract(monkey, sessions)
    savez_atomic(path, **data)
    print(f"Saved {path} ({len(data['session'])} trials, {len(data['fix_row'])} looks)")
    return data


def main():
    parser = argparse.ArgumentParser(description="Build the post-choice exit cache(s).")
    parser.add_argument("--monkey", nargs="*", default=list(MONKEYS), choices=list(MONKEYS))
    parser.add_argument("--session", nargs="*", help="subset of the monkey's publication sessions")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    for monkey in args.monkey:
        sessions = SESSIONS[monkey]
        if args.session:
            sessions = tuple(s for s in sessions if s in args.session)
        data = load(monkey, sessions=sessions, refresh=args.refresh)
        lengths = np.array([len(q) for q in data["exit_seq"]])
        print(f"{monkey}: {len(lengths)} trials; exit-list length median {np.median(lengths):.0f}, "
              f"max {lengths.max() if lengths.size else 0}; "
              f"{int((lengths > 1).sum())} trials with a post-choice exit look; "
              f"landing on chosen {data['landing_on_chosen'].mean():.0%}; "
              f"straddling looks {int(data['n_straddle'].sum())}")


if __name__ == "__main__":
    main()
