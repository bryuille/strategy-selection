"""Per-fixation table against the fixed codebook, one cache per monkey.

The pipeline is msp's (see `msp/features.py`): looks from `data.looks`.

1. **Trials**: unfaded (``trial_fade == 0``, checked explicitly), then
   `data.builder.trial_qc_ok` (photodiode QC, ``path_type != -99``), maze 1-6,
   ``fix_start > geo_present``, a usable saccade threshold.
2. **Looks** from the ``fix`` epoch cache (maze onset -> `fix_start`) of the
   advisor's `timestamp_preprocessing`: centroid in unit H, stable duration.
3. **Assign** each look's centroid (on-maze, within ``MAZE_SCREEN_LIM``) to
   the nearest prototype within ``ASSIGN_RADIUS``; dwell = stable support
   where assigned, 0 otherwise.

Caches live in ``data/processed/`` (same convention as msp), as
``<Monkey>_reg_looks5_r1.npz`` (the ``*_reg_fixed5_*`` caches are orphaned). Session subsets are filtered on load;
they do not get their own files. The three measures are derived on load by
`trial_measures`, so changing how they are defined never needs a re-extraction.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from data.looks import LOOKS_SCHEMA, load_looks, looks_of, usable_rows
from data.looks import source_sha256 as looks_source_sha256

from regression.config import (
    ASSIGN_RADIUS,
    CODEBOOK_XY,
    K,
    MAZE_SCREEN_LIM,
    N_MAZES,
    SESSIONS,
    STATE_NAMES,
    cache_stem,
    window_bounds,
)

SCHEMA_VERSION = 2
EVENTS_SOURCE = "timestamp_preprocessing"


# ---- helpers copied from msp/features.py -------------------------------------


def assign_states(xy, codebook_xy, radius=ASSIGN_RADIUS):
    """Nearest prototype within scalar `radius`, else -1 (first index on a tie)."""
    xy = np.asarray(xy, dtype=np.float32)
    codebook_xy = np.asarray(codebook_xy, dtype=np.float32)
    if xy.size == 0:
        return np.empty(0, dtype=np.int16)
    dist = np.linalg.norm(xy[:, None, :] - codebook_xy[None, :, :], axis=-1)
    eligible = dist <= float(radius)
    has = eligible.any(axis=1)
    state = np.full(xy.shape[0], -1, dtype=np.int16)
    if not has.any():
        return state
    pick = np.where(eligible, dist, np.inf).argmin(axis=1)
    state[has] = pick[has].astype(np.int16, copy=False)
    return state


# ---- one trial -----------------------------------------------------------------


def trial_fixations(looks, r, *, radius=ASSIGN_RADIUS):
    """Looks of cache row `r`, assigned: ``centroid`` (unit H), ``state``,
    ``dwell_ms`` (stable support where assigned, 0 otherwise), ``onset_ms``."""
    sl = looks_of(looks, r)
    xy = looks["look_xy_u"][sl].astype(np.float32).reshape(-1, 2)
    stable = looks["look_stable_ms"][sl].astype(float)
    onset = looks["look_onset_ms"][sl].astype(float)
    on_maze = (np.abs(xy) <= MAZE_SCREEN_LIM).all(axis=1)
    xy, stable, onset = xy[on_maze], stable[on_maze], onset[on_maze]
    state = assign_states(xy, CODEBOOK_XY, radius=radius)
    return {
        "centroid": xy,
        "state": state.astype(np.int16),
        "dwell_ms": np.where(state >= 0, stable, 0.0).astype(float),
        "onset_ms": onset,
    }


# ---- extraction ------------------------------------------------------------------


def extract(monkey, sessions, *, radius=ASSIGN_RADIUS):
    """Per-trial and per-fixation tables for `monkey`."""
    from data.builder import behavioral_lookup, fix_start_ms, trial_qc_ok
    from data.loader import load_eye_behavioral_data

    behavioral = load_eye_behavioral_data(monkey)
    beh = behavioral_lookup(behavioral)
    looks = load_looks(monkey, "fix", tuple(sessions))
    usable = usable_rows(looks)

    rows = {k: [] for k in ("session", "trial", "maze", "fix_ms", "lo", "hi")}
    fix = {k: [] for k in ("row", "state", "dwell_ms", "onset_ms", "xy")}
    drops = {}

    for session in sessions:
        tally = {"no_behavioral": 0, "faded": 0, "qc": 0, "maze": 0,
                 "no_fix_start": 0, "saccade_status": 0, "kept": 0}
        for r in np.flatnonzero(looks["session"].astype(str) == session):
            trial_id = int(looks["trial_indices_all"][r])
            beh_i = beh.get((session, trial_id))
            if beh_i is None:
                tally["no_behavioral"] += 1
                continue
            if float(behavioral["trial_fade"][beh_i]) != 0:
                tally["faded"] += 1
                continue
            if not trial_qc_ok(behavioral, beh_i):
                tally["qc"] += 1
                continue
            maze = int(behavioral["geo_type"][beh_i])
            if not 1 <= maze <= N_MAZES:
                tally["maze"] += 1
                continue
            fix_ms = float(fix_start_ms(behavioral, beh_i))
            if not np.isfinite(fix_ms) or fix_ms <= 0:
                tally["no_fix_start"] += 1
                continue
            if not usable[r]:
                tally["saccade_status"] += 1
                continue
            lo, hi = window_bounds(fix_ms)
            out = trial_fixations(looks, r, radius=radius)

            row = len(rows["session"])
            rows["session"].append(session)
            rows["trial"].append(trial_id)
            rows["maze"].append(maze)
            rows["fix_ms"].append(fix_ms)
            rows["lo"].append(lo)
            rows["hi"].append(hi)
            nf = out["state"].size
            fix["row"].append(np.full(nf, row, dtype=np.int64))
            fix["state"].append(out["state"])
            fix["dwell_ms"].append(out["dwell_ms"])
            fix["onset_ms"].append(out["onset_ms"])
            fix["xy"].append(out["centroid"].reshape(-1, 2))
            tally["kept"] += 1

        drops[session] = tally
        print(f"  {monkey} {session}: " + ", ".join(f"{k}={v}" for k, v in tally.items()))

    def cat(parts, dtype, shape=(0,)):
        return np.concatenate(parts).astype(dtype) if parts else np.empty(shape, dtype=dtype)

    return {
        "session": np.asarray(rows["session"], dtype=str),
        "trial_indices_all": np.asarray(rows["trial"], dtype=int),
        "maze_id": np.asarray(rows["maze"], dtype=int),
        "fix_ms": np.asarray(rows["fix_ms"], dtype=float),
        "window_lo_ms": np.asarray(rows["lo"], dtype=float),
        "window_hi_ms": np.asarray(rows["hi"], dtype=float),
        "fix_row": cat(fix["row"], np.int64),
        "fix_state": cat(fix["state"], np.int16),
        "fix_dwell_ms": cat(fix["dwell_ms"], float),
        "fix_onset_ms": cat(fix["onset_ms"], float),
        "fix_xy": cat(fix["xy"], np.float32, (0, 2)),
        "codebook_xy": np.asarray(CODEBOOK_XY, dtype=np.float32),
        "state_names": np.asarray(STATE_NAMES, dtype=str),
        "assign_radius": np.float64(radius),
        "sessions": np.asarray(tuple(sessions), dtype=str),
        "drops_json": np.asarray(json.dumps(drops)),
        "schema_version": np.int64(SCHEMA_VERSION),
        "events_source": np.asarray(EVENTS_SOURCE),
        "looks_schema": np.int64(LOOKS_SCHEMA),
        "looks_source_sha256": np.asarray(looks_source_sha256()),
    }


def _cache_matches(data, sessions, radius):
    try:
        return (
            int(data["schema_version"]) == SCHEMA_VERSION
            and tuple(np.asarray(data["sessions"]).astype(str)) == tuple(sessions)
            and np.isclose(float(data["assign_radius"]), float(radius))
            and np.array_equal(np.asarray(data["codebook_xy"], dtype=np.float32), CODEBOOK_XY)
            and tuple(np.asarray(data["state_names"]).astype(str)) == STATE_NAMES
            and str(np.asarray(data["events_source"]).reshape(-1)[0]) == EVENTS_SOURCE
            and int(data["looks_schema"]) == LOOKS_SCHEMA
            and str(np.asarray(data["looks_source_sha256"]).reshape(-1)[0]) == looks_source_sha256()
        )
    except (KeyError, ValueError, TypeError):
        return False


def _subset_sessions(data, sessions):
    """Keep only trials (and their fixations) whose session is in `sessions`."""
    wanted = set(sessions)
    keep = np.asarray([str(s) in wanted for s in np.asarray(data["session"])], dtype=bool)
    if keep.all():
        out = dict(data)
        out["sessions"] = np.asarray(tuple(sessions), dtype=str)
        return out
    old_row = np.flatnonzero(keep)
    new_of = {int(o): i for i, o in enumerate(old_row)}
    fix_row = np.asarray(data["fix_row"], dtype=int)
    fkeep = np.asarray([int(r) in new_of for r in fix_row], dtype=bool)
    out = {
        "session": np.asarray(data["session"])[keep],
        "trial_indices_all": np.asarray(data["trial_indices_all"])[keep],
        "maze_id": np.asarray(data["maze_id"])[keep],
        "fix_ms": np.asarray(data["fix_ms"])[keep],
        "window_lo_ms": np.asarray(data["window_lo_ms"])[keep],
        "window_hi_ms": np.asarray(data["window_hi_ms"])[keep],
        "fix_row": np.asarray([new_of[int(r)] for r in fix_row[fkeep]], dtype=np.int64),
        "fix_state": np.asarray(data["fix_state"])[fkeep],
        "fix_dwell_ms": np.asarray(data["fix_dwell_ms"])[fkeep],
        "fix_onset_ms": np.asarray(data["fix_onset_ms"])[fkeep],
        "fix_xy": np.asarray(data["fix_xy"])[fkeep],
        "codebook_xy": np.asarray(data["codebook_xy"]),
        "state_names": np.asarray(data["state_names"]),
        "assign_radius": data["assign_radius"],
        "sessions": np.asarray(tuple(sessions), dtype=str),
        "drops_json": data["drops_json"],
        "schema_version": data["schema_version"],
        "events_source": data["events_source"],
        "looks_schema": data["looks_schema"],
        "looks_source_sha256": data["looks_source_sha256"],
    }
    return out


def load_fixations(monkey, *, sessions=None, radius=ASSIGN_RADIUS, refresh=False):
    """Cached tables for `monkey` under ``data/processed/``.

    Always built for the publication session set; ``sessions`` subsets are
    filtered after load (no per-subset cache files).
    """
    from data.config import processed_npz
    from data.loader import savez_atomic
    from data.mat import load_npz

    sessions = tuple(sessions) if sessions else SESSIONS[monkey]
    full = SESSIONS[monkey]
    path = processed_npz(cache_stem(monkey, radius))
    if path.exists() and not refresh:
        data = load_npz(path)
        if _cache_matches(data, full, radius):
            return _subset_sessions(data, sessions)
        print(f"  {path.name}: stale (schema/codebook/radius/sessions); rebuilding")
    print(f"Extracting {monkey} for {', '.join(full)} ...")
    data = extract(monkey, full, radius=radius)
    savez_atomic(path, **data)
    print(f"Saved {path} ({data['session'].size} trials, {data['fix_state'].size} fixations)")
    return _subset_sessions(data, sessions)


# ---- the three measures -------------------------------------------------------------


def trial_measures(data):
    """Per (trial, state) measures from the per-fixation table.

    ``occ``      1 if the state received any assigned dwell, else 0
    ``visits``   runs of consecutive same-state assigned fixations, so a detector
                 drift split into fragments counts once
    ``dur_ms``   dwell / visits; NaN where unvisited
    ``dwell_ms`` total assigned dwell (msp's ``occ_ms``)
    ``n_fix``    raw assigned fixation count, for comparison with ``visits``
    """
    from data.builder import collapse_state_runs

    n = np.asarray(data["session"]).size
    rows = np.asarray(data["fix_row"], dtype=int)
    state = np.asarray(data["fix_state"], dtype=int)
    dwell = np.asarray(data["fix_dwell_ms"], dtype=float)
    onset = np.asarray(data["fix_onset_ms"], dtype=float)

    dwell_ms = np.zeros((n, K))
    n_fix = np.zeros((n, K))
    visits = np.zeros((n, K))
    a = state >= 0
    np.add.at(dwell_ms, (rows[a], state[a]), dwell[a])
    np.add.at(n_fix, (rows[a], state[a]), 1.0)

    order = np.lexsort((onset, rows))
    rows_o, state_o = rows[order], state[order]
    bounds = np.flatnonzero(np.diff(rows_o)) + 1
    for seg_rows, seg_state in zip(np.split(rows_o, bounds), np.split(state_o, bounds)):
        if seg_rows.size == 0:
            continue
        for sid in collapse_state_runs(seg_state[seg_state >= 0]):
            visits[seg_rows[0], sid] += 1.0

    occ = (dwell_ms > 0).astype(float)
    # a zero-dwell assigned fixation (all samples masked) is not a visit
    visits = np.where(occ > 0, np.maximum(visits, 1.0), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        dur_ms = np.where(occ > 0, dwell_ms / visits, np.nan)
    window_ms = np.asarray(data["window_hi_ms"], float) - np.asarray(data["window_lo_ms"], float)
    return {
        "occ": occ,
        "visits": visits,
        "dur_ms": dur_ms,
        "dwell_ms": dwell_ms,
        "n_fix": n_fix,
        "window_ms": window_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="Build the regression fixation caches.")
    parser.add_argument("--monkey", nargs="*", default=list(SESSIONS), choices=list(SESSIONS))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    for monkey in args.monkey:
        load_fixations(monkey, refresh=args.refresh)


if __name__ == "__main__":
    main()
