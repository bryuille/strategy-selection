"""Per-fixation table against the fixed codebook, one cache per (monkey, window).

The pipeline is msp's (see `msp/features.py`): the gaze is re-detected and
re-warped per trial inside the analysis window.

1. **Trials**: unfaded (``trial_fade == 0``, checked explicitly), then
   `data.builder.trial_qc_ok` (photodiode QC, ``path_type != -99``), maze 1-6,
   ``fix_start > geo_present``.
2. **Clip** samples and pupil to the window's ``[lo, hi]`` (ms from
   geo_present), exactly as the windowed branch of
   `data.builder.clean_eye_data` does, including its >= 6-sample floor.
3. **Detect** saccades and I-DT fixations on the clipped segment with
   `data.builder.trial_events` -- the detector behind every events cache.
4. **Warp + validity** with `data.builder.prepare_trial`, given those events:
   unit-H position and a mask of on-screen, non-blink fixation samples.
5. **Assign** each fixation's mean valid warped position to the nearest
   prototype within ``ASSIGN_RADIUS`` (helpers copied from msp).

Caches live in ``data/processed/`` (same convention as msp), as
``<Monkey>_reg_fixed5_r1_<window>.npz``. Session subsets are filtered on load;
they do not get their own files. The three measures are derived on load by
`trial_measures`, so changing how they are defined never needs a re-extraction.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from regression.config import (
    ASSIGN_RADIUS,
    CODEBOOK_XY,
    K,
    MAZE_SCREEN_LIM,
    MIN_WINDOW_SAMPLES,
    N_MAZES,
    SESSIONS,
    STATE_NAMES,
    WINDOW_NAMES,
    WINDOWS,
    cache_stem,
)

SCHEMA_VERSION = 1
FIXATION_EVENT = "fixation_idt"


# ---- helpers copied from msp/features.py -------------------------------------


def _sample_period_ms(t_ms):
    """Median positive sample interval (ms) of a full, unmasked time vector."""
    t_ms = np.asarray(t_ms, dtype=float)
    if t_ms.size < 2:
        return 1.0
    d = np.diff(t_ms)
    pos = d[d > 0]
    return float(np.median(pos)) if pos.size else 1.0


def _sample_dt_seconds(t_ms, period_ms=None):
    """Per-sample durations (s), each capped at the sample period, so gaps in
    the masked timebase (saccades, blinks, unassigned fixations) contribute
    nothing to dwell."""
    t_ms = np.asarray(t_ms, dtype=float)
    if t_ms.size == 0:
        return np.empty(0, dtype=float)
    if period_ms is None:
        period_ms = _sample_period_ms(t_ms)
    if t_ms.size == 1:
        return np.array([period_ms / 1000.0])
    dt = np.empty(t_ms.size, dtype=float)
    dt[:-1] = np.diff(t_ms)
    dt[-1] = period_ms
    dt = np.clip(dt, 0.0, period_ms)
    return np.where(dt > 0, dt, 0.0) / 1000.0


def clip_fixation_spans(event_rows, lo, hi, min_duration_ms):
    """Fixation rows intersected with ``[lo, hi]`` and re-tested against the
    minimum duration."""
    out = []
    for name, onset, offset in event_rows or ():
        if str(name).lower() != FIXATION_EVENT:
            continue
        o0, o1 = max(float(onset), lo), min(float(offset), hi)
        if o1 - o0 >= min_duration_ms:
            out.append((name, o0, o1))
    return out


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


def fixation_centroids(xy, t_ms, valid, event_rows):
    """``(centroids (F, 2), onsets (F,), hit masks)`` for fixations with >= 1
    valid sample, earliest first."""
    valid = np.asarray(valid, dtype=bool)
    t_ms = np.asarray(t_ms)
    spans = sorted(
        (float(on), float(off))
        for name, on, off in event_rows or ()
        if str(name).lower() == FIXATION_EVENT
    )
    centroids, onsets, hits = [], [], []
    if len(xy) == 0 or not valid.any():
        return np.empty((0, 2), dtype=np.float32), np.empty(0), hits
    for onset, offset in spans:
        hit = valid & (t_ms >= onset) & (t_ms <= offset)
        if not hit.any():
            continue
        centroids.append(np.asarray(xy[hit], dtype=np.float32).mean(axis=0))
        onsets.append(onset)
        hits.append(hit)
    if not centroids:
        return np.empty((0, 2), dtype=np.float32), np.empty(0), hits
    return np.stack(centroids).astype(np.float32), np.asarray(onsets), hits


# ---- one trial -----------------------------------------------------------------


def trial_fixations(t_s, x, y, h, pupil, lo, hi, *, experiment, radius=ASSIGN_RADIUS):
    """Detect, warp and assign one trial inside ``[lo, hi]`` (ms from geo_present).

    Returns None when the trial is unusable, else a dict with the window's
    fixations: ``centroid``, ``state``, ``dwell_ms`` (assigned dwell, 0 for an
    unassigned fixation) and ``onset_ms``.
    """
    from data.builder import prepare_trial
    from data.builder import FIXATION_MIN_DURATION_MS, trial_events

    t_s = np.asarray(t_s, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(t_s.size, x.size, y.size)
    t_s, x, y = t_s[:n], x[:n], y[:n]

    # ---- Step 2: clip, as clean_eye_data's windowed branch.
    lo_s, hi_s = lo / 1000.0, hi / 1000.0
    keep = np.isfinite(t_s) & (t_s >= lo_s) & (t_s <= hi_s)
    if keep.sum() < MIN_WINDOW_SAMPLES:
        return None
    pupil_win = pupil
    if pupil is not None and np.asarray(pupil).ndim == 2 and len(pupil):
        pu = np.asarray(pupil, dtype=float)
        pk = np.isfinite(pu[:, 1]) & (pu[:, 1] >= lo_s) & (pu[:, 1] <= hi_s)
        pupil_win = pu[pk] if pk.any() else np.empty((0, 2))

    # ---- Step 3: detect on the clipped segment.
    frame = trial_events(t_s[keep], x[keep], y[keep], pupil_win, experiment)
    event_rows = (
        []
        if frame is None
        else [
            (str(name), float(on), float(off))
            for name, on, off in frame.select("name", "onset", "offset").rows()
        ]
    )

    # ---- Step 4: warp the whole trial, as data.builder does, with these events.
    p = prepare_trial(t_s, x, y, h, pupil, event_rows=event_rows)
    if p is None:
        return None
    t_ms = np.asarray(p["t"], dtype=float) * 1000.0
    wx = np.asarray(p["x"], dtype=float)
    wy = np.asarray(p["y"], dtype=float)
    valid = np.asarray(p["valid_orig"], dtype=bool)
    m = min(t_ms.size, valid.size, wx.size, wy.size)
    t_ms, valid, wx, wy = t_ms[:m], valid[:m], wx[:m], wy[:m]

    in_window = np.isfinite(t_ms) & (t_ms >= lo) & (t_ms <= hi)
    finite = np.isfinite(wx) & np.isfinite(wy)
    on_screen = (np.abs(wx) <= MAZE_SCREEN_LIM) & (np.abs(wy) <= MAZE_SCREEN_LIM)
    if (in_window & valid & finite & on_screen).sum() < 2:
        return None  # msp's usable-trial rule
    valid_win = valid & in_window & finite
    if valid_win.sum() < 2:
        return None

    # ---- Step 5: assign fixation centroids.
    spans = clip_fixation_spans(event_rows, lo, hi, FIXATION_MIN_DURATION_MS)
    xy = np.stack([wx, wy], axis=1).astype(np.float32)
    centroids, onsets, hits = fixation_centroids(xy, t_ms, valid_win, spans)
    fix_state = assign_states(centroids, CODEBOOK_XY, radius=radius)

    sample_state = np.full(m, -1, dtype=int)
    sample_fix = np.full(m, -1, dtype=int)
    for j, (hit, sid) in enumerate(zip(hits, fix_state)):
        if sid >= 0:
            sample_state[hit] = sid
            sample_fix[hit] = j
    assigned = valid_win & (sample_state >= 0)

    dwell_ms = np.zeros(len(hits))
    if assigned.any():
        dt = _sample_dt_seconds(t_ms[assigned], _sample_period_ms(t_ms))
        dwell_ms = np.bincount(sample_fix[assigned], weights=dt, minlength=len(hits)) * 1000.0

    return {
        "centroid": centroids,
        "state": fix_state.astype(np.int16),
        "dwell_ms": dwell_ms.astype(float),
        "onset_ms": np.asarray(onsets, dtype=float),
    }


# ---- extraction ------------------------------------------------------------------


def extract(monkey, window, sessions, *, radius=ASSIGN_RADIUS):
    """Per-trial and per-fixation tables for `monkey` under `window`."""
    import pymovements as pm

    from data.builder import EYE_SAMPLING_RATE_HZ, gaze_arrays, trial_qc_ok
    from data.mat import load_eye
    from data.config import eye_mat_path
    from data.loader import load_eye_behavioral_data
    from data.builder import behavioral_lookup, fix_start_ms

    window_fn = WINDOWS[window]
    behavioral = load_eye_behavioral_data(monkey)
    beh = behavioral_lookup(behavioral)
    experiment = pm.Experiment(sampling_rate=EYE_SAMPLING_RATE_HZ)

    rows = {k: [] for k in ("session", "trial", "maze", "fix_ms", "lo", "hi")}
    fix = {k: [] for k in ("row", "state", "dwell_ms", "onset_ms", "xy")}
    drops = {}

    for session in sessions:
        path = eye_mat_path(monkey, session)
        if not path.exists():
            raise FileNotFoundError(f"missing eye mat for {session}: {path}")
        raw = load_eye(monkey, session)
        gaze = raw["gaze"]
        pupils = list(raw["pupil_size"]) if "pupil_size" in raw else None
        tally = {"no_behavioral": 0, "faded": 0, "qc": 0, "maze": 0,
                 "no_fix_start": 0, "unusable": 0, "kept": 0}

        for trial_id, g in enumerate(gaze, start=1):
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
            lo, hi = window_fn(fix_ms)

            t_s, x, y = gaze_arrays(g)
            pupil = (
                pupils[trial_id - 1]
                if pupils is not None and trial_id - 1 < len(pupils)
                else np.empty((0, 2))
            )
            h = tuple(float(behavioral[f"h{k}"][beh_i]) for k in range(1, 7))
            out = trial_fixations(
                t_s, x, y, h, pupil, lo, hi, experiment=experiment, radius=radius
            )
            if out is None:
                tally["unusable"] += 1
                continue

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
        print(f"  {monkey} {session} [{window}]: " + ", ".join(f"{k}={v}" for k, v in tally.items()))

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
        "window": np.asarray(window),
        "sessions": np.asarray(tuple(sessions), dtype=str),
        "drops_json": np.asarray(json.dumps(drops)),
        "schema_version": np.int64(SCHEMA_VERSION),
    }


def _cache_matches(data, window, sessions, radius):
    try:
        return (
            int(data["schema_version"]) == SCHEMA_VERSION
            and str(data["window"]) == window
            and tuple(np.asarray(data["sessions"]).astype(str)) == tuple(sessions)
            and np.isclose(float(data["assign_radius"]), float(radius))
            and np.array_equal(np.asarray(data["codebook_xy"], dtype=np.float32), CODEBOOK_XY)
            and tuple(np.asarray(data["state_names"]).astype(str)) == STATE_NAMES
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
        "window": data["window"],
        "sessions": np.asarray(tuple(sessions), dtype=str),
        "drops_json": data["drops_json"],
        "schema_version": data["schema_version"],
    }
    return out


def load_fixations(monkey, window, *, sessions=None, radius=ASSIGN_RADIUS, refresh=False):
    """Cached tables for (monkey, window) under ``data/processed/``.

    Always built for the publication session set; ``sessions`` subsets are
    filtered after load (no per-subset cache files).
    """
    from data.config import processed_npz
    from data.loader import savez_atomic
    from data.mat import load_npz

    if window not in WINDOWS:
        raise ValueError(f"unknown window {window!r}; choose from {WINDOW_NAMES}")
    sessions = tuple(sessions) if sessions else SESSIONS[monkey]
    full = SESSIONS[monkey]
    path = processed_npz(cache_stem(monkey, window, radius))
    if path.exists() and not refresh:
        data = load_npz(path)
        if _cache_matches(data, window, full, radius):
            return _subset_sessions(data, sessions)
        print(f"  {path.name}: stale (schema/codebook/radius/sessions); rebuilding")
    print(f"Extracting {monkey} [{window}] for {', '.join(full)} ...")
    data = extract(monkey, window, full, radius=radius)
    savez_atomic(path, **data)
    print(f"Saved {path} ({data['session'].size} trials, {data['fix_state'].size} fixations)")
    return _subset_sessions(data, sessions)


# ---- the three measures -------------------------------------------------------------


def trial_measures(data):
    """Per (trial, state) measures from the per-fixation table.

    ``occ``      1 if the state received any assigned dwell, else 0
    ``visits``   runs of consecutive same-state assigned fixations, so an I-DT
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
    parser.add_argument("--window", nargs="*", default=list(WINDOW_NAMES), choices=list(WINDOW_NAMES))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    for monkey in args.monkey:
        for window in args.window:
            load_fixations(monkey, window, refresh=args.refresh)


if __name__ == "__main__":
    main()
