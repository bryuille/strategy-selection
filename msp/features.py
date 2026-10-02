"""Per-trial occupancy of the five fixed states, one cache per monkey.

Pipeline (no k-means; the codebook is `config.CODEBOOK_XY`):

1. **Clip** each trial to the analysis window (default ``geofix``,
   ``[0, fix_start]`` from maze onset; legacy ``pre1466`` is
   ``[fix_start - 1466 ms, fix_start]``).
2. **Warp** to unit H.
3. **Fixations** by I-DT on the clipped, unwarped data.
4. **Assign** each fixation's mean warped position to the nearest prototype
   within `ASSIGN_RADIUS`; every sample of that fixation inherits the state.
   Saccades, blinks and fixations outside every ball are omitted.

Under ``geofix`` (and any window other than ``pre1466``) gaze is re-detected
and re-warped per trial inside the window, the same path
`regression.features` uses; this covers the svm/top_ten sessions only. Under
the legacy ``pre1466`` window, warp and events come from the shared
attractor / clean-eye caches, whose validity is tied to the s1466 detection
segment, over every attractor session.

Blocks (one row per usable trial, ``float32``):

``occ_ms``   milliseconds of assigned fixation per state (d = 5)
``occ_bin``  visited / not visited per state, {0, 1} (d = 5) -- what `build` uses

Which trials become rows is decided before the codebook is consulted (QC,
maze 1-6, a usable window, >= 2 valid in-window on-screen samples), so the
row population equals that of the fitted-codebook caches at any K. This is
what makes `n_H`/`n_S` comparable across packages.

Diagnostics kept per trial: `n_fix` (fixation spans with >= 1 valid
in-window sample) and `n_fix_assigned`. Every fixation's centroid is also
stored (`fix_xy`), with its state (`fix_state`) and the index of the trial row
it belongs to (`fix_row`), so the figures can split fixations by maze and by
the trial's strategy label.

Cache: ``$STRATEGY_DATA_ROOT/processed/<Monkey>_msp_fixed5_r<radius>.npz`` for
the default ``geofix`` window; ``…_msp_fixed5_r<radius>_unith_s1466_e0.npz``
(the historical name) for ``pre1466``. The codebook, radius and window are
verified on load (stored ``assign_radius`` / ``window``, not the tag string)
and the cache rebuilt if `config.py` has changed under it.
"""

from __future__ import annotations

import argparse

import numpy as np

from data.builder import FIXATION_MIN_DURATION_MS, trial_qc_ok
from data.config import MONKEYS, PRE_FIX_END_MS, PRE_FIX_START_MS, processed_npz
from data.convert import load_npz
from data.loader import (
    load_attractor_eye_data,
    load_clean_eye_data,
    load_eye_behavioral_data,
    savez_atomic,
)
from data.builder import behavioral_lookup, fix_start_ms
from msp.config import (
    ASSIGN_RADIUS,
    CODEBOOK_XY,
    DEFAULT_WINDOW,
    PRE1466,
    K,
    MAZE_SCREEN_LIM,
    ORIGIN_STATE,
    AssignmentSpec,
    STATE_NAMES,
    WINDOW_NAMES,
    resolve_assignment,
    resolve_tag,
)

DEFAULT_START_MS = PRE_FIX_START_MS
DEFAULT_END_MS = PRE_FIX_END_MS
N_MAZES = 6
MIN_WINDOW_SAMPLES = 6  # clean_eye_data's per-trial floor before detection
FIXATION_EVENT = "fixation_idt"
BLOCK_NAMES = ("occ_ms", "occ_bin")
VARIANTS = ("full", "no_origin", "mean_removed")


# ---- helpers -----------------------------------------------------------------


def _event_lookup(monkey, sessions, start_ms=DEFAULT_START_MS, end_ms=DEFAULT_END_MS):
    """``(session, trial) -> [(name, onset, offset), ...]`` from the clean events."""
    events = load_clean_eye_data(monkey, start_ms=start_ms, end_ms=end_ms)
    names = np.asarray(events["name"]).astype(str)
    sess = np.asarray(events["session"]).astype(str)
    trials = np.asarray(events["trial_indices_all"]).astype(int)
    onset = np.asarray(events["onset"], dtype=float)
    offset = np.asarray(events["offset"], dtype=float)
    keep = np.isin(sess, list(sessions))
    lookup = {}
    for i in np.flatnonzero(keep):
        lookup.setdefault((sess[i], int(trials[i])), []).append(
            (names[i].lower(), onset[i], offset[i])
        )
    return lookup


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


def _trial_arrays(attractor, i):
    t_ms = np.asarray(attractor["time"][i], dtype=float) * 1000.0
    valid = np.asarray(attractor["valid"][i], dtype=bool)
    x = np.asarray(attractor["eye_x"][i], dtype=float)
    y = np.asarray(attractor["eye_y"][i], dtype=float)
    n = min(t_ms.size, valid.size, x.size, y.size)
    return t_ms[:n], valid[:n], x[:n], y[:n]


def _trial_window(behavioral, beh_i, start_ms, end_ms):
    """``(lo, hi)`` in trial ms, or None if the trial has no usable fixation onset."""
    fix_ms = fix_start_ms(behavioral, beh_i)
    if not np.isfinite(fix_ms) or fix_ms <= 0:
        return None
    return fix_ms - start_ms, fix_ms - end_ms


def clip_fixation_spans(
    event_rows, lo, hi, name_wanted=FIXATION_EVENT, min_duration_ms=FIXATION_MIN_DURATION_MS
):
    """`name_wanted` rows intersected with ``[lo, hi]`` and re-tested against
    the minimum duration -- equivalent to re-running I-DT on the clipped
    segment (see `classifier.features.clip_fixation_spans`)."""
    out = []
    for name, onset, offset in event_rows or ():
        if str(name).lower() != name_wanted:
            continue
        o0, o1 = max(float(onset), lo), min(float(offset), hi)
        if o1 - o0 >= min_duration_ms:
            out.append((name, o0, o1))
    return out


# ---- helpers (also in data.builder) ------------------------------------------


def assign_states(xy, codebook_xy, radius=ASSIGN_RADIUS):
    """Nearest prototype within scalar `radius`, else -1.

    The radius rejects points outside every ball; among the balls containing
    the point, nearest wins (first index on an exact tie).
    """
    xy = np.asarray(xy, dtype=np.float32)
    codebook_xy = np.asarray(codebook_xy, dtype=np.float32)
    if xy.size == 0:
        return np.empty(0, dtype=np.int16)

    radius = float(radius)
    dist = np.linalg.norm(xy[:, None, :] - codebook_xy[None, :, :], axis=-1)
    eligible = dist <= radius
    has = eligible.any(axis=1)
    state = np.full(xy.shape[0], -1, dtype=np.int16)
    if not has.any():
        return state

    dist_pick = np.where(eligible, dist, np.inf)
    pick = dist_pick.argmin(axis=1)
    state[has] = pick[has].astype(np.int16, copy=False)
    return state


def fixation_event_spans(event_rows):
    """I-DT fixation `(onset, offset)` spans, earliest first."""
    spans = []
    for name, onset, offset in event_rows or ():
        if str(name).lower() != FIXATION_EVENT:
            continue
        spans.append((float(onset), float(offset)))
    spans.sort()
    return spans


def fixation_centroids(xy, t_ms, valid, event_rows):
    """``(centroids (F, 2) float32, hit_masks)`` for the fixations with >= 1
    valid sample. Returned instead of consumed so the per-fixation state can be
    kept for diagnostics."""
    valid = np.asarray(valid, dtype=bool)
    t_ms = np.asarray(t_ms)
    centroids, hits = [], []
    if len(xy) == 0 or not valid.any():
        return np.empty((0, 2), dtype=np.float32), hits
    for onset, offset in fixation_event_spans(event_rows):
        hit = valid & (t_ms >= onset) & (t_ms <= offset)
        if not hit.any():
            continue
        centroids.append(np.asarray(xy[hit], dtype=np.float32).mean(axis=0))
        hits.append(hit)
    if not centroids:
        return np.empty((0, 2), dtype=np.float32), hits
    return np.stack(centroids).astype(np.float32), hits


def assign_trial(xy, t_ms, valid, event_rows, codebook_xy, assignment: AssignmentSpec):
    """``(state per sample, centroids, state per fixation)``.

    Every valid sample of a fixation inherits its centroid's state; everything
    else stays -1.
    """
    state = np.full(len(xy), -1, dtype=np.int16)
    centroids, hits = fixation_centroids(xy, t_ms, valid, event_rows)
    if not hits:
        return state, centroids, np.empty(0, dtype=np.int16)
    fix_state = assign_states(centroids, codebook_xy, radius=assignment.radius)
    for hit, sid in zip(hits, fix_state):
        if sid >= 0:
            state[hit] = sid
    return state, centroids, fix_state


# ---- extraction --------------------------------------------------------------


def _pack_feature_dict(
    blocks,
    rows_session,
    rows_trial,
    rows_maze,
    rows_n_fix,
    rows_n_fix_assigned,
    fix_xy,
    fix_state,
    fix_row,
    *,
    assignment: AssignmentSpec,
):
    """Assemble the on-disk / in-memory feature dict shared by both paths."""
    codebook = np.asarray(CODEBOOK_XY, dtype=np.float32)
    out = {
        name: (
            np.asarray(vals, dtype=np.float32)
            if vals
            else np.empty((0, K), dtype=np.float32)
        )
        for name, vals in blocks.items()
    }
    out["session"] = np.asarray(rows_session, dtype=str)
    out["trial_indices_all"] = np.asarray(rows_trial, dtype=int)
    out["maze_id"] = np.asarray(rows_maze, dtype=int)
    out["n_fix"] = np.asarray(rows_n_fix, dtype=int)
    out["n_fix_assigned"] = np.asarray(rows_n_fix_assigned, dtype=int)

    if fix_xy:
        out["fix_xy"] = np.concatenate(fix_xy, axis=0).astype(np.float32)
        out["fix_state"] = np.concatenate(fix_state)
        out["fix_row"] = np.concatenate(fix_row)
    else:
        out["fix_xy"] = np.empty((0, 2), dtype=np.float32)
        out["fix_state"] = np.empty(0, dtype=np.int16)
        out["fix_row"] = np.empty(0, dtype=np.int64)

    out["codebook_xy"] = codebook
    out["state_names"] = np.asarray(STATE_NAMES, dtype=str)
    out["codebook_k"] = np.int64(K)
    out["radius_tag"] = np.asarray(assignment.tag)
    out["assign_radius"] = np.float64(assignment.radius)
    out["window"] = np.asarray(assignment.window)
    if assignment.window == PRE1466:
        out["window_start_ms"] = np.int64(DEFAULT_START_MS)
        out["window_end_ms"] = np.int64(DEFAULT_END_MS)
    return out


def _extract_pre1466(monkey, assignment: AssignmentSpec):
    """Default window: reuse attractor warp + s1466 clean-eye events."""
    attractor = load_attractor_eye_data(monkey)
    behavioral = load_eye_behavioral_data(monkey)
    beh = behavioral_lookup(behavioral)
    sessions = np.asarray(attractor["session"]).astype(str)
    trials = np.asarray(attractor["trial_indices_all"]).astype(int)
    events = _event_lookup(
        monkey, set(sessions.tolist()), DEFAULT_START_MS, DEFAULT_END_MS
    )
    codebook = np.asarray(CODEBOOK_XY, dtype=np.float32)

    usable = []
    for i in range(sessions.size):
        session, trial_id = sessions[i], int(trials[i])
        beh_i = beh.get((session, trial_id))
        if not trial_qc_ok(behavioral, beh_i):
            continue
        maze = int(behavioral["geo_type"][beh_i])
        if not 1 <= maze <= N_MAZES:
            continue
        bounds = _trial_window(behavioral, beh_i, DEFAULT_START_MS, DEFAULT_END_MS)
        if bounds is None:
            continue
        lo, hi = bounds

        t_ms, valid, x, y = _trial_arrays(attractor, i)
        if t_ms.size < 2:
            continue

        in_window = np.isfinite(t_ms) & (t_ms >= lo) & (t_ms <= hi)
        keep_u = in_window & valid & np.isfinite(x) & np.isfinite(y)
        keep_u &= (np.abs(x) <= MAZE_SCREEN_LIM) & (np.abs(y) <= MAZE_SCREEN_LIM)
        if keep_u.sum() < 2:
            continue
        usable.append((i, session, trial_id, maze, lo, hi))

    print(
        f"  {monkey}: {len(usable)} usable trials; fixed K={K} codebook, "
        f"tag={assignment.tag}, {assignment.label()}"
    )

    blocks = {name: [] for name in BLOCK_NAMES}
    rows_session, rows_trial, rows_maze = [], [], []
    rows_n_fix, rows_n_fix_assigned = [], []
    fix_xy, fix_state, fix_row = [], [], []

    for i, session, trial_id, maze, lo, hi in usable:
        t_ms, valid, x, y = _trial_arrays(attractor, i)
        in_window = np.isfinite(t_ms) & (t_ms >= lo) & (t_ms <= hi)
        valid_win = valid & in_window & np.isfinite(x) & np.isfinite(y)
        spans = clip_fixation_spans(events.get((session, trial_id), ()), lo, hi)
        state, centroids, states_f = assign_trial(
            np.stack([x, y], axis=1).astype(np.float32),
            t_ms,
            valid_win,
            spans,
            codebook,
            assignment,
        )
        state = np.where(valid_win, state, -1).astype(int)

        if valid_win.sum() < 2:
            continue
        assigned = valid_win & (state >= 0) & (state < K)

        if assigned.any():
            dt = _sample_dt_seconds(t_ms[assigned], _sample_period_ms(t_ms))
            occ_ms = np.bincount(state[assigned], weights=dt, minlength=K) * 1000.0
        else:
            occ_ms = np.zeros(K)

        row = len(rows_session)
        blocks["occ_ms"].append(occ_ms)
        blocks["occ_bin"].append((occ_ms > 0).astype(float))
        rows_session.append(session)
        rows_trial.append(trial_id)
        rows_maze.append(maze)
        rows_n_fix.append(int(states_f.size))
        rows_n_fix_assigned.append(int((states_f >= 0).sum()))
        if states_f.size:
            fix_xy.append(centroids)
            fix_state.append(states_f.astype(np.int16))
            fix_row.append(np.full(states_f.size, row, dtype=np.int64))

    return _pack_feature_dict(
        blocks,
        rows_session,
        rows_trial,
        rows_maze,
        rows_n_fix,
        rows_n_fix_assigned,
        fix_xy,
        fix_state,
        fix_row,
        assignment=assignment,
    )


def redetect_trace(t_s, x, y, h, pupil, lo, hi, *, experiment):
    """Detect fixations and warp one trial inside ``[lo, hi]`` (ms from geo_present).

    Returns None when the trial is unusable, else a dict with the warped trace
    (``t_ms``, ``x``, ``y``, ``valid`` -- the detector's validity, whole trial),
    ``valid_win`` (valid, in-window, finite) and the in-window fixation
    ``spans``. Shared by the feature extraction and `msp.saccades`.
    """
    from data.builder import prepare_trial, trial_events

    t_s = np.asarray(t_s, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(t_s.size, x.size, y.size)
    t_s, x, y = t_s[:n], x[:n], y[:n]

    lo_s, hi_s = lo / 1000.0, hi / 1000.0
    keep = np.isfinite(t_s) & (t_s >= lo_s) & (t_s <= hi_s)
    if keep.sum() < MIN_WINDOW_SAMPLES:
        return None
    pupil_win = pupil
    if pupil is not None and np.asarray(pupil).ndim == 2 and len(pupil):
        pu = np.asarray(pupil, dtype=float)
        pk = np.isfinite(pu[:, 1]) & (pu[:, 1] >= lo_s) & (pu[:, 1] <= hi_s)
        pupil_win = pu[pk] if pk.any() else np.empty((0, 2))

    frame = trial_events(t_s[keep], x[keep], y[keep], pupil_win, experiment)
    event_rows = (
        []
        if frame is None
        else [
            (str(name), float(on), float(off))
            for name, on, off in frame.select("name", "onset", "offset").rows()
        ]
    )

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
        return None
    valid_win = valid & in_window & finite
    if valid_win.sum() < 2:
        return None
    return dict(
        t_ms=t_ms, x=wx, y=wy, valid=valid, valid_win=valid_win,
        spans=clip_fixation_spans(event_rows, lo, hi),
    )


def _trial_redetect(t_s, x, y, h, pupil, lo, hi, *, experiment, assignment: AssignmentSpec):
    """Detect, warp and assign one trial inside ``[lo, hi]`` (ms from geo_present).

    Returns None when the trial is unusable, else
    ``(occ_ms, centroids, fix_state)``.
    """
    tr = redetect_trace(t_s, x, y, h, pupil, lo, hi, experiment=experiment)
    if tr is None:
        return None
    t_ms, valid_win = tr["t_ms"], tr["valid_win"]
    xy = np.stack([tr["x"], tr["y"]], axis=1).astype(np.float32)
    state, centroids, states_f = assign_trial(
        xy, t_ms, valid_win, tr["spans"], CODEBOOK_XY, assignment
    )
    state = np.where(valid_win, state, -1).astype(int)
    assigned = valid_win & (state >= 0) & (state < K)
    if assigned.any():
        dt = _sample_dt_seconds(t_ms[assigned], _sample_period_ms(t_ms))
        occ_ms = np.bincount(state[assigned], weights=dt, minlength=K) * 1000.0
    else:
        occ_ms = np.zeros(K)
    return occ_ms, centroids, states_f


def _extract_redetect(monkey, assignment: AssignmentSpec):
    """Non-default windows: re-detect and re-warp from pooled ``*_eye_data``.

    Restricted to the svm/top_ten sessions (the estimator's label scope). A
    full-monkey re-detect would re-process every attractor session; those
    extra rows never enter the 2x2. Uses ``load_eye_data`` rather than
    per-session npzs so a laptop checkout that only has the pooled caches
    still runs. Pupil (blink) data is optional -- empty when the per-session
    eye file is absent, the same fallback ``clean_eye_data`` uses.
    """
    import pymovements as pm

    from data.builder import EYE_SAMPLING_RATE_HZ, pupil_for_trial, unpack_eye
    from data.loader import load_eye_data
    from msp.labels import top_ten_sessions

    sessions_wanted = set(top_ten_sessions(monkey))
    eye = load_eye_data(monkey)
    behavioral = load_eye_behavioral_data(monkey)
    beh = behavioral_lookup(behavioral)
    experiment = pm.Experiment(sampling_rate=EYE_SAMPLING_RATE_HZ)
    pupils_by_session = {}

    blocks = {name: [] for name in BLOCK_NAMES}
    rows_session, rows_trial, rows_maze = [], [], []
    rows_n_fix, rows_n_fix_assigned = [], []
    fix_xy, fix_state, fix_row = [], [], []
    kept_by_session = {s: 0 for s in sorted(sessions_wanted)}

    n_trials = len(eye["session"])
    for i in range(n_trials):
        t_s, x, y, session, trial_id = unpack_eye(eye, i)
        if session not in sessions_wanted:
            continue
        beh_i = beh.get((session, trial_id))
        if not trial_qc_ok(behavioral, beh_i):
            continue
        maze = int(behavioral["geo_type"][beh_i])
        if not 1 <= maze <= N_MAZES:
            continue
        fix_ms = float(fix_start_ms(behavioral, beh_i))
        if not np.isfinite(fix_ms) or fix_ms <= 0:
            continue
        lo, hi = assignment.window_bounds(fix_ms)

        pupil = pupil_for_trial(pupils_by_session, monkey, session, trial_id)
        h = tuple(float(behavioral[f"h{k}"][beh_i]) for k in range(1, 7))
        out = _trial_redetect(
            t_s, x, y, h, pupil, lo, hi,
            experiment=experiment, assignment=assignment,
        )
        if out is None:
            continue
        occ_ms, centroids, states_f = out

        row = len(rows_session)
        blocks["occ_ms"].append(occ_ms)
        blocks["occ_bin"].append((occ_ms > 0).astype(float))
        rows_session.append(session)
        rows_trial.append(trial_id)
        rows_maze.append(maze)
        rows_n_fix.append(int(states_f.size))
        rows_n_fix_assigned.append(int((states_f >= 0).sum()))
        if states_f.size:
            fix_xy.append(centroids)
            fix_state.append(states_f.astype(np.int16))
            fix_row.append(np.full(states_f.size, row, dtype=np.int64))
        kept_by_session[session] += 1

    for session, kept in kept_by_session.items():
        print(f"  {monkey} {session} [{assignment.window}]: kept={kept}")

    print(
        f"  {monkey}: {len(rows_session)} usable trials; fixed K={K} codebook, "
        f"tag={assignment.tag}, {assignment.label()}"
    )
    return _pack_feature_dict(
        blocks,
        rows_session,
        rows_trial,
        rows_maze,
        rows_n_fix,
        rows_n_fix_assigned,
        fix_xy,
        fix_state,
        fix_row,
        assignment=assignment,
    )


def extract_monkey_features(
    monkey,
    *,
    radius=ASSIGN_RADIUS,
    assignment=None,
    start_ms=DEFAULT_START_MS,
    end_ms=DEFAULT_END_MS,
):
    """Feature blocks for every usable trial of `monkey` against the fixed codebook.

    Pass a scalar ``radius`` or a full ``assignment``. ``resolve_assignment``
    builds the default (uniform balls, ``geofix`` window). ``start_ms`` /
    ``end_ms`` are only used by the legacy ``pre1466`` path and must match the
    package defaults.
    """
    if assignment is None:
        assignment = resolve_assignment(radius=radius)
    if assignment.window != PRE1466:
        if start_ms != DEFAULT_START_MS or end_ms != DEFAULT_END_MS:
            raise ValueError(
                f"start_ms/end_ms only apply to {PRE1466!r}; "
                f"got window={assignment.window!r}"
            )
        return _extract_redetect(monkey, assignment)
    if start_ms != DEFAULT_START_MS or end_ms != DEFAULT_END_MS:
        # Historical callers could override; keep the attractor path but
        # refuse combinations that would diverge from the assignment tag.
        raise ValueError(
            "non-default start_ms/end_ms are no longer supported; "
            "pass assignment.window='geofix' (or another named window) instead"
        )
    return _extract_pre1466(monkey, assignment)


def cache_stem(
    monkey,
    radius=ASSIGN_RADIUS,
    *,
    tag=None,
    assignment=None,
    start_ms=DEFAULT_START_MS,
    end_ms=DEFAULT_END_MS,
):
    """``geofix``: ``<M>_msp_fixed5_r<radius>``; other named windows append
    ``_<window>``; ``pre1466`` keeps its historical ``…_unith_s1466_e0`` name."""
    if assignment is None:
        assignment = resolve_tag(tag) if tag is not None else resolve_assignment(radius=radius)
    base = f"{monkey}_msp_fixed{K}_r{float(assignment.radius):g}"
    if assignment.window == PRE1466:
        return f"{base}_unith_s{int(start_ms)}_e{int(end_ms)}"
    if assignment.window == DEFAULT_WINDOW:
        return base
    return f"{base}_{assignment.window}"


def _cache_matches(data, assignment: AssignmentSpec):
    """False if the cache was built under a different codebook, radius or
    window, or predates the per-fixation arrays. Radius and window are read
    from their own fields rather than the tag, whose spelling depends on which
    window is the default (``r0.5`` meant pre1466 before 2026-10-02)."""
    try:
        stored = np.asarray(data["assign_radius"], dtype=float).ravel()
        if stored.size != 1 or not np.isfinite(stored[0]):
            return False
        window_ok = (
            "window" not in data and assignment.window == PRE1466  # predates the field
        ) or (
            "window" in data
            and str(np.asarray(data["window"]).astype(str).reshape(-1)[0]) == assignment.window
        )
        return (
            "fix_row" in data
            and int(data["codebook_k"]) == K
            and np.array_equal(
                np.asarray(data["codebook_xy"], dtype=np.float32),
                np.asarray(CODEBOOK_XY, dtype=np.float32),
            )
            and window_ok
            and tuple(np.asarray(data["state_names"]).astype(str)) == tuple(STATE_NAMES)
            and bool(np.isclose(float(stored[0]), float(assignment.radius)))
        )
    except KeyError:
        return False


def load_features(
    monkey,
    *,
    radius=ASSIGN_RADIUS,
    assignment=None,
    start_ms=DEFAULT_START_MS,
    end_ms=DEFAULT_END_MS,
    refresh=False,
):
    """Cached fixed-codebook feature blocks for `monkey`, rebuilt on mismatch."""
    if assignment is None:
        assignment = resolve_assignment(radius=radius)
    path = processed_npz(
        cache_stem(monkey, assignment=assignment, start_ms=start_ms, end_ms=end_ms)
    )
    if path.exists() and not refresh:
        data = load_npz(path)
        if _cache_matches(data, assignment):
            return data
        print(f"  {path.name}: built under a different codebook, radius or schema; rebuilding")
    print(
        f"Extracting {monkey} fixed-codebook features "
        f"(tag={assignment.tag}, {assignment.label()}) ..."
    )
    data = extract_monkey_features(
        monkey, assignment=assignment, start_ms=start_ms, end_ms=end_ms
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    savez_atomic(path, **data)
    print(f"Saved {path} ({len(data['session'])} trials)")
    return data


# ---- analysis variants on the occupancy block ------------------------------


def no_origin_dims(k=K, origin=ORIGIN_STATE):
    """Dimension indices to KEEP under `no_origin`."""
    return [d for d in range(k) if d != origin]


def apply_no_origin(X, k=K, origin=ORIGIN_STATE):
    dims = no_origin_dims(k, origin)
    return np.asarray(X, dtype=float)[:, dims], dims


def fit_grand_mean(X):
    """Mean feature vector over the finite rows of `X` (each trial once)."""
    X = np.asarray(X, dtype=float)
    finite = np.isfinite(X).all(axis=1)
    Xf = X[finite]
    if Xf.shape[0] == 0:
        return np.zeros(X.shape[1])
    return Xf.mean(axis=0)


def apply_mean_removed(X, mean_profile):
    return np.asarray(X, dtype=float) - np.asarray(mean_profile, dtype=float)


def apply_variant(X, variant, *, mean_profile=None):
    """Apply one named variant to an (n, K) occupancy block.

    ``full``          unmodified.
    ``no_origin``     drop the origin state, leaving the four exits.
    ``mean_removed``  subtract the maze grand mean (label-free; fit once
                      before the estimator). Absolute r is uninterpretable;
                      read Δ / z, and `no_origin` alongside.

    Returns ``(X_variant, dims)`` where `dims` are the original dimension
    indices retained (identity except under `no_origin`).
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    X = np.asarray(X, dtype=float)
    if variant == "full":
        return X, list(range(X.shape[1]))
    if variant == "no_origin":
        return apply_no_origin(X, X.shape[1], ORIGIN_STATE)
    if mean_profile is None:
        mean_profile = fit_grand_mean(X)
    return apply_mean_removed(X, mean_profile), list(range(X.shape[1]))


def main():
    parser = argparse.ArgumentParser(description="Build the fixed-codebook feature cache(s).")
    parser.add_argument("--monkey", nargs="*", default=list(MONKEYS), choices=MONKEYS)
    parser.add_argument(
        "--radius", type=float, default=None,
        help="uniform assignment radius around each fixed prototype (unit H)",
    )
    parser.add_argument(
        "--window", default=DEFAULT_WINDOW, choices=list(WINDOW_NAMES),
        help="analysis clip: geofix (default; re-detects from raw eye, top-ten "
             "sessions) or pre1466 (attractor caches, every session)",
    )
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    assignment = resolve_assignment(radius=args.radius, window=args.window)

    for monkey in args.monkey:
        data = load_features(monkey, assignment=assignment, refresh=args.refresh)
        mazes = np.asarray(data["maze_id"], dtype=int)
        counts = {m: int((mazes == m).sum()) for m in range(1, N_MAZES + 1)}
        n_fix = int(data["n_fix"].sum())
        n_ok = int(data["n_fix_assigned"].sum())
        print(f"{monkey}: {len(mazes)} trials {counts}")
        print(f"  fixations assigned: {n_ok}/{n_fix} ({n_ok / max(n_fix, 1):.1%})")


if __name__ == "__main__":
    main()
