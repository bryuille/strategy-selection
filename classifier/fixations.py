"""Fixation table per monkey and label scope.

Extraction, window maze onset -> fixation onset:

1. **Clip** each trial to ``[0, fix_start]`` ms from maze onset.
2. **Detect** I-DT fixations and saccades on the clipped, unwarped gaze
   (`data.builder.trial_events`), blink events masked from the pupil trace.
3. **Warp** to unit H (`data.builder.prepare_trial`).
4. **Centroids**: every fixation with >= 1 valid in-window sample is kept
   with its centroid in unit H *and* in degrees, its onset / offset and its
   valid dwell (ms). No codebook / region is consulted here: assignment
   happens in `classifier.features`.

Pupil per trial (raw tracker units; z-scored within session later): mean
over the window, and late minus early (last / first `PUPIL_*_MS`). Samples
within 50 ms of a tracker-loss run are dropped. Pupil traces come from
``classifier/out/pupil/<session>_pupil.npz`` (`classifier.pupil_extract`,
cluster) or, failing that, the eye mats; with neither, the trial's pupil
features are NaN and blink masking is off -- the extraction then prints a
loud warning, because a cache built that way is not the cluster's.

Cache: ``<PROCESSED_ROOT>/<Monkey>_msp_revised_fix[_svm].npz`` (legacy name,
see `cache_path`), rebuilt when the stored schema, sessions or window differ.

    uv run python -m classifier.fixations [--labels dendro svm] [--monkey ...] [--refresh]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

from data.builder import (
    EYE_SAMPLING_RATE_HZ,
    FIXATION_MIN_DURATION_MS,
    behavioral_lookup,
    fix_start_ms,
    prepare_trial,
    pupil_for_trial,
    trial_events,
    trial_qc_ok,
    unpack_eye,
)
from data.config import MAT_ROOT, processed_npz
from data.loader import load_eye_behavioral_data, load_eye_data, savez_atomic
from data.mat import load_npz
from classifier.config import (
    LABEL_SCOPES,
    MAZES,
    MONKEYS,
    PUPIL_DIR,
    PUPIL_EARLY_MS,
    PUPIL_LATE_MS,
    PUPIL_MIN_VALID,
)

# Schema and window tag are the on-disk contract of the existing caches;
# change them only together with a rebuild.
SCHEMA = 2
WINDOW_TAG = "onset_to_fix"
MIN_WINDOW_SAMPLES = 6
FIXATION_EVENT = "fixation_idt"
PUPIL_EDGE_MS = 50
UNIT_LIM = 3.0  # unit-H box; warped samples outside are off-maze


def window_bounds(fix_ms):
    """``(lo, hi)`` in ms from geo_present: maze onset to fixation onset."""
    return 0.0, float(fix_ms)


# ---- helpers -----------------------------------------------------------------


def _sample_period_ms(t_ms):
    t_ms = np.asarray(t_ms, dtype=float)
    if t_ms.size < 2:
        return 1.0
    d = np.diff(t_ms)
    pos = d[d > 0]
    return float(np.median(pos)) if pos.size else 1.0


def _sample_dt_ms(t_ms, period_ms):
    """Per-sample durations (ms), each capped at the period, so gaps between
    masked samples contribute nothing."""
    t_ms = np.asarray(t_ms, dtype=float)
    if t_ms.size == 0:
        return np.empty(0)
    if t_ms.size == 1:
        return np.array([period_ms])
    dt = np.empty(t_ms.size)
    dt[:-1] = np.diff(t_ms)
    dt[-1] = period_ms
    return np.clip(dt, 0.0, period_ms)


def clip_fixation_spans(event_rows, lo, hi, min_duration_ms=FIXATION_MIN_DURATION_MS):
    out = []
    for name, onset, offset in event_rows or ():
        if str(name).lower() != FIXATION_EVENT:
            continue
        o0, o1 = max(float(onset), lo), min(float(offset), hi)
        if o1 - o0 >= min_duration_ms:
            out.append((o0, o1))
    out.sort()
    return out


def redetect_trace(t_s, x, y, h, pupil, lo, hi, *, experiment):
    """Detect, warp and validate one trial inside ``[lo, hi]`` ms. None when
    unusable, else a dict with ``t_ms``, warped ``xu``/``yu``, raw ``xd``/``yd``
    (degrees, same sample alignment), ``valid_win`` and fixation ``spans``."""
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
    xu = np.asarray(p["x"], dtype=float)
    yu = np.asarray(p["y"], dtype=float)
    valid = np.asarray(p["valid_orig"], dtype=bool)
    m = min(t_ms.size, valid.size, xu.size, yu.size)
    t_ms, valid, xu, yu = t_ms[:m], valid[:m], xu[:m], yu[:m]
    xd, yd = x[:m], y[:m]

    in_window = np.isfinite(t_ms) & (t_ms >= lo) & (t_ms <= hi)
    finite = np.isfinite(xu) & np.isfinite(yu)
    on_screen = (np.abs(xu) <= UNIT_LIM) & (np.abs(yu) <= UNIT_LIM)
    if (in_window & valid & finite & on_screen).sum() < 2:
        return None
    valid_win = valid & in_window & finite
    if valid_win.sum() < 2:
        return None
    return dict(t_ms=t_ms, xu=xu, yu=yu, xd=xd, yd=yd, valid_win=valid_win,
                spans=clip_fixation_spans(event_rows, lo, hi))


def fixation_rows(tr):
    """Per fixation with >= 1 valid sample: ``(xy_u, xy_d, onset, offset, valid_ms)``."""
    t_ms, valid = tr["t_ms"], tr["valid_win"]
    period = _sample_period_ms(t_ms)
    rows = []
    for onset, offset in tr["spans"]:
        hit = valid & (t_ms >= onset) & (t_ms <= offset)
        if not hit.any():
            continue
        rows.append((
            (float(tr["xu"][hit].mean()), float(tr["yu"][hit].mean())),
            (float(tr["xd"][hit].mean()), float(tr["yd"][hit].mean())),
            float(t_ms[hit].min()), float(t_ms[hit].max()),
            float(_sample_dt_ms(t_ms[hit], period).sum()),
        ))
    return rows


# ---- pupil -------------------------------------------------------------------


def _dilate(mask, n):
    if n <= 0 or not mask.any():
        return mask
    k = np.ones(2 * n + 1)
    return np.convolve(mask.astype(float), k, mode="same") > 0


def pupil_summary(pupil, lo, hi):
    """``(mean, change, valid_frac)`` of the pupil trace inside ``[lo, hi]`` ms,
    NaN when under `PUPIL_MIN_VALID` of the window is usable. (The fixation
    cache's own variant; `classifier.features.pupil_summary` is the raw
    branch's.)"""
    nan3 = (np.nan, np.nan, 0.0)
    if pupil is None:
        return nan3
    pu = np.asarray(pupil, dtype=float)
    if pu.ndim != 2 or pu.shape[0] == 0:
        return nan3
    if pu.shape[1] != 2:
        pu = pu.T
    size, t_ms = pu[:, 0], pu[:, 1] * 1000.0
    win = np.isfinite(t_ms) & (t_ms >= lo) & (t_ms <= hi)
    if not win.any():
        return nan3
    size, t_ms = size[win], t_ms[win]
    period = _sample_period_ms(t_ms)
    bad = _dilate(~np.isfinite(size), int(round(PUPIL_EDGE_MS / period)))
    ok = ~bad
    expected = max((hi - lo) / period, 1.0)
    frac = float(ok.sum() / expected)
    if frac < PUPIL_MIN_VALID:
        return np.nan, np.nan, frac
    early = ok & (t_ms <= lo + PUPIL_EARLY_MS)
    late = ok & (t_ms >= hi - PUPIL_LATE_MS)
    change = (
        float(size[late].mean() - size[early].mean()) if early.any() and late.any() else np.nan
    )
    return float(size[ok].mean()), change, frac


def session_pupils(monkey, session, mat_cache):
    """Per-trial pupil list for `session`: the extracted table, else the mat,
    else None (and a warning)."""
    path = PUPIL_DIR / f"{session}_pupil.npz"
    if path.exists():
        with np.load(path, allow_pickle=True) as f:
            return list(f["pupil"]), "table"
    if MAT_ROOT is not None:
        pupil = pupil_for_trial(mat_cache, monkey, session, 1)
        if mat_cache.get(session) is not None:
            return mat_cache[session], "mat"
        del pupil
    warnings.warn(
        f"{session}: no pupil table under {PUPIL_DIR} and no eye mat; blink masking "
        "is OFF and pupil features are NaN for this session (not the cluster's cache)",
        stacklevel=2,
    )
    return None, "none"


# ---- extraction --------------------------------------------------------------


def extract(monkey, sessions):
    import pymovements as pm

    sessions = tuple(sessions)
    eye = load_eye_data(monkey)
    behavioral = load_eye_behavioral_data(monkey)
    beh = behavioral_lookup(behavioral)
    experiment = pm.Experiment(sampling_rate=EYE_SAMPLING_RATE_HZ)
    pupils, source, mat_cache = {}, {}, {}
    for s in sessions:
        pupils[s], source[s] = session_pupils(monkey, s, mat_cache)

    tr_session, tr_trial, tr_maze, tr_h, tr_fix_ms = [], [], [], [], []
    tr_pupil = []
    fx_row, fx_u, fx_d, fx_on, fx_off, fx_ms = [], [], [], [], [], []
    kept = {s: 0 for s in sessions}
    n_trials = len(eye["session"])
    for i in range(n_trials):
        session = str(eye["session"][i])
        if session not in sessions:
            continue
        t_s, x, y, _, trial_id = unpack_eye(eye, i)
        beh_i = beh.get((session, trial_id))
        if not trial_qc_ok(behavioral, beh_i):
            continue
        maze = int(behavioral["geo_type"][beh_i])
        if maze not in MAZES:
            continue
        fix_ms = float(fix_start_ms(behavioral, beh_i))
        if not np.isfinite(fix_ms) or fix_ms <= 0:
            continue
        lo, hi = window_bounds(fix_ms)
        sp = pupils[session]
        pupil = (
            sp[trial_id - 1] if sp is not None and trial_id - 1 < len(sp) else np.empty((0, 2))
        )
        h = tuple(float(behavioral[f"h{k}"][beh_i]) for k in range(1, 7))
        tr = redetect_trace(t_s, x, y, h, pupil, lo, hi, experiment=experiment)
        if tr is None:
            continue
        row = len(tr_session)
        tr_session.append(session)
        tr_trial.append(trial_id)
        tr_maze.append(maze)
        tr_h.append(h)
        tr_fix_ms.append(fix_ms)
        tr_pupil.append(pupil_summary(pupil, lo, hi))
        for xy_u, xy_d, on, off, ms in fixation_rows(tr):
            fx_row.append(row)
            fx_u.append(xy_u)
            fx_d.append(xy_d)
            fx_on.append(on)
            fx_off.append(off)
            fx_ms.append(ms)
        kept[session] += 1

    for s in sessions:
        print(f"  {monkey} {s}: kept={kept[s]} pupil={source[s]}")
    pupil_arr = np.asarray(tr_pupil, dtype=np.float32).reshape(-1, 3)
    # Every key below is part of the on-disk schema (SCHEMA); classifier
    # reads a subset, but the full set keeps old and new caches identical.
    return {
        "session": np.asarray(tr_session, dtype=str),
        "trial_indices_all": np.asarray(tr_trial, dtype=int),
        "maze_id": np.asarray(tr_maze, dtype=int),
        "h": np.asarray(tr_h, dtype=np.float32).reshape(-1, 6),
        "fix_ms": np.asarray(tr_fix_ms, dtype=np.float32),
        "pupil_mean_raw": pupil_arr[:, 0],
        "pupil_change_raw": pupil_arr[:, 1],
        "pupil_valid_frac": pupil_arr[:, 2],
        "fix_row": np.asarray(fx_row, dtype=np.int64),
        "fix_xy_u": np.asarray(fx_u, dtype=np.float32).reshape(-1, 2),
        "fix_xy_d": np.asarray(fx_d, dtype=np.float32).reshape(-1, 2),
        "fix_onset_ms": np.asarray(fx_on, dtype=np.float32),
        "fix_offset_ms": np.asarray(fx_off, dtype=np.float32),
        "fix_valid_ms": np.asarray(fx_ms, dtype=np.float32),
        "sessions": np.asarray(sessions, dtype=str),
        "pupil_source": np.asarray([source[s] for s in sessions], dtype=str),
        "window": np.asarray(WINDOW_TAG),
        "schema": np.int64(SCHEMA),
    }


# ---- cache -------------------------------------------------------------------


def cache_path(monkey, labels="dendro"):
    # Legacy stem: these caches were first written by the (now removed)
    # msp_revised package; the name is kept so existing caches stay valid.
    stem = f"{monkey}_msp_revised_fix"
    tag = LABEL_SCOPES[labels]["tag"]
    return processed_npz(f"{stem}_{tag}" if tag else stem)


def cache_ok(data, sessions):
    try:
        return (
            int(data["schema"]) == SCHEMA
            and str(np.asarray(data["window"]).reshape(-1)[0]) == WINDOW_TAG
            and tuple(np.asarray(data["sessions"]).astype(str)) == tuple(sessions)
        )
    except KeyError:
        return False


def load(monkey, *, labels="dendro", refresh=False):
    """The fixation cache over the `labels` scope's sessions; built (needs
    pymovements and the eye caches) when missing, stale or `refresh`."""
    sessions = tuple(LABEL_SCOPES[labels]["sessions"][monkey])
    path = cache_path(monkey, labels)
    if path.exists() and not refresh:
        data = load_npz(path)
        if cache_ok(data, sessions):
            return data
        print(f"  {path.name}: schema / sessions / window differ; rebuilding")
    print(f"Extracting {monkey} fixation table ({', '.join(sessions)}) ...", flush=True)
    data = extract(monkey, sessions)
    savez_atomic(path, **data)
    print(f"Saved {path} ({len(data['session'])} trials, {len(data['fix_row'])} fixations)")
    return data


def main():
    parser = argparse.ArgumentParser(description="Build the fixation-table cache(s).")
    parser.add_argument("--labels", nargs="*", default=["dendro"], choices=list(LABEL_SCOPES))
    parser.add_argument("--monkey", nargs="*", default=list(MONKEYS), choices=list(MONKEYS))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    for labels in args.labels:
        for monkey in args.monkey:
            data = load(monkey, labels=labels, refresh=args.refresh)
            mazes = np.asarray(data["maze_id"], dtype=int)
            print(f"{monkey} [{labels}]: {len(mazes)} trials "
                  f"{ {m: int((mazes == m).sum()) for m in MAZES} }; "
                  f"{len(data['fix_row'])} fixations; pupil sources "
                  f"{dict(zip(data['sessions'].tolist(), data['pupil_source'].tolist()))}; "
                  f"pupil valid on {np.isfinite(data['pupil_mean_raw']).mean():.0%} of trials")


if __name__ == "__main__":
    main()
