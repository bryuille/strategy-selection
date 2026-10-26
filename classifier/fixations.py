"""Look table per monkey and label scope.

Looks come from `data.looks` (the `timestamp_preprocessing` package, epoch
``fix`` = maze onset -> fixation onset): qualified looks with >= 75 ms of
stable support, each kept with its centroid in unit H *and* in degrees, its
onset / offset and its stable duration (``fix_valid_ms``). No codebook /
region is consulted here: assignment happens in `classifier.features`.
Trials without an estimated saccade threshold are dropped (tallied).

Pupil per trial (raw tracker units; z-scored within session later): mean
over the window, and late minus early (last / first `PUPIL_*_MS`). Samples
within 50 ms of a tracker-loss run are dropped. Pupil traces come from
``classifier/out/pupil/<session>_pupil.npz`` (`classifier.pupil_extract`,
cluster) or, failing that, the eye mats; with neither, the trial's pupil
features are NaN (blink masking of the looks is the package's own, from the
eye mat) -- the extraction then prints a loud warning.

Cache: ``<PROCESSED_ROOT>/<Monkey>_msp_revised_fix[_svm].npz`` (legacy name,
see `cache_path`), rebuilt when the stored schema, sessions, window or looks
source differ.

    uv run python -m classifier.fixations [--labels dendro svm] [--monkey ...] [--refresh]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

from data.builder import (
    behavioral_lookup,
    pupil_for_trial,
    trial_qc_ok,
)
from data.config import MAT_ROOT, processed_npz
from data.loader import load_eye_behavioral_data, savez_atomic
from data.looks import LOOKS_SCHEMA, load_looks, looks_of, usable_rows
from data.looks import source_sha256 as looks_source_sha256
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
SCHEMA = 4  # looks from data.looks (timestamp_preprocessing)
WINDOW_TAG = "onset_to_fix_looks"
EVENTS_SOURCE = "timestamp_preprocessing"
PUPIL_EDGE_MS = 50


def window_bounds(fix_ms):
    """``(lo, hi)`` in ms from geo_present: maze onset to fixation onset."""
    return 0.0, float(fix_ms)


def _sample_period_ms(t_ms):
    """Median positive sample interval (ms) of a time vector (pupil trace)."""
    t_ms = np.asarray(t_ms, dtype=float)
    if t_ms.size < 2:
        return 1.0
    d = np.diff(t_ms)
    pos = d[d > 0]
    return float(np.median(pos)) if pos.size else 1.0


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
    sessions = tuple(sessions)
    looks = load_looks(monkey, "fix", sessions)
    usable = usable_rows(looks)
    behavioral = load_eye_behavioral_data(monkey)
    beh = behavioral_lookup(behavioral)
    pupils, source, mat_cache = {}, {}, {}
    for s in sessions:
        pupils[s], source[s] = session_pupils(monkey, s, mat_cache)

    tr_session, tr_trial, tr_maze, tr_h, tr_fix_ms = [], [], [], [], []
    tr_pupil = []
    fx_row, fx_u, fx_d, fx_on, fx_off, fx_ms, fx_roi, fx_amb = [], [], [], [], [], [], [], []
    kept = {s: 0 for s in sessions}
    n_status = 0
    wanted = set(sessions)
    for r in range(len(looks["session"])):
        session = str(looks["session"][r])
        if session not in wanted:
            continue
        trial_id = int(looks["trial_indices_all"][r])
        beh_i = beh.get((session, trial_id))
        if not trial_qc_ok(behavioral, beh_i):
            continue
        maze = int(behavioral["geo_type"][beh_i])
        if maze not in MAZES:
            continue
        fix_ms = float(looks["fix_ms"][r])
        if not np.isfinite(fix_ms) or fix_ms <= 0:
            continue
        if not usable[r]:
            n_status += 1
            continue
        lo, hi = window_bounds(fix_ms)
        sp = pupils[session]
        pupil = (
            sp[trial_id - 1] if sp is not None and trial_id - 1 < len(sp) else np.empty((0, 2))
        )
        h = tuple(float(behavioral[f"h{k}"][beh_i]) for k in range(1, 7))
        row = len(tr_session)
        tr_session.append(session)
        tr_trial.append(trial_id)
        tr_maze.append(maze)
        tr_h.append(h)
        tr_fix_ms.append(fix_ms)
        tr_pupil.append(pupil_summary(pupil, lo, hi))
        sl = looks_of(looks, r)
        for k in range(sl.start, sl.stop):
            fx_row.append(row)
            fx_u.append(tuple(looks["look_xy_u"][k]))
            fx_d.append(tuple(looks["look_xy_d"][k]))
            fx_on.append(float(looks["look_onset_ms"][k]))
            fx_off.append(float(looks["look_offset_ms"][k]))
            fx_ms.append(float(looks["look_stable_ms"][k]))
            fx_roi.append(str(looks["look_roi_adv"][k]))
            fx_amb.append(bool(looks["look_ambiguous"][k]))
        kept[session] += 1

    for s in sessions:
        print(f"  {monkey} {s}: kept={kept[s]} pupil={source[s]}")
    if n_status:
        print(f"  {monkey}: {n_status} trial(s) dropped for a degenerate saccade threshold")
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
        "fix_roi_adv": np.asarray(fx_roi, dtype=str),
        "fix_ambiguous": np.asarray(fx_amb, dtype=bool),
        "sessions": np.asarray(sessions, dtype=str),
        "pupil_source": np.asarray([source[s] for s in sessions], dtype=str),
        "window": np.asarray(WINDOW_TAG),
        "schema": np.int64(SCHEMA),
        "events_source": np.asarray(EVENTS_SOURCE),
        "looks_schema": np.int64(LOOKS_SCHEMA),
        "looks_source_sha256": np.asarray(looks_source_sha256()),
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
            and str(np.asarray(data["looks_source_sha256"]).reshape(-1)[0]) == looks_source_sha256()
        )
    except KeyError:
        return False


def load(monkey, *, labels="dendro", refresh=False):
    """The look table over the `labels` scope's sessions; built from the
    looks cache (`data.looks`) when missing, stale or `refresh`."""
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
    parser = argparse.ArgumentParser(description="Build the look-table cache(s).")
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
