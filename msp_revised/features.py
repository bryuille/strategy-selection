"""Fixation table per monkey, and the per-trial feature matrices built from it.

Extraction (copied from `msp.features`, publication sessions only):

1. **Clip** each trial to ``[0, fix_start]`` ms from maze onset.
2. **Detect** I-DT fixations and saccades on the clipped, unwarped gaze
   (`data.builder.trial_events`), blink events masked from the pupil trace.
3. **Warp** to unit H (`data.builder.prepare_trial`).
4. **Centroids**: every fixation with >= 1 valid in-window sample is kept
   with its centroid in unit H *and* in degrees, its onset / offset and its
   valid dwell (ms). No codebook is consulted here: assignment happens in
   `trial_features`, so a new codebook costs nothing.

Pupil per trial (raw tracker units; z-scored within session later): mean
over the window, and late minus early (last / first `PUPIL_*_MS`). Samples
within 50 ms of a tracker-loss run are dropped. Pupil traces come from
``out/pupil/<session>_pupil.npz`` (`pupil_extract`, cluster) or, failing
that, the eye mats; with neither, the trial's pupil features are NaN and
blink masking is off -- the extraction then prints a loud warning, because
a cache built that way is not the cluster's.

Cache: ``<PROCESSED_ROOT>/<Monkey>_msp_revised_fix.npz``, rebuilt when the
stored schema, sessions or window differ.

Feature kinds (`KINDS`; one row per trial, states in codebook order):

``binary``  visited / not per state
``dwell``   share of the trial's assigned dwell per state (all-zero if none)
``count``   share of the trial's assigned fixations per state
``first``   one-hot state of the first assigned fixation after maze onset
``last``    one-hot state of the last assigned fixation before fix_start
``both``    binary then dwell, concatenated

``pupil=True`` appends `PUPIL_FEATURES` (within-session z, NaN -> 0).
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
from msp_revised.config import (
    LABEL_STEM,
    N_MAZES,
    PUPIL_DIR,
    PUPIL_EARLY_MS,
    PUPIL_FEATURES,
    PUPIL_LATE_MS,
    PUPIL_MIN_VALID,
    SESSIONS,
    UNIT_LIM,
    VARIANTS,
    window_bounds,
)

SCHEMA = 2
WINDOW_TAG = "onset_to_fix"
MIN_WINDOW_SAMPLES = 6
FIXATION_EVENT = "fixation_idt"
PUPIL_EDGE_MS = 50
KINDS = ("binary", "dwell", "count", "first", "last", "both")
BASE_KINDS = ("binary", "dwell", "count", "first", "last")


# ---- helpers (copied from msp.features) --------------------------------------


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
    NaN when under `PUPIL_MIN_VALID` of the window is usable."""
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


def extract(monkey, sessions=None):
    import pymovements as pm

    sessions = tuple(sessions) if sessions is not None else SESSIONS[monkey]
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
        if not 1 <= maze <= N_MAZES:
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


def cache_path(monkey, cache_name=None):
    return processed_npz(cache_name or f"{monkey}_msp_revised_fix")


def _cache_ok(data, sessions):
    try:
        return (
            int(data["schema"]) == SCHEMA
            and str(np.asarray(data["window"]).reshape(-1)[0]) == WINDOW_TAG
            and tuple(np.asarray(data["sessions"]).astype(str)) == tuple(sessions)
        )
    except KeyError:
        return False


def load(monkey, *, sessions=None, cache_name=None, refresh=False):
    """The fixation cache. msp_revised's own scope by default; another
    session list (e.g. `classifier`'s svm scope) goes to its own `cache_name`
    so the publication cache is never overwritten."""
    sessions = tuple(sessions) if sessions is not None else SESSIONS[monkey]
    if sessions != SESSIONS[monkey] and cache_name is None:
        raise ValueError("a non-default session list needs its own cache_name")
    path = cache_path(monkey, cache_name)
    if path.exists() and not refresh:
        data = load_npz(path)
        if _cache_ok(data, sessions):
            return data
        print(f"  {path.name}: schema / sessions / window differ; rebuilding")
    print(f"Extracting {monkey} fixation table ({', '.join(sessions)}) ...", flush=True)
    data = extract(monkey, sessions)
    savez_atomic(path, **data)
    print(f"Saved {path} ({len(data['session'])} trials, {len(data['fix_row'])} fixations)")
    return data


# ---- labels and pooling ------------------------------------------------------


def label_lookup(sessions):
    """``(session, trial_id) -> 0.0 / 1.0`` from ``<session>_strategy_choices.npz``
    (index i is trial i + 1). Read directly: never built here."""
    missing = [s for s in sessions if not processed_npz(f"{s}_{LABEL_STEM}").exists()]
    if missing:
        raise FileNotFoundError(f"missing {LABEL_STEM} label cache(s) for: {', '.join(missing)}")
    lookup = {}
    for session in sessions:
        with np.load(processed_npz(f"{session}_{LABEL_STEM}"), allow_pickle=True) as f:
            choices = np.asarray(f["strategy_choices"], dtype=float)
        for idx in np.flatnonzero(np.isfinite(choices)):
            lookup[(session, int(idx) + 1)] = float(choices[idx])
    return lookup


def labels_for(data, lookup):
    sess = np.asarray(data["session"]).astype(str)
    trials = np.asarray(data["trial_indices_all"], dtype=int)
    return np.asarray([lookup.get((s, int(t)), np.nan) for s, t in zip(sess, trials)])


def pool_maze(data, y_full, maze):
    """``(mask, y, session_ids)`` for every labelled trial of one maze."""
    mazes = np.asarray(data["maze_id"], dtype=int)
    mask = np.isfinite(y_full) & (mazes == maze)
    return mask, y_full[mask].astype(int), np.asarray(data["session"]).astype(str)[mask]


# ---- feature matrices --------------------------------------------------------


def fixation_states(data, codebook):
    """`codebook` state per fixation (-1 = unassigned)."""
    rows = np.asarray(data["fix_row"], dtype=int)
    fix = dict(
        xy_u=np.asarray(data["fix_xy_u"], dtype=float),
        xy_d=np.asarray(data["fix_xy_d"], dtype=float),
        h=np.asarray(data["h"], dtype=float)[rows],
    )
    return codebook.assign(fix)


def _share(M):
    tot = M.sum(axis=1, keepdims=True)
    return np.divide(M, tot, out=np.zeros_like(M), where=tot > 0)


def gaze_block(data, codebook, kind, states=None):
    """``(n_trials, k)`` block of `kind` (``(n, 2k)`` for ``both``)."""
    if kind == "both":
        return np.hstack([gaze_block(data, codebook, k_, states) for k_ in ("binary", "dwell")])
    if kind not in BASE_KINDS:
        raise ValueError(f"unknown kind {kind!r}; choose from {KINDS}")
    if states is None:
        states = fixation_states(data, codebook)
    n, k = len(data["session"]), codebook.k
    rows = np.asarray(data["fix_row"], dtype=int)
    ok = states >= 0
    r, s = rows[ok], states[ok].astype(int)
    if kind == "binary":
        occ = np.zeros((n, k))
        np.add.at(occ, (r, s), np.asarray(data["fix_valid_ms"], dtype=float)[ok])
        return (occ > 0).astype(float)
    if kind == "dwell":
        occ = np.zeros((n, k))
        np.add.at(occ, (r, s), np.asarray(data["fix_valid_ms"], dtype=float)[ok])
        return _share(occ)
    if kind == "count":
        cnt = np.zeros((n, k))
        np.add.at(cnt, (r, s), 1.0)
        return _share(cnt)
    onset = np.asarray(data["fix_onset_ms"], dtype=float)[ok]
    out = np.zeros((n, k))
    # fixations are stored in onset order per trial; pick first / last per row
    order = np.lexsort((onset if kind == "first" else -onset, r))
    r_o, s_o = r[order], s[order]
    first_idx = np.unique(r_o, return_index=True)[1]
    out[r_o[first_idx], s_o[first_idx]] = 1.0
    return out


def pupil_block(data):
    """``(n, 2)`` within-session z of `PUPIL_FEATURES`; NaN -> 0 (session mean)."""
    sess = np.asarray(data["session"]).astype(str)
    cols = []
    for key in ("pupil_mean_raw", "pupil_change_raw"):
        raw = np.asarray(data[key], dtype=float)
        z = np.zeros_like(raw)
        for s in np.unique(sess):
            m = sess == s
            v = raw[m]
            fin = np.isfinite(v)
            if fin.sum() >= 2 and np.nanstd(v) > 0:
                z[m] = np.where(fin, (v - np.nanmean(v)) / np.nanstd(v), 0.0)
        cols.append(z)
    return np.stack(cols, 1)


def feature_names(codebook, kind, pupil):
    if kind == "both":
        names = [f"{n} bin" for n in codebook.names] + [f"{n} dwell" for n in codebook.names]
    else:
        names = list(codebook.names)
    return tuple(names + (list(PUPIL_FEATURES) if pupil else []))


def trial_features(data, codebook, kind, *, pupil=False, states=None):
    X = gaze_block(data, codebook, kind, states)
    if pupil:
        X = np.hstack([X, pupil_block(data)])
    return X


# ---- variants ----------------------------------------------------------------


def apply_variant(X, variant):
    """``full`` unmodified; ``mean_removed`` minus the column mean of the rows
    given (the maze's label-free grand mean when called per maze)."""
    X = np.asarray(X, dtype=float)
    if variant == "full":
        return X
    if variant == "mean_removed":
        return X - X.mean(axis=0, keepdims=True)
    raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")


def main():
    parser = argparse.ArgumentParser(description="Build the fixation-table cache(s).")
    parser.add_argument("--monkey", nargs="*", default=list(SESSIONS), choices=list(SESSIONS))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    for monkey in args.monkey:
        data = load(monkey, refresh=args.refresh)
        mazes = np.asarray(data["maze_id"], dtype=int)
        print(f"{monkey}: {len(mazes)} trials "
              f"{ {m: int((mazes == m).sum()) for m in range(1, N_MAZES + 1)} }; "
              f"{len(data['fix_row'])} fixations; pupil sources "
              f"{dict(zip(data['sessions'].tolist(), data['pupil_source'].tolist()))}; "
              f"pupil valid on {np.isfinite(data['pupil_mean_raw']).mean():.0%} of trials")


if __name__ == "__main__":
    main()
