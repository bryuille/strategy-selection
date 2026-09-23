"""Per-trial occupancy of the five fixed states, one cache per monkey.

The pipeline is the repo's standard one (see `eye_pre_flash/classifier/
features.py`, whose helpers are copied here verbatim) with the k-means step
removed:

1. **Clip** each trial to ``[fix_start - 1466 ms, fix_start]``.
2. **Warp** to unit H (already cached by `data.attractor`).
3. ~~K-means~~ -- the codebook is `codebook.CODEBOOK_XY`, fixed.
4. **Fixations** by I-DT on the clipped, unwarped data (cached events).
5. **Assign** each fixation's mean warped position to the nearest prototype
   within `ASSIGN_RADIUS`; every sample of that fixation inherits the state.
   Saccades, blinks and fixations outside every ball are omitted.

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

Cache: ``$STRATEGY_DATA_ROOT/processed/<Monkey>_msp_fixed5_r<radius>_unith_s1466_e0.npz``.
The radius is in the stem; the codebook itself is verified on load and the
cache rebuilt if `codebook.py` has changed under it.
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
from data.occupancy import behavioral_lookup, fix_start_ms
from eye_pre_flash.msp.codebook import (
    ASSIGN_RADIUS,
    CODEBOOK_XY,
    K,
    MAZE_SCREEN_LIM,
    STATE_NAMES,
)

DEFAULT_START_MS = PRE_FIX_START_MS
DEFAULT_END_MS = PRE_FIX_END_MS
N_MAZES = 6
FIXATION_EVENT = "fixation_idt"
BLOCK_NAMES = ("occ_ms", "occ_bin")


# ---- helpers copied from eye_pre_flash/classifier/features.py --------------


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


# ---- helpers copied from data/attractor.py ----------------------------------


def assign_states_inference(xy, codebook_xy, radius=ASSIGN_RADIUS):
    """Nearest prototype within `radius`, else -1. The radius rejects points
    outside every ball; among the balls containing the point, nearest wins
    (first index on an exact tie)."""
    xy = np.asarray(xy, dtype=np.float32)
    codebook_xy = np.asarray(codebook_xy, dtype=np.float32)
    if xy.size == 0:
        return np.empty(0, dtype=np.int16)

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
    valid sample. Same centroid as `data.attractor.assign_fixation_states`
    computes, returned instead of consumed so the per-fixation state can be
    kept for the diagnostics."""
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


def assign_trial(xy, t_ms, valid, event_rows, codebook_xy, radius):
    """``(state per sample, centroids, state per fixation)``.

    Identical sample-level result to `data.attractor.assign_fixation_states`:
    every valid sample of a fixation inherits its centroid's state; everything
    else stays -1.
    """
    state = np.full(len(xy), -1, dtype=np.int16)
    centroids, hits = fixation_centroids(xy, t_ms, valid, event_rows)
    if not hits:
        return state, centroids, np.empty(0, dtype=np.int16)
    fix_state = assign_states_inference(centroids, codebook_xy, radius=radius)
    for hit, sid in zip(hits, fix_state):
        if sid >= 0:
            state[hit] = sid
    return state, centroids, fix_state


# ---- extraction --------------------------------------------------------------


def extract_monkey_features(
    monkey, *, radius=ASSIGN_RADIUS, start_ms=DEFAULT_START_MS, end_ms=DEFAULT_END_MS
):
    """Feature blocks for every usable trial of `monkey` against the fixed codebook."""
    attractor = load_attractor_eye_data(monkey)
    behavioral = load_eye_behavioral_data(monkey)
    beh = behavioral_lookup(behavioral)
    sessions = np.asarray(attractor["session"]).astype(str)
    trials = np.asarray(attractor["trial_indices_all"]).astype(int)
    events = _event_lookup(monkey, set(sessions.tolist()), start_ms, end_ms)
    codebook = np.asarray(CODEBOOK_XY, dtype=np.float32)
    radius = float(radius)

    # ---- Steps 1-2: the usable-trial test, unchanged from the fitted pipeline.
    # It includes the on-screen box because it decided the k-means pool there,
    # and it decides the row population here; keeping it keeps n_H/n_S equal.
    usable = []
    for i in range(sessions.size):
        session, trial_id = sessions[i], int(trials[i])
        beh_i = beh.get((session, trial_id))
        if not trial_qc_ok(behavioral, beh_i):
            continue
        maze = int(behavioral["geo_type"][beh_i])
        if not 1 <= maze <= N_MAZES:
            continue
        bounds = _trial_window(behavioral, beh_i, start_ms, end_ms)
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

    print(f"  {monkey}: {len(usable)} usable trials; fixed K={K} codebook, radius={radius:g}")

    blocks = {name: [] for name in BLOCK_NAMES}
    rows_session, rows_trial, rows_maze = [], [], []
    rows_n_fix, rows_n_fix_assigned = [], []
    fix_xy, fix_state, fix_row = [], [], []

    # ---- Steps 4-5: clip fixation spans, assign their centroids -------------
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
            radius,
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
    out["assign_radius"] = np.float64(radius)
    out["window_start_ms"] = np.int64(start_ms)
    out["window_end_ms"] = np.int64(end_ms)
    return out


def cache_stem(monkey, radius=ASSIGN_RADIUS, start_ms=DEFAULT_START_MS, end_ms=DEFAULT_END_MS):
    return f"{monkey}_msp_fixed{K}_r{float(radius):g}_unith_s{int(start_ms)}_e{int(end_ms)}"


def _cache_matches(data, radius):
    """False if the cache was built under a different codebook or radius, or
    predates the per-fixation arrays."""
    try:
        return (
            "fix_row" in data
            and int(data["codebook_k"]) == K
            and np.array_equal(
                np.asarray(data["codebook_xy"], dtype=np.float32),
                np.asarray(CODEBOOK_XY, dtype=np.float32),
            )
            and float(data["assign_radius"]) == float(radius)
            and tuple(np.asarray(data["state_names"]).astype(str)) == tuple(STATE_NAMES)
        )
    except KeyError:
        return False


def load_features(
    monkey,
    *,
    radius=ASSIGN_RADIUS,
    start_ms=DEFAULT_START_MS,
    end_ms=DEFAULT_END_MS,
    refresh=False,
):
    """Cached fixed-codebook feature blocks for `monkey`, rebuilt on mismatch."""
    path = processed_npz(cache_stem(monkey, radius, start_ms, end_ms))
    if path.exists() and not refresh:
        data = load_npz(path)
        if _cache_matches(data, radius):
            return data
        print(f"  {path.name}: built under a different codebook, radius or schema; rebuilding")
    print(f"Extracting {monkey} fixed-codebook features (radius={float(radius):g}) ...")
    data = extract_monkey_features(monkey, radius=radius, start_ms=start_ms, end_ms=end_ms)
    path.parent.mkdir(parents=True, exist_ok=True)
    savez_atomic(path, **data)
    print(f"Saved {path} ({len(data['session'])} trials)")
    return data


def main():
    parser = argparse.ArgumentParser(description="Build the fixed-codebook feature cache(s).")
    parser.add_argument("--monkey", nargs="*", default=list(MONKEYS), choices=MONKEYS)
    parser.add_argument("--radius", type=float, default=ASSIGN_RADIUS)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    for monkey in args.monkey:
        data = load_features(monkey, radius=args.radius, refresh=args.refresh)
        mazes = np.asarray(data["maze_id"], dtype=int)
        counts = {m: int((mazes == m).sum()) for m in range(1, N_MAZES + 1)}
        n_fix = int(data["n_fix"].sum())
        n_ok = int(data["n_fix_assigned"].sum())
        print(f"{monkey}: {len(mazes)} trials {counts}")
        print(f"  fixations assigned: {n_ok}/{n_fix} ({n_ok / max(n_fix, 1):.1%})")


if __name__ == "__main__":
    main()
