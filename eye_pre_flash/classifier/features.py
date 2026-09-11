"""Per-trial eye-data feature blocks, and the one codebook they all use.

Every block describes the pre-fixation free-viewing window,
``fix_start - START_MS`` through ``fix_start``, in unit-H maze coordinates:
``data.attractor.to_maze`` warps degrees by the trial's arm lengths so every
maze becomes the same unit H with exits at (+-1, +-1).

Blocks (one row per trial, ``float32``):

``occ_ms``   milliseconds of gaze in each cluster (d = K)  -- time occupancy
``occ_bin``  visited / not visited per cluster, {0, 1} (d = K) -- binary occupancy
``bigram``   proportions of ordered cluster-run pairs (d = K*K) -- state bigrams

The pipeline, in order:

1. **Clip** each trial to ``[fix_start - start_ms, fix_start - end_ms]``.
2. **Warp** to unit H (already cached by ``data.attractor``).
3. **K-means** over the pooled in-window samples, once per (monkey, K), across
   sessions.
4. **Fixations** by I-DT only, on the clipped but *unwarped* data.
5. **Assign** each fixation's mean warped position to the nearest prototype whose
   unit ball contains it; saccades, blinks and fixations outside every ball are
   omitted.

Step 3 depends on step 4 despite coming first in the statement: the pool is the
in-window *fixation* samples, so prototypes land where gaze dwells rather than
along saccade trajectories. Detection (step 4) stays in degree space because
I-DT's dispersion threshold is physical, while unit H is dimensionless and
warped per trial -- detecting on warped data would make the effective threshold
vary by maze, reintroducing the geometry dependence the warp exists to remove.

This module fits **the** codebook. ``data.attractor`` holds no codebook of its
own; it is read only for warped positions, validity mask and timebase, which is
why its cache is K-independent.

Caches land under ``data/processed/<Monkey>_clf2_k<K>_unith_s<start>_e<end>.npz``.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from data.attractor import (
    ASSIGN_RADIUS,
    MAZE_SCREEN_LIM,
    assign_fixation_states,
)
from data.builder import FIXATION_MIN_DURATION_MS, trial_qc_ok
from data.config import (
    MONKEYS,
    PRE_FIX_END_MS,
    PRE_FIX_START_MS,
    processed_npz,
)
from data.convert import load_npz
from data.loader import (
    load_attractor_eye_data,
    savez_atomic,
    load_clean_eye_data,
    load_eye_behavioral_data,
)
from data.occupancy import (
    behavioral_lookup,
    collapse_state_runs,
    fix_start_ms,
    ngram_proportions,
)

KS = (6, 12)
DEFAULT_K = 12

DEFAULT_START_MS = PRE_FIX_START_MS
DEFAULT_END_MS = PRE_FIX_END_MS

# Samples pooled per monkey for the codebook fit, capped so one long trial
# cannot dominate the k-means.
POOL_PER_TRIAL = 50
POOL_MAX = 80_000
SEED = 0

N_MAZES = 6


def block_dims(k):
    return {
        "occ_ms": k,
        "occ_bin": k,
        "bigram": k * k,
    }


BLOCK_NAMES = tuple(block_dims(1).keys())


def _event_lookup(monkey, sessions):
    """``(session, trial) -> [(name, onset, offset), ...]`` from the clean events."""
    # Windowed detection: I-DT runs on the clipped, unwarped segment.
    events = load_clean_eye_data(monkey, start_ms=DEFAULT_START_MS, end_ms=DEFAULT_END_MS)
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
    """Per-sample durations (s), each capped at the sample period.

    `t_ms` here is the *masked* time vector -- unassigned samples (saccades,
    blinks, gaze outside every codebook ball) are already removed -- so a raw
    `np.diff` charges the whole of any gap to the last sample before it. One
    sample sitting in front of a 500 ms blink would contribute 500 ms of
    "dwell", and `occ_ms` would sum to the full window regardless of how much
    of that window was actually spent in an assigned fixation.

    Capping each interval at the sample period fixes that: the total becomes
    the time genuinely spent in assigned fixations, and gaps contribute
    nothing. `period_ms` should come from the trial's *unmasked* timebase
    (`_sample_period_ms`), since the median interval of a heavily masked vector
    is itself biased by the gaps.

    Summing span durations instead would also miscount, since each sample
    carries exactly one state and contributes exactly one period.
    """
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
    """``(lo, hi)`` window bounds in trial ms, or None if the trial has no
    usable fixation onset."""
    fix_ms = fix_start_ms(behavioral, beh_i)
    if not np.isfinite(fix_ms) or fix_ms <= 0:
        return None
    return fix_ms - start_ms, fix_ms - end_ms


FIXATION_EVENT = "fixation_idt"


def clip_fixation_spans(
    event_rows, lo, hi, name_wanted=FIXATION_EVENT, min_duration_ms=FIXATION_MIN_DURATION_MS
):
    """`name_wanted` event rows intersected with ``[lo, hi]``, then re-tested
    against `min_duration_ms`.

    Step 1 of the pipeline is the window clip, so a fixation only contributes
    the part of itself falling inside the window, and only if that part would
    have been detected on its own.

    This is equivalent to re-running I-DT on the clipped segment, without
    paying for a second pymovements pass, because of two properties of the
    algorithm. Dispersion is monotone under subsetting, so any sub-window of a
    span that already passed the 2 deg test still passes it -- a truncated
    fixation is still a fixation. And I-DT grows a window until dispersion
    exceeds the threshold, which on the clipped segment happens at exactly the
    boundary the whole-trial pass already found, so span *edges* are preserved.
    The only thing a genuine re-detect would do differently is refuse to emit a
    remnant shorter than the minimum duration, since its initial window can
    never be filled -- which is precisely the test applied here.

    Measured cost of that test: it drops 6.4% of in-window fixations for Faure
    and 5.6% for Nielsen, but only 1.40% and 0.84% of in-window *fixation
    time*, and no trial loses all of its fixations.

    Spans arrive unmerged and are kept that way -- each one is assigned on its
    own, and consecutive fragments landing on the same cluster collapse into a
    single run downstream rather than being stitched together here.
    """
    out = []
    for name, onset, offset in event_rows or ():
        if str(name).lower() != name_wanted:
            continue
        o0, o1 = max(float(onset), lo), min(float(offset), hi)
        if o1 - o0 >= min_duration_ms:
            out.append((name, o0, o1))
    return out


def _fit_window_codebook(pool_xy, k, *, seed=SEED):
    """K-means over the pooled in-window fixation samples, clipped to the box."""
    rng = np.random.default_rng(seed)
    xy = np.concatenate(pool_xy, axis=0).astype(np.float32)
    if xy.shape[0] > POOL_MAX:
        xy = xy[rng.choice(xy.shape[0], POOL_MAX, replace=False)]
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    km.fit(xy)
    centers = np.clip(km.cluster_centers_, -MAZE_SCREEN_LIM, MAZE_SCREEN_LIM)
    return centers.astype(np.float32), xy.shape[0]


def extract_monkey_features(
    monkey,
    *,
    k,
    start_ms=DEFAULT_START_MS,
    end_ms=DEFAULT_END_MS,
):
    """All feature blocks for every usable trial of `monkey`.

    Steps 1-2 (clip, warp) are already baked into the attractor cache, which
    holds warped positions and a validity mask restricted to fixation samples.
    Step 3 fits the codebook on what survives the window; steps 4-5 clip the
    I-DT spans to the same window and assign each one's mean to a cluster.
    """
    attractor = load_attractor_eye_data(monkey)
    behavioral = load_eye_behavioral_data(monkey)
    k = int(k)
    beh = behavioral_lookup(behavioral)
    sessions = np.asarray(attractor["session"]).astype(str)
    trials = np.asarray(attractor["trial_indices_all"]).astype(int)
    events = _event_lookup(monkey, set(sessions.tolist()))

    # ---- Steps 1-2: keep in-window samples and pool what survives ----------
    rng = np.random.default_rng(SEED)
    usable, pool = [], []
    for i in range(sessions.size):
        session, trial_id = sessions[i], int(trials[i])
        beh_i = beh.get((session, trial_id))
        if not trial_qc_ok(behavioral, beh_i):
            # `path_type != -99` *and* no fade *and* photodiode QC passed --
            # the same pool `data.labeler` labels from.
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

        idx = np.flatnonzero(keep_u)
        if idx.size > POOL_PER_TRIAL:
            idx = rng.choice(idx, POOL_PER_TRIAL, replace=False)
        pool.append(np.stack([x[idx], y[idx]], axis=1))
        usable.append((i, session, trial_id, maze, lo, hi))

    dims = block_dims(k)
    if not pool:
        out = {
            name: np.empty((0, dims[name]), dtype=np.float32) for name in BLOCK_NAMES
        }
        out["session"] = np.asarray([])
        out["trial_indices_all"] = np.asarray([], dtype=int)
        out["maze_id"] = np.asarray([], dtype=int)
        out["codebook_k"] = np.int64(k)
        return out

    # ---- Step 3: k-means across sessions, on in-window samples only --------
    codebook, n_pool = _fit_window_codebook(pool, k)
    radius = ASSIGN_RADIUS
    print(
        f"  windowed unit-H codebook: K={k}, n={n_pool} in-window samples, "
        f"radius={radius:.2f} unit-H"
    )

    blocks = {name: [] for name in BLOCK_NAMES}
    rows_session, rows_trial, rows_maze = [], [], []

    # ---- Steps 4-5: clip fixation spans, assign their means ----------------
    for i, session, trial_id, maze, lo, hi in usable:
        t_ms, valid, x, y = _trial_arrays(attractor, i)

        in_window = np.isfinite(t_ms) & (t_ms >= lo) & (t_ms <= hi)
        valid_win = valid & in_window & np.isfinite(x) & np.isfinite(y)
        spans = clip_fixation_spans(events.get((session, trial_id), ()), lo, hi)
        state = assign_fixation_states(
            np.stack([x, y], axis=1).astype(np.float32),
            t_ms,
            valid_win,
            spans,
            codebook,
            radius=radius,
        )
        state = np.where(valid_win, state, -1).astype(int)

        if valid_win.sum() < 2:
            continue
        assigned = valid_win & (state >= 0) & (state < k)

        if assigned.any():
            dt = _sample_dt_seconds(t_ms[assigned], _sample_period_ms(t_ms))
            occ_ms = np.bincount(state[assigned], weights=dt, minlength=k) * 1000.0
            runs = collapse_state_runs(state[assigned])
        else:
            occ_ms, runs = np.zeros(k), []

        blocks["occ_ms"].append(occ_ms)
        blocks["occ_bin"].append((occ_ms > 0).astype(float))
        blocks["bigram"].append(ngram_proportions(runs, k, 2).ravel())
        rows_session.append(session)
        rows_trial.append(trial_id)
        rows_maze.append(maze)

    out = {
        name: (
            np.asarray(vals, dtype=np.float32)
            if vals
            else np.empty((0, dims[name]), dtype=np.float32)
        )
        for name, vals in blocks.items()
    }
    out["session"] = np.asarray(rows_session)
    out["trial_indices_all"] = np.asarray(rows_trial, dtype=int)
    out["maze_id"] = np.asarray(rows_maze, dtype=int)
    out["codebook_k"] = np.int64(k)
    out["codebook_xy"] = codebook
    out["assign_radius"] = np.float64(radius)
    return out


def cache_stem(monkey, k, start_ms=DEFAULT_START_MS, end_ms=DEFAULT_END_MS):
    return f"{monkey}_clf2_k{int(k)}_unith_s{int(start_ms)}_e{int(end_ms)}"


def load_features(
    monkey,
    *,
    k,
    start_ms=DEFAULT_START_MS,
    end_ms=DEFAULT_END_MS,
    refresh=False,
):
    """Cached feature blocks for every trial of `monkey` at `k`."""
    path = processed_npz(cache_stem(monkey, k, start_ms, end_ms))
    if path.exists() and not refresh:
        return load_npz(path)
    print(f"Extracting {monkey} features (k={k}) ...")
    data = extract_monkey_features(monkey, k=k, start_ms=start_ms, end_ms=end_ms)
    path.parent.mkdir(parents=True, exist_ok=True)
    savez_atomic(path, **data)
    print(f"Saved {path} ({len(data['session'])} trials)")
    return data


def feature_rows(monkey, *, k, block, start_ms=DEFAULT_START_MS, end_ms=DEFAULT_END_MS):
    """One block as `(rows, sessions, mazes)`, for the figures that plot it."""
    data = load_features(monkey, k=k, start_ms=start_ms, end_ms=end_ms)
    return (
        np.asarray(data[block], dtype=float),
        np.asarray(data["session"]).astype(str),
        np.asarray(data["maze_id"], dtype=int),
    )


def main():
    """Build (or rebuild) every feature cache: both monkeys, every K.

    `load_features` pulls in `data.attractor`, so this also builds the warped
    attractor cache if it does not exist yet, and fits the per-monkey unit-H
    k-means codebook at each K.
    """
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--monkey", choices=MONKEYS, default=None)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    for monkey in [args.monkey] if args.monkey else list(MONKEYS):
        for k in KS:
            data = load_features(monkey, k=k, refresh=args.refresh)
            print(f"{monkey} k={k}: {len(data['session'])} trials")


if __name__ == "__main__":
    main()
