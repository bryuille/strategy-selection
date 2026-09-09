"""Per-trial eye-data feature blocks for the strategy-decoding tables.

Every block describes the pre-fixation free-viewing window,
``fix_start - START_MS`` through ``fix_start``, and is extracted in one of two
coordinate **spaces**:

``unith``   unit-H maze coordinates: ``data.attractor.to_maze`` warps degrees by
            the trial's arm lengths so every maze becomes the same unit H with
            exits at (+-1, +-1). Used by the **across-maze** tables, where the
            warp removes the visual-geometry differences between mazes.
``deg``     raw screen degrees, recovered by inverting the warp with the same
            arm lengths. Used by the **per-maze** tables, where geometry is
            constant inside each maze and the warp would only distort gaze.

Blocks (one row per trial, ``float32``):

``occ_ms``      milliseconds of gaze in each codebook state (d = K)
``occ_bin``     visited / not visited per state, {0, 1} (d = K)
``bigram``      proportions of ordered state-run pairs (d = K*K)
``heatmap_lo``  gaze histogram on a coarse GRID_LO x GRID_LO grid
``heatmap_hi``  the same on a fine GRID_HI x GRID_HI grid

The ``unith`` codebook and state ids come straight from ``data.attractor``
(k-means in unit-H space, fixation-centroid assignment). The ``deg`` codebook is
fit here the same way -- per-monkey k-means on pooled valid samples, fixation-
centroid assignment within a radius -- but in degree space, with the radius
scaled by the pooled degrees-per-unit-H ratio so the two spaces assign a
comparable share of samples.

Caches land under ``data/processed/<Monkey>_clf2_k<K>_<space>_s<start>_e<end>.npz``.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from data.attractor import (
    ASSIGN_RADIUS,
    MAZE_SCREEN_LIM,
    assign_fixation_states,
    h_lookup,
)
from data.config import MONKEYS, processed_npz
from data.convert import load_npz
from data.loader import (
    load_attractor_eye_data,
    load_clean_eye_data,
    load_eye_behavioral_data,
)
from data.occupancy import (
    behavioral_lookup,
    collapse_state_runs,
    fix_start_ms,
    ngram_proportions,
)
from eye_pre_flash.plotting.plot_io import PRE_FIX_END_MS, PRE_FIX_START_MS

KS = (6, 12)
SPACES = ("unith", "deg")

DEFAULT_START_MS = PRE_FIX_START_MS
DEFAULT_END_MS = PRE_FIX_END_MS

GRID_LO = 5
GRID_HI = 10
# Histogram box half-widths. Unit-H mirrors the attractor's maze box; degrees
# covers the maze extent (arms reach 10 deg) with margin.
UNITH_LIM = MAZE_SCREEN_LIM
DEG_LIM = 15.0

# Degree-space codebook fit: samples pooled per monkey, capped like
# `data.attractor.codebook_xy_pool`.
POOL_PER_TRIAL = 50
POOL_MAX = 80_000
SEED = 0

N_MAZES = 6


def block_dims(k):
    return {
        "occ_ms": k,
        "occ_bin": k,
        "bigram": k * k,
        "heatmap_lo": GRID_LO * GRID_LO,
        "heatmap_hi": GRID_HI * GRID_HI,
    }


BLOCK_NAMES = tuple(block_dims(1).keys())


def from_maze(x_n, y_n, h):
    """Invert ``data.attractor.to_maze``: unit-H coordinates back to degrees.

    The warp is exact and per-trial: x is scaled by the left or right arm
    length, y by an arm length interpolated along x. Both steps invert in
    closed form, so no information is lost going unit-H -> degrees.
    """
    from data.attractor import STEM_DEG, arm_lengths

    h1, h2, h3, h4, h5, h6 = arm_lengths(h)
    x_n = np.asarray(x_n, dtype=np.float64)
    y_n = np.asarray(y_n, dtype=np.float64)
    x = np.where(x_n <= 0, x_n * h1, x_n * h4)
    s_pos = np.interp(x_n, [-1.0, 0.0, 1.0], [h2, STEM_DEG, h5])
    s_neg = np.interp(x_n, [-1.0, 0.0, 1.0], [h3, 0.5 * (h3 + h6), h6])
    y = np.where(y_n >= 0, y_n * s_pos, y_n * s_neg)
    return x, y


def _event_lookup(monkey, sessions):
    """``(session, trial) -> [(name, onset, offset), ...]`` from the clean events."""
    events = load_clean_eye_data(monkey)
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


def _sample_dt_seconds(t_ms):
    """Per-sample durations (s) for a kept-sample time vector."""
    dt = np.empty(t_ms.size, dtype=float)
    if t_ms.size == 1:
        return np.array([0.001])
    dt[:-1] = np.diff(t_ms)
    pos = dt[:-1][dt[:-1] > 0]
    dt[-1] = float(np.median(pos)) if pos.size else 1.0
    return np.where(dt > 0, dt, 0.0) / 1000.0


def _trial_arrays(attractor, i):
    t_ms = np.asarray(attractor["time"][i], dtype=float) * 1000.0
    state = np.asarray(attractor["state_id"][i], dtype=int)
    valid = np.asarray(attractor["valid"][i], dtype=bool)
    x = np.asarray(attractor["eye_x"][i], dtype=float)
    y = np.asarray(attractor["eye_y"][i], dtype=float)
    n = min(t_ms.size, state.size, valid.size, x.size, y.size)
    return t_ms[:n], state[:n], valid[:n], x[:n], y[:n]


def _fit_degree_codebook(attractor, hs, k, *, seed=SEED):
    """Per-monkey k-means codebook in degree space, plus its assignment radius.

    Mirrors `data.attractor`: pool valid samples across all trials (capped),
    k-means, clip to the box. The radius is `ASSIGN_RADIUS` (1.0 unit-H)
    scaled by the pooled median degrees-per-unit-H ratio, so the degree-space
    book leaves a comparable share of gaze unassigned.
    """
    rng = np.random.default_rng(seed)
    pool, ratios = [], []
    sessions = np.asarray(attractor["session"]).astype(str)
    trials = np.asarray(attractor["trial_indices_all"]).astype(int)
    for i in range(sessions.size):
        h = hs.get((sessions[i], int(trials[i])))
        if h is None or not np.all(np.isfinite(h)):
            continue
        _t, _s, valid, x_u, y_u = _trial_arrays(attractor, i)
        keep = valid & np.isfinite(x_u) & np.isfinite(y_u)
        keep &= (np.abs(x_u) <= UNITH_LIM) & (np.abs(y_u) <= UNITH_LIM)
        idx = np.flatnonzero(keep)
        if idx.size == 0:
            continue
        if idx.size > POOL_PER_TRIAL:
            idx = rng.choice(idx, POOL_PER_TRIAL, replace=False)
        x_d, y_d = from_maze(x_u[idx], y_u[idx], h)
        pool.append(np.stack([x_d, y_d], axis=1))
        r_u = np.hypot(x_u[idx], y_u[idx])
        r_d = np.hypot(x_d, y_d)
        far = r_u > 0.25
        if far.any():
            ratios.append(r_d[far] / r_u[far])
    xy = np.concatenate(pool, axis=0).astype(np.float32)
    if xy.shape[0] > POOL_MAX:
        xy = xy[rng.choice(xy.shape[0], POOL_MAX, replace=False)]
    scale = float(np.median(np.concatenate(ratios)))
    radius = ASSIGN_RADIUS * scale
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    km.fit(xy)
    codebook = np.clip(km.cluster_centers_, -DEG_LIM, DEG_LIM).astype(np.float32)
    print(
        f"  degree codebook: K={k}, n={xy.shape[0]} samples, "
        f"scale={scale:.2f} deg/unit-H, radius={radius:.2f} deg"
    )
    return codebook, radius


def extract_monkey_features(
    monkey,
    *,
    k,
    space,
    start_ms=DEFAULT_START_MS,
    end_ms=DEFAULT_END_MS,
):
    """All feature blocks for every usable trial of `monkey` in `space`."""
    if space not in SPACES:
        raise ValueError(f"unknown space {space!r}; choose from {SPACES}")
    attractor = load_attractor_eye_data(monkey, k=k)
    behavioral = load_eye_behavioral_data(monkey)
    k = int(np.asarray(attractor["codebook_k"]))
    beh = behavioral_lookup(behavioral)
    sessions = np.asarray(attractor["session"]).astype(str)
    trials = np.asarray(attractor["trial_indices_all"]).astype(int)

    if space == "deg":
        hs = h_lookup(behavioral)
        codebook_deg, radius_deg = _fit_degree_codebook(attractor, hs, k)
        events = _event_lookup(monkey, set(sessions.tolist()))
        lim = DEG_LIM
    else:
        codebook_deg, radius_deg = None, np.nan
        lim = UNITH_LIM
    edges_lo = np.linspace(-lim, lim, GRID_LO + 1)
    edges_hi = np.linspace(-lim, lim, GRID_HI + 1)

    blocks = {name: [] for name in BLOCK_NAMES}
    rows_session, rows_trial, rows_maze = [], [], []

    for i in range(sessions.size):
        session, trial_id = sessions[i], int(trials[i])
        beh_i = beh.get((session, trial_id))
        if beh_i is None or behavioral["path_type"][beh_i] == -99:
            continue
        maze = int(behavioral["geo_type"][beh_i])
        if not 1 <= maze <= N_MAZES:
            continue
        fix_ms = fix_start_ms(behavioral, beh_i)
        if not np.isfinite(fix_ms) or fix_ms <= 0:
            continue

        t_ms, state, valid, x, y = _trial_arrays(attractor, i)
        if t_ms.size < 2:
            continue

        if space == "deg":
            h = hs.get((session, trial_id))
            if h is None or not np.all(np.isfinite(h)):
                continue
            x, y = from_maze(x, y, h)
            state = assign_fixation_states(
                np.stack([x, y], axis=1).astype(np.float32),
                t_ms,
                valid,
                events.get((session, trial_id), ()),
                codebook_deg,
                radius=radius_deg,
            )
            state = np.where(valid, state, -1).astype(int)

        lo, hi = fix_ms - start_ms, fix_ms - end_ms
        in_window = np.isfinite(t_ms) & (t_ms >= lo) & (t_ms <= hi)
        keep = in_window & valid & np.isfinite(x) & np.isfinite(y)
        if keep.sum() < 2:
            continue
        assigned = keep & (state >= 0) & (state < k)

        if assigned.any():
            dt = _sample_dt_seconds(t_ms[assigned])
            occ_ms = np.bincount(state[assigned], weights=dt, minlength=k) * 1000.0
            runs = collapse_state_runs(state[assigned])
        else:
            occ_ms, runs = np.zeros(k), []

        hist_lo, _, _ = np.histogram2d(x[keep], y[keep], bins=[edges_lo, edges_lo])
        hist_hi, _, _ = np.histogram2d(x[keep], y[keep], bins=[edges_hi, edges_hi])

        blocks["occ_ms"].append(occ_ms)
        blocks["occ_bin"].append((occ_ms > 0).astype(float))
        blocks["bigram"].append(ngram_proportions(runs, k, 2).ravel())
        blocks["heatmap_lo"].append(hist_lo.ravel() / max(hist_lo.sum(), 1.0))
        blocks["heatmap_hi"].append(hist_hi.ravel() / max(hist_hi.sum(), 1.0))
        rows_session.append(session)
        rows_trial.append(trial_id)
        rows_maze.append(maze)

    dims = block_dims(k)
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
    out["space"] = np.str_(space)
    if space == "deg":
        out["codebook_deg_xy"] = codebook_deg
        out["assign_radius_deg"] = np.float64(radius_deg)
    return out


def cache_stem(monkey, k, space, start_ms=DEFAULT_START_MS, end_ms=DEFAULT_END_MS):
    return f"{monkey}_clf2_k{int(k)}_{space}_s{int(start_ms)}_e{int(end_ms)}"


def load_features(
    monkey,
    *,
    k,
    space,
    start_ms=DEFAULT_START_MS,
    end_ms=DEFAULT_END_MS,
    refresh=False,
):
    """Cached feature blocks for every trial of `monkey` in `space` at `k`."""
    path = processed_npz(cache_stem(monkey, k, space, start_ms, end_ms))
    if path.exists() and not refresh:
        return load_npz(path)
    print(f"Extracting {monkey} features (k={k}, space={space}) ...")
    data = extract_monkey_features(
        monkey, k=k, space=space, start_ms=start_ms, end_ms=end_ms
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    print(f"Saved {path} ({len(data['session'])} trials)")
    return data


def main():
    """Build (or rebuild) every feature cache: both monkeys, both spaces, all K.

    `load_features` pulls in `data.attractor`, so this also fits and caches the
    per-monkey unit-H k-means codebook at each K -- the k=12 attractor cache is
    built here if it does not exist yet.

    `--space` narrows the sweep for a caller that only needs one space, so it
    does not pay for caches it will never read (`eye_pre_flash.corr` is unit-H
    only). Default is unchanged: every space.
    """
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--monkey", choices=MONKEYS, default=None)
    parser.add_argument("--space", choices=SPACES, default=None, help="default: all")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    spaces = (args.space,) if args.space else SPACES
    for monkey in [args.monkey] if args.monkey else list(MONKEYS):
        for k in KS:
            for space in spaces:
                data = load_features(monkey, k=k, space=space, refresh=args.refresh)
                print(
                    f"{monkey} k={k} {space}: {len(data['session'])} trials"
                )


if __name__ == "__main__":
    main()
