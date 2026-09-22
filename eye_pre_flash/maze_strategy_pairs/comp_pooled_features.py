"""Per-trial feature blocks with the codebook fitted per (monkey, maze).

The same pipeline as `classifier.features`, with step 3 narrowed one level:
instead of one k-means per (monkey, K) pooled over every session and every
maze (`classifier.features`), or one per (session, maze, K) pooled over
neither (`comp_features`), this fits **one k-means per (monkey, maze, K)** --
pooling that maze's in-window fixation samples across every one of the
monkey's usable sessions. Every other step -- the window clip, the unit-H
warp, I-DT fixation detection, the nearest-prototype assignment -- is
unchanged and reuses `classifier.features`' own helpers, exactly as
`comp_features` already does.

The fit itself never sees a label or a label source: it pools every session
that clears QC and the window, the same way `classifier.features` and
`comp_features` do. A label-source scope (e.g. `svm/top_ten`) is applied only
downstream, at estimation time in `comp_pooled_build`, by masking which of
this maze's pooled rows enter the estimator -- not by restricting which
sessions' samples the codebook itself is fit on.

`classifier.features` and `comp_features` are both left completely alone.
This module writes to its own cache stem and never touches theirs.

Consequences of this fit, which callers have to respect:

* A dimension means nothing across mazes. State 3 of maze 2 is not state 3 of
  maze 5 -- different k-means, different centres. Features may only ever be
  compared *within* one maze, which is exactly what `comp_pooled_build` does.
* `no_origin`'s origin state and `mean_removed`'s grand mean are therefore
  per-maze, and must be recomputed from that maze's own codebook and its own
  pooled trials rather than inherited from a monkey-wide fit.
* A maze with fewer pooled samples than K cannot be fitted at all. Those are
  dropped, and the maze recorded in `dropped_groups`.

Caches land under
``data/processed/<Monkey>_mspcomp_pooled_k<K>_unith_s<start>_e<end>.npz``.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from data.attractor import ASSIGN_RADIUS, MAZE_SCREEN_LIM, assign_fixation_states
from data.builder import trial_qc_ok
from data.config import PRE_FIX_END_MS, PRE_FIX_START_MS, processed_npz
from data.convert import load_npz
from data.loader import (
    load_attractor_eye_data,
    load_eye_behavioral_data,
    savez_atomic,
)
from data.occupancy import behavioral_lookup
from eye_pre_flash.classifier.features import (
    N_MAZES,
    POOL_MAX,
    POOL_PER_TRIAL,
    SEED,
    _event_lookup,
    _sample_dt_seconds,
    _sample_period_ms,
    _trial_arrays,
    _trial_window,
    clip_fixation_spans,
)

DEFAULT_START_MS = PRE_FIX_START_MS
DEFAULT_END_MS = PRE_FIX_END_MS

# Only the two blocks `maze_strategy_pairs` reads. `bigram` is K*K wide and
# would dominate the cache at K = 30 for a block no caller here uses -- same
# call `comp_features` already makes, kept for consistency between the two.
BLOCK_NAMES = ("occ_ms", "occ_bin")


def _fit_group_codebook(pool_xy, k, *, seed=SEED):
    """K-means over one maze's pooled, cross-session in-window fixation samples.

    Identical to `classifier.features._fit_window_codebook` and
    `comp_features._fit_group_codebook` except for what it is handed. Returns
    ``(centers, n_pool)``, or ``(None, n_pool)`` when the maze has fewer
    pooled samples than K and no k-means is possible.
    """
    rng = np.random.default_rng(seed)
    xy = np.concatenate(pool_xy, axis=0).astype(np.float32)
    if xy.shape[0] > POOL_MAX:
        xy = xy[rng.choice(xy.shape[0], POOL_MAX, replace=False)]
    if xy.shape[0] < k:
        return None, xy.shape[0]
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    km.fit(xy)
    centers = np.clip(km.cluster_centers_, -MAZE_SCREEN_LIM, MAZE_SCREEN_LIM)
    return centers.astype(np.float32), xy.shape[0]


def extract_monkey_features(
    monkey, *, k, start_ms=DEFAULT_START_MS, end_ms=DEFAULT_END_MS, mazes=None
):
    """Feature blocks for every usable trial of `monkey`, per-maze codebooks.

    `mazes` restricts which mazes are fitted at all; the default is all of
    them. Rows come back with a `group_index` pointing into the parallel
    `group_maze` / `codebook_xy` arrays, so a caller can recover the codebook
    any given row was assigned against.
    """
    attractor = load_attractor_eye_data(monkey)
    behavioral = load_eye_behavioral_data(monkey)
    k = int(k)
    wanted = None if mazes is None else set(int(m) for m in mazes)
    beh = behavioral_lookup(behavioral)
    sessions = np.asarray(attractor["session"]).astype(str)
    trials = np.asarray(attractor["trial_indices_all"]).astype(int)
    events = _event_lookup(monkey, set(sessions.tolist()))

    # ---- Steps 1-2: keep in-window samples, pooled **per maze** -----------
    rng = np.random.default_rng(SEED)
    usable = []
    pool = {}
    for i in range(sessions.size):
        session, trial_id = sessions[i], int(trials[i])
        beh_i = beh.get((session, trial_id))
        if not trial_qc_ok(behavioral, beh_i):
            continue
        maze = int(behavioral["geo_type"][beh_i])
        if not 1 <= maze <= N_MAZES:
            continue
        if wanted is not None and maze not in wanted:
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
        pool.setdefault(maze, []).append(np.stack([x[idx], y[idx]], axis=1))
        usable.append((i, session, trial_id, maze, lo, hi))

    radius = ASSIGN_RADIUS

    # ---- Step 3: one k-means per maze, pooled across every session --------
    codebooks, group_maze, group_n_pool, dropped = {}, [], [], []
    for maze in sorted(pool):
        centers, n_pool = _fit_group_codebook(pool[maze], k)
        if centers is None:
            dropped.append(f"maze{maze}: {n_pool} samples < K={k}")
            continue
        codebooks[maze] = (len(group_maze), centers)
        group_maze.append(maze)
        group_n_pool.append(n_pool)

    if dropped:
        print(f"  K={k}: {len(dropped)} maze(s) unfittable: {'; '.join(dropped)}")
    print(f"  K={k}: {len(codebooks)} per-maze codebooks fitted")

    blocks = {name: [] for name in BLOCK_NAMES}
    rows_session, rows_trial, rows_maze, rows_group = [], [], [], []

    # ---- Steps 4-5: assign each trial against *its own maze's* codebook ---
    for i, session, trial_id, maze, lo, hi in usable:
        entry = codebooks.get(maze)
        if entry is None:
            continue
        group_index, codebook = entry

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
        else:
            occ_ms = np.zeros(k)

        blocks["occ_ms"].append(occ_ms)
        blocks["occ_bin"].append((occ_ms > 0).astype(float))
        rows_session.append(session)
        rows_trial.append(trial_id)
        rows_maze.append(maze)
        rows_group.append(group_index)

    out = {
        name: (
            np.asarray(vals, dtype=np.float32)
            if vals
            else np.empty((0, k), dtype=np.float32)
        )
        for name, vals in blocks.items()
    }
    out["session"] = np.asarray(rows_session, dtype=str)
    out["trial_indices_all"] = np.asarray(rows_trial, dtype=int)
    out["maze_id"] = np.asarray(rows_maze, dtype=int)
    out["group_index"] = np.asarray(rows_group, dtype=int)
    out["group_maze"] = np.asarray(group_maze, dtype=int)
    out["group_n_pool"] = np.asarray(group_n_pool, dtype=int)
    out["codebook_xy"] = (
        np.stack([codebooks[maze][1] for maze in sorted(codebooks)], axis=0)
        if codebooks
        else np.empty((0, k, 2), dtype=np.float32)
    )
    out["codebook_k"] = np.int64(k)
    out["assign_radius"] = np.float64(radius)
    out["dropped_groups"] = np.asarray(dropped, dtype=str)
    return out


def cache_stem(monkey, k, start_ms=DEFAULT_START_MS, end_ms=DEFAULT_END_MS):
    """Distinct from both `classifier.features` and `comp_features`.

    `mspcomp_pooled` rather than `clf2` or `mspcomp`, so a per-maze codebook
    can never be served to a caller that asked for the whole-monkey or the
    per-(session, maze) one, at any K the three share.
    """
    return f"{monkey}_mspcomp_pooled_k{int(k)}_unith_s{int(start_ms)}_e{int(end_ms)}"


def load_features(
    monkey, *, k, start_ms=DEFAULT_START_MS, end_ms=DEFAULT_END_MS, refresh=False
):
    """Cached per-(monkey, maze) feature blocks for `monkey` at `k`."""
    path = processed_npz(cache_stem(monkey, k, start_ms, end_ms))
    if path.exists() and not refresh:
        return load_npz(path)
    print(f"Extracting {monkey} per-(monkey, maze) features (k={k}) ...")
    data = extract_monkey_features(monkey, k=k, start_ms=start_ms, end_ms=end_ms)
    path.parent.mkdir(parents=True, exist_ok=True)
    savez_atomic(path, **data)
    print(f"Saved {path} ({len(data['session'])} trials)")
    return data
