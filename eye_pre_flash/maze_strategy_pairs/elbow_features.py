"""Per-trial feature blocks with a per-maze codebook at a per-maze K.

The same pipeline as `classifier.features`, with step 3 narrowed on two axes at
once. `comp_pooled_features` already fits one k-means per (monkey, maze, K);
this differs from it in exactly two ways, both required by
`elbow_select`'s criterion:

* **K differs per maze.** `elbow_select` chooses one K for each maze from its
  own held-out EV curve, so there is no single K for the monkey. `k_by_maze`
  carries the mapping.
* **The pool is scope-restricted.** `comp_pooled_features` fits label-blind
  over every session that clears QC -- up to 27 for Faure. Here only the
  in-scope sessions contribute, so the codebook is fitted on the same
  population the EV curve sized it against and the estimator then analyses.

Everything else -- the window clip, the unit-H warp, I-DT detection, the
nearest-within-radius assignment, the dwell-time accumulation -- is unchanged
and reuses `classifier.features`' own helpers, exactly as both `comp_features`
and `comp_pooled_features` do.

`classifier.features`, `comp_features` and `comp_pooled_features` are all left
completely alone. This module writes to its own cache stem and never touches
theirs.

## Why the cache is keyed the way it is

Every other feature cache in the repo puts K in the filename. That is no longer
possible: K differs per maze, so one file holds several. The blocks are
therefore right-padded to `max(K_m)` and each row carries its own `row_k`,
with ragged codebooks stored concatenated plus offsets. Padding is safe because
a dimension already means nothing across mazes -- state 3 of maze 2 is not
state 3 of maze 5 -- so nothing was ever allowed to compare two mazes' blocks.

The cost is one new failure mode. With K inside the file rather than in the
stem, a stale cache built at a different `k_by_maze` (or a different session
scope) would otherwise be served silently. `load_features` therefore validates
both against what was asked for and re-extracts on any mismatch. Per
`cloud.md`'s rule that caches do not encode the settings they were built under,
this is the one place that rule has to be enforced in code rather than by the
filename.

Caches land under
``data/processed/<Monkey>_mspelbow_unith_s<start>_e<end>.npz``.
"""

from __future__ import annotations

import numpy as np

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
    _event_lookup,
    _sample_dt_seconds,
    _sample_period_ms,
    _trial_arrays,
    _trial_window,
    clip_fixation_spans,
)
from eye_pre_flash.maze_strategy_pairs.elbow_select import (
    POOL_PER_TRIAL,
    _trial_seed,
    fit_codebook,
)

DEFAULT_START_MS = PRE_FIX_START_MS
DEFAULT_END_MS = PRE_FIX_END_MS

# Only the two blocks `maze_strategy_pairs` reads, matching both comp modules.
BLOCK_NAMES = ("occ_ms", "occ_bin")


def extract_monkey_features(
    monkey,
    *,
    k_by_maze,
    keep_sessions,
    start_ms=DEFAULT_START_MS,
    end_ms=DEFAULT_END_MS,
):
    """Feature blocks for every in-scope trial, each maze at its own K.

    One pass over the attractor rows serves every maze. The scan is the
    expensive part and it is per-monkey, not per-maze, so fitting four
    codebooks from one scan rather than scanning four times is the whole reason
    this function takes a mapping instead of a single K.
    """
    attractor = load_attractor_eye_data(monkey)
    behavioral = load_eye_behavioral_data(monkey)
    beh = behavioral_lookup(behavioral)
    sessions = np.asarray(attractor["session"]).astype(str)
    trials = np.asarray(attractor["trial_indices_all"]).astype(int)
    events = _event_lookup(monkey, set(sessions.tolist()))

    keep = set(str(s) for s in keep_sessions)
    k_by_maze = {int(m): int(k) for m, k in k_by_maze.items()}

    # ---- Steps 1-2: in-scope, in-window samples, pooled per maze -----------
    usable, pool = [], {}
    for i in range(sessions.size):
        session, trial_id = sessions[i], int(trials[i])
        if session not in keep:
            continue
        beh_i = beh.get((session, trial_id))
        if not trial_qc_ok(behavioral, beh_i):
            continue
        maze = int(behavioral["geo_type"][beh_i])
        if not 1 <= maze <= N_MAZES or maze not in k_by_maze:
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
            # Same per-trial seeding as `elbow_select.maze_pools`, so the
            # codebook is fitted on exactly the samples the EV curve sized.
            rng = np.random.default_rng(_trial_seed(session, trial_id))
            idx = rng.choice(idx, POOL_PER_TRIAL, replace=False)
        pool.setdefault(maze, []).append(np.stack([x[idx], y[idx]], axis=1))
        usable.append((i, session, trial_id, maze, lo, hi))

    radius = ASSIGN_RADIUS

    # ---- Step 3: one k-means per maze, each at that maze's own K -----------
    codebooks, group_maze, group_k, group_n_pool, dropped = {}, [], [], [], []
    for maze in sorted(pool):
        k = k_by_maze[maze]
        xy = np.concatenate(pool[maze], axis=0).astype(np.float32)
        centers = fit_codebook(xy, k)
        if centers is None:
            dropped.append(f"maze{maze}: {xy.shape[0]} samples < K={k}")
            continue
        codebooks[maze] = (len(group_maze), centers)
        group_maze.append(maze)
        group_k.append(k)
        group_n_pool.append(xy.shape[0])

    if dropped:
        print(f"  {len(dropped)} maze(s) unfittable: {'; '.join(dropped)}")
    print(
        "  codebooks fitted: "
        + ", ".join(f"maze{m}:K={k_by_maze[m]}" for m in sorted(codebooks))
    )

    max_k = max(group_k) if group_k else 1
    blocks = {name: [] for name in BLOCK_NAMES}
    rows_session, rows_trial, rows_maze, rows_group, rows_k = [], [], [], [], []

    # ---- Steps 4-5: assign each trial against its own maze's codebook ------
    for i, session, trial_id, maze, lo, hi in usable:
        entry = codebooks.get(maze)
        if entry is None:
            continue
        group_index, codebook = entry
        k = k_by_maze[maze]

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

        # Right-padded to `max_k`; `row_k` says how much of the row is real.
        padded = np.zeros(max_k, dtype=float)
        padded[:k] = occ_ms
        blocks["occ_ms"].append(padded)
        bin_padded = np.zeros(max_k, dtype=float)
        bin_padded[:k] = (occ_ms > 0).astype(float)
        blocks["occ_bin"].append(bin_padded)

        rows_session.append(session)
        rows_trial.append(trial_id)
        rows_maze.append(maze)
        rows_group.append(group_index)
        rows_k.append(k)

    out = {
        name: (
            np.asarray(vals, dtype=np.float32)
            if vals
            else np.empty((0, max_k), dtype=np.float32)
        )
        for name, vals in blocks.items()
    }
    out["session"] = np.asarray(rows_session, dtype=str)
    out["trial_indices_all"] = np.asarray(rows_trial, dtype=int)
    out["maze_id"] = np.asarray(rows_maze, dtype=int)
    out["group_index"] = np.asarray(rows_group, dtype=int)
    out["row_k"] = np.asarray(rows_k, dtype=int)
    out["group_maze"] = np.asarray(group_maze, dtype=int)
    out["group_k"] = np.asarray(group_k, dtype=int)
    out["group_n_pool"] = np.asarray(group_n_pool, dtype=int)

    # Ragged codebooks, concatenated with offsets: maze m's centres are
    # rows [offset[j], offset[j] + group_k[j]) of `codebook_xy`.
    ordered = sorted(codebooks)
    if ordered:
        stacked = np.concatenate(
            [codebooks[m][1] for m in ordered], axis=0
        ).astype(np.float32)
        offsets = np.cumsum([0] + [codebooks[m][1].shape[0] for m in ordered[:-1]])
    else:
        stacked = np.empty((0, 2), dtype=np.float32)
        offsets = np.empty(0, dtype=int)
    out["codebook_xy"] = stacked
    out["codebook_offset"] = np.asarray(offsets, dtype=int)

    # Two session lists, not one. `scope_sessions` is what was *asked for* and
    # is what `_cache_matches` validates against, so the check is stable.
    # `data_sessions` is what actually contributed a row, which is the only way
    # to notice that the attractor cache is partial -- a laptop checkout can
    # hold one session while the scope names ten, and a cache keyed only on the
    # request would look valid.
    out["scope_sessions"] = np.asarray(sorted(keep), dtype=str)
    out["data_sessions"] = np.asarray(sorted(set(rows_session)), dtype=str)
    out["assign_radius"] = np.float64(radius)
    out["dropped_groups"] = np.asarray(dropped, dtype=str)

    missing = sorted(keep - set(rows_session))
    if missing:
        print(
            f"  WARNING {len(missing)} of {len(keep)} in-scope sessions "
            f"contributed no row: {', '.join(missing)}. The attractor cache is "
            "partial, so every number below rests on fewer sessions than the "
            "scope names."
        )
    return out


def maze_codebook(data, maze):
    """The ``(K_m, 2)`` codebook for `maze`, or `None` if it did not fit."""
    group_maze = np.asarray(data["group_maze"], dtype=int)
    hit = np.flatnonzero(group_maze == int(maze))
    if hit.size == 0:
        return None
    j = int(hit[0])
    start = int(np.asarray(data["codebook_offset"], dtype=int)[j])
    k = int(np.asarray(data["group_k"], dtype=int)[j])
    return np.asarray(data["codebook_xy"], dtype=np.float32)[start : start + k]


def maze_block(data, maze, block):
    """One maze's rows of `block`, trimmed to that maze's real K.

    The stored blocks are right-padded to `max(K_m)`; slicing here is the only
    place a caller should have to know that.
    """
    maze_id = np.asarray(data["maze_id"], dtype=int)
    mask = maze_id == int(maze)
    if not mask.any():
        return mask, np.empty((0, 0), dtype=float)
    k = int(np.asarray(data["row_k"], dtype=int)[mask][0])
    return mask, np.asarray(data[block], dtype=float)[mask][:, :k]


def cache_stem(monkey, start_ms=DEFAULT_START_MS, end_ms=DEFAULT_END_MS):
    """No K in the stem, because K differs per maze -- see the module docstring.

    `mspelbow` rather than `clf2`, `mspcomp` or `mspcomp_pooled`, so a
    scope-restricted per-maze-K codebook can never be served to a caller that
    asked for one of the other three.
    """
    return f"{monkey}_mspelbow_unith_s{int(start_ms)}_e{int(end_ms)}"


def _cache_matches(data, k_by_maze, keep_sessions):
    """Whether a loaded cache was built at this `k_by_maze` and this scope."""
    want_k = {int(m): int(k) for m, k in k_by_maze.items()}
    have_k = dict(
        zip(
            np.asarray(data["group_maze"], dtype=int).tolist(),
            np.asarray(data["group_k"], dtype=int).tolist(),
        )
    )
    # A maze that was asked for and did not fit is absent from `have_k`, which
    # is a legitimate state -- so only disagree where both sides have an entry,
    # and require every fitted maze to be one that was asked for at that K.
    if any(have_k.get(m) not in (None, k) for m, k in want_k.items()):
        return False
    if any(m not in want_k or want_k[m] != k for m, k in have_k.items()):
        return False
    have_sessions = set(np.asarray(data["scope_sessions"]).astype(str).tolist())
    return have_sessions == set(str(s) for s in keep_sessions)


def load_features(
    monkey,
    *,
    k_by_maze,
    keep_sessions,
    start_ms=DEFAULT_START_MS,
    end_ms=DEFAULT_END_MS,
    refresh=False,
):
    """Cached blocks, re-extracted whenever the cache's K or scope disagrees."""
    path = processed_npz(cache_stem(monkey, start_ms, end_ms))
    if path.exists() and not refresh:
        data = load_npz(path)
        if _cache_matches(data, k_by_maze, keep_sessions):
            return data
        print(
            f"{path.name} was built at a different K-per-maze or session scope; "
            "re-extracting."
        )
    print(f"Extracting {monkey} elbow features ...")
    data = extract_monkey_features(
        monkey,
        k_by_maze=k_by_maze,
        keep_sessions=keep_sessions,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    savez_atomic(path, **data)
    print(f"Saved {path} ({len(data['session'])} trials)")
    return data
