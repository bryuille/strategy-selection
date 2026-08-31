import numpy as np

from eye_data_plotting.plot_io import PRE_FIX_END_MS, PRE_FIX_START_MS


def behavioral_lookup(behavioral):
    return {
        (str(session), int(trial_id)): i
        for session, trial_id, i in zip(
            behavioral["session"],
            behavioral["trial_indices_all"],
            range(len(behavioral["session"])),
        )
    }


def fix_start_ms(behavioral, beh_i):
    return (
        behavioral["fix_start"][beh_i] - behavioral["geo_present"][beh_i]
    ) * 1000.0


def pre_fix_assigned_mask(
    t_ms,
    fix_ms,
    valid,
    state,
    k,
    start_ms=PRE_FIX_START_MS,
    end_ms=PRE_FIX_END_MS,
):
    """Samples in [fix_start - start_ms, fix_start - end_ms] with assigned states."""
    keep = (
        np.isfinite(t_ms)
        & (t_ms >= fix_ms - start_ms)
        & (t_ms <= fix_ms - end_ms)
    )
    keep &= valid & (state >= 0) & (state < k)
    return keep


def attractor_pre_fix_state_times(
    attractor,
    behavioral,
    start_ms=PRE_FIX_START_MS,
    end_ms=PRE_FIX_END_MS,
):
    """Seconds per codebook state from fix_start-start_ms through fix_start-end_ms."""
    lookup = behavioral_lookup(behavioral)
    k = int(np.asarray(attractor["codebook_k"]))
    n = len(attractor["session"])
    times = np.full((n, k), np.nan)
    for i in range(n):
        beh_i = lookup.get(
            (str(attractor["session"][i]), int(attractor["trial_indices_all"][i]))
        )
        if beh_i is None or behavioral["path_type"][beh_i] == -99:
            continue
        t_ms = np.asarray(attractor["time"][i], dtype=float) * 1000.0
        state = np.asarray(attractor["state_id"][i], dtype=int)
        valid = np.asarray(attractor["valid"][i], dtype=bool)
        n_samp = min(t_ms.size, state.size, valid.size)
        if n_samp < 2:
            continue
        t_ms = t_ms[:n_samp]
        state = state[:n_samp]
        valid = valid[:n_samp]
        fix_ms = fix_start_ms(behavioral, beh_i)
        if not np.isfinite(fix_ms) or fix_ms <= 0:
            continue
        keep = pre_fix_assigned_mask(
            t_ms, fix_ms, valid, state, k, start_ms=start_ms, end_ms=end_ms
        )
        if keep.sum() < 2:
            continue
        t_keep = t_ms[keep]
        dt = np.empty(t_keep.size, dtype=float)
        dt[:-1] = np.diff(t_keep)
        pos = dt[:-1][dt[:-1] > 0]
        dt[-1] = float(np.median(pos)) if pos.size else 1.0
        dt = np.where(dt > 0, dt, 0.0)
        times[i] = np.bincount(state[keep], weights=dt, minlength=k) / 1000.0
    return times


def attractor_pre_fix_occupancy_matrix(
    attractor,
    behavioral,
    start_ms=PRE_FIX_START_MS,
    end_ms=PRE_FIX_END_MS,
):
    """Pre-fixation codebook occupancy (seconds per state) aligned to attractor rows."""
    return attractor_pre_fix_state_times(
        attractor, behavioral, start_ms=start_ms, end_ms=end_ms
    )


def attractor_pre_fix_occupancy_bin_matrix(
    attractor,
    behavioral,
    start_ms=PRE_FIX_START_MS,
    end_ms=PRE_FIX_END_MS,
):
    """Pre-fixation visit/no-visit occupancy (1 if a state was used, else 0)."""
    occ = attractor_pre_fix_occupancy_matrix(
        attractor, behavioral, start_ms=start_ms, end_ms=end_ms
    )
    out = np.full_like(occ, np.nan)
    finite = np.isfinite(occ).all(axis=1)
    out[finite] = (occ[finite] > 0).astype(np.float64)
    return out


def occupancy_rows_from_matrix(attractor, behavioral, features):
    beh_lookup = behavioral_lookup(behavioral)
    rows_x, rows_g, rows_trial, rows_maze = [], [], [], []
    for i in range(len(attractor["session"])):
        session = str(attractor["session"][i])
        trial_id = int(attractor["trial_indices_all"][i])
        beh_i = beh_lookup.get((session, trial_id))
        if beh_i is None or behavioral["path_type"][beh_i] == -99:
            continue
        if not np.isfinite(features[i]).all():
            continue
        rows_x.append(features[i])
        rows_g.append(session)
        rows_trial.append(trial_id)
        rows_maze.append(int(behavioral["geo_type"][beh_i]))
    if not rows_x:
        empty_k = int(np.asarray(attractor["codebook_k"]))
        return (
            np.empty((0, empty_k), dtype=float),
            np.asarray([], dtype=object),
            np.asarray([], dtype=int),
            np.asarray([], dtype=int),
        )
    return (
        np.asarray(rows_x, dtype=float),
        np.asarray(rows_g),
        np.asarray(rows_trial, dtype=int),
        np.asarray(rows_maze, dtype=int),
    )


def attractor_occupancy_rows(
    attractor,
    behavioral,
    start_ms=PRE_FIX_START_MS,
    end_ms=PRE_FIX_END_MS,
):
    """Seconds-per-state occupancy for every valid-path trial with finite occupancy."""
    return occupancy_rows_from_matrix(
        attractor,
        behavioral,
        attractor_pre_fix_occupancy_matrix(
            attractor, behavioral, start_ms=start_ms, end_ms=end_ms
        ),
    )


def attractor_occupancy_bin_rows(
    attractor,
    behavioral,
    start_ms=PRE_FIX_START_MS,
    end_ms=PRE_FIX_END_MS,
):
    """Binary visit/no-visit occupancy for every valid-path trial with finite occupancy."""
    return occupancy_rows_from_matrix(
        attractor,
        behavioral,
        attractor_pre_fix_occupancy_bin_matrix(
            attractor, behavioral, start_ms=start_ms, end_ms=end_ms
        ),
    )


def collapse_state_runs(states):
    runs = []
    for state in states:
        sid = int(state)
        if not runs or runs[-1] != sid:
            runs.append(sid)
    return runs


def ngram_proportions(runs, k, order):
    """Proportion of length-`order` state-run n-grams; zeros if too few runs."""
    counts = np.zeros((k,) * order, dtype=np.float64)
    n = len(runs) - order + 1
    if n <= 0:
        return counts
    for i in range(n):
        counts[tuple(runs[i : i + order])] += 1.0
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def transition_feature_dim(k):
    """Bigram (K×K) plus trigram (K×K×K) proportions."""
    return k * k + k * k * k


def attractor_pre_fix_transition_matrix(
    attractor,
    behavioral,
    start_ms=PRE_FIX_START_MS,
    end_ms=PRE_FIX_END_MS,
):
    """Pre-fixation transition-order features (bigram + trigram proportions)."""
    lookup = behavioral_lookup(behavioral)
    k = int(np.asarray(attractor["codebook_k"]))
    n = len(attractor["session"])
    n_bigrams = k * k
    features = np.full((n, transition_feature_dim(k)), np.nan)
    for i in range(n):
        beh_i = lookup.get(
            (str(attractor["session"][i]), int(attractor["trial_indices_all"][i]))
        )
        if beh_i is None or behavioral["path_type"][beh_i] == -99:
            continue
        t_ms = np.asarray(attractor["time"][i], dtype=float) * 1000.0
        state = np.asarray(attractor["state_id"][i], dtype=int)
        valid = np.asarray(attractor["valid"][i], dtype=bool)
        n_samp = min(t_ms.size, state.size, valid.size)
        if n_samp < 1:
            continue
        t_ms = t_ms[:n_samp]
        state = state[:n_samp]
        valid = valid[:n_samp]
        fix_ms = fix_start_ms(behavioral, beh_i)
        if not np.isfinite(fix_ms) or fix_ms <= 0:
            continue
        keep = pre_fix_assigned_mask(
            t_ms, fix_ms, valid, state, k, start_ms=start_ms, end_ms=end_ms
        )
        if not keep.any():
            continue
        runs = collapse_state_runs(state[keep])
        bigrams = ngram_proportions(runs, k, 2)
        trigrams = ngram_proportions(runs, k, 3)
        features[i, :n_bigrams] = bigrams.ravel()
        features[i, n_bigrams:] = trigrams.ravel()
    return features


def attractor_transition_rows(
    attractor,
    behavioral,
    start_ms=PRE_FIX_START_MS,
    end_ms=PRE_FIX_END_MS,
):
    """Transition n-gram features for trials with at least one state change."""
    beh_lookup = behavioral_lookup(behavioral)
    features = attractor_pre_fix_transition_matrix(
        attractor, behavioral, start_ms=start_ms, end_ms=end_ms
    )
    rows_x, rows_g, rows_trial, rows_maze = [], [], [], []
    for i in range(len(attractor["session"])):
        session = str(attractor["session"][i])
        trial_id = int(attractor["trial_indices_all"][i])
        beh_i = beh_lookup.get((session, trial_id))
        if beh_i is None or behavioral["path_type"][beh_i] == -99:
            continue
        row = features[i]
        if not np.isfinite(row).all() or not np.any(row > 0):
            continue
        rows_x.append(row)
        rows_g.append(session)
        rows_trial.append(trial_id)
        rows_maze.append(int(behavioral["geo_type"][beh_i]))
    if not rows_x:
        empty_k = int(np.asarray(attractor["codebook_k"]))
        return (
            np.empty((0, transition_feature_dim(empty_k)), dtype=float),
            np.asarray([], dtype=object),
            np.asarray([], dtype=int),
            np.asarray([], dtype=int),
        )
    return (
        np.asarray(rows_x, dtype=float),
        np.asarray(rows_g),
        np.asarray(rows_trial, dtype=int),
        np.asarray(rows_maze, dtype=int),
    )
