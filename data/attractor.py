from __future__ import annotations

import argparse

import numpy as np
from pymovements.events.detection import out_of_screen
from sklearn.cluster import KMeans

from data.builder import (
    POSITION_LIMIT_DEG,
    qc_mask,
    trial_blink_mask,
    trial_fixation_mask,
)
from data.config import eye_npz_path, processed_npz
from data.convert import load_npz
from eye_pre_flash.plotting.plot_io import PRE_FIX_END_MS, PRE_FIX_START_MS

CODEBOOK_SWEEP = (4, 5, 6, 7, 8, 10, 12)
DEFAULT_K = 12
# Maze plot box; k-means samples and prototypes are clipped to this square.
MAZE_SCREEN_LIM = 3.0
# Assign a fixation only when gaze is within this radius of a prototype.
ASSIGN_RADIUS = 1.0
CODE_X_LIM = MAZE_SCREEN_LIM
CODE_Y_LIM = MAZE_SCREEN_LIM
BLINK_PADDING_MS = 50
ARM_EPS = 1e-6
# Physical height of the center vertical stem (fixation target → junction).
STEM_DEG = 7.0
MAZE_EXITS = (
    ("LU", (-1.0, 1.0)),
    ("LD", (-1.0, -1.0)),
    ("RU", (1.0, 1.0)),
    ("RD", (1.0, -1.0)),
)
MAZE_ORIGIN = ("origin", (0.0, 0.0))
EXTRA_HOLD_XY = (2.25, 0.25)

ATTRACTOR_EYE_DATA_KEYS = (
    "time",
    "eye_x",
    "eye_y",
    "recon_x",
    "recon_y",
    "state_id",
    "valid",
    "artifact_prob",
    "session",
    "trial_indices_all",
    "codebook_xy",
    "codebook_k",
    "codebook_name",
    "codebook_usage",
    "valid_mae",
    "transition_state",
    "transition_onset",
    "transition_offset",
    "transition_duration",
    "transition_location_x",
    "transition_location_y",
    "transition_session",
    "transition_trial_indices_all",
)


def in_maze_screen(x, y, lim=MAZE_SCREEN_LIM):
    """True where maze `(x, y)` lies in `[-lim, lim] × [-lim, lim]`."""
    x = np.asarray(x)
    y = np.asarray(y)
    return np.isfinite(x) & np.isfinite(y) & (np.abs(x) <= lim) & (np.abs(y) <= lim)


def clip_codebook_xy(codebook_xy, lim=MAZE_SCREEN_LIM):
    """Clip prototype coordinates into `[-lim, lim]²`."""
    xy = np.asarray(codebook_xy, dtype=np.float32)
    clipped = np.clip(xy, -lim, lim)
    if xy.size and not np.allclose(xy, clipped):
        n = int(np.any(np.abs(xy) > lim, axis=1).sum())
        print(f"Clipped {n} codebook point(s) into [{-lim:g}, {lim:g}]^2")
    return clipped.astype(np.float32, copy=False)


def assign_states_inference(xy, codebook_xy, radius=ASSIGN_RADIUS):
    """Pick the nearest prototype within `radius`, else return -1."""
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


def contiguous_true_runs(mask):
    """Inclusive-exclusive sample slices `[start, end)` of contiguous True runs."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], cuts + 1))
    ends = np.concatenate((cuts + 1, [idx.size]))
    return [(int(idx[s]), int(idx[e - 1]) + 1) for s, e in zip(starts, ends)]


def fixation_event_spans(event_rows):
    """Fixation `(duration, onset, offset)` spans, shortest first."""
    spans = []
    for name, onset, offset in event_rows or ():
        if "fixation" not in str(name).lower():
            continue
        onset = float(onset)
        offset = float(offset)
        spans.append((offset - onset, onset, offset))
    spans.sort()
    return spans


def assign_fixation_states(
    xy, t_ms, valid, event_rows, codebook_xy, radius=ASSIGN_RADIUS
):
    """Assign each pymovements fixation as a whole to one prototype.

    The representative point is the mean of valid maze-space samples in the
    event. That centroid is snapped with `assign_states_inference`; every valid
    sample in the event inherits the result. Overlapping events keep the
    longest event's assignment so a slow drift is not split at a Voronoi edge.
    """
    n = len(xy)
    state = np.full(n, -1, dtype=np.int16)
    valid = np.asarray(valid, dtype=bool)
    if n == 0 or not valid.any():
        return state

    spans = fixation_event_spans(event_rows)
    if spans:
        t_ms = np.asarray(t_ms)
        groups = [valid & (t_ms >= onset) & (t_ms <= offset) for _, onset, offset in spans]
    else:
        groups = []
        for start, end in contiguous_true_runs(valid):
            hit = np.zeros(n, dtype=bool)
            hit[start:end] = True
            groups.append(hit)

    for hit in groups:
        if not hit.any():
            continue
        centroid = np.asarray(xy[hit], dtype=np.float32).mean(axis=0, keepdims=True)
        sid = int(assign_states_inference(centroid, codebook_xy, radius=radius)[0])
        if sid >= 0:
            state[hit] = np.int16(sid)
    return state


def events_mask(events, t_ms, n, padding_ms=0):
    mask = np.zeros(n, dtype=bool)
    if events is None or events.frame.height == 0:
        return mask
    for onset, offset in events.frame.select("onset", "offset").rows():
        mask |= (t_ms >= onset - padding_ms) & (t_ms <= offset + padding_ms)
    return mask


def out_of_screen_mask(x, y, t_ms, x_min, x_max, y_min, y_max):
    xy = np.stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)], axis=1)
    xy = np.where(np.isfinite(xy), xy, 1e6)
    return events_mask(
        out_of_screen(
            xy,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            timesteps=t_ms,
        ),
        t_ms,
        t_ms.size,
    )


def samples_in_fixations(t_ms, event_rows):
    """Fixation samples that do not also fall inside a saccade."""
    n = t_ms.size
    in_fix = np.zeros(n, dtype=bool)
    in_sac = np.zeros(n, dtype=bool)
    for name, onset, offset in event_rows:
        hit = (t_ms >= onset) & (t_ms <= offset)
        name = str(name).lower()
        if "saccade" in name:
            in_sac |= hit
        elif "fixation" in name:
            in_fix |= hit
    return in_fix & ~in_sac


def event_lookup(events):
    lookup = {}
    for i in range(len(events["name"])):
        key = (str(events["session"][i]), int(events["trial_indices_all"][i]))
        lookup.setdefault(key, []).append(
            (
                str(events["name"][i]),
                float(events["onset"][i]),
                float(events["offset"][i]),
            )
        )
    return lookup


def arm_lengths(h):
    return tuple(max(float(v), ARM_EPS) for v in h)


def to_maze(x, y, h):
    """Warp degrees so exits sit at the unit-square corners and the stem top at (0, 1)."""
    h1, h2, h3, h4, h5, h6 = arm_lengths(h)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_n = np.where(x <= 0, x / h1, x / h4)
    s_pos = np.interp(x_n, [-1.0, 0.0, 1.0], [h2, STEM_DEG, h5])
    s_neg = np.interp(x_n, [-1.0, 0.0, 1.0], [h3, 0.5 * (h3 + h6), h6])
    y_n = np.where(y >= 0, y / s_pos, y / s_neg)
    return x_n, y_n


def prepare_trial(
    t, x, y, h, pupil, event_rows=None, experiment=None, *, require_fixation=True
):
    """Maze-normalize a trial; by default keep only on-screen fixation samples."""
    n = min(t.size, x.size, y.size)
    t = np.asarray(t[:n], dtype=np.float64)
    x = np.asarray(x[:n], dtype=np.float64)
    y = np.asarray(y[:n], dtype=np.float64)
    if n < 2 or h is None or not np.all(np.isfinite(h)):
        return None

    x_m, y_m = to_maze(x, y, h)
    t_ms = np.round(t * 1000.0).astype(np.int64)
    blink = trial_blink_mask(t, pupil, padding_ms=BLINK_PADDING_MS)
    if blink.size < n:
        blink = np.pad(blink, (0, n - blink.size))
    oos_deg = out_of_screen_mask(
        x, y, t_ms, -POSITION_LIMIT_DEG, POSITION_LIMIT_DEG, -POSITION_LIMIT_DEG, POSITION_LIMIT_DEG
    )
    valid = (
        np.isfinite(t)
        & np.isfinite(x_m)
        & np.isfinite(y_m)
        & ~blink[:n]
        & ~oos_deg
    )
    if require_fixation:
        if event_rows is None:
            in_fix = trial_fixation_mask(t, x, y, pupil, experiment=experiment)
        else:
            in_fix = samples_in_fixations(t_ms, event_rows)
        valid &= in_fix[:n]
    return {
        "t": t.astype(np.float32),
        "x": x_m.astype(np.float32),
        "y": y_m.astype(np.float32),
        "valid_orig": np.asarray(valid, dtype=bool),
        "event_rows": tuple(event_rows) if event_rows is not None else (),
    }


def inlier_xy(xy):
    xy = np.asarray(xy, dtype=np.float32)
    return (np.abs(xy[:, 0]) <= CODE_X_LIM) & (np.abs(xy[:, 1]) <= CODE_Y_LIM)


def codebook_xy_pool(packed, max_samples=80000, seed=0):
    """In-maze valid fixation samples used to fit k-means codes."""
    chunks = []
    for p in packed:
        if p is None:
            continue
        keep = p["valid_orig"]
        if not keep.any():
            continue
        xy = np.stack([p["x"][keep], p["y"][keep]], axis=1)
        xy = xy[inlier_xy(xy)]
        if xy.size:
            chunks.append(xy)
    if not chunks:
        return np.empty((0, 2), dtype=np.float32)
    xy = np.concatenate(chunks, axis=0)
    rng = np.random.default_rng(seed)
    if xy.shape[0] > max_samples:
        xy = xy[rng.choice(xy.shape[0], max_samples, replace=False)]
    return xy.astype(np.float32, copy=False)


def fit_code_xy_kmeans(xy, k, seed=0):
    """Hard k-means on in-maze (x, y) samples."""
    rng = np.random.default_rng(seed)
    xy = np.asarray(xy, dtype=np.float32)
    if k <= 0:
        return np.empty((0, 2), dtype=np.float32)
    if xy.shape[0] == 0:
        return rng.uniform(-1.0, 1.0, size=(k, 2)).astype(np.float32)
    if xy.shape[0] < k:
        reps = np.tile(xy, (int(np.ceil(k / xy.shape[0])), 1))[:k]
        return reps.astype(np.float32, copy=False)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    km.fit(xy)
    return km.cluster_centers_.astype(np.float32)


def fit_code_xy(xy, k, seed=0):
    """K-means codebook in maze units, clipped to the ±3 maze box."""
    centers = clip_codebook_xy(fit_code_xy_kmeans(xy, k, seed=seed))
    names = [f"k{i}" for i in range(k)]
    return centers, names


def pinned_landmarks(k):
    """Geometric maze landmarks for diagnostics."""
    exits = [(name, np.asarray(xy, dtype=np.float32)) for name, xy in MAZE_EXITS]
    if k <= 4:
        return exits[:k]
    origin = (MAZE_ORIGIN[0], np.asarray(MAZE_ORIGIN[1], dtype=np.float32))
    return [origin, *exits]


def trial_inference(packed, codebook_xy):
    n = min(packed["valid_orig"].size, packed["x"].size, packed["y"].size)
    valid_orig = np.asarray(packed["valid_orig"][:n], dtype=bool)
    xy = np.stack([packed["x"][:n], packed["y"][:n]], axis=1).astype(np.float32)
    t_ms = np.round(np.asarray(packed["t"][:n], dtype=np.float64) * 1000.0).astype(
        np.int64
    )
    state = assign_fixation_states(
        xy, t_ms, valid_orig, packed.get("event_rows"), codebook_xy
    )
    state = np.where(valid_orig, state, np.int16(-1))

    recon = np.full((n, 2), np.nan, dtype=np.float32)
    ok = state >= 0
    recon[ok] = codebook_xy[state[ok]]
    artifact = np.where(valid_orig, 0.0, 1.0).astype(np.float32)
    return state, recon, artifact, valid_orig, xy


def state_runs(t_ms, state, codebook_xy, session, trial_id):
    n = state.size
    if n == 0:
        return []
    rows = []
    start = 0
    n_codes = len(codebook_xy)
    for i in range(1, n + 1):
        if i == n or state[i] != state[start]:
            sid = int(state[start])
            if 0 <= sid < n_codes:
                onset = float(t_ms[start])
                offset = float(t_ms[i - 1] if i - 1 < t_ms.size else t_ms[-1])
                x, y = codebook_xy[sid]
                rows.append(
                    {
                        "state": sid,
                        "onset": onset,
                        "offset": offset,
                        "duration": max(offset - onset, 0.0),
                        "location_x": float(x),
                        "location_y": float(y),
                        "session": session,
                        "trial_indices_all": int(trial_id),
                    }
                )
            start = i
    return rows


def h_lookup(behavioral):
    return {
        (str(session), int(trial_id)): tuple(
            float(behavioral[f"h{k}"][i]) for k in range(1, 7)
        )
        for i, (session, trial_id) in enumerate(
            zip(behavioral["session"], behavioral["trial_indices_all"])
        )
    }


def pupil_for_trial(pupils_by_session, monkey, session, trial_id):
    if session not in pupils_by_session:
        path = eye_npz_path(monkey, session)
        if path.exists():
            raw = load_npz(path)
            pupils_by_session[session] = (
                list(raw["pupil_size"]) if "pupil_size" in raw else None
            )
        else:
            pupils_by_session[session] = None
    session_pupils = pupils_by_session[session]
    if session_pupils is None or trial_id - 1 >= len(session_pupils):
        return np.empty((0, 2))
    return session_pupils[trial_id - 1]


def unpack_eye(eye, i):
    return (
        np.asarray(eye["time"][i], dtype=float),
        np.asarray(eye["eye_x"][i], dtype=float),
        np.asarray(eye["eye_y"][i], dtype=float),
        str(eye["session"][i]),
        int(eye["trial_indices_all"][i]),
    )


def pack_trials(
    eye_data,
    behavioral,
    *,
    monkey,
    max_trials=None,
    events="auto",
    require_fixation=True,
):
    events_by_trial = None
    experiment = None
    if require_fixation:
        if events == "auto":
            from data.loader import load_clean_eye_data

            events_by_trial = event_lookup(
                load_clean_eye_data(
                    monkey, start_ms=PRE_FIX_START_MS, end_ms=PRE_FIX_END_MS
                )
            )
        elif events is not None:
            events_by_trial = event_lookup(events)

    n_trials = len(eye_data["session"])
    if max_trials is not None:
        n_trials = min(n_trials, int(max_trials))
    hs = h_lookup(behavioral)
    # QC gate. A failing trial is packed as `None`, the signal `prepare_trial`
    # already uses for an unusable trial, so row indices stay aligned with
    # `eye_data` while `codebook_xy_pool` and `build_attractor_eye_data` skip
    # it. Without this the whole-trial k-means codebook is fitted on faded and
    # photodiode-bad gaze.
    qc_ok = qc_mask(behavioral)
    qc_row = {
        (str(s_), int(t_)): j
        for j, (s_, t_) in enumerate(
            zip(behavioral["session"], behavioral["trial_indices_all"])
        )
    }
    pupils_by_session = {}
    packed = []
    n_qc_dropped = 0
    for i in range(n_trials):
        t, x, y, session, trial_id = unpack_eye(eye_data, i)
        beh_i = qc_row.get((str(session), int(trial_id)))
        if beh_i is None or not qc_ok[beh_i]:
            packed.append(None)
            n_qc_dropped += 1
            continue
        pupil = pupil_for_trial(pupils_by_session, monkey, session, trial_id)
        event_rows = (
            None if events_by_trial is None else events_by_trial.get((session, trial_id), ())
        )
        packed.append(
            prepare_trial(
                t,
                x,
                y,
                hs.get((session, trial_id)),
                pupil,
                event_rows=event_rows,
                experiment=experiment,
                require_fixation=require_fixation,
            )
        )
        if (i + 1) % 2000 == 0:
            print(f"  packed {i + 1}/{n_trials}")
    print(f"  {n_qc_dropped}/{n_trials} trial(s) dropped by QC before packing")
    return packed, n_trials


def print_codebook_diagnostics(codebook_xy, codebook_name, usage, k):
    print("K-means prototypes (maze units):")
    for j, ((px, py), name) in enumerate(zip(codebook_xy, codebook_name)):
        print(f"  state {j:2d} {name:8s}: ({px:7.2f}, {py:7.2f})  usage={usage[j]:.3f}")
    diagnostic = list(pinned_landmarks(max(k, 5))) + [
        ("extra", np.asarray(EXTRA_HOLD_XY, dtype=np.float32))
    ]
    for name, pt in diagnostic:
        d = np.hypot(codebook_xy[:, 0] - pt[0], codebook_xy[:, 1] - pt[1])
        j = int(np.argmin(d))
        print(f"  nearest to {name} ({pt[0]:g}, {pt[1]:g}): state {j} at {d[j]:.2f}")
    if k > 1:
        dmat = np.sqrt(
            (codebook_xy[:, None, 0] - codebook_xy[None, :, 0]) ** 2
            + (codebook_xy[:, None, 1] - codebook_xy[None, :, 1]) ** 2
        )
        np.fill_diagonal(dmat, np.inf)
        print(f"  min prototype separation: {dmat.min():.2f}")


def infer_all_trials(packed, eye_data, codebook_xy, codebook_name):
    codebook_xy = clip_codebook_xy(codebook_xy)
    k = codebook_xy.shape[0]
    times, xs, ys = [], [], []
    recon_xs, recon_ys = [], []
    state_ids, valids, artifacts = [], [], []
    sessions, trial_ids = [], []
    transition_rows = []
    usage = np.zeros(k, dtype=np.float64)
    usage_n = 0.0
    abs_err = 0.0
    err_n = 0.0
    n_trials = len(packed)

    for i, p in enumerate(packed):
        t, x, y, session, trial_id = unpack_eye(eye_data, i)
        if p is None:
            n = min(t.size, x.size, y.size)
            times.append(t[:n].astype(np.float32))
            xs.append(np.full(n, np.nan, dtype=np.float32))
            ys.append(np.full(n, np.nan, dtype=np.float32))
            recon_xs.append(np.full(n, np.nan, dtype=np.float32))
            recon_ys.append(np.full(n, np.nan, dtype=np.float32))
            state_ids.append(np.full(n, -1, dtype=np.int16))
            valids.append(np.zeros(n, dtype=bool))
            artifacts.append(np.ones(n, dtype=np.float32))
            sessions.append(session)
            trial_ids.append(trial_id)
            continue

        state, recon, artifact, valid_orig, xy = trial_inference(p, codebook_xy)
        n = state.size
        t_ms = p["t"][:n] * 1000.0

        times.append(p["t"][:n].astype(np.float32))
        xs.append(p["x"][:n].astype(np.float32))
        ys.append(p["y"][:n].astype(np.float32))
        recon_xs.append(recon[:, 0].astype(np.float32))
        recon_ys.append(recon[:, 1].astype(np.float32))
        state_ids.append(state)
        valids.append(valid_orig)
        artifacts.append(artifact.astype(np.float32))
        sessions.append(session)
        trial_ids.append(trial_id)
        transition_rows.extend(state_runs(t_ms, state, codebook_xy, session, trial_id))

        assigned = valid_orig & (state >= 0)
        if assigned.any():
            one_hot = np.zeros(k, dtype=np.float64)
            for sid in state[assigned]:
                one_hot[int(sid)] += 1.0
            usage += one_hot
            usage_n += float(assigned.sum())
            err = np.hypot(
                codebook_xy[state[assigned], 0] - xy[assigned, 0],
                codebook_xy[state[assigned], 1] - xy[assigned, 1],
            )
            abs_err += float(err.sum())
            err_n += float(err.size)

        if (i + 1) % 2000 == 0:
            print(f"  inferred {i + 1}/{n_trials}")

    usage = (usage / max(usage_n, 1.0)).astype(np.float32)
    mae = np.float32(abs_err / max(err_n, 1.0))
    print(f"Assigned-sample MAE: {mae:.3f} maze units (radius={ASSIGN_RADIUS:g})")

    if transition_rows:
        transitions = {
            key: np.asarray([r[key] for r in transition_rows]) for key in transition_rows[0]
        }
    else:
        transitions = {
            "state": np.asarray([], dtype=np.int16),
            "onset": np.asarray([], dtype=np.float32),
            "offset": np.asarray([], dtype=np.float32),
            "duration": np.asarray([], dtype=np.float32),
            "location_x": np.asarray([], dtype=np.float32),
            "location_y": np.asarray([], dtype=np.float32),
            "session": np.asarray([], dtype=object),
            "trial_indices_all": np.asarray([], dtype=int),
        }

    return {
        "time": np.asarray(times, dtype=object),
        "eye_x": np.asarray(xs, dtype=object),
        "eye_y": np.asarray(ys, dtype=object),
        "recon_x": np.asarray(recon_xs, dtype=object),
        "recon_y": np.asarray(recon_ys, dtype=object),
        "state_id": np.asarray(state_ids, dtype=object),
        "valid": np.asarray(valids, dtype=object),
        "artifact_prob": np.asarray(artifacts, dtype=object),
        "session": np.asarray(sessions),
        "trial_indices_all": np.asarray(trial_ids, dtype=int),
        "codebook_xy": codebook_xy,
        "codebook_k": np.int32(k),
        "codebook_name": codebook_name,
        "codebook_usage": usage,
        "valid_mae": mae,
        "transition_state": transitions["state"].astype(np.int16, copy=False),
        "transition_onset": transitions["onset"].astype(np.float32, copy=False),
        "transition_offset": transitions["offset"].astype(np.float32, copy=False),
        "transition_duration": transitions["duration"].astype(np.float32, copy=False),
        "transition_location_x": transitions["location_x"].astype(np.float32, copy=False),
        "transition_location_y": transitions["location_y"].astype(np.float32, copy=False),
        "transition_session": np.asarray(transitions["session"]),
        "transition_trial_indices_all": transitions["trial_indices_all"].astype(
            int, copy=False
        ),
    }


def fit_codebook_from_trials(packed, k, seed=0):
    pool = codebook_xy_pool(packed, seed=seed)
    codebook_xy, codebook_name = fit_code_xy(pool, k, seed=seed)
    print(f"Fitted k-means codebook (K={k}, n={pool.shape[0]} fixation samples):")
    for j, ((px, py), name) in enumerate(zip(codebook_xy, codebook_name)):
        print(f"  state {j:2d} {name:8s}: ({px:7.2f}, {py:7.2f})")
    return codebook_xy, codebook_name


def build_attractor_eye_data(
    eye_data,
    *,
    monkey="Faure",
    k=DEFAULT_K,
    max_trials=None,
    seed=0,
    behavioral=None,
    events="auto",
):
    if behavioral is None:
        from data.loader import load_eye_behavioral_data

        behavioral = load_eye_behavioral_data(monkey)
    packed, n_trials = pack_trials(
        eye_data,
        behavioral,
        monkey=monkey,
        max_trials=max_trials,
        events=events,
    )
    print(f"Preparing {n_trials} maze-normalized fixation trials (K={k})")
    codebook_xy, codebook_name = fit_codebook_from_trials(packed, k, seed=seed)
    data = infer_all_trials(packed, eye_data, codebook_xy, codebook_name)
    print_codebook_diagnostics(codebook_xy, codebook_name, data["codebook_usage"], k)
    return data


def produce_attractor_eye_data(
    eye_data,
    *,
    monkey="Faure",
    k=DEFAULT_K,
    max_trials=None,
    seed=0,
    behavioral=None,
    events="auto",
):
    return build_attractor_eye_data(
        eye_data,
        monkey=monkey,
        k=k,
        max_trials=max_trials,
        seed=seed,
        behavioral=behavioral,
        events=events,
    )


def save_attractor_eye_data(data, monkey, stem=None):
    path = processed_npz(stem or f"{monkey}_attractor_eye_data")
    path.parent.mkdir(parents=True, exist_ok=True)
    from data.loader import savez_atomic

    savez_atomic(path, **data)
    print(f"Wrote {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monkey", default="Faure", choices=("Faure", "Nielsen"))
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help=f"Run K in {CODEBOOK_SWEEP}",
    )
    args = parser.parse_args()

    from data.loader import load_eye_data

    eye = load_eye_data(args.monkey)
    ks = CODEBOOK_SWEEP if args.sweep else (args.k,)
    for k in ks:
        data = produce_attractor_eye_data(
            eye,
            monkey=args.monkey,
            k=k,
            max_trials=args.max_trials,
            seed=args.seed,
        )
        save_attractor_eye_data(
            data, args.monkey, stem=f"{args.monkey}_attractor_k{k}_eye_data"
        )
        if k == DEFAULT_K:
            save_attractor_eye_data(
                data, args.monkey, stem=f"{args.monkey}_attractor_eye_data"
            )


if __name__ == "__main__":
    main()
