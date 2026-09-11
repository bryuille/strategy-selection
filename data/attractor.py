"""Unit-H warping and fixation-to-cluster assignment.

This module owns steps 2 and 5 of the pipeline: it warps raw degrees into the
unit H that makes every maze comparable, and it snaps a fixation to a cluster
point. It does **not** fit the codebook -- `eye_pre_flash.classifier.features`
does that, on clipped in-window samples, and is the single source of every
codebook in the analysis.

The cache this module writes (`{monkey}_attractor_eye_data.npz`) is therefore
K-independent: warped positions, a validity mask and a timebase, nothing more.
"""

from __future__ import annotations

import argparse

import numpy as np
from pymovements.events.detection import out_of_screen

from data.builder import (
    POSITION_LIMIT_DEG,
    qc_mask,
    trial_blink_mask,
    trial_fixation_mask,
)
from data.config import (
    PRE_FIX_END_MS,
    PRE_FIX_START_MS,
    eye_npz_path,
    processed_npz,
)
from data.convert import load_npz

# Maze plot box. Warped samples outside it are off-maze.
MAZE_SCREEN_LIM = 3.0
# Assign a fixation only when its mean lies within this radius of a prototype --
# the "unit ball" of step 5.
ASSIGN_RADIUS = 1.0
BLINK_PADDING_MS = 50
ARM_EPS = 1e-6
# Physical height of the center vertical stem (fixation target -> junction).
STEM_DEG = 7.0

FIXATION_EVENT = "fixation_idt"

ATTRACTOR_EYE_DATA_KEYS = (
    "time",
    "eye_x",
    "eye_y",
    "valid",
    "session",
    "trial_indices_all",
)


def in_maze_screen(x, y, lim=MAZE_SCREEN_LIM):
    """True where maze `(x, y)` lies in `[-lim, lim] × [-lim, lim]`."""
    x = np.asarray(x)
    y = np.asarray(y)
    return np.isfinite(x) & np.isfinite(y) & (np.abs(x) <= lim) & (np.abs(y) <= lim)


def assign_states_inference(xy, codebook_xy, radius=ASSIGN_RADIUS):
    """Nearest prototype within `radius`, else -1.

    Two steps, in this order: the radius rejects points sitting outside every
    ball, and among the prototypes that do contain the point the nearest one
    wins. The tie-break is not an edge case -- prototypes routinely sit closer
    together than `2 * radius`, so overlapping balls are the normal situation
    and most central fixations are eligible for several at once.
    """
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
    """I-DT fixation `(onset, offset)` spans, earliest first.

    `fixation_idt` by name, not by substring: `data.builder` emits exactly one
    fixation detector now, and naming it here keeps that true if another is ever
    added alongside.
    """
    spans = []
    for name, onset, offset in event_rows or ():
        if str(name).lower() != FIXATION_EVENT:
            continue
        spans.append((float(onset), float(offset)))
    spans.sort()
    return spans


def assign_fixation_states(
    xy, t_ms, valid, event_rows, codebook_xy, radius=ASSIGN_RADIUS
):
    """Assign each fixation as a whole to one prototype.

    The representative point is the mean of the fixation's valid warped samples
    -- "based on its average", step 5. That mean is snapped with
    `assign_states_inference`, and every valid sample in the fixation inherits
    the result. Samples outside any fixation (saccades, blinks, gaps) keep -1
    and are dropped from every feature.

    Fixations are assigned independently. Nothing merges them: two consecutive
    fragments landing on the same prototype already collapse into one run
    downstream, because `collapse_state_runs` sees only assigned samples.
    """
    n = len(xy)
    state = np.full(n, -1, dtype=np.int16)
    valid = np.asarray(valid, dtype=bool)
    if n == 0 or not valid.any():
        return state

    t_ms = np.asarray(t_ms)
    for onset, offset in fixation_event_spans(event_rows):
        hit = valid & (t_ms >= onset) & (t_ms <= offset)
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
    # `eye_data` while the assembly below skips it.
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


def assemble_trials(packed, eye_data):
    """Warped positions, validity and timebase, one row per `eye_data` trial."""
    times, xs, ys, valids = [], [], [], []
    sessions, trial_ids = [], []

    for i, p in enumerate(packed):
        t, x, y, session, trial_id = unpack_eye(eye_data, i)
        if p is None:
            n = min(t.size, x.size, y.size)
            times.append(t[:n].astype(np.float32))
            xs.append(np.full(n, np.nan, dtype=np.float32))
            ys.append(np.full(n, np.nan, dtype=np.float32))
            valids.append(np.zeros(n, dtype=bool))
        else:
            n = p["valid_orig"].size
            times.append(p["t"][:n].astype(np.float32))
            xs.append(p["x"][:n].astype(np.float32))
            ys.append(p["y"][:n].astype(np.float32))
            valids.append(np.asarray(p["valid_orig"], dtype=bool))
        sessions.append(session)
        trial_ids.append(trial_id)

    return {
        "time": np.asarray(times, dtype=object),
        "eye_x": np.asarray(xs, dtype=object),
        "eye_y": np.asarray(ys, dtype=object),
        "valid": np.asarray(valids, dtype=object),
        "session": np.asarray(sessions),
        "trial_indices_all": np.asarray(trial_ids, dtype=int),
    }


def build_attractor_eye_data(
    eye_data,
    *,
    monkey="Faure",
    max_trials=None,
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
    print(f"Preparing {n_trials} maze-normalized fixation trials")
    return assemble_trials(packed, eye_data)


def produce_attractor_eye_data(
    eye_data,
    *,
    monkey="Faure",
    max_trials=None,
    behavioral=None,
    events="auto",
):
    return build_attractor_eye_data(
        eye_data,
        monkey=monkey,
        max_trials=max_trials,
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
    parser.add_argument("--max-trials", type=int, default=None)
    args = parser.parse_args()

    from data.loader import load_eye_data

    eye = load_eye_data(args.monkey)
    data = produce_attractor_eye_data(
        eye, monkey=args.monkey, max_trials=args.max_trials
    )
    save_attractor_eye_data(data, args.monkey)


if __name__ == "__main__":
    main()
