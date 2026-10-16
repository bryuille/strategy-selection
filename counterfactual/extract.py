"""Post-choice fixation table and exit sequences, one cache per monkey.

Per trial of the publication sessions (`config.SESSIONS`) that passes
`data.builder.trial_qc_ok`:

1. **Window** ``[saccade_init, trial_end]`` ms from `geo_present` (first
   finite of `WINDOW_START_FIELDS` for the start), clipped to the eye trace.
2. **Detect** saccades and fixations on the clipped samples with
   `data.builder.trial_events`, blink events masked from the pupil trace when
   one is available.
3. **Centroids** in degrees of every detected fixation, from the unwarped
   samples in its span.
4. **States**: nearest exit within `EXIT_RADIUS_DEG` (1-4), origin (0) or
   elsewhere (-1), with the exits placed from the trial's arm lengths.
5. **Sequence**: the decoded chosen exit, then the exit-state fixations in
   order (`counterfactual.exits.exit_sequence`). The first fixation in the
   window is where the choice saccade landed; when it is on the chosen exit
   (`landing_on_chosen`, the sanity check of the decoding) item 0 already
   stands for it and the post-choice items start with the next fixation.

The dendrogram strategy label (0 = H, 1 = S) is attached when the session's
``<session>_strategy_choices.npz`` cache exists; it is read, never built here.

Cache: ``<PROCESSED_ROOT>/<Monkey>_cf_exits.npz``, rebuilt when the schema or
session list differs.

    uv run python -m counterfactual.extract [--monkey ...] [--refresh]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

from data.builder import (
    EYE_SAMPLING_RATE_HZ,
    FIXATION_EVENT,
    behavioral_lookup,
    pupil_for_trial,
    trial_events,
    trial_qc_ok,
    unpack_eye,
)
from data.config import MAT_ROOT, processed_npz
from data.loader import load_eye_behavioral_data, load_eye_data, savez_atomic
from data.mat import load_npz
from counterfactual.config import (
    MIN_WINDOW_SAMPLES,
    MONKEYS,
    PUPIL_DIR,
    SCHEMA,
    SESSIONS,
    WINDOW_END_FIELD,
    WINDOW_START_FIELDS,
    cache_path,
)
from counterfactual.exits import assign_states, chosen_exit, correct_exit, exit_sequence

DROP_REASONS = ("qc", "no_chosen_exit", "no_window", "no_events")


# ---- helpers ------------------------------------------------------------------


def rel_ms(behavioral, field, i):
    """`field` of row `i` in ms from `geo_present`; NaN when absent."""
    if field not in behavioral:
        return np.nan
    v = float(behavioral[field][i])
    g = float(behavioral["geo_present"][i])
    return (v - g) * 1000.0 if np.isfinite(v) and np.isfinite(g) else np.nan


def window_ms(behavioral, i, t_ms):
    """``(lo, hi)`` of the post-choice window, clipped to the trace; None when
    no start field is finite or the window is empty."""
    lo = next((v for v in (rel_ms(behavioral, f, i) for f in WINDOW_START_FIELDS) if np.isfinite(v)), np.nan)
    if not np.isfinite(lo):
        return None
    hi = rel_ms(behavioral, WINDOW_END_FIELD, i)
    finite = t_ms[np.isfinite(t_ms)]
    if finite.size == 0:
        return None
    hi = float(finite.max()) if not np.isfinite(hi) else min(hi, float(finite.max()))
    return (lo, hi) if hi > lo else None


def post_choice_fixations(t_s, x, y, pupil, lo, hi, *, experiment):
    """``(onset_ms, offset_ms, cx, cy)`` per fixation detected on the samples
    inside ``[lo, hi]`` (so every event lies in the window); None when unusable."""
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
    if frame is None:
        return []
    t_ms = np.round(t_s[keep] * 1000.0)
    xk, yk = x[keep], y[keep]
    rows = []
    for name, onset, offset in frame.select("name", "onset", "offset").rows():
        if str(name).lower() != FIXATION_EVENT:
            continue
        hit = (t_ms >= float(onset)) & (t_ms <= float(offset)) & np.isfinite(xk) & np.isfinite(yk)
        if not hit.any():
            continue
        rows.append((float(onset), float(offset), float(xk[hit].mean()), float(yk[hit].mean())))
    rows.sort()
    return rows


def session_pupils(monkey, session, mat_cache):
    path = PUPIL_DIR / f"{session}_pupil.npz"
    if path.exists():
        with np.load(path, allow_pickle=True) as f:
            return list(f["pupil"]), "table"
    if MAT_ROOT is not None:
        pupil_for_trial(mat_cache, monkey, session, 1)
        if mat_cache.get(session) is not None:
            return mat_cache[session], "mat"
    warnings.warn(
        f"{session}: no pupil table under {PUPIL_DIR} and no eye mat; blink masking is OFF",
        stacklevel=2,
    )
    return None, "none"


def strategy_labels(session):
    """``trial_id -> 0/1`` from the session's dendrogram label cache, or {}."""
    path = processed_npz(f"{session}_strategy_choices")
    if not path.exists():
        return {}
    lab = np.asarray(load_npz(path)["strategy_choices"], dtype=float)
    return {i + 1: float(v) for i, v in enumerate(lab) if np.isfinite(v)}


# ---- extraction ----------------------------------------------------------------


def extract(monkey, sessions):
    import pymovements as pm

    sessions = tuple(sessions)
    eye = load_eye_data(monkey)
    behavioral = load_eye_behavioral_data(monkey)
    beh = behavioral_lookup(behavioral)
    experiment = pm.Experiment(sampling_rate=EYE_SAMPLING_RATE_HZ)
    pupils, source, mat_cache, labels = {}, {}, {}, {}
    for s in sessions:
        pupils[s], source[s] = session_pupils(monkey, s, mat_cache)
        labels[s] = strategy_labels(s)

    tr = {k: [] for k in ("session", "trial", "maze", "h", "chosen", "correct", "strategy",
                          "lo", "hi", "n_fix", "landing", "exit_seq", "state_seq")}
    fx = {k: [] for k in ("row", "xy", "on", "off", "state")}
    dropped = {s: dict.fromkeys(DROP_REASONS, 0) for s in sessions}
    kept = dict.fromkeys(sessions, 0)
    for i in range(len(eye["session"])):
        session = str(eye["session"][i])
        if session not in sessions:
            continue
        t_s, x, y, _, trial_id = unpack_eye(eye, i)
        beh_i = beh.get((session, trial_id))
        if not trial_qc_ok(behavioral, beh_i):
            dropped[session]["qc"] += 1
            continue
        flags = tuple(float(behavioral[f"trial_answer{k}"][beh_i]) for k in range(1, 5))
        chosen = chosen_exit(flags, float(behavioral["LR"][beh_i]), float(behavioral["LR2"][beh_i]))
        if chosen is None:
            dropped[session]["no_chosen_exit"] += 1
            continue
        win = window_ms(behavioral, beh_i, np.round(t_s * 1000.0))
        if win is None:
            dropped[session]["no_window"] += 1
            continue
        lo, hi = win
        sp = pupils[session]
        pupil = sp[trial_id - 1] if sp is not None and trial_id - 1 < len(sp) else np.empty((0, 2))
        fixations = post_choice_fixations(t_s, x, y, pupil, lo, hi, experiment=experiment)
        if fixations is None:
            dropped[session]["no_events"] += 1
            continue
        h = tuple(float(behavioral[f"h{k}"][beh_i]) for k in range(1, 7))
        xy = np.array([[cx, cy] for _, _, cx, cy in fixations], dtype=float).reshape(-1, 2)
        states = assign_states(xy, h)
        row = len(tr["session"])
        tr["session"].append(session)
        tr["trial"].append(trial_id)
        tr["maze"].append(int(behavioral["geo_type"][beh_i]))
        tr["h"].append(h)
        tr["chosen"].append(chosen)
        tr["correct"].append(correct_exit(float(behavioral["path_type"][beh_i])) or -1)
        tr["strategy"].append(labels[session].get(trial_id, np.nan))
        tr["lo"].append(lo)
        tr["hi"].append(hi)
        tr["n_fix"].append(len(fixations))
        # The first fixation is the landing of the choice saccade; when it sits
        # on the decoded chosen exit it is the choice itself, represented by
        # item 0 of the list, and the post-choice items start after it.
        landing = bool(states.size) and int(states[0]) == chosen
        tr["landing"].append(landing)
        tr["exit_seq"].append(exit_sequence(chosen, states[1:] if landing else states))
        tr["state_seq"].append([int(s) for s in states])
        for (on, off, _, _), s, p in zip(fixations, states, xy):
            fx["row"].append(row)
            fx["xy"].append(p)
            fx["on"].append(on)
            fx["off"].append(off)
            fx["state"].append(int(s))
        kept[session] += 1

    for s in sessions:
        print(f"  {monkey} {s}: kept={kept[s]} dropped={dropped[s]} pupil={source[s]} "
              f"labels={len(labels[s])}")
    return {
        "session": np.asarray(tr["session"], dtype=str),
        "trial_indices_all": np.asarray(tr["trial"], dtype=int),
        "maze_id": np.asarray(tr["maze"], dtype=int),
        "h": np.asarray(tr["h"], dtype=np.float32).reshape(-1, 6),
        "chosen_exit": np.asarray(tr["chosen"], dtype=int),
        "correct_exit": np.asarray(tr["correct"], dtype=int),
        "strategy": np.asarray(tr["strategy"], dtype=float),
        "window_lo_ms": np.asarray(tr["lo"], dtype=np.float32),
        "window_hi_ms": np.asarray(tr["hi"], dtype=np.float32),
        "n_fix": np.asarray(tr["n_fix"], dtype=int),
        "landing_on_chosen": np.asarray(tr["landing"], dtype=bool),
        "exit_seq": np.asarray(tr["exit_seq"] + [None], dtype=object)[:-1],
        "state_seq": np.asarray(tr["state_seq"] + [None], dtype=object)[:-1],
        "fix_row": np.asarray(fx["row"], dtype=np.int64),
        "fix_xy_d": np.asarray(fx["xy"], dtype=np.float32).reshape(-1, 2),
        "fix_onset_ms": np.asarray(fx["on"], dtype=np.float32),
        "fix_offset_ms": np.asarray(fx["off"], dtype=np.float32),
        "fix_state": np.asarray(fx["state"], dtype=np.int16),
        "dropped": np.asarray([[dropped[s][r] for r in DROP_REASONS] for s in sessions], dtype=int),
        "drop_reasons": np.asarray(DROP_REASONS, dtype=str),
        "sessions": np.asarray(sessions, dtype=str),
        "pupil_source": np.asarray([source[s] for s in sessions], dtype=str),
        "schema": np.int64(SCHEMA),
    }


# ---- cache -----------------------------------------------------------------------


def cache_ok(data, sessions):
    try:
        return (
            int(data["schema"]) == SCHEMA
            and tuple(np.asarray(data["sessions"]).astype(str)) == tuple(sessions)
        )
    except KeyError:
        return False


def load(monkey, *, sessions=None, refresh=False):
    """The post-choice table for `monkey`; built (needs pymovements and the
    eye caches) when missing, stale or `refresh`."""
    sessions = tuple(SESSIONS[monkey] if sessions is None else sessions)
    path = cache_path(monkey)
    if path.exists() and not refresh:
        data = load_npz(path)
        if cache_ok(data, sessions):
            return data
        print(f"  {path.name}: schema / sessions differ; rebuilding")
    print(f"Extracting {monkey} post-choice exits ({', '.join(sessions)}) ...", flush=True)
    data = extract(monkey, sessions)
    savez_atomic(path, **data)
    print(f"Saved {path} ({len(data['session'])} trials, {len(data['fix_row'])} fixations)")
    return data


def main():
    parser = argparse.ArgumentParser(description="Build the post-choice exit cache(s).")
    parser.add_argument("--monkey", nargs="*", default=list(MONKEYS), choices=list(MONKEYS))
    parser.add_argument("--session", nargs="*", help="subset of the monkey's publication sessions")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    for monkey in args.monkey:
        sessions = SESSIONS[monkey]
        if args.session:
            sessions = tuple(s for s in sessions if s in args.session)
        data = load(monkey, sessions=sessions, refresh=args.refresh)
        lengths = np.array([len(q) for q in data["exit_seq"]])
        print(f"{monkey}: {len(lengths)} trials; exit-list length median {np.median(lengths):.0f}, "
              f"max {lengths.max() if lengths.size else 0}; "
              f"{int((lengths > 1).sum())} trials with a post-choice exit fixation")


if __name__ == "__main__":
    main()
