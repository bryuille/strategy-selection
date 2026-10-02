"""Derived arrays: neural trial tensors, eye-data assembly and QC, event
detection, the unit-H warp, and the behavioral-table / state-run helpers the
feature extractors share. Nothing here touches disk caches: every builder
returns arrays, and `data.loader` owns the save-and-load of each one.
"""

from __future__ import annotations

import warnings
from collections import Counter

import numpy as np
import polars as pl
import pymovements as pm
from pymovements.events import blink as blink_fn
from pymovements.events.detection import out_of_screen

from data.config import PRE_FIX_END_MS, PRE_FIX_START_MS, eye_npz_path, npz_dir
from data.convert import load_npz

# Per path-type timing (ms from flash 1). Indices 0-23 map to path_type 1-24,
# so blocks follow published geo_type: maze = ceil(path_type/4).
# (first arm, second arm, stop = first+second+300). Arm times are h/vel at 10 deg/s:
# LU (h1,h2), LD (h1,h3), RU (h4,h5), RD (h4,h6).
########### Decoder Data

def build_trial_timebins(data):
    trials = data["trial_indices_all"].astype(int)
    nrns = data["nrns"].astype(int)

    n_trials = np.max(trials)
    n_neurons = np.max(nrns)

    geo_present = data["geo_present"]
    flash_one = data["flash_one"]
    flash_three = data["flash_three"]
    fr_raw = data["FR_WH"].T

    start_bins = np.floor((flash_one - geo_present) * 1000).astype(int)
    end_bins = np.floor((flash_three - geo_present) * 1000).astype(int) + 300

    max_length = np.max(end_bins - start_bins)

    tensor = np.full((n_trials, n_neurons, max_length), np.nan)

    for k in range(len(nrns)):
        ti = trials[k] - 1  # matlab indexing correction
        ni = nrns[k] - 1
        segment = fr_raw[k, start_bins[k] : end_bins[k]]
        tensor[ti, ni, : len(segment)] = segment

    return tensor


########### Eye Data

EYE_BEHAVIORAL_FIELDS = (
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "trial_fade",
    "path_type",
    "geo_present",
    "fix_start",
    "flash_one",
    "flash_two",
    "flash_three",
    "photodiode_qc_bad",
    "vel",
    "trial_answer1",
    "trial_answer2",
    "trial_answer3",
    "trial_answer4",
    "LR",
    "LR2",
    "geo_type",
    "fixation_off",
)

# Post-flash event fields. Older behavioral npz predate their addition to
# `data.convert.BEHAVIORAL_FIELDS`; sessions converted before then contribute
# NaN, and the post-flash analyses tell the user to re-convert.
# Per-trial optional fields. `feedback_time` and `reward` are deliberately
# NOT here: in the raw struct they are (1, 1) scalars -- session-level protocol
# parameters in ms, constant across all 61 sessions (feedback_time = 1500,
# reward = 250) -- not per-trial timestamps like `fix_start` or `answer_time`,
# which are (1, n_trials). Listing `feedback_time` here produced an all-NaN
# column at best, and `collapse_to_trials` raised IndexError on its 0-d array
# the moment a converted npz actually contained it.
EYE_BEHAVIORAL_OPTIONAL_FIELDS = (
    "answer_time",
    "fixation_cue_present",
    "saccade_init",
    "trial_end",
)

# ---- The analysis pool -------------------------------------------------
# Three flags, screened wherever a trial enters a pooled fit, a cache or a
# figure -- not only at plot time. The distinction matters: a k-means codebook
# or a PCA fitted on excluded trials contaminates every trial that later reads
# it, so filtering at the point of *display* is too late.
#
#   `path_type != -99`        the trial has a real condition
#   `trial_fade == 0`         the maze display did not fade mid-trial
#   `photodiode_qc_bad == 0`  flash timing within one 120-Hz frame of expected
#                             (`DATA_DICTIONARY.md`)
#
# The same predicate `data.labels.strategy_pc_scores` already applied to the
# *label* pool, and the one the reference requires: the
# `apply_photodiode_qc` argument of `Compute_All_Session_SNR.m` defaults true
# and raises if set false ("Photodiode QC is required for the publication
# analysis"). Factored here so the eye pipeline screens on exactly the same
# three flags rather than each stage keeping its own subset -- which is what it
# did until 2026-09-10, when the (now archived) `classifier.features` and `data.builder`
# screened only `path_type` and so fitted their codebooks on faded and
# photodiode-bad trials.
#
# Verified present and finite in all 61 converted behavioral npz. A non-finite
# flag rejects the trial: "unknown QC" is not "QC passed".
QC_FIELDS = ("path_type", "trial_fade", "photodiode_qc_bad")


def qc_mask(behavioral):
    """Boolean row mask over a trial-level behavioral table."""
    path = np.asarray(behavioral["path_type"], dtype=float)
    fade = np.asarray(behavioral["trial_fade"], dtype=float)
    bad = np.asarray(behavioral["photodiode_qc_bad"], dtype=float)
    return (
        np.isfinite(path)
        & (path != -99)
        & np.isfinite(fade)
        & (fade == 0)
        & np.isfinite(bad)
        & (bad == 0)
    )


def trial_qc_ok(behavioral, i):
    """`qc_mask` for one row, for the per-trial loops that carry a row index."""
    if i is None:
        return False
    path = float(behavioral["path_type"][i])
    fade = float(behavioral["trial_fade"][i])
    bad = float(behavioral["photodiode_qc_bad"][i])
    return (
        np.isfinite(path)
        and path != -99
        and np.isfinite(fade)
        and fade == 0
        and np.isfinite(bad)
        and bad == 0
    )


EYE_SAMPLING_RATE_HZ = 1000.0
POSITION_LIMIT_DEG = 30.0
BLINK_PADDING_MS = 5
SACCADE_MIN_DURATION_MS = 12
SACCADE_THRESHOLD_FACTOR = 8
SACCADE_MERGE_GAP_MS = 40
# Maximum dispersion -- (max x - min x) + (max y - min y) -- I-DT tolerates
# inside one fixation. Fragments are kept exactly as detected: a slow drift that
# crosses the threshold ends one fixation and begins the next, and each fragment
# is assigned to a cluster on its own. Nothing stitches them back together, so a
# reported fixation never spans more gaze travel than this.
#
# An earlier version merged consecutive fragments separated by a sub-saccade
# gap, which let a chain of them report as one fixation: june_24_g0 trial 308
# came back as a single 1387 ms "fixation" covering 21.2 deg of drift, carrying
# the dispersion of only its first fragment because merging never recomputed the
# event properties. Assigning fragments independently removes that failure mode.
IDT_DISPERSION_THRESHOLD_DEG = 2.0
FIXATION_MIN_DURATION_MS = 100
CLEAN_EVENT_PROPERTIES = ("peak_velocity", "amplitude", "dispersion", "location")
CLEAN_EYE_DATA_KEYS = (
    "name",
    "onset",
    "offset",
    "duration",
    "peak_velocity",
    "amplitude",
    "dispersion",
    "location_x",
    "location_y",
    "session",
    "trial_indices_all",
)


def gaze_arrays(trial):
    g = np.asarray(trial, dtype=float)
    if g.ndim == 1:
        g = g.reshape(1, -1) if g.size else np.empty((0, 3))
    if g.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty
    return g[:, 2].copy(), g[:, 0].copy(), g[:, 1].copy()


def collapse_to_trials(session_data, fields):
    """One row per trial_indices_all value (neuron–trial rows collapsed)."""
    trials = np.asarray(session_data["trial_indices_all"]).astype(int)
    n_trials = int(np.max(trials))
    row = np.full(n_trials, -1, dtype=int)
    row[trials - 1] = np.arange(len(trials))
    present = row >= 0
    out = {}
    for field in fields:
        vals = np.asarray(session_data[field])
        collapsed = np.full(n_trials, np.nan, dtype=np.float64)
        # A field the recording never populated comes back from MATLAB as a
        # 0-d empty rather than a per-trial vector (`feedback_time` is like
        # this in all 61 sessions). That is "no data for this field", not an
        # error, so it stays NaN -- the same contract
        # `EYE_BEHAVIORAL_OPTIONAL_FIELDS` already documents for fields a
        # session's npz predates. Indexing it would raise IndexError, which is
        # what used to happen the moment such a field appeared in a converted
        # npz at all.
        if vals.ndim >= 1 and vals.shape[0] == trials.size:
            collapsed[present] = vals[row[present]].astype(np.float64)
        out[field] = collapsed
    return out, n_trials


def build_eye_data(monkey="Faure"):
    """Concatenate all eye npz sessions for `monkey`."""
    times, xs, ys, sessions, trial_ids = [], [], [], [], []
    for path in sorted(npz_dir("eye", monkey).glob("Eye_Data_*.npz")):
        raw = load_npz(path)
        session = path.stem.removeprefix("Eye_Data_")
        t_trials, x_trials, y_trials = [], [], []
        for trial in raw["gaze"]:
            t, x, y = gaze_arrays(trial)
            t_trials.append(t)
            x_trials.append(x)
            y_trials.append(y)
        n = len(t_trials)
        times.extend(t_trials)
        xs.extend(x_trials)
        ys.extend(y_trials)
        sessions.extend([session] * n)
        trial_ids.extend(range(1, n + 1))
    return {
        "time": np.asarray(times, dtype=object),
        "eye_x": np.asarray(xs, dtype=object),
        "eye_y": np.asarray(ys, dtype=object),
        "session": np.asarray(sessions),
        "trial_indices_all": np.asarray(trial_ids, dtype=int),
    }


def build_eye_behavioral_data(monkey="Faure"):
    """Concatenate trial-level behavioral fields across all sessions for `monkey`."""
    fields = EYE_BEHAVIORAL_FIELDS + EYE_BEHAVIORAL_OPTIONAL_FIELDS
    chunks = {f: [] for f in fields}
    sessions = []
    trial_ids = []
    for path in sorted(npz_dir("behavioral", monkey).glob("*_good_trials_concat.npz")):
        raw = load_npz(path)
        collapsed, n_trials = collapse_to_trials(raw, [f for f in fields if f in raw])
        for field in fields:
            chunks[field].append(
                collapsed.get(field, np.full(n_trials, np.nan))
            )
        sessions.append(np.full(n_trials, path.name.removesuffix("_good_trials_concat.npz")))
        trial_ids.append(np.arange(1, n_trials + 1, dtype=int))
    return {
        "session": np.concatenate(sessions),
        "trial_indices_all": np.concatenate(trial_ids),
        **{f: np.concatenate(chunks[f]) for f in fields},
    }


def empty_clean_events():
    return {
        "name": np.array([], dtype=object),
        "onset": np.array([], dtype=float),
        "offset": np.array([], dtype=float),
        "duration": np.array([], dtype=float),
        "peak_velocity": np.array([], dtype=float),
        "amplitude": np.array([], dtype=float),
        "dispersion": np.array([], dtype=float),
        "location_x": np.array([], dtype=float),
        "location_y": np.array([], dtype=float),
        "session": np.array([], dtype=object),
        "trial_indices_all": np.array([], dtype=int),
    }


def pupil_signal(pupil):
    p = np.asarray(pupil, dtype=float)
    if p.ndim == 2 and min(p.shape) == 2:
        if p.shape[1] != 2:
            p = p.T
        return p[:, 0], np.round(p[:, 1] * 1000.0).astype(np.int64)
    return None, None


def trial_blink_mask(time, pupil, padding_ms=BLINK_PADDING_MS):
    """True where gaze time falls inside a pymovements pupil-blink window."""
    t = np.asarray(time, dtype=float)
    mask = np.zeros(t.size, dtype=bool)
    p_size, p_t_ms = pupil_signal(pupil)
    if p_size is None or not np.isfinite(p_size).any():
        return mask
    blinks = blink_fn(
        pupil=p_size,
        timesteps=p_t_ms,
        minimum_duration=20,
        maximum_duration=None,
        minimum_gap=10,
    )
    if blinks.frame.height == 0:
        return mask
    t_ms = np.round(t * 1000.0).astype(np.int64)
    for onset, offset in blinks.frame.select("onset", "offset").rows():
        start = onset - padding_ms
        stop = offset + padding_ms
        mask |= (t_ms >= start) & (t_ms <= stop)
    return mask


def drop_blink_overlaps(events, blinks):
    if blinks.frame.height == 0 or events.frame.height == 0:
        return events
    kept = events.frame
    for blink_onset, blink_offset in blinks.frame.select("onset", "offset").rows():
        start = blink_onset - BLINK_PADDING_MS
        stop = blink_offset + BLINK_PADDING_MS
        kept = kept.filter(~((pl.col("onset") < stop) & (pl.col("offset") > start)))
    events.frame = kept
    return events


# Trials where microsaccade detection could not estimate a threshold. Module
# level so a whole `clean_eye_data` pass can report one tally at the end.
SACCADE_SKIPS = Counter()


def trial_events(time, eye_x, eye_y, pupil, experiment):
    """Detect saccades and I-DT fixations, then drop events that overlap blinks."""
    t = np.asarray(time, dtype=float)
    x = np.asarray(eye_x, dtype=float)
    y = np.asarray(eye_y, dtype=float)
    n = min(t.size, x.size, y.size)
    if n < 6:
        return None

    t = t[:n]
    x = x[:n].copy()
    y = y[:n].copy()
    oob = (
        ~np.isfinite(x)
        | ~np.isfinite(y)
        | (np.abs(x) > POSITION_LIMIT_DEG)
        | (np.abs(y) > POSITION_LIMIT_DEG)
    )
    x[oob] = np.nan
    y[oob] = np.nan
    if not np.isfinite(x).any():
        return None

    t_ms = np.round(t * 1000.0).astype(np.int64)
    gaze = pm.Gaze(
        samples=pl.DataFrame({
            "time": np.arange(n, dtype=np.int64),
            "x": x,
            "y": y,
        }),
        experiment=experiment,
        time_column="time",
        time_unit="ms",
        position_columns=["x", "y"],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        gaze.pos2vel("smooth")
        gaze.detect(
            "idt",
            name="fixation_idt",
            dispersion_threshold=IDT_DISPERSION_THRESHOLD_DEG,
            minimum_duration=FIXATION_MIN_DURATION_MS,
        )
        # The microsaccade detector estimates its threshold from velocity
        # variance and raises if either component has none. On a whole trial
        # that never happens; on a clipped window it does -- gaze held still in
        # one axis for the whole segment gives an exactly-zero y threshold.
        # Fixations are unaffected and are what the pipeline consumes, so a
        # degenerate segment yields no saccade events for that trial rather
        # than killing the run. Counted, not swallowed: `trial_events` reports
        # the tally through `SACCADE_SKIPS` so it cannot go unnoticed.
        try:
            gaze.detect(
                "microsaccades",
                name="saccade",
                minimum_duration=SACCADE_MIN_DURATION_MS,
                threshold_factor=SACCADE_THRESHOLD_FACTOR,
            )
        except ValueError as exc:
            if "min_threshold" not in str(exc):
                raise
            SACCADE_SKIPS["degenerate_threshold"] += 1
        if gaze.events.frame.filter(pl.col("name") == "saccade").height:
            gaze.events.merge_subsequent_close_events(
                name="saccade", max_gap=SACCADE_MERGE_GAP_MS
            )

        if len(gaze.events):
            gaze.compute_event_properties(list(CLEAN_EVENT_PROPERTIES))
            onset_i = np.clip(gaze.events.frame["onset"].to_numpy().astype(int), 0, n - 1)
            offset_i = np.clip(gaze.events.frame["offset"].to_numpy().astype(int), 0, n - 1)
            gaze.events.frame = gaze.events.frame.with_columns(
                pl.Series("onset", t_ms[onset_i]),
                pl.Series("offset", t_ms[offset_i]),
                pl.Series("duration", t_ms[offset_i] - t_ms[onset_i]),
            )

        p_size, p_t_ms = pupil_signal(pupil)
        if p_size is not None and np.isfinite(p_size).any():
            blinks = blink_fn(
                pupil=p_size,
                timesteps=p_t_ms,
                minimum_duration=20,
                maximum_duration=None,
                minimum_gap=10,
            )
            gaze.events = drop_blink_overlaps(gaze.events, blinks)

    gaze.events.frame = gaze.events.frame.filter(
        pl.col("name").str.contains("saccade") | pl.col("name").str.contains("fixation")
    )
    if len(gaze.events) == 0:
        return None
    return gaze.events.frame


def trial_fixation_mask(time, eye_x, eye_y, pupil, experiment=None):
    """True on samples inside a fixation and outside saccades/blinks."""
    t = np.asarray(time, dtype=float)
    x = np.asarray(eye_x, dtype=float)
    y = np.asarray(eye_y, dtype=float)
    n = min(t.size, x.size, y.size)
    mask = np.zeros(n, dtype=bool)
    if n < 6:
        return mask
    if experiment is None:
        experiment = pm.Experiment(sampling_rate=EYE_SAMPLING_RATE_HZ)
    frame = trial_events(t[:n], x[:n], y[:n], pupil, experiment)
    if frame is None:
        return mask
    t_ms = np.round(t[:n] * 1000.0).astype(np.int64)
    in_fix = np.zeros(n, dtype=bool)
    in_sac = np.zeros(n, dtype=bool)
    for name, onset, offset in frame.select("name", "onset", "offset").rows():
        hit = (t_ms >= float(onset)) & (t_ms <= float(offset))
        name = str(name).lower()
        if "saccade" in name:
            in_sac |= hit
        elif "fixation" in name:
            in_fix |= hit
    return in_fix & ~in_sac


def clean_eye_data(eye_data, monkey="Faure", *, start_ms=None, end_ms=None):
    """Saccade and fixation events; blink-overlapping events are dropped.

    With `start_ms` given, each trial's samples are clipped to
    ``[fix_start - start_ms, fix_start - end_ms]`` *before* detection, so the
    detectors run on the analysis window rather than the whole trial. That is
    not a shortcut for the same answer -- I-DT is a greedy sequential scan, so
    restarting it at the window edge re-partitions the whole segment. Measured
    on 400 trials, detect-then-clip and clip-then-detect agree on only 7.8% of
    trials, so the two orders are genuinely different analyses and the window
    has to be applied here to mean anything.

    It is also ~5x cheaper: the window is 1466 ms against a median trial span
    of 7322 ms, and detection dominates this stage.

    `start_ms=None` keeps the whole trial, for an analysis whose fixations lie
    outside the pre-fixation window.

    Either way the trial must pass `qc_mask`, so no faded or photodiode-bad
    trial has events in this table and none can reach a downstream fit through
    it.
    """
    experiment = pm.Experiment(sampling_rate=EYE_SAMPLING_RATE_HZ)
    windowed = start_ms is not None
    # Behavioral is always loaded because `qc_mask` gates every trial. A trial
    # with no behavioral row cannot be QC-checked, so it is dropped rather than
    # trusted.
    behavioral = build_eye_behavioral_data(monkey)
    beh_row = {
        (str(s_), int(t_)): j
        for j, (s_, t_) in enumerate(
            zip(behavioral["session"], behavioral["trial_indices_all"])
        )
    }
    qc_ok = qc_mask(behavioral)
    n_qc_dropped = 0
    if windowed:
        beh_fix = np.asarray(behavioral["fix_start"], dtype=float)
        beh_geo = np.asarray(behavioral["geo_present"], dtype=float)
    frames = []
    n_trials = len(eye_data["time"])
    pupils_by_session = {}
    for i in range(n_trials):
        if i and i % 1000 == 0:
            print(f"clean_eye_data: {i}/{n_trials} trials", flush=True)
        session = str(eye_data["session"][i])
        if session not in pupils_by_session:
            raw = load_npz(eye_npz_path(monkey, session))
            pupils_by_session[session] = (
                list(raw["pupil_size"]) if "pupil_size" in raw else None
            )
        trial_id = int(eye_data["trial_indices_all"][i])
        beh_i = beh_row.get((session, trial_id))
        if beh_i is None or not qc_ok[beh_i]:
            n_qc_dropped += 1
            continue
        session_pupils = pupils_by_session[session]
        pupil = (
            session_pupils[trial_id - 1]
            if session_pupils is not None and trial_id - 1 < len(session_pupils)
            else np.empty((0, 2))
        )
        t_arr = np.asarray(eye_data["time"][i], dtype=float)
        x_arr = np.asarray(eye_data["eye_x"][i], dtype=float)
        y_arr = np.asarray(eye_data["eye_y"][i], dtype=float)
        if windowed:
            j = beh_i
            fix_ms = (beh_fix[j] - beh_geo[j]) * 1000.0
            if not np.isfinite(fix_ms) or fix_ms <= 0:
                continue
            lo = (fix_ms - float(start_ms)) / 1000.0
            hi = (fix_ms - float(end_ms)) / 1000.0
            keep = np.isfinite(t_arr) & (t_arr >= lo) & (t_arr <= hi)
            if keep.sum() < 6:
                continue
            t_arr, x_arr, y_arr = t_arr[keep], x_arr[keep], y_arr[keep]
            if pupil is not None and np.asarray(pupil).ndim == 2 and len(pupil):
                pu = np.asarray(pupil, dtype=float)
                pk = np.isfinite(pu[:, 1]) & (pu[:, 1] >= lo) & (pu[:, 1] <= hi)
                pupil = pu[pk] if pk.any() else np.empty((0, 2))
        frame = trial_events(
            t_arr,
            x_arr,
            y_arr,
            pupil,
            experiment,
        )
        if frame is None or frame.height == 0:
            continue
        frames.append(
            frame.with_columns(
                pl.lit(str(eye_data["session"][i])).alias("session"),
                pl.lit(int(eye_data["trial_indices_all"][i])).alias(
                    "trial_indices_all"
                ),
            )
        )

    print(
        f"clean_eye_data: {n_qc_dropped} trial(s) dropped by QC "
        "(path_type/trial_fade/photodiode_qc_bad)",
        flush=True,
    )
    if not frames:
        return empty_clean_events()

    events = pl.concat(frames, how="diagonal_relaxed")
    location = np.array(events["location"].to_list(), dtype=float)
    if location.ndim != 2 or location.shape[1] != 2:
        location = np.full((events.height, 2), np.nan)
    if SACCADE_SKIPS:
        print(
            "clean_eye_data: saccade detection skipped on "
            f"{SACCADE_SKIPS['degenerate_threshold']} trial(s) with no "
            "velocity variance; their fixations are unaffected",
            flush=True,
        )
    return {
        "name": events["name"].to_numpy(),
        "onset": events["onset"].to_numpy().astype(float),
        "offset": events["offset"].to_numpy().astype(float),
        "duration": events["duration"].to_numpy().astype(float),
        "peak_velocity": events["peak_velocity"].to_numpy().astype(float),
        "amplitude": events["amplitude"].to_numpy().astype(float),
        "dispersion": events["dispersion"].to_numpy().astype(float),
        "location_x": location[:, 0],
        "location_y": location[:, 1],
        "session": events["session"].to_numpy(),
        "trial_indices_all": events["trial_indices_all"].to_numpy().astype(int),
    }


########### Behavioral-table and state-run helpers


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


def collapse_state_runs(states):
    """Consecutive identical states collapsed to one entry each.

    Called on the *assigned-only* subsequence, so unassigned samples (saccades,
    blinks, gaze outside every ball) never appear and cannot split a run. That
    is what makes a merge step unnecessary: two consecutive fixation fragments
    that landed on the same cluster are already one run here.
    """
    runs = []
    for state in states:
        sid = int(state)
        if not runs or runs[-1] != sid:
            runs.append(sid)
    return runs


########### Unit-H warp

WARP_BLINK_PADDING_MS = 50
ARM_EPS = 1e-6
# Physical height of the center vertical stem (fixation target -> junction).
STEM_DEG = 7.0

ATTRACTOR_EYE_DATA_KEYS = (
    "time",
    "eye_x",
    "eye_y",
    "valid",
    "session",
    "trial_indices_all",
)


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
    blink = trial_blink_mask(t, pupil, padding_ms=WARP_BLINK_PADDING_MS)
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
