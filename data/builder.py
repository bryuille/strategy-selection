import warnings

from collections import Counter

import numpy as np
import polars as pl
import pymovements as pm
from pymovements.events import blink as blink_fn

from data.config import eye_npz_path, npz_dir
from data.convert import load_npz

# Per path-type timing (ms from flash 1). Indices 0-23 map to path_type 1-24,
# so blocks follow published geo_type: maze = ceil(path_type/4).
# (first arm, second arm, stop = first+second+300). Arm times are h/vel at 10 deg/s:
# LU (h1,h2), LD (h1,h3), RU (h4,h5), RD (h4,h6).
PATH_TYPE_LEN_MS = [
    (500, 700, 1500),   # 0: m1, LU
    (500, 1100, 1900),  # 1: m1, LD
    (1000, 900, 2200),  # 2: m1, RU
    (1000, 500, 1800),  # 3: m1, RD
    (500, 900, 1700),   # 4: m2, LU
    (500, 1100, 1900),  # 5: m2, LD
    (1000, 700, 2000),  # 6: m2, RU
    (1000, 500, 1800),  # 7: m2, RD
    (500, 700, 1500),   # 8: m3, LU
    (500, 1100, 1900),  # 9: m3, LD
    (750, 900, 1950),   # 10: m3, RU
    (750, 500, 1550),   # 11: m3, RD
    (500, 900, 1700),   # 12: m4, LU
    (500, 1100, 1900),  # 13: m4, LD
    (750, 700, 1750),   # 14: m4, RU
    (750, 500, 1550),   # 15: m4, RD
    (500, 700, 1500),   # 16: m5, LU
    (500, 1100, 1900),  # 17: m5, LD
    (500, 900, 1700),   # 18: m5, RU
    (500, 500, 1300),   # 19: m5, RD
    (500, 900, 1700),   # 20: m6, LU
    (500, 1100, 1900),  # 21: m6, LD
    (500, 700, 1500),   # 22: m6, RU
    (500, 500, 1300),   # 23: m6, RD
]

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


def build_trial_metadata(data):
    trials = data["trial_indices_all"].astype(int)
    n_trials = np.max(trials)

    path_type = np.full(n_trials, np.nan)
    flash2_ms = np.full(n_trials, np.nan)
    flash3_ms = np.full(n_trials, np.nan)

    for k in range(len(trials)):
        ti = trials[k] - 1
        path_type[ti] = data["path_type"][k]
        flash2_ms[ti] = (data["flash_two"][k] - data["flash_one"][k]) * 1000
        flash3_ms[ti] = (data["flash_three"][k] - data["flash_one"][k]) * 1000

    trial_mask = path_type != -99
    return path_type, flash2_ms, flash3_ms, trial_mask


def build_post_flash_timebins(data):
    trials = data["trial_indices_all"].astype(int)
    nrns = data["nrns"].astype(int)

    n_trials = np.max(trials)
    n_neurons = np.max(nrns)

    geo_present = data["geo_present"]
    flash_three = data["flash_three"]
    fr_raw = data["FR_WH"].T

    start_bins = np.floor((flash_three - geo_present) * 1000).astype(int)
    end_bins = start_bins + 500

    tensor = np.full((n_trials, n_neurons, 500), np.nan)

    for k in range(len(nrns)):
        ti = trials[k] - 1
        ni = nrns[k] - 1
        segment = fr_raw[k, start_bins[k] : end_bins[k]]
        tensor[ti, ni, : len(segment)] = segment

    return tensor


def build_post_flash_metadata(data):
    trials = data["trial_indices_all"].astype(int)
    n_trials = np.max(trials)

    path_type = np.full(n_trials, np.nan)

    for k in range(len(trials)):
        ti = trials[k] - 1
        path_type[ti] = data["path_type"][k]

    trial_mask = path_type != -99
    return path_type, trial_mask


def build_pre_flash_timebins(data):
    trials = data["trial_indices_all"].astype(int)
    nrns = data["nrns"].astype(int)

    n_trials = np.max(trials)
    n_neurons = np.max(nrns)

    geo_present = data["geo_present"]
    flash_one = data["flash_one"]
    fr_raw = data["FR_WH"].T

    start_bins = np.floor((geo_present - geo_present) * 1000).astype(int)
    end_bins = np.floor((flash_one - geo_present) * 1000).astype(int)

    max_length = np.max(end_bins - start_bins)

    tensor = np.full((n_trials, n_neurons, max_length), np.nan)

    for k in range(len(nrns)):
        ti = trials[k] - 1  # matlab indexing correction
        ni = nrns[k] - 1
        segment = fr_raw[k, start_bins[k] : end_bins[k]]
        tensor[ti, ni, : len(segment)] = segment

    return tensor


def build_pre_flash_metadata(data):
    trials = data["trial_indices_all"].astype(int)
    n_trials = np.max(trials)

    path_type = np.full(n_trials, np.nan)
    flash1_ms = np.full(n_trials, np.nan)

    for k in range(len(trials)):
        ti = trials[k] - 1
        path_type[ti] = data["path_type"][k]
        flash1_ms[ti] = (data["flash_one"][k] - data["geo_present"][k]) * 1000

    trial_mask = path_type != -99
    return path_type, flash1_ms, trial_mask


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
# The same predicate `data.labeler.strategy_pc_scores` already applied to the
# *label* pool, and the one the reference requires: the
# `apply_photodiode_qc` argument of `Compute_All_Session_SNR.m` defaults true
# and raises if set false ("Photodiode QC is required for the publication
# analysis"). Factored here so the eye pipeline screens on exactly the same
# three flags rather than each stage keeping its own subset -- which is what it
# did until 2026-09-10, when `classifier.features` and `data.attractor`
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
# Consecutive fixations of the same kind are one fixation when the gap between
# them is too short to hold a saccade (`SACCADE_MIN_DURATION_MS`) or a blink
# (the 20 ms floor passed to `blink_fn`) *and* no saccade event actually spans
# it. I-DT ends a fixation the moment accumulated drift crosses
# `IDT_DISPERSION_THRESHOLD_DEG` and immediately starts another, so a single
# steady fixation comes back as a run of fragments 1 ms apart -- 45.3% of all
# consecutive fixation pairs sit under 2 ms, and only 1.81% of those have a
# saccade between them (measured at the old 2 deg threshold; a wider threshold
# fragments less, so both figures are upper bounds now). Measured the other
# way round: pairs with no intervening saccade have a median gap of 1 ms,
# pairs with one have 42 ms.
# The distribution is cleanly bimodal, so this recovers whole fixations
# without ever bridging a real eye movement.
FIXATION_MERGE_GAP_MS = SACCADE_MIN_DURATION_MS
IVT_VELOCITY_THRESHOLD_DEG_S = 40.0
# Maximum dispersion -- (max x - min x) + (max y - min y) -- I-DT tolerates
# inside one fixation. Doubled from the 2.0 deg this analysis used through
# 2026-09-09: at 2.0 deg a slow drift was split into fragments 1 ms apart that
# `merge_fragmented_fixations` then glued back into a single event, since the
# microsaccade detector flags no saccade across a smooth drift -- so the event
# table reported one long "fixation" spanning gaze that had travelled well
# past 2 deg (june_24_g0 trial 581: 4.13 deg of net drift inside one 1292 ms
# fixation_idt). Widening the threshold makes that drift a fixation by the
# detector's own definition instead of by merge artifact. It also makes I-DT
# more permissive in general: gaze excursions up to 4 deg now stay inside one
# fixation rather than ending it.
IDT_DISPERSION_THRESHOLD_DEG = 4.0
FIXATION_MIN_DURATION_MS = 50
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


def merge_fragmented_fixations(frame, max_gap_ms=FIXATION_MERGE_GAP_MS):
    """Join consecutive same-name fixations that no saccade separates.

    Operates on the finished event frame, in ms, *after* blink-overlapping
    events have been dropped -- so a saccade removed for overlapping a blink
    leaves a saccade-free gap, which the `max_gap_ms` test then refuses to
    bridge (a blink is at least 20 ms, a saccade at least
    `SACCADE_MIN_DURATION_MS`).

    Each fixation name is merged independently; `fixation_idt` and
    `fixation_ivt` fragment differently and must not be mixed.
    """
    if frame is None or frame.height == 0:
        return frame
    sacc = [
        (float(o), float(f))
        for o, f, nm in zip(frame["onset"], frame["offset"], frame["name"])
        if "saccade" in str(nm).lower()
    ]

    def saccade_spans(lo, hi):
        return any(o < hi and f > lo for o, f in sacc)

    keep_rows, out = [], []
    for i, (nm, o, f) in enumerate(zip(frame["name"], frame["onset"], frame["offset"])):
        out.append([str(nm), float(o), float(f), i])
    out.sort(key=lambda r: (r[0], r[1]))

    merged, drop = {}, set()
    prev = None
    for name, o, f, idx in out:
        if (
            prev is not None
            and prev[0] == name
            and "fixation" in name.lower()
            and o - prev[2] < max_gap_ms
            and not saccade_spans(prev[2], o)
        ):
            merged[prev[3]] = max(merged.get(prev[3], prev[2]), f)
            drop.add(idx)
            prev = [name, prev[1], f, prev[3]]
            continue
        prev = [name, o, f, idx]

    if not drop:
        return frame
    idx_col = pl.Series("__i", range(frame.height))
    frame = frame.with_columns(idx_col)
    new_off = [
        merged.get(i, float(frame["offset"][i])) for i in range(frame.height)
    ]
    frame = frame.with_columns(pl.Series("offset", new_off))
    frame = frame.with_columns(
        (pl.col("offset") - pl.col("onset")).alias("duration")
    )
    return frame.filter(~pl.col("__i").is_in(sorted(drop))).drop("__i")


def trial_events(time, eye_x, eye_y, pupil, experiment):
    """Detect saccades and fixations, then drop events that overlap blinks."""
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
            "ivt",
            name="fixation_ivt",
            velocity_threshold=IVT_VELOCITY_THRESHOLD_DEG_S,
            minimum_duration=FIXATION_MIN_DURATION_MS,
        )
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
    return merge_fragmented_fixations(gaze.events.frame)


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

    `start_ms=None` keeps the whole trial, which is what `eye_post_flash`
    needs -- its fixations lie in a post-feedback window that a pre-fixation
    clip would discard entirely.

    Either way the trial must pass `qc_mask`, so no faded or photodiode-bad
    trial has events in this table and none can reach a downstream fit through
    it.
    """
    experiment = pm.Experiment(sampling_rate=EYE_SAMPLING_RATE_HZ)
    windowed = start_ms is not None
    # Behavioral is loaded either way now, because `qc_mask` gates every trial
    # -- not just the windowed ones. A trial with no behavioral row cannot be
    # QC-checked, so it is dropped rather than trusted.
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