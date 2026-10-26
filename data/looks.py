"""Looks and saccades from the `timestamp_preprocessing` package,
cached per monkey and epoch. This is the eye-event source for every analysis
package; `data.builder.trial_events` (pymovements) is legacy.

`timestamp_preprocessing.current.preprocess_trial` (repo root, never edited
here) takes a trial's raw x / y / pupil arrays with their own timestamps,
matches them, flags invalid samples, breaks segments at gaps, detects saccades
(Engbert threshold, factor 8, 12 ms minimum, 40 ms merge) and emits
*qualified looks*: one candidate interval between saccades / breaks with at
least 75 ms of stable support (maximum pairwise spread <= 2 deg); position is
the time-weighted raw centroid over the stable islands and the counted
duration is the stable support. See timestamp_preprocessing/HANDOFF.md.

Two epoch policies, each its own cache (`<Monkey>_looks_<epoch>.npz`):

``fix``    epoch = maze onset -> `fix_start`; the pre-fixation analyses
           (msp, classifier, msp.saccades).
``trial``  epoch = maze onset -> `trial_end`; post-choice (counterfactual) and
           whole-trial uses.

The saccade threshold is estimated from the samples inside the epoch, so the
two caches are different analyses over the shared span: never slice the
``trial`` cache for pre-fixation work.

Exit / state assignment stays with the consumers (their codebooks on
``look_xy_u`` / ``look_xy_d``); the package's own 2.5 deg centre / left /
right ROI is stored as ``look_roi_adv`` for reference only.

    uv run python -m data.looks --monkey Faure Nielsen --epoch fix trial [--sessions ...]
    uv run python -m data.looks --check                     # synthetic, no data
    uv run python -m data.looks --compare Oct_22_g0         # looks vs legacy fixations
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    from timestamp_preprocessing.core import Config
    from timestamp_preprocessing.current import preprocess_trial
except ModuleNotFoundError:  # run from another cwd: the package is a root-level namespace package
    sys.path.insert(0, str(REPO_ROOT))
    from timestamp_preprocessing.core import Config
    from timestamp_preprocessing.current import preprocess_trial

from data.builder import behavioral_lookup, fix_start_ms, qc_mask, to_maze
from data.config import eye_mat_path, processed_npz
from data.loader import load_eye_behavioral_data, savez_atomic
from data.mat import load_eye_raw, load_npz

LOOKS_SCHEMA = 1
EPOCHS = ("fix", "trial")
PACKAGE_DIR = REPO_ROOT / "timestamp_preprocessing"
PACKAGE_SOURCES = ("core.py", "current.py", "stable_support.py")
# Fixed in the package (HANDOFF.md "Final definitions"); recorded in the cache.
LOOK_MIN_MS = 75.0
LOOK_SPREAD_DEG = 2.0
RAW_ROI_DEG = 2.5
SKIP_REASONS = ("no_eye_cell", "no_pupil", "bad_epoch", "bad_geometry", "preprocess_error")


def source_sha256():
    h = hashlib.sha256()
    for name in PACKAGE_SOURCES:
        h.update((PACKAGE_DIR / name).read_bytes())
    return h.hexdigest()


def config_json():
    cfg = asdict(Config())
    cfg.update(look_min_ms=LOOK_MIN_MS, look_spread_deg=LOOK_SPREAD_DEG, raw_roi_deg=RAW_ROI_DEG)
    return json.dumps(cfg, sort_keys=True)


# ---- epochs -------------------------------------------------------------------


def epoch_end_ms(behavioral, i, epoch):
    """Epoch end in ms from geo_present for behavioral row `i`; NaN if absent."""
    if epoch == "fix":
        return float(fix_start_ms(behavioral, i))
    if epoch == "trial":
        end = float(behavioral["trial_end"][i])
        geo = float(behavioral["geo_present"][i])
        return (end - geo) * 1000.0 if np.isfinite(end) and np.isfinite(geo) else np.nan
    raise ValueError(f"unknown epoch {epoch!r}; known: {EPOCHS}")


# ---- one trial ------------------------------------------------------------------


def trial_looks(xraw, yraw, praw, h, epoch_end):
    """`preprocess_trial` on one trial, flattened to plain records.

    Returns ``(trial, looks, saccades)``: `trial` a dict of scalars, `looks`
    and `saccades` lists of dicts (time in ms from geo_present, positions in
    degrees and unit H). Raises whatever the package raises.
    """
    r = preprocess_trial(np.asarray(xraw), np.asarray(yraw), np.asarray(praw), h, float(epoch_end))
    q = r["quality"]
    seg = np.asarray(r["epoch_segment"])
    t = np.asarray(q["t"], dtype=float)
    trial = dict(
        epoch_end_ms=float(epoch_end),
        saccade_status=str(r["saccade_status"]),
        gap_ms=float(q["gap_ms"]) if np.isfinite(q["gap_ms"]) else np.nan,
        n_samples=int(np.sum(np.isfinite(t) & (t >= 0) & (t <= epoch_end))),
        n_valid=int(np.sum(seg >= 0)),
        n_looks=len(r["looks"]),
        n_saccades=len(r["saccades"]),
    )
    looks = [
        dict(
            onset_ms=float(e["onset_ms"]),
            offset_ms=float(e["offset_ms"]),
            stable_ms=float(e["stable_duration_ms"]),
            cand_start_ms=float(e["candidate_start_ms"]),
            cand_end_ms=float(e["candidate_end_ms"]),
            xy_d=(float(e["raw_x"]), float(e["raw_y"])),
            xy_u=(float(e["normalized_x"]), float(e["normalized_y"])),
            roi_adv=str(e["roi"]),
            ambiguous=bool(e["island_roi_disagreement"]),
            segment=int(e["segment"]),
            witness_deg=float(e["witness_diameter_deg"]),
            islands=[(float(a), float(b)) for a, b in e["stable_intervals"]],
        )
        for e in r["looks"]
    ]
    saccades = [
        dict(
            onset_ms=float(s["onset_ms"]),
            offset_ms=float(s["offset_ms"]),
            duration_ms=float(s["duration_ms"]),
            peak_deg_s=float(s["peak_speed_deg_s"]) if np.isfinite(s["peak_speed_deg_s"]) else np.nan,
            xy_u=(float(s["centroid_unit_x"]), float(s["centroid_unit_y"])),
            segment=int(s["segment"]),
        )
        for s in r["saccades"]
    ]
    return trial, looks, saccades


# ---- cache build ----------------------------------------------------------------


class _Rows:
    """Column accumulators for the long-format cache."""

    def __init__(self):
        self.trial = {k: [] for k in (
            "session", "trial", "maze", "qc_ok", "h", "fix_ms", "epoch_end_ms", "saccade_status",
            "gap_ms", "n_samples", "n_valid", "n_looks", "n_saccades",
            "look_start", "look_stop", "sac_start", "sac_stop")}
        self.look = {k: [] for k in (
            "row", "onset", "offset", "stable", "cand_start", "cand_end", "xy_d", "xy_u",
            "roi_adv", "ambiguous", "segment", "witness", "n_islands")}
        self.island = {k: [] for k in ("look", "start", "end")}
        self.sac = {k: [] for k in ("row", "onset", "offset", "duration", "peak", "xy_u", "segment")}
        self.skip = {k: [] for k in ("session", "trial", "reason")}

    def add_trial(self, session, trial_id, maze, qc_ok, h, fix_ms, tr, looks, sacs):
        row = len(self.trial["session"])
        t = self.trial
        t["session"].append(session)
        t["trial"].append(trial_id)
        t["maze"].append(maze)
        t["qc_ok"].append(qc_ok)
        t["h"].append(h)
        t["fix_ms"].append(fix_ms)
        for k in ("epoch_end_ms", "saccade_status", "gap_ms", "n_samples", "n_valid", "n_looks", "n_saccades"):
            t[k].append(tr[k])
        t["look_start"].append(len(self.look["row"]))
        for e in looks:
            k = len(self.look["row"])
            L = self.look
            L["row"].append(row)
            L["onset"].append(e["onset_ms"])
            L["offset"].append(e["offset_ms"])
            L["stable"].append(e["stable_ms"])
            L["cand_start"].append(e["cand_start_ms"])
            L["cand_end"].append(e["cand_end_ms"])
            L["xy_d"].append(e["xy_d"])
            L["xy_u"].append(e["xy_u"])
            L["roi_adv"].append(e["roi_adv"])
            L["ambiguous"].append(e["ambiguous"])
            L["segment"].append(e["segment"])
            L["witness"].append(e["witness_deg"])
            L["n_islands"].append(len(e["islands"]))
            for a, b in e["islands"]:
                self.island["look"].append(k)
                self.island["start"].append(a)
                self.island["end"].append(b)
        t["look_stop"].append(len(self.look["row"]))
        t["sac_start"].append(len(self.sac["row"]))
        for s in sacs:
            S = self.sac
            S["row"].append(row)
            S["onset"].append(s["onset_ms"])
            S["offset"].append(s["offset_ms"])
            S["duration"].append(s["duration_ms"])
            S["peak"].append(s["peak_deg_s"])
            S["xy_u"].append(s["xy_u"])
            S["segment"].append(s["segment"])
        t["sac_stop"].append(len(self.sac["row"]))

    def add_skip(self, session, trial_id, reason):
        self.skip["session"].append(session)
        self.skip["trial"].append(trial_id)
        self.skip["reason"].append(reason)

    def arrays(self, *, monkey, epoch, sessions):
        t, L, I, S, K = self.trial, self.look, self.island, self.sac, self.skip
        f32 = np.float32
        return {
            "session": np.asarray(t["session"], dtype=str),
            "trial_indices_all": np.asarray(t["trial"], dtype=int),
            "maze_id": np.asarray(t["maze"], dtype=int),
            "qc_ok": np.asarray(t["qc_ok"], dtype=bool),
            "h": np.asarray(t["h"], dtype=f32).reshape(-1, 6),
            "fix_ms": np.asarray(t["fix_ms"], dtype=float),
            "epoch_end_ms": np.asarray(t["epoch_end_ms"], dtype=float),
            "saccade_status": np.asarray(t["saccade_status"], dtype=str),
            "gap_ms": np.asarray(t["gap_ms"], dtype=float),
            "n_samples": np.asarray(t["n_samples"], dtype=int),
            "n_valid": np.asarray(t["n_valid"], dtype=int),
            "n_looks": np.asarray(t["n_looks"], dtype=int),
            "n_saccades": np.asarray(t["n_saccades"], dtype=int),
            "look_start": np.asarray(t["look_start"], dtype=np.int64),
            "look_stop": np.asarray(t["look_stop"], dtype=np.int64),
            "sac_start": np.asarray(t["sac_start"], dtype=np.int64),
            "sac_stop": np.asarray(t["sac_stop"], dtype=np.int64),
            "look_row": np.asarray(L["row"], dtype=np.int64),
            "look_onset_ms": np.asarray(L["onset"], dtype=float),
            "look_offset_ms": np.asarray(L["offset"], dtype=float),
            "look_stable_ms": np.asarray(L["stable"], dtype=float),
            "look_cand_start_ms": np.asarray(L["cand_start"], dtype=float),
            "look_cand_end_ms": np.asarray(L["cand_end"], dtype=float),
            "look_xy_d": np.asarray(L["xy_d"], dtype=f32).reshape(-1, 2),
            "look_xy_u": np.asarray(L["xy_u"], dtype=f32).reshape(-1, 2),
            "look_roi_adv": np.asarray(L["roi_adv"], dtype=str),
            "look_ambiguous": np.asarray(L["ambiguous"], dtype=bool),
            "look_segment": np.asarray(L["segment"], dtype=int),
            "look_witness_deg": np.asarray(L["witness"], dtype=f32),
            "look_n_islands": np.asarray(L["n_islands"], dtype=int),
            "island_look": np.asarray(I["look"], dtype=np.int64),
            "island_start_ms": np.asarray(I["start"], dtype=float),
            "island_end_ms": np.asarray(I["end"], dtype=float),
            "sac_row": np.asarray(S["row"], dtype=np.int64),
            "sac_onset_ms": np.asarray(S["onset"], dtype=float),
            "sac_offset_ms": np.asarray(S["offset"], dtype=float),
            "sac_duration_ms": np.asarray(S["duration"], dtype=float),
            "sac_peak_deg_s": np.asarray(S["peak"], dtype=f32),
            "sac_xy_u": np.asarray(S["xy_u"], dtype=f32).reshape(-1, 2),
            "sac_segment": np.asarray(S["segment"], dtype=int),
            "skip_session": np.asarray(K["session"], dtype=str),
            "skip_trial": np.asarray(K["trial"], dtype=int),
            "skip_reason": np.asarray(K["reason"], dtype=str),
            "schema": np.int64(LOOKS_SCHEMA),
            "epoch": np.asarray(epoch),
            "monkey": np.asarray(monkey),
            "sessions": np.asarray(tuple(sessions), dtype=str),
            "config_json": np.asarray(config_json()),
            "source_sha256": np.asarray(source_sha256()),
        }


def _skip_reason(exc):
    msg = str(exc)
    if "epoch" in msg.lower():
        return "bad_epoch"
    if "geometry" in msg.lower() or "arm lengths" in msg.lower():
        return "bad_geometry"
    return f"preprocess_error:{type(exc).__name__}"


def build_looks(monkey, sessions, epoch, *, verbose=True):
    """Every behavioral row of `sessions` (no QC filter; `qc_ok` is stored)."""
    if epoch not in EPOCHS:
        raise ValueError(f"unknown epoch {epoch!r}; known: {EPOCHS}")
    sessions = tuple(sessions)
    behavioral = load_eye_behavioral_data(monkey)
    qc = qc_mask(behavioral)
    beh_session = behavioral["session"].astype(str)
    rows = _Rows()
    for session in sessions:
        path = eye_mat_path(monkey, session)
        if not path.exists():
            raise FileNotFoundError(f"{monkey} {session}: eye mat missing at {path}")
        raw = load_eye_raw(monkey, session)
        xs, ys, ps = raw["eye_x"], raw["eye_y"], raw["pupil_size"]
        beh_rows = np.flatnonzero(beh_session == session)
        beh_rows = beh_rows[np.argsort(behavioral["trial_indices_all"][beh_rows].astype(int))]
        n_kept = 0
        for i in beh_rows:
            trial_id = int(behavioral["trial_indices_all"][i])
            if trial_id < 1 or trial_id > len(xs) or trial_id > len(ys):
                rows.add_skip(session, trial_id, "no_eye_cell")
                continue
            if ps is None or trial_id > len(ps) or ps[trial_id - 1].shape[0] == 0:
                # An empty pupil array makes every sample `unmatched_pupil`
                # and yields zero looks silently; refuse instead.
                rows.add_skip(session, trial_id, "no_pupil")
                continue
            h = tuple(float(behavioral[f"h{k}"][i]) for k in range(1, 7))
            end = epoch_end_ms(behavioral, i, epoch)
            if not np.isfinite(end) or end <= 0:
                rows.add_skip(session, trial_id, "bad_epoch")
                continue
            if not all(np.isfinite(h)) or min(h) <= 0:
                rows.add_skip(session, trial_id, "bad_geometry")
                continue
            try:
                tr, looks, sacs = trial_looks(xs[trial_id - 1], ys[trial_id - 1], ps[trial_id - 1], h, end)
            except Exception as exc:  # the package raises ValueError / AssertionError per trial
                rows.add_skip(session, trial_id, _skip_reason(exc))
                continue
            rows.add_trial(session, trial_id, int(behavioral["geo_type"][i]), bool(qc[i]), h,
                           float(fix_start_ms(behavioral, i)), tr, looks, sacs)
            n_kept += 1
        if verbose:
            skips = [r for s_, r in zip(rows.skip["session"], rows.skip["reason"]) if s_ == session]
            print(f"  {monkey} {session} [{epoch}]: {n_kept} trials, {len(skips)} skipped"
                  + (f" ({', '.join(sorted(set(skips)))})" if skips else ""), flush=True)
    return rows.arrays(monkey=monkey, epoch=epoch, sessions=sessions)


# ---- cache ----------------------------------------------------------------------


def cache_path(monkey, epoch):
    return processed_npz(f"{monkey}_looks_{epoch}")


def cache_ok(data, sessions):
    try:
        return (
            int(data["schema"]) == LOOKS_SCHEMA
            and str(np.asarray(data["source_sha256"]).reshape(-1)[0]) == source_sha256()
            and str(np.asarray(data["config_json"]).reshape(-1)[0]) == config_json()
            and set(sessions) <= set(np.asarray(data["sessions"]).astype(str).tolist())
        )
    except KeyError:
        return False


def _concat(a, b):
    """Append cache `b` to cache `a`, offsetting the row references."""
    out = {}
    n_tr, n_lk, n_sac = len(a["session"]), len(a["look_row"]), len(a["sac_row"])
    offsets = {"look_row": n_tr, "sac_row": n_tr, "look_start": n_lk, "look_stop": n_lk,
               "sac_start": n_sac, "sac_stop": n_sac, "island_look": n_lk}
    meta = {"schema", "epoch", "monkey", "config_json", "source_sha256"}
    for k in a:
        if k in meta:
            out[k] = a[k]
        elif k == "sessions":
            out[k] = np.asarray(list(a[k].astype(str)) + list(b[k].astype(str)), dtype=str)
        else:
            bb = b[k] + offsets[k] if k in offsets else b[k]
            out[k] = np.concatenate([a[k], bb]) if a[k].ndim else a[k]
    return out


def load_looks(monkey, epoch, sessions=None, *, refresh=False, build=True):
    """The looks cache for `monkey` / `epoch` covering `sessions` (default:
    every session of the monkey with an eye mat under MAT_ROOT). Missing
    sessions are built and appended; a stale cache (schema, package source or
    config changed) is rebuilt for the requested sessions."""
    if sessions is None:
        behavioral = load_eye_behavioral_data(monkey)
        sessions = tuple(s for s in dict.fromkeys(behavioral["session"].astype(str).tolist())
                         if eye_mat_path(monkey, s).exists())
    sessions = tuple(sessions)
    path = cache_path(monkey, epoch)
    data = None
    if path.exists() and not refresh:
        data = load_npz(path)
        if cache_ok(data, sessions):
            return data
        stale = not cache_ok(data, ())
        if stale:
            print(f"  {path.name}: schema / package source / config differ; rebuilding")
            data = None
    missing = sessions if data is None else tuple(
        s for s in sessions if s not in set(np.asarray(data["sessions"]).astype(str).tolist()))
    if not build:
        raise FileNotFoundError(
            f"{path} lacks {', '.join(missing)}; run: uv run python -m data.looks "
            f"--monkey {monkey} --epoch {epoch} --sessions {' '.join(missing)}")
    print(f"Building {monkey} looks [{epoch}] for {', '.join(missing)} ...", flush=True)
    built = build_looks(monkey, missing, epoch)
    data = built if data is None else _concat(data, built)
    savez_atomic(path, **data)
    print(f"Saved {path} ({len(data['session'])} trials, {len(data['look_row'])} looks, "
          f"{len(data['skip_trial'])} skipped)")
    return data


# ---- consumer helpers --------------------------------------------------------------


def trial_index(d):
    """``(session, trial_id) -> row``."""
    return {(str(s), int(t)): i for i, (s, t) in enumerate(zip(d["session"], d["trial_indices_all"]))}


def looks_of(d, row):
    return slice(int(d["look_start"][row]), int(d["look_stop"][row]))


def saccades_of(d, row):
    return slice(int(d["sac_start"][row]), int(d["sac_stop"][row]))


def islands_of(d, look_index):
    """``(starts, ends)`` ms of the stable islands of one look."""
    m = d["island_look"] == look_index
    return d["island_start_ms"][m], d["island_end_ms"][m]


def usable_rows(d, *, min_valid=2, status_ok=True):
    """Rows with enough valid samples and (by default) an estimated saccade
    threshold. HANDOFF.md: with a degenerate threshold the absence of
    saccades is not confirmed absence of movement, so those trials are
    dropped by consumers, with a tally."""
    ok = d["n_valid"] >= min_valid
    if status_ok:
        ok &= d["saccade_status"].astype(str) == "ok"
    return ok


_RAW_CACHE = {}


def sample_quality(monkey, session, trial_id):
    """Whole-trial `timestamp_preprocessing.core.quality` for one trial:
    ``dict(t ms, x, y, pupil, valid, flags, segment)``; None if the trial has
    no eye cell or no pupil. Mats are read once per session."""
    from timestamp_preprocessing.core import quality

    key = (monkey, session)
    if key not in _RAW_CACHE:
        _RAW_CACHE[key] = load_eye_raw(monkey, session)
    raw = _RAW_CACHE[key]
    xs, ys, ps = raw["eye_x"], raw["eye_y"], raw["pupil_size"]
    if trial_id < 1 or trial_id > min(len(xs), len(ys)) or ps is None or trial_id > len(ps):
        return None
    if ps[trial_id - 1].shape[0] == 0:
        return None
    q = quality(np.asarray(xs[trial_id - 1]), np.asarray(ys[trial_id - 1]), np.asarray(ps[trial_id - 1]), Config())
    return {k: q[k] for k in ("t", "x", "y", "pupil", "valid", "flags", "segment")}


# ---- checks ------------------------------------------------------------------------


def _synthetic_trial(n=1500, jump_at=600, jump_to=(5.0, 5.0), ramp_ms=20, jitter=0.03, seed=0):
    """1.5 s at the origin, a 20 ms step to `jump_to`, hold; `jitter` deg of
    tracker noise (a noiseless trace gives a degenerate saccade threshold)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 1000.0  # seconds
    frac = np.clip((np.arange(n) - jump_at) / ramp_ms, 0.0, 1.0)
    x = jump_to[0] * (3 * frac**2 - 2 * frac**3) + rng.normal(0, jitter, n)
    y = jump_to[1] * (3 * frac**2 - 2 * frac**3) + rng.normal(0, jitter, n)
    xraw = np.column_stack([x, t])
    yraw = np.column_stack([y, t])
    praw = np.column_stack([np.full(n, 1000.0), t])
    return xraw, yraw, praw


def check_synthetic():
    """Adapter flattening on synthetic trials; no data needed."""
    h = (5.0, 9.0, 11.0, 5.0, 7.0, 5.0)
    xraw, yraw, praw = _synthetic_trial()
    tr, looks, sacs = trial_looks(xraw, yraw, praw, h, 1500.0)
    assert tr["saccade_status"] == "ok", tr
    assert tr["n_valid"] == 1500 and tr["n_samples"] == 1500, tr
    assert len(looks) == 2, [(e["onset_ms"], e["offset_ms"]) for e in looks]
    assert len(sacs) == 1 and 590 <= sacs[0]["onset_ms"] <= 610, sacs
    a, b = looks
    assert a["onset_ms"] < b["onset_ms"] and a["offset_ms"] < sacs[0]["onset_ms"] < b["onset_ms"]
    assert np.allclose(a["xy_d"], (0.0, 0.0), atol=0.02) and a["roi_adv"] == "center", a
    assert np.allclose(b["xy_d"], (5.0, 5.0), atol=0.02) and b["roi_adv"] == "right", b
    xu, yu = to_maze(np.array([b["xy_d"][0]]), np.array([b["xy_d"][1]]), h)
    assert np.allclose(b["xy_u"], (xu[0], yu[0]), atol=1e-9), (b["xy_u"], xu, yu)
    assert 560 <= a["stable_ms"] <= 600 and len(a["islands"]) == 1, a
    assert all(not e["ambiguous"] for e in looks)
    # Round trip through the cache arrays with consistent row ranges.
    rows = _Rows()
    rows.add_trial("synthetic", 1, 1, True, h, 1500.0, tr, looks, sacs)
    rows.add_trial("synthetic", 2, 1, True, h, 1500.0, dict(tr, n_looks=0, n_saccades=0), [], [])
    rows.add_skip("synthetic", 3, "no_pupil")
    d = rows.arrays(monkey="Synthetic", epoch="fix", sessions=("synthetic",))
    assert d["look_start"].tolist() == [0, 2] and d["look_stop"].tolist() == [2, 2]
    assert d["look_row"].tolist() == [0, 0] and d["sac_row"].tolist() == [0]
    assert d["island_look"].tolist() == [0, 1] and d["look_xy_u"].shape == (2, 2)
    assert d["skip_reason"].tolist() == ["no_pupil"]
    # Zero-look trial (jitter far above the 2 deg spread) flattens to empty rows.
    xraw, yraw, praw = _synthetic_trial(jitter=3.0)
    tr0, looks0, _ = trial_looks(xraw, yraw, praw, h, 1500.0)
    assert len(looks0) == 0 and tr0["n_looks"] == 0
    # Package guards surface as skip reasons.
    try:
        trial_looks(*_synthetic_trial(), h, 0.0)
    except Exception as exc:
        assert _skip_reason(exc) == "bad_epoch", exc
    else:
        raise AssertionError("epoch_end = 0 should raise")
    print("data.looks: adapter flattening, cache row ranges, skip reasons (synthetic)")


def compare_with_legacy(session, *, max_trials=None):
    """Looks (`fix` epoch) vs the legacy pymovements fixations on one session
    whose eye mat is local: counts, durations, warp agreement."""
    from data.builder import EYE_SAMPLING_RATE_HZ, trial_events, trial_qc_ok, unpack_eye
    from data.config import monkey_for_session
    from data.loader import load_eye_data

    import pymovements as pm

    monkey = monkey_for_session(session)
    d = load_looks(monkey, "fix", (session,))
    eye = load_eye_data(monkey)
    behavioral = load_eye_behavioral_data(monkey)
    beh = behavioral_lookup(behavioral)
    eye_ix = {(str(s), int(t)): i for i, (s, t) in enumerate(zip(eye["session"], eye["trial_indices_all"]))}
    exp = pm.Experiment(sampling_rate=EYE_SAMPLING_RATE_HZ)
    diffs, look_ms, fix_ms, warp_err = [], [], [], []
    rows = [r for r in np.flatnonzero(d["session"].astype(str) == session) if d["qc_ok"][r]]
    rows = rows[:max_trials] if max_trials else rows
    for r in rows:
        tid = int(d["trial_indices_all"][r])
        if not trial_qc_ok(behavioral, beh.get((session, tid))):
            continue
        sl = looks_of(d, r)
        look_ms += d["look_stable_ms"][sl].tolist()
        xd = d["look_xy_d"][sl].astype(float)
        xu, yu = to_maze(xd[:, 0], xd[:, 1], d["h"][r].astype(float))
        if xd.size:
            warp_err.append(float(np.max(np.abs(np.column_stack([xu, yu]) - d["look_xy_u"][sl]))))
        t, x, y, _, _ = unpack_eye(eye, eye_ix[(session, tid)])
        hi = float(d["fix_ms"][r]) / 1000.0
        keep = np.isfinite(t) & (t >= 0) & (t <= hi)
        n_fix = 0
        if keep.sum() >= 6:
            fr = trial_events(t[keep], x[keep], y[keep], np.empty((0, 2)), exp)
            if fr is not None:
                for name, on, off in fr.select("name", "onset", "offset").rows():
                    if "fixation" in str(name):
                        n_fix += 1
                        fix_ms.append(float(off - on))
        diffs.append(int(d["n_looks"][r]) - n_fix)
    diffs = np.asarray(diffs)
    status = dict(zip(*np.unique(d["saccade_status"][rows].astype(str), return_counts=True)))
    roi = dict(zip(*np.unique(d["look_roi_adv"][np.concatenate([np.arange(looks_of(d, r).start, looks_of(d, r).stop) for r in rows])].astype(str), return_counts=True)))
    print(f"{monkey} {session}: {len(rows)} QC trials | looks {sum(d['n_looks'][rows])} vs legacy fixations "
          f"{len(fix_ms)} | per-trial difference (looks - fixations): "
          f"{dict(zip(*np.unique(diffs, return_counts=True)))} | median stable {np.median(look_ms):.0f} ms vs "
          f"median fixation {np.median(fix_ms):.0f} ms | package ROI {roi} | ambiguous "
          f"{int(d['look_ambiguous'][np.concatenate([np.arange(looks_of(d, r).start, looks_of(d, r).stop) for r in rows])].sum())} "
          f"| max |to_maze - normalized| {max(warp_err) if warp_err else float('nan'):.2e} | status {status} | "
          f"skipped {dict(zip(*np.unique(d['skip_reason'], return_counts=True))) if len(d['skip_reason']) else {}}")


def main():
    parser = argparse.ArgumentParser(description="Build the looks caches (timestamp_preprocessing).")
    parser.add_argument("--monkey", nargs="*", default=["Faure", "Nielsen"])
    parser.add_argument("--epoch", nargs="*", default=list(EPOCHS), choices=list(EPOCHS))
    parser.add_argument("--sessions", nargs="*", help="default: every session with a local eye mat")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--check", action="store_true", help="synthetic checks only")
    parser.add_argument("--compare", metavar="SESSION", help="looks vs legacy fixations on one session")
    args = parser.parse_args()
    if args.check:
        check_synthetic()
        return
    if args.compare:
        compare_with_legacy(args.compare)
        return
    for monkey in args.monkey:
        for epoch in args.epoch:
            d = load_looks(monkey, epoch, args.sessions, refresh=args.refresh)
            n_ok = int(usable_rows(d).sum())
            print(f"{monkey} [{epoch}]: {len(d['session'])} trials ({n_ok} usable), "
                  f"{len(d['look_row'])} looks, {len(d['sac_row'])} saccades, "
                  f"sessions {', '.join(d['sessions'].astype(str))}")


if __name__ == "__main__":
    main()
