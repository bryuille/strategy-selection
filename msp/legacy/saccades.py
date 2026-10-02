"""Legacy pre-fixation gaze plots: the retired fixed ``pre1466`` window.

The window is ``[fix_start − 1466 ms, fix_start]``, and gaze comes from the
shared caches rather than being re-detected per trial: the attractor warp
(``<Monkey>_attractor_eye_data.npz``) and the s1466 clean events
(``<Monkey>_clean_eye_data_s1466_e0.npz``). Fixation onset is ~1534 ms after
maze onset in nearly every trial (never below 1466 ms), so this window misses
the first ~70 ms after maze onset that `msp.saccades` covers.

This is the only surviving pre1466 code; the feature caches and 2x2 figures
under that window were removed on 2026-10-02. Plotting is `msp.saccades`'s,
with this module's trial collector and output root.

Outputs land under ``msp/legacy/out/saccades/r<radius>_pre1466/<Monkey>/<session>/``.

Usage:
    uv run python -m msp.legacy.saccades --radius 0.5 --session june_24_g0 --maze 2
    uv run python -m msp.legacy.saccades --radius 0.5 --session june_24_g0 --maze 2 --view trials
    uv run python -m msp.legacy.saccades --all-analyses
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np

from data.builder import behavioral_lookup, fix_start_ms, trial_qc_ok
from data.config import PRE_FIX_END_MS, PRE_FIX_START_MS, PRE_FIX_WINDOW_MS
from data.loader import (
    load_attractor_eye_data,
    load_clean_eye_data,
    load_eye_behavioral_data,
)
from msp import saccades
from msp.config import CODEBOOK_XY, AssignmentSpec
from msp.features import clip_fixation_spans

WINDOW = "pre1466"
RADII = (0.5, 1.0)
OUT_ROOT = Path(__file__).resolve().parent / "out"


def assignment(radius) -> AssignmentSpec:
    """msp's uniform balls under the legacy window (tag ``r<radius>_pre1466``)."""
    radius = float(radius)
    return AssignmentSpec(tag=f"r{radius:g}_{WINDOW}", radius=radius, window=WINDOW)


@lru_cache(maxsize=4)
def _event_rows(monkey):
    ev = load_clean_eye_data(monkey, start_ms=PRE_FIX_START_MS, end_ms=PRE_FIX_END_MS)
    names = np.asarray(ev["name"]).astype(str)
    sess = np.asarray(ev["session"]).astype(str)
    trials = np.asarray(ev["trial_indices_all"]).astype(int)
    onset = np.asarray(ev["onset"], dtype=float)
    offset = np.asarray(ev["offset"], dtype=float)
    out = {}
    for i in range(names.size):
        out.setdefault((sess[i], int(trials[i])), []).append(
            (names[i].lower(), onset[i], offset[i])
        )
    return out


def collect(behavioral, attractor, maze, session, assignment, *, monkey):
    """Attractor warp + s1466 clean events, ``fix_start − 1466 ms``."""
    lookup = behavioral_lookup(behavioral)
    events = _event_rows(monkey)
    codebook_xy = np.asarray(CODEBOOK_XY, dtype=np.float32)
    raw = []
    sessions = np.asarray(attractor["session"]).astype(str)
    trial_ids = np.asarray(attractor["trial_indices_all"]).astype(int)
    for att_i in range(sessions.size):
        if sessions[att_i] != session:
            continue
        trial_id = int(trial_ids[att_i])
        beh_i = lookup.get((session, trial_id))
        if beh_i is None or not trial_qc_ok(behavioral, beh_i):
            continue
        if int(behavioral["geo_type"][beh_i]) != maze:
            continue

        t_ms = np.asarray(attractor["time"][att_i], dtype=float) * 1000
        x = np.asarray(attractor["eye_x"][att_i], dtype=float)
        y = np.asarray(attractor["eye_y"][att_i], dtype=float)
        valid = np.asarray(attractor["valid"][att_i], dtype=bool)
        fix_ms = fix_start_ms(behavioral, beh_i)
        if not np.isfinite(fix_ms) or fix_ms <= 0:
            continue
        h = tuple(behavioral[f"h{i}"][beh_i] for i in range(1, 7))
        n = min(t_ms.size, x.size, y.size, valid.size)
        x, y, valid, t_ms = x[:n], y[:n], valid[:n], t_ms[:n]
        lo, hi = fix_ms - PRE_FIX_WINDOW_MS, fix_ms
        in_win = np.isfinite(t_ms) & (t_ms >= lo) & (t_ms <= hi)
        valid_win = valid & in_win & np.isfinite(x) & np.isfinite(y)
        spans = clip_fixation_spans(events.get((session, trial_id), ()), lo, hi)
        raw.append(saccades._raw_entry(
            trial_id, t_ms, x, y, valid, valid_win, spans, lo, hi, fix_ms, h,
            saccades.cue_rel_ms(behavioral, beh_i), assignment, codebook_xy,
        ))
    return raw


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--radius", type=float, default=None,
                        help=f"ball radius (unit H); required unless --all-analyses ({RADII})")
    parser.add_argument("--session", default=None)
    parser.add_argument("--monkey", default=None, choices=saccades.MONKEYS)
    parser.add_argument("--maze", type=int, default=None, choices=range(1, 7))
    parser.add_argument("--view", choices=("average", "trials", "trial"), default="average")
    parser.add_argument("--trial-id", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--all-analyses", action="store_true",
        help="avg_2d_maze_<M> and trial_<id>_maze_<M> for every "
             "radius in RADII × publication session × mazes 2-5",
    )
    args = parser.parse_args()

    if args.all_analyses:
        saccades.run_all_analyses(
            assignments=[assignment(r) for r in RADII], dpi=args.dpi,
            load_source=load_attractor_eye_data, collect=collect, out_root=OUT_ROOT,
        )
        return

    if args.radius is None or args.session is None or args.maze is None:
        parser.error("--radius, --session and --maze are required unless --all-analyses")
    monkey = args.monkey or saccades._monkey_for_session(args.session)
    saccades.plot_session(
        args.maze, args.session, view=args.view, trial_id=args.trial_id,
        monkey=monkey, assignment=assignment(args.radius), dpi=args.dpi,
        behavioral=load_eye_behavioral_data(monkey),
        source=load_attractor_eye_data(monkey),
        collect=collect, out_root=OUT_ROOT,
    )


if __name__ == "__main__":
    main()
