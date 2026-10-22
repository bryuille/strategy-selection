"""LEGACY per-sample assignment from pymovements fixation spans, kept only for
`msp.legacy.saccades` (the retired ``pre1466`` window over the
``*_clean_eye_data_s1466_e0`` caches). The current pipeline assigns looks
from `data.looks` (`msp.features`)."""

from __future__ import annotations

import numpy as np

from msp.config import AssignmentSpec
from msp.features import assign_by_rule

FIXATION_EVENT = "fixation"


def fixation_event_spans(event_rows):
    """Fixation `(onset, offset)` spans, earliest first."""
    spans = []
    for name, onset, offset in event_rows or ():
        if "fixation" not in str(name).lower():
            continue
        spans.append((float(onset), float(offset)))
    spans.sort()
    return spans


def fixation_centroids(xy, t_ms, valid, event_rows):
    """``(centroids (F, 2) float32, hit_masks)`` for the fixations with >= 1
    valid sample."""
    valid = np.asarray(valid, dtype=bool)
    t_ms = np.asarray(t_ms)
    centroids, hits = [], []
    if len(xy) == 0 or not valid.any():
        return np.empty((0, 2), dtype=np.float32), hits
    for onset, offset in fixation_event_spans(event_rows):
        hit = valid & (t_ms >= onset) & (t_ms <= offset)
        if not hit.any():
            continue
        centroids.append(np.asarray(xy[hit], dtype=np.float32).mean(axis=0))
        hits.append(hit)
    if not centroids:
        return np.empty((0, 2), dtype=np.float32), hits
    return np.stack(centroids).astype(np.float32), hits


def clip_fixation_spans(event_rows, lo, hi, name_wanted=FIXATION_EVENT, min_duration_ms=100.0):
    """Fixation rows intersected with ``[lo, hi]`` and re-tested against the
    legacy minimum duration."""
    out = []
    for name, onset, offset in event_rows or ():
        if "fixation" not in str(name).lower():
            continue
        o0, o1 = max(float(onset), lo), min(float(offset), hi)
        if o1 - o0 >= min_duration_ms:
            out.append((name, o0, o1))
    return out


def assign_trial(xy, t_ms, valid, event_rows, codebook_xy, assignment: AssignmentSpec):
    """``(state per sample, centroids, state per fixation)``: every valid
    sample of a fixation inherits its centroid's state; everything else -1."""
    state = np.full(len(xy), -1, dtype=np.int16)
    centroids, hits = fixation_centroids(xy, t_ms, valid, event_rows)
    if not hits:
        return state, centroids, np.empty(0, dtype=np.int16)
    fix_state = assign_by_rule(centroids, assignment, codebook_xy)
    for hit, sid in zip(hits, fix_state):
        if sid >= 0:
            state[hit] = sid
    return state, centroids, fix_state
