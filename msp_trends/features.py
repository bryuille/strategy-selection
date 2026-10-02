"""Codebook blocks derived from msp's five-state features.

Extraction is msp's (`msp.features.extract_monkey_features` under the package
window). It assigns each fixation centroid to one of five msp states (origin,
LU, LD, RU, RD); every codebook here is a relabelling of those
(`Codebook.state_map`). Two caches per (monkey, radius):

``<Monkey>_msp_trends_r<radius>.npz``             msp's balls (``balls``)
``<Monkey>_msp_trends_r<radius>_region.npz``      origin ball + maze quadrants
                                                  (``quads``; ``halves`` merges it)

For the region cache msp's ball rule is swapped for `region_states` for the
duration of the extraction (msp's code is not edited; `_rule_context` scopes
and undoes the swap).
Every cache is verified on load with msp's codebook / radius / window / state
count check, under the same context, plus the stored ``assignment_rule``. The
merge is a relabelling, so

``binary``  visited(left) = visited(LU) or visited(LD), etc. (column max)
``dwell``   assigned ms per merged state (column sum), divided by the trial's
            total assigned ms; a trial with no assigned dwell stays all-zero
            and is kept (counted as degenerate by the estimator, as in msp)

Row population is msp's under its ``geofix`` window. Under the default ``svm``
label scope, labels and sessions are msp's too, so `n_H` / `n_S` per maze equal
an msp run at the same radius (geofix is msp's default too); ``dendro`` swaps in the
dendrogram labels on the publication sessions.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest import mock

import numpy as np

from data.config import processed_npz
from data.mat import load_npz
from data.loader import savez_atomic
from msp import build as msp_build
from msp import config as msp_cfg
from msp import features as msp_features
from msp import labels as msp_labels
from msp_trends.config import (
    BLOCKS,
    EXIT_HALF_WIDTH,
    LABEL_SCOPES,
    PERIMETER_RADIUS,
    WINDOW,
    parse_tag,
)

_O, _LU, _LD, _RU, _RD = 0, 1, 2, 3, 4  # msp state indices
BALLS_RULE = "balls"
REGION_RULE = f"region_exit{EXIT_HALF_WIDTH:g}_perim{PERIMETER_RADIUS:g}"


def assignment_for(tag):
    """msp's `AssignmentSpec` for a tag: its radius under the package window."""
    return msp_cfg.resolve_assignment(radius=parse_tag(tag).radius, window=WINDOW)


def inside_maze(xy):
    """Within `PERIMETER_RADIUS` of the square the exits span: the outer edge
    of the four r1 exit balls, joined by straight tangents."""
    xy = np.asarray(xy, dtype=float)
    excess = np.clip(np.abs(xy) - EXIT_HALF_WIDTH, 0.0, None)
    return np.linalg.norm(excess, axis=-1) <= PERIMETER_RADIUS


def region_states(xy, codebook_xy=None, radius=None):
    """msp state per centroid under the region rule; drop-in for
    `msp.features.assign_states` (`codebook_xy` is ignored).

    Origin ball of `radius` first; then, inside the maze, the quadrant
    (x >= 0 is right, y >= 0 is up, so the stem and the crossbar break ties
    toward RU); else -1.
    """
    xy = np.asarray(xy, dtype=np.float32)
    state = np.full(xy.shape[0], -1, dtype=np.int16)
    if xy.size == 0:
        return state
    origin = np.linalg.norm(xy, axis=1) <= float(radius)
    inside = inside_maze(xy) & ~origin
    right, up = xy[:, 0] >= 0, xy[:, 1] >= 0
    quad = np.where(right, np.where(up, _RU, _RD), np.where(up, _LU, _LD))
    state[inside] = quad[inside]
    state[origin] = _O
    return state


def rule_for(codebook):
    return REGION_RULE if codebook.region else BALLS_RULE


@contextmanager
def _rule_context(rule):
    """Inside msp's extraction (and its cache check): swap the ball rule for
    `region_states`."""
    if rule == BALLS_RULE:
        yield
        return
    with mock.patch.object(msp_features, "assign_states", region_states):
        yield


_SUFFIX = {BALLS_RULE: "", REGION_RULE: "_region"}


def cache_path(monkey, tag):
    """One cache per (radius, rule); label scopes and codebooks of a rule share it."""
    t = parse_tag(tag)
    return processed_npz(f"{monkey}_msp_trends_r{t.radius:g}{_SUFFIX[rule_for(t.codebook)]}")


def _rule_of(data):
    return str(np.asarray(data["assignment_rule"]).reshape(-1)[0]) if "assignment_rule" in data else BALLS_RULE


def load(monkey, tag, *, refresh=False):
    """Extracted features for `monkey` under `tag`'s radius and rule (built on first use)."""
    t = parse_tag(tag)
    assignment = assignment_for(tag)
    rule = rule_for(t.codebook)
    path = cache_path(monkey, tag)
    if path.exists() and not refresh:
        data = load_npz(path)
        with _rule_context(rule):
            ok = msp_features._cache_matches(data, assignment)
        if ok and _rule_of(data) == rule:
            return data
        print(f"  {path.name}: built under a different codebook, radius, window or rule; rebuilding")
    print(f"Extracting {monkey} features ({assignment.label()}, {rule}) ...")
    with _rule_context(rule):
        data = msp_features.extract_monkey_features(monkey, assignment=assignment)
    data["assignment_rule"] = np.asarray(rule)
    path.parent.mkdir(parents=True, exist_ok=True)
    savez_atomic(path, **data)
    print(f"Saved {path} ({len(data['session'])} trials)")
    return data


def merge_columns(X5, how, codebook):
    """``(n, 5) -> (n, k)`` through ``codebook.state_map``; `how` is ``"max"`` or ``"sum"``."""
    X5 = np.asarray(X5, dtype=float)
    smap = codebook.map_array
    out = np.zeros((X5.shape[0], codebook.k))
    for s in range(codebook.k):
        cols = X5[:, smap == s]
        out[:, s] = cols.max(axis=1) if how == "max" else cols.sum(axis=1)
    return out


def dwell_share(ms3):
    total = ms3.sum(axis=1, keepdims=True)
    return np.divide(ms3, total, out=np.zeros_like(ms3), where=total > 0)


def merge_block(data, block, codebook):
    """The ``(n, k)`` block for every cache row under `codebook`."""
    raw = np.asarray(data[BLOCKS[block]], dtype=float)
    if block == "binary":
        return merge_columns(raw, "max", codebook)
    return dwell_share(merge_columns(raw, "sum", codebook))


def merged_fix_state(data, codebook):
    """`codebook` state per fixation centroid (-1 stays unassigned)."""
    fs = np.asarray(data["fix_state"], dtype=int)
    out = np.full(fs.shape, -1, dtype=np.int16)
    ok = fs >= 0
    out[ok] = codebook.map_array[fs[ok]]
    return out


def label_lookup(sessions, stem):
    """``(session, trial_id) -> 0.0/1.0`` from ``<session>_<stem>.npz``; the
    `msp.labels.svm_lookup` convention (index i is trial i + 1), any label file."""
    missing = [s for s in sessions if not processed_npz(f"{s}_{stem}").exists()]
    if missing:
        raise FileNotFoundError(f"missing {stem} label cache(s) for: {', '.join(missing)}")
    lookup = {}
    for session in sessions:
        with np.load(processed_npz(f"{session}_{stem}"), allow_pickle=True) as f:
            choices = np.asarray(f["strategy_choices"], dtype=float)
        for idx in np.flatnonzero(np.isfinite(choices)):
            lookup[(session, int(idx) + 1)] = float(choices[idx])
    return lookup


def scope(monkey, tag):
    """``(sessions, lookup)`` for the tag's label scope."""
    spec = LABEL_SCOPES[parse_tag(tag).scope]
    if spec["sessions"] is None:
        return msp_build.scope(monkey)
    sessions = tuple(spec["sessions"][monkey])
    in_cache = set(msp_labels.top_ten_sessions(monkey))
    outside = [s for s in sessions if s not in in_cache]
    if outside:
        # The geofix extraction only re-detects the top-ten sessions.
        raise ValueError(f"{monkey}: {outside} are outside the top-ten feature scope")
    return sessions, label_lookup(sessions, spec["stem"])


pool_maze = msp_build.pool_maze


def apply_variant(X, variant):
    """``full`` unmodified; ``mean_removed`` minus the maze grand mean."""
    if variant == "full":
        return np.asarray(X, dtype=float)
    if variant == "mean_removed":
        return msp_features.apply_mean_removed(X, msp_features.fit_grand_mean(X))
    raise ValueError(f"unknown variant {variant!r}")
