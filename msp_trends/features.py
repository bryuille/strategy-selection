"""Codebook blocks derived from msp's five-state features.

Extraction is msp's (`msp.features.extract_monkey_features` under the package
window). It assigns each fixation centroid to one of five msp states (origin,
LU, LD, RU, RD), or six with the stem; every codebook here is a relabelling of
those (`Codebook.state_map`). Three caches per (monkey, radius):

``<Monkey>_msp_trends_r<radius>.npz``             msp's balls (``balls``)
``<Monkey>_msp_trends_r<radius>_region.npz``      origin ball + maze quadrants
                                                  (``quads``; ``halves`` merges it)
``<Monkey>_msp_trends_r<radius>_regionstem.npz``  the same plus a stem state
                                                  (``quadsstem``, ``halvesstem``)

For the region caches msp's ball rule is swapped for `region_states` for the
duration of the extraction, and for the stem cache msp's state count is raised
to six (msp's code is not edited; `_rule_context` scopes and undoes both).
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
from data.convert import load_npz
from data.loader import savez_atomic
from msp import build as msp_build
from msp import config as msp_cfg
from msp import features as msp_features
from msp import labels as msp_labels
from msp_trends.config import (
    BLOCKS,
    EXIT_HALF_WIDTH,
    LABEL_SCOPES,
    N_STATES,
    N_STATES_STEM,
    PERIMETER_RADIUS,
    STEM_HALF_WIDTH,
    STEM_STATE,
    STEM_TOP,
    WINDOW,
    parse_tag,
)

_O, _LU, _LD, _RU, _RD = 0, 1, 2, 3, 4  # msp state indices
BALLS_RULE = "balls"
REGION_RULE = f"region_exit{EXIT_HALF_WIDTH:g}_perim{PERIMETER_RADIUS:g}"
STEM_RULE = f"{REGION_RULE}_stem{STEM_HALF_WIDTH:g}x{STEM_TOP:g}"


def assignment_for(tag):
    """msp's `AssignmentSpec` for a tag: its radius under the package window."""
    return msp_cfg.resolve_assignment(radius=parse_tag(tag).radius, window=WINDOW)


def inside_maze(xy):
    """Within `PERIMETER_RADIUS` of the square the exits span: the outer edge
    of the four r1 exit balls, joined by straight tangents."""
    xy = np.asarray(xy, dtype=float)
    excess = np.clip(np.abs(xy) - EXIT_HALF_WIDTH, 0.0, None)
    return np.linalg.norm(excess, axis=-1) <= PERIMETER_RADIUS


def in_stem(xy):
    """The upper stem strip: |x| <= `STEM_HALF_WIDTH`, 0 <= y <= `STEM_TOP`."""
    xy = np.asarray(xy, dtype=float)
    return (np.abs(xy[..., 0]) <= STEM_HALF_WIDTH) & (xy[..., 1] >= 0) & (xy[..., 1] <= STEM_TOP)


def region_states(xy, codebook_xy=None, radius=None, *, stem=False):
    """msp state per centroid under the region rule; drop-in for
    `msp.features.assign_states` (`codebook_xy` is ignored).

    Origin ball of `radius` first; then, with `stem`, the stem strip
    (`STEM_STATE`); else, inside the maze, the quadrant (x >= 0 is right,
    y >= 0 is up, so the stem and the crossbar break ties toward RU); else -1.
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
    if stem:
        state[inside & in_stem(xy)] = STEM_STATE
    state[origin] = _O
    return state


def rule_for(codebook):
    if not codebook.region:
        return BALLS_RULE
    return STEM_RULE if codebook.stem else REGION_RULE


@contextmanager
def _rule_context(rule):
    """Inside msp's extraction (and its cache check): swap the ball rule for
    `region_states`, and for the stem rule raise msp's state count to six so
    the stem's dwell is bincounted rather than dropped."""
    if rule == BALLS_RULE:
        yield
        return
    stem = rule == STEM_RULE
    assign = (lambda xy, codebook_xy=None, radius=None:
              region_states(xy, codebook_xy, radius, stem=True)) if stem else region_states
    with mock.patch.object(msp_features, "assign_states", assign), \
            mock.patch.object(msp_features, "K", N_STATES_STEM if stem else N_STATES):
        yield


_SUFFIX = {BALLS_RULE: "", REGION_RULE: "_region", STEM_RULE: "_regionstem"}


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
    """``(n, 5 or 6) -> (n, k)`` through ``codebook.state_map``; `how` is ``"max"`` or ``"sum"``."""
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


def balanced_centre(X, labels):
    """Centre ``c = w_H·μ_H + w_S·μ_S`` with ``w_X ∝ sqrt(n_X)``.

    Each strategy's expected residual is ``w_other · (μ_X − μ_other)`` and its
    split-half noise falls as ``1/sqrt(n_X)``, so sqrt(n) weights give both
    diagonals the same expected signal-to-noise. The grand mean (``w ∝ n``)
    favours the minority by ``(n_maj/n_min)^1.5``; equal weights favour the
    majority by ``sqrt(n_maj/n_min)``. Uses the labels, so it must be refit
    under every shuffle (`variant_transform`).
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    means, roots = [], []
    for strategy in (0, 1):
        rows = X[labels == strategy]
        means.append(rows.mean(axis=0))
        roots.append(np.sqrt(rows.shape[0]))
    w = np.asarray(roots) / sum(roots)
    return w[0] * means[0] + w[1] * means[1]


def balanced_mean_removed(X, labels):
    return np.asarray(X, dtype=float) - balanced_centre(X, labels)


def apply_variant(X, variant):
    """The label-free part: ``full`` and ``mean_removed_balanced`` unmodified
    (the latter centres in `variant_transform`); ``mean_removed`` minus the
    maze grand mean."""
    if variant in ("full", "mean_removed_balanced"):
        return np.asarray(X, dtype=float)
    if variant == "mean_removed":
        return msp_features.apply_mean_removed(X, msp_features.fit_grand_mean(X))
    raise ValueError(f"unknown variant {variant!r}")


def variant_transform(variant):
    """Label-dependent preprocessing for `estimator.run_maze`, or None."""
    return balanced_mean_removed if variant == "mean_removed_balanced" else None
