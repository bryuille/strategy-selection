"""Feature matrices for both branches, one cache per monkey, branch and scope.

Raw: `<Monkey>_clf_raw[_svm].npz`, built from the pooled eye cache and the
pupil tables. Regions: computed on the fly from the fixation table of
`classifier.fixations` (which also folds in the pupil tables). Both return ``(X, names, meta)`` with `meta` holding session,
trial, maze and label per row; rows are the labelled, QC-passing trials of
the scope's mazes. Sessions without a pupil table get NaN pupil features
(logistic imputes to the session mean, the tree takes NaN).
"""

from __future__ import annotations

import warnings

import numpy as np

from data.builder import behavioral_lookup, fix_start_ms, trial_qc_ok, unpack_eye
from data.config import processed_npz
from data.loader import load_eye_behavioral_data, load_eye_data, savez_atomic
from data.looks import sample_quality
from data.mat import load_npz
from classifier.config import (
    BALLS_ORIGIN_DEG,
    CAPSULE_DEG,
    CAPSULE_ORIGIN_DEG,
    CAPSULE_ORIGIN_TIGHT_DEG,
    EXIT_BALL_DEG,
    FAR_DEG,
    GRID_LIM_DEG,
    LABEL_SCOPES,
    LOR_ORIGIN_DEG,
    LOR_PERIMETER_DEG,
    MAZES,
    ORIGIN_DISC_DEG,
    POSITION_LIMIT_DEG,
    PUPIL_DIR,
    PUPIL_EARLY_MS,
    PUPIL_LATE_MS,
    PUPIL_MIN_VALID,
    RAW_BIN_MS,
    RAW_GRID,
    RAW_WINDOW_MS,
)

RAW_SCHEMA = 3  # raw samples with timestamp_preprocessing validity; widths derived on load


# ---- labels ------------------------------------------------------------------


def label_lookup(sessions, stem):
    """``(session, trial_id) -> 0.0 / 1.0`` from ``<session>_<stem>.npz``
    (index i is trial i + 1; both label kinds share the key). Read, never built."""
    lookup = {}
    for s in sessions:
        path = processed_npz(f"{s}_{stem}")
        if not path.exists():
            raise FileNotFoundError(f"missing label cache {path}")
        with np.load(path, allow_pickle=True) as f:
            c = np.asarray(f["strategy_choices"], dtype=float)
        for i in np.flatnonzero(np.isfinite(c)):
            lookup[(s, int(i) + 1)] = float(c[i])
    return lookup


# ---- pupil -------------------------------------------------------------------


def pupil_tables(sessions):
    """Per-session pupil lists from ``PUPIL_DIR``; a session without a table
    maps to ``None`` (its pupil features are NaN) with a warning."""
    out = {}
    for s in sessions:
        path = PUPIL_DIR / f"{s}_pupil.npz"
        if not path.exists():
            warnings.warn(f"{path} missing (classifier.pupil_extract --session {s} on the cluster, "
                          f"then pull classifier/out/pupil/); pupil is NaN for {s}", stacklevel=2)
            out[s] = None
            continue
        with np.load(path, allow_pickle=True) as f:
            out[s] = list(f["pupil"])
    return out


def _clean_pupil(pupil, lo, hi):
    """``(size, t_ms)`` inside ``[lo, hi]`` with samples near tracker loss dropped."""
    pu = np.asarray(pupil, dtype=float)
    if pu.ndim != 2 or pu.shape[0] == 0:
        return np.empty(0), np.empty(0)
    if pu.shape[1] != 2:
        pu = pu.T
    size, t_ms = pu[:, 0], pu[:, 1] * 1000.0
    win = np.isfinite(t_ms) & (t_ms >= lo) & (t_ms <= hi)
    size, t_ms = size[win], t_ms[win]
    bad = ~np.isfinite(size)
    if bad.any():
        bad = np.convolve(bad.astype(float), np.ones(101), mode="same") > 0  # +-50 ms at 1 kHz
    return size[~bad], t_ms[~bad]


def pupil_summary(pupil, lo, hi):
    size, t_ms = _clean_pupil(pupil, lo, hi)
    frac = size.size / max(hi - lo, 1.0)
    if frac < PUPIL_MIN_VALID:
        return np.nan, np.nan
    early = t_ms <= lo + PUPIL_EARLY_MS
    late = t_ms >= hi - PUPIL_LATE_MS
    change = size[late].mean() - size[early].mean() if early.any() and late.any() else np.nan
    return float(size.mean()), float(change)


def within_session_z(values, sessions):
    """z within session over the finite values; NaN stays NaN. The logistic
    pipeline imputes NaN to 0 (the session mean), the tree model takes NaN."""
    out = np.full(np.shape(values), np.nan, dtype=float)
    for s in np.unique(sessions):
        m = sessions == s
        v = values[m]
        fin = np.isfinite(v)
        if fin.sum() >= 2 and np.nanstd(v) > 0:
            out[m] = np.where(fin, (v - np.nanmean(v)) / np.nanstd(v), np.nan)
        elif fin.any():
            out[m] = np.where(fin, 0.0, np.nan)
    return out


# ---- raw branch --------------------------------------------------------------


def n_bins(bin_ms):
    return int(round(RAW_WINDOW_MS / bin_ms))


def _bin_trace(t_ms, v, lo, hi, bin_ms=RAW_BIN_MS):
    """Mean of `v` in `bin_ms` bins over ``[hi - RAW_WINDOW_MS, hi]``.
    Empty bins stay NaN. ``bin_ms = 1`` is the whole 1 kHz trace."""
    nb = n_bins(bin_ms)
    edges = hi - RAW_WINDOW_MS + np.arange(nb + 1) * bin_ms
    idx = np.digitize(t_ms, edges) - 1
    ok = (idx >= 0) & (idx < nb) & np.isfinite(v)
    sums = np.bincount(idx[ok], weights=v[ok], minlength=nb)
    cnt = np.bincount(idx[ok], minlength=nb)
    out = np.divide(sums, cnt, out=np.full(nb, np.nan), where=cnt > 0)
    return out, float(np.isfinite(out).mean())


def _grid(x, y):
    edges = np.linspace(-GRID_LIM_DEG, GRID_LIM_DEG, RAW_GRID + 1)
    ok = np.isfinite(x) & np.isfinite(y)
    h, _, _ = np.histogram2d(x[ok], y[ok], bins=[edges, edges])
    tot = h.sum()
    return (h / tot if tot > 0 else h).ravel()


def build_raw(monkey, sessions, bin_ms=1.0):
    nb = n_bins(bin_ms)
    sessions = tuple(sessions)
    eye = load_eye_data(monkey)
    beh = load_eye_behavioral_data(monkey)
    rows_beh = behavioral_lookup(beh)
    pupils = pupil_tables(sessions)
    recs = []
    for i in range(len(eye["session"])):
        session = str(eye["session"][i])
        if session not in sessions:
            continue
        t_s, x, y, _, trial = unpack_eye(eye, i)
        bi = rows_beh.get((session, trial))
        if not trial_qc_ok(beh, bi):
            continue
        maze = int(beh["geo_type"][bi])
        if maze not in MAZES:
            continue
        fix_ms = float(fix_start_ms(beh, bi))
        if not np.isfinite(fix_ms) or fix_ms <= 0:
            continue
        # Validity from the timestamp_preprocessing package (matched
        # timestamps, finite, on screen, outside blink padding), on the raw
        # samples; no event detection enters the raw branch.
        q = sample_quality(monkey, session, trial)
        if q is None:
            continue
        t_ms = np.asarray(q["t"], dtype=float)
        x = np.where(q["valid"], q["x"], np.nan)
        y = np.where(q["valid"], q["y"], np.nan)
        lo, hi = fix_ms - RAW_WINDOW_MS, fix_ms
        win = (t_ms >= lo) & (t_ms <= hi)
        if win.sum() < 100:
            continue
        bx, cov = _bin_trace(t_ms[win], x[win], lo, hi, bin_ms)
        by, _ = _bin_trace(t_ms[win], y[win], lo, hi, bin_ms)
        if cov < 0.5:
            continue
        sp = pupils[session]
        pupil = sp[trial - 1] if sp is not None and trial - 1 < len(sp) else np.empty((0, 2))
        size, pt = _clean_pupil(pupil, lo, hi)
        bp, pcov = _bin_trace(pt, size, lo, hi, bin_ms) if size.size else (np.full(nb, np.nan), 0.0)
        recs.append((session, trial, maze, fix_ms, bx, by, bp, pcov, _grid(x[win], y[win])))
    out = {
        "session": np.asarray([r[0] for r in recs], dtype=str),
        "trial_indices_all": np.asarray([r[1] for r in recs], dtype=int),
        "maze_id": np.asarray([r[2] for r in recs], dtype=int),
        "fix_ms": np.asarray([r[3] for r in recs], dtype=np.float32),
        "trace_x": np.asarray([r[4] for r in recs], dtype=np.float32),
        "trace_y": np.asarray([r[5] for r in recs], dtype=np.float32),
        "trace_pupil": np.asarray([r[6] for r in recs], dtype=np.float32),
        "pupil_cov": np.asarray([r[7] for r in recs], dtype=np.float32),
        "grid": np.asarray([r[8] for r in recs], dtype=np.float32),
        "sessions": np.asarray(sessions, dtype=str),
        "bin_ms": np.float64(bin_ms),
        "schema": np.int64(RAW_SCHEMA),
    }
    print(f"  {monkey}: {len(recs)} raw trials over {len(sessions)} sessions, "
          f"{nb} bins x (x, y, pupil) + {RAW_GRID}x{RAW_GRID} grid")
    return out


def _scoped(stem, scope):
    tag = LABEL_SCOPES[scope]["tag"]
    return f"{stem}_{tag}" if tag else stem


def load_raw(monkey, *, labels="dendro", bin_ms=RAW_BIN_MS, refresh=False):
    """The 1 kHz cache ``<Monkey>_clf_raw[_svm]`` over the scope's sessions,
    block-averaged to `bin_ms`."""
    sessions = tuple(LABEL_SCOPES[labels]["sessions"][monkey])
    path = processed_npz(_scoped(f"{monkey}_clf_raw", labels))
    d = None
    if path.exists() and not refresh:
        d = load_npz(path)
        if not (int(d.get("schema", -1)) == RAW_SCHEMA and tuple(d["sessions"].astype(str)) == sessions
                and float(d.get("bin_ms", 0)) == 1.0):
            d = None
    if d is None:
        print(f"Building raw 1 kHz traces for {monkey} ({', '.join(sessions)}) ...", flush=True)
        d = build_raw(monkey, sessions, 1.0)
        savez_atomic(path, **d)
    return d if float(bin_ms) == 1.0 else rebin(d, bin_ms)


def raw_matrix(d):
    """Gaze x / y / pupil traces over the pre-fixation window, each bin
    z-scored within session over its finite values. Empty bins stay NaN
    (logistic imputes to the column mean; the tree takes NaN)."""
    sess = d["session"].astype(str)
    bin_ms = float(d.get("bin_ms", RAW_BIN_MS))
    blocks, names = [], []
    for key, tag in (("trace_x", "gaze x"), ("trace_y", "gaze y"), ("trace_pupil", "pupil")):
        T = np.asarray(d[key], dtype=float)
        T = np.stack([within_session_z(T[:, j], sess) for j in range(T.shape[1])], 1)
        blocks.append(T)
        names += [f"{tag}, {RAW_WINDOW_MS - (j + 0.5) * bin_ms:.0f} ms before fixation"
                  for j in range(T.shape[1])]
    return np.hstack(blocks), names


def grid_matrix(d):
    """The time-collapsed gaze map stored with the raw cache: share of
    in-window samples per cell of a RAW_GRID x RAW_GRID grid over
    [-GRID_LIM_DEG, GRID_LIM_DEG] deg (``np.histogram2d`` order, x outer)."""
    edges = np.linspace(-GRID_LIM_DEG, GRID_LIM_DEG, RAW_GRID + 1)
    mid = (edges[:-1] + edges[1:]) / 2
    names = [f"gaze map x {mx:+.1f}, y {my:+.1f} deg" for mx in mid for my in mid]
    return np.asarray(d["grid"], dtype=float), names


def rebin(d, bin_ms):
    """A coarser copy of the 1 kHz cache by NaN-aware block-averaging of traces."""
    src = float(d.get("bin_ms", 1.0))
    k = int(round(bin_ms / src))
    assert abs(k * src - bin_ms) < 1e-9 and d["trace_x"].shape[1] % k == 0, (src, bin_ms)
    out = dict(d)
    for key in ("trace_x", "trace_y", "trace_pupil"):
        T = np.asarray(d[key], dtype=float)
        with np.errstate(invalid="ignore"):
            out[key] = np.nanmean(T.reshape(T.shape[0], T.shape[1] // k, k), axis=2).astype(np.float32)
    out["bin_ms"] = np.float64(bin_ms)
    return out


# ---- fixation table and maze landmarks ---------------------------------------


def load_fix(monkey, *, labels="dendro", refresh=False):
    """The fixation table over the scope's sessions (`classifier.fixations`):
    read from its cache when schema / window / sessions match, else built
    (needs pymovements)."""
    from classifier.fixations import load as load_fixations
    return load_fixations(monkey, labels=labels, refresh=refresh)


def _exit_landmarks(h):
    h1, h2, h3, h4, h5, h6 = (h[:, k] for k in range(6))
    return np.stack([np.stack([-h1, h2], 1), np.stack([-h1, -h3], 1),
                     np.stack([h4, h5], 1), np.stack([h4, -h6], 1)], 1)  # LU, LD, RU, RD


# ---- regions branch ----------------------------------------------------------
#
# Two families of gaze summary, over one partition of the screen in degrees:
# the four quadrants of the origin disc (ORIGIN_DISC_DEG), the four exit balls
# (EXIT_BALL_DEG), and everything else. Two sets over the same nine regions,
# as in msp's binary / dwell pair: `dwell` (share of in-window fixation time,
# sums to 1) and `binary` (visited or not). No ordering, latency or pupil
# features.
#
# `regions_plus` adds the two engineered terms that kept their sign in both
# monkeys and through the trial-order control: "elsewhere" is split at
# FAR_DEG into near (ORIGIN_DISC_DEG to FAR_DEG, outside the exit balls) and
# far, so the dwell shares still sum to 1, and pupil change (late minus early,
# z within session) is appended once.
#
# `exits` keeps only the four exit-ball columns of the same partition (dwell
# share is still of all in-window fixation time, so the four shares sum to
# the share of time spent at any exit, not to 1).
#
# `lor` is a different partition, msp's `halves` codebook in degrees: an origin
# ball (LOR_ORIGIN_DEG), then the left (x < 0) and right half of "inside the
# maze", every point within LOR_PERIMETER_DEG of the quadrilateral the exits
# span. Fixations outside the perimeter are unassigned, so shares sum to the
# share of time inside the maze.
#
# `balls` is msp's r0.5 codebook in degrees: an origin ball (BALLS_ORIGIN_DEG)
# and the four exit balls (EXIT_BALL_DEG) at each maze's exits, nearest ball
# wins, everything else unassigned.
#
# `capsules` joins each exit ball to the origin ball with parallel tangents:
# a capsule is every point within CAPSULE_DEG of the segment origin -> exit.
# The origin ball is its own region (it wins where it overlaps a capsule) and
# everything outside is "elsewhere", so the dwell shares sum to 1.
# `capsules_tight` keeps the same capsules and shrinks the origin ball.
#
# Everything is in degrees around each maze's own exit locations; nothing in
# classifier is warped to unit H.

REGION_NAMES = ("origin LU", "origin LD", "origin RU", "origin RD",
                "exit LU", "exit LD", "exit RU", "exit RD", "elsewhere")
REGION_PLUS_NAMES = REGION_NAMES[:-1] + (f"near {ORIGIN_DISC_DEG:g}-{FAR_DEG:g} deg", f"far > {FAR_DEG:g} deg")
EXIT_IDS = range(4, 8)  # the exit-ball rows of REGION_NAMES
LOR_NAMES = ("origin", "left", "right")
BALLS_NAMES = ("origin", "exit LU", "exit LD", "exit RU", "exit RD")
CAPSULE_NAMES = ("origin", "capsule LU", "capsule LD", "capsule RU", "capsule RD", "elsewhere")
RAW_KINDS = ("raw", "raw_plus")
REGIONS_KINDS = {"regions": ("dwell", "binary"), "regions_dwell": ("dwell",), "regions_binary": ("binary",),
                 "regions_plus": ("dwell", "binary"), "regions_plus_dwell": ("dwell",),
                 "regions_plus_binary": ("binary",),
                 # dwell share + visited + visit count (separate entries into the region)
                 "regions_all": ("dwell", "binary", "count"), "regions_plus_all": ("dwell", "binary", "count"),
                 # the four exit balls alone
                 "exits": ("dwell", "binary"), "exits_dwell": ("dwell",), "exits_binary": ("binary",),
                 "exits_all": ("dwell", "binary", "count"),
                 # origin ball + left / right half of the maze
                 "lor": ("dwell", "binary"), "lor_dwell": ("dwell",), "lor_binary": ("binary",),
                 "lor_all": ("dwell", "binary", "count"),
                 # msp's r0.5 codebook in degrees: origin ball + four exit balls
                 "balls": ("dwell", "binary"), "balls_dwell": ("dwell",), "balls_binary": ("binary",),
                 "balls_all": ("dwell", "binary", "count"),
                 # origin ball + four origin-to-exit capsules + elsewhere
                 "capsules": ("dwell", "binary"), "capsules_dwell": ("dwell",), "capsules_binary": ("binary",),
                 "capsules_all": ("dwell", "binary", "count"),
                 "capsules_tight": ("dwell", "binary"), "capsules_tight_dwell": ("dwell",),
                 "capsules_tight_binary": ("binary",), "capsules_tight_all": ("dwell", "binary", "count")}


def is_plus(branch):
    return branch.startswith("regions_plus")


def is_exits(branch):
    return branch.startswith("exits")


def is_lor(branch):
    return branch.startswith("lor")


def is_balls(branch):
    return branch.startswith("balls")


def is_capsules_tight(branch):
    return branch.startswith("capsules_tight")


def is_capsules(branch):
    return branch.startswith("capsules") and not is_capsules_tight(branch)


def region_index(xy, h, *, split_far=False):
    """Region id per fixation (``REGION_NAMES`` order, or ``REGION_PLUS_NAMES``
    with `split_far`). Exit balls win over the origin disc where they overlap
    (they do not at the default radii). `xy` and `h` in degrees."""
    exits, disc_r, ball_r = _exit_landmarks(h), ORIGIN_DISC_DEG, EXIT_BALL_DEG
    rad = np.linalg.norm(xy, axis=1)
    right, up = xy[:, 0] >= 0, xy[:, 1] >= 0
    idx = np.full(len(xy), 8, dtype=int)
    if split_far:
        idx[rad > FAR_DEG] = 9
    in_disc = rad <= disc_r
    idx[in_disc & ~right & up] = 0
    idx[in_disc & ~right & ~up] = 1
    idx[in_disc & right & up] = 2
    idx[in_disc & right & ~up] = 3
    dist = np.linalg.norm(xy[:, None, :] - exits, axis=-1)  # (F, 4): LU, LD, RU, RD
    near = dist.min(1) <= ball_r
    idx[near] = 4 + dist.argmin(1)[near]
    return idx


def exit_quad(h):
    """The convex quadrilateral the exits span, ``(F, 4, 2)`` in the order
    LU, RU, RD, LD (one winding) from the per-fixation arm lengths `h`."""
    return _exit_landmarks(h)[:, [0, 2, 3, 1], :]


def dist_to_polygon(xy, poly):
    """Distance from each point of `xy` ``(F, 2)`` to the convex polygon
    ``poly[f]`` ``(F, V, 2)`` (vertices in order): 0 inside, else the
    distance to the nearest edge."""
    a = poly
    ab = np.roll(poly, -1, axis=1) - a  # edge vectors
    ap = xy[:, None, :] - a
    t = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0.0, 1.0)
    seg = np.linalg.norm(ap - t[..., None] * ab, axis=-1).min(1)
    cross = ab[..., 0] * ap[..., 1] - ab[..., 1] * ap[..., 0]
    inside = (cross >= 0).all(1) | (cross <= 0).all(1)
    return np.where(inside, 0.0, seg)


def lor_index(xy, h, *, origin_deg=LOR_ORIGIN_DEG, perimeter_deg=LOR_PERIMETER_DEG):
    """Region id per fixation in ``LOR_NAMES`` order, -1 outside the maze.
    Origin ball first; then, within `perimeter_deg` of the exit quadrilateral,
    left (x < 0) or right. `xy` and `h` in degrees."""
    idx = np.full(len(xy), -1, dtype=int)
    inside = dist_to_polygon(xy, exit_quad(h)) <= perimeter_deg
    idx[inside] = np.where(xy[inside, 0] < 0, 1, 2)
    idx[np.linalg.norm(xy, axis=1) <= origin_deg] = 0
    return idx


def balls_index(xy, h, *, origin_deg=BALLS_ORIGIN_DEG, exit_deg=EXIT_BALL_DEG):
    """msp's ball assignment in degrees: ``BALLS_NAMES`` id of the nearest
    ball (origin at (0, 0), exits at the maze's own landmarks) that holds the
    point, else -1. `xy` and `h` in degrees."""
    protos = np.concatenate([np.zeros((len(xy), 1, 2)), _exit_landmarks(h)], axis=1)  # (F, 5, 2)
    radii = np.array([origin_deg, exit_deg, exit_deg, exit_deg, exit_deg])
    dist = np.linalg.norm(xy[:, None, :] - protos, axis=-1)
    dist = np.where(dist <= radii, dist, np.inf)
    return np.where(np.isfinite(dist.min(1)), dist.argmin(1), -1)


def capsule_index(xy, h, *, origin_deg=CAPSULE_ORIGIN_DEG, capsule_deg=CAPSULE_DEG):
    """``CAPSULE_NAMES`` id per fixation, in degrees. A capsule is every point
    within `capsule_deg` of the segment from the origin to one exit (the two
    balls joined by parallel tangents); the nearest segment wins where they
    overlap. The origin ball wins over every capsule, and the rest is elsewhere."""
    exits = _exit_landmarks(h)  # (F, 4, 2)
    dot = (xy[:, None, :] * exits).sum(-1)
    t = np.clip(dot / np.maximum((exits * exits).sum(-1), 1e-12), 0.0, 1.0)
    dist = np.linalg.norm(xy[:, None, :] - t[..., None] * exits, axis=-1)
    dist = np.where(dist <= capsule_deg, dist, np.inf)
    idx = np.full(len(xy), CAPSULE_NAMES.index("elsewhere"), dtype=int)
    has = np.isfinite(dist.min(1))
    idx[has] = 1 + dist.argmin(1)[has]
    idx[np.linalg.norm(xy, axis=1) <= origin_deg] = 0
    return idx


def regions_matrix(d, kinds=("dwell", "binary"), *, plus=False, exits=False, lor=False, balls=False,
                   capsule_origin=None):
    """One row per cache trial: per-region dwell share and / or visited flag;
    with `plus`, over ``REGION_PLUS_NAMES`` and with pupil change appended;
    with `exits`, only the four exit-ball columns are kept; with `lor`, the
    partition is origin ball / left half / right half instead; with `balls`,
    msp's five balls (origin + exits) in degrees; with `capsule_origin`, the
    origin ball, four capsules and elsewhere."""
    n = len(d["session"])
    rows = np.asarray(d["fix_row"], dtype=int)
    xy = np.asarray(d["fix_xy_d"], dtype=float)
    ms = np.asarray(d["fix_valid_ms"], dtype=float)
    h = np.asarray(d["h"], dtype=float)[rows]
    if capsule_origin is not None:
        reg, all_names, ids = capsule_index(xy, h, origin_deg=capsule_origin), CAPSULE_NAMES, range(len(CAPSULE_NAMES))
    elif balls:
        reg, all_names, ids = balls_index(xy, h), BALLS_NAMES, range(len(BALLS_NAMES))
    elif lor:
        reg, all_names, ids = lor_index(xy, h), LOR_NAMES, range(len(LOR_NAMES))
    else:
        reg = region_index(xy, h, split_far=plus)
        all_names = REGION_PLUS_NAMES if plus else REGION_NAMES
        ids = EXIT_IDS if exits else range(len(all_names))
    tot = np.bincount(rows, weights=ms, minlength=n)
    if "count" in kinds:
        # a visit starts at a trial's first fixation or wherever the region changes,
        # so consecutive fixations inside one region count once
        order = np.lexsort((np.asarray(d["fix_onset_ms"], dtype=float), rows))
        r_o, g_o = rows[order], reg[order]
        starts = np.ones(len(order), dtype=bool)
        starts[1:] = (r_o[1:] != r_o[:-1]) | (g_o[1:] != g_o[:-1])
    X, names = [], []
    for kind in kinds:
        for k in ids:
            name = all_names[k]
            sel = reg == k
            if kind == "dwell":
                X.append(np.divide(np.bincount(rows[sel], weights=ms[sel], minlength=n), tot,
                                   out=np.zeros(n), where=tot > 0))
            elif kind == "binary":
                X.append((np.bincount(rows[sel], minlength=n) > 0).astype(float))
            elif kind == "count":
                X.append(np.bincount(r_o[starts & (g_o == k)], minlength=n).astype(float))
            else:
                raise ValueError(kind)
            names.append(f"{kind}: {name}")
    if plus:
        X.append(within_session_z(np.asarray(d["pupil_change_raw"], dtype=float), d["session"].astype(str)))
        names.append("pupil change z")
    return np.stack(X, 1), names


# ---- assembly ----------------------------------------------------------------


def dataset(monkey, branch, *, labels="dendro", bin_ms=RAW_BIN_MS, refresh=False):
    """``(X, names, meta)`` over the labelled trials of `branch` in the
    `labels` scope (its sessions, label stem and mazes)."""
    scope = LABEL_SCOPES[labels]
    if branch in RAW_KINDS:
        d = load_raw(monkey, labels=labels, bin_ms=bin_ms, refresh=refresh)
        X, names = raw_matrix(d)
        if branch == "raw_plus":  # timing (traces) + static (gaze map)
            G, gnames = grid_matrix(d)
            X, names = np.hstack([X, G]), names + gnames
    elif branch in REGIONS_KINDS:
        d = load_fix(monkey, labels=labels, refresh=refresh)
        origin = (CAPSULE_ORIGIN_TIGHT_DEG if is_capsules_tight(branch) else CAPSULE_ORIGIN_DEG
                  if is_capsules(branch) else None)
        X, names = regions_matrix(d, REGIONS_KINDS[branch], plus=is_plus(branch), exits=is_exits(branch),
                                  lor=is_lor(branch), balls=is_balls(branch), capsule_origin=origin)
    else:
        raise ValueError(branch)
    sess = d["session"].astype(str)
    trials = np.asarray(d["trial_indices_all"], dtype=int)
    lookup = label_lookup(scope["sessions"][monkey], scope["stem"])
    y = np.asarray([lookup.get((s, int(t)), np.nan) for s, t in zip(sess, trials)])
    keep = np.isfinite(y) & np.isin(np.asarray(d["maze_id"], dtype=int), scope["mazes"])
    meta = dict(session=sess[keep], trial=trials[keep], maze=np.asarray(d["maze_id"], dtype=int)[keep],
                y=y[keep].astype(int))
    return X[keep], names, meta
