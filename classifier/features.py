"""Feature matrices for both branches, one cache per monkey, branch and scope.

Raw: `<Monkey>_clf_raw[_svm].npz`, built from the pooled eye cache and the
pupil tables. Engineered: computed on the fly from msp_revised's fixation
cache (`<Monkey>_msp_revised_fix[_svm].npz`, read as an npz) and the same
pupil tables. Both return ``(X, names, meta)`` with `meta` holding session,
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
from data.mat import load_npz
from classifier.config import (
    EXIT_BALL_DEG,
    GRID_LIM_DEG,
    LABEL_SCOPES,
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
    RING_EDGES_DEG,
)

RAW_SCHEMA = 2  # traces stored un-interpolated at 1 kHz; widths derived on load


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
            warnings.warn(f"{path} missing (msp_revised.pupil_extract --session {s} on the cluster, "
                          f"then pull out/pupil/); pupil is NaN for {s}", stacklevel=2)
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
        n = min(t_s.size, x.size, y.size)
        t_ms = t_s[:n] * 1000.0
        x, y = x[:n].copy(), y[:n].copy()
        oob = ~np.isfinite(x) | ~np.isfinite(y) | (np.abs(x) > POSITION_LIMIT_DEG) | (np.abs(y) > POSITION_LIMIT_DEG)
        x[oob] = np.nan
        y[oob] = np.nan
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


# ---- engineered branch -------------------------------------------------------


def load_fix(monkey, *, labels="dendro", refresh=False):
    """msp_revised's fixation cache over the scope's sessions. The dendro
    cache is msp_revised's own and is only read; the svm one
    (``<Monkey>_msp_revised_fix_svm``) is built through
    `msp_revised.features.load` on first use (needs pymovements)."""
    sessions = tuple(LABEL_SCOPES[labels]["sessions"][monkey])
    path = processed_npz(_scoped(f"{monkey}_msp_revised_fix", labels))
    if path.exists() and not refresh:
        d = load_npz(path)
        if tuple(np.asarray(d["sessions"]).astype(str)) == sessions:
            return d
    if labels == "dendro":
        raise FileNotFoundError(f"{path} missing or stale; run `python -m msp_revised.features` first")
    from msp_revised.features import load as build_fix
    return build_fix(monkey, sessions=sessions, cache_name=path.stem, refresh=refresh)


def _exit_landmarks(h):
    h1, h2, h3, h4, h5, h6 = (h[:, k] for k in range(6))
    return np.stack([np.stack([-h1, h2], 1), np.stack([-h1, -h3], 1),
                     np.stack([h4, h5], 1), np.stack([h4, -h6], 1)], 1)  # LU, LD, RU, RD


ENGINEERED_NAMES = (
    "origin dwell share", "o-LU share", "o-LD share", "o-RU share", "o-RD share",
    "LU visited", "LD visited", "RU visited", "RD visited",
    "LU dwell", "LD dwell", "RU dwell", "RD dwell",
    "ring 2-6 share", "ring 6-10 share", "ring >10 share",
    "n fixations", "first fix: origin", "first fix: left", "first fix: right",
    "last fix: origin", "last fix: left", "last fix: right",
    "first fix onset (s)", "pupil mean z", "pupil change z",
)


def engineered_matrix(d):
    """One row per cache trial (all of them; the caller selects labelled rows)."""
    n = len(d["session"])
    rows = np.asarray(d["fix_row"], dtype=int)
    xy = np.asarray(d["fix_xy_d"], dtype=float)
    ms = np.asarray(d["fix_valid_ms"], dtype=float)
    onset = np.asarray(d["fix_onset_ms"], dtype=float)
    h = np.asarray(d["h"], dtype=float)[rows]
    rad = np.linalg.norm(xy, axis=1)
    tot = np.bincount(rows, weights=ms, minlength=n)
    share = lambda mask: np.divide(np.bincount(rows[mask], weights=ms[mask], minlength=n), tot,
                                   out=np.zeros(n), where=tot > 0)
    in_disc = rad <= ORIGIN_DISC_DEG
    right, up = xy[:, 0] >= 0, xy[:, 1] >= 0
    X = [share(in_disc),
         share(in_disc & ~right & up), share(in_disc & ~right & ~up),
         share(in_disc & right & up), share(in_disc & right & ~up)]
    dist = np.linalg.norm(xy[:, None, :] - _exit_landmarks(h), axis=-1)  # (F, 4)
    for k in range(4):
        X.append((np.bincount(rows[dist[:, k] <= EXIT_BALL_DEG], minlength=n) > 0).astype(float))
    for k in range(4):
        X.append(share(dist[:, k] <= EXIT_BALL_DEG))
    e = RING_EDGES_DEG
    X += [share((rad > e[0]) & (rad <= e[1])), share((rad > e[1]) & (rad <= e[2])), share(rad > e[2])]
    X.append(np.bincount(rows, minlength=n).astype(float))
    order = np.lexsort((onset, rows))
    first_i = order[np.unique(rows[order], return_index=True)[1]]
    last_i = order[::-1][np.unique(rows[order[::-1]], return_index=True)[1]]
    for idx in (first_i, last_i):
        for cond in (rad[idx] <= ORIGIN_DISC_DEG,
                     (rad[idx] > ORIGIN_DISC_DEG) & (xy[idx, 0] < 0),
                     (rad[idx] > ORIGIN_DISC_DEG) & (xy[idx, 0] >= 0)):
            v = np.zeros(n)
            v[rows[idx][cond]] = 1.0
            X.append(v)
    first_on = np.full(n, np.nan)
    first_on[rows[first_i]] = onset[first_i] / 1000.0
    X.append(np.where(np.isfinite(first_on), first_on, np.nanmedian(first_on)))
    sess = d["session"].astype(str)
    X.append(within_session_z(np.asarray(d["pupil_mean_raw"], dtype=float), sess))
    X.append(within_session_z(np.asarray(d["pupil_change_raw"], dtype=float), sess))
    X = np.stack(X, 1)
    assert X.shape[1] == len(ENGINEERED_NAMES)
    return X, list(ENGINEERED_NAMES)


# ---- regions branch ----------------------------------------------------------
#
# The engineered fits put their weight on two families, in both monkeys and
# both scopes: dwell-time shares over the origin-disc quadrants and the exit
# balls, and whether an exit ball was visited at all. `regions` keeps exactly
# those, over one partition of the screen in degrees: the four quadrants of
# the origin disc (ORIGIN_DISC_DEG), the four exit balls (EXIT_BALL_DEG), and
# everything else. Two sets over the same nine regions, as in msp's
# binary / dwell pair: `dwell` (share of in-window fixation time, sums to 1)
# and `binary` (visited or not). No ordering, latency or pupil features.

REGION_NAMES = ("origin LU", "origin LD", "origin RU", "origin RD",
                "exit LU", "exit LD", "exit RU", "exit RD", "elsewhere")
REGIONS_KINDS = {"regions": ("dwell", "binary"), "regions_dwell": ("dwell",), "regions_binary": ("binary",)}


def region_index(xy, h):
    """Region id per fixation (``REGION_NAMES`` order). Exit balls win over the
    origin disc where they overlap (they do not at the default radii)."""
    rad = np.linalg.norm(xy, axis=1)
    right, up = xy[:, 0] >= 0, xy[:, 1] >= 0
    idx = np.full(len(xy), 8, dtype=int)
    in_disc = rad <= ORIGIN_DISC_DEG
    idx[in_disc & ~right & up] = 0
    idx[in_disc & ~right & ~up] = 1
    idx[in_disc & right & up] = 2
    idx[in_disc & right & ~up] = 3
    dist = np.linalg.norm(xy[:, None, :] - _exit_landmarks(h), axis=-1)  # (F, 4): LU, LD, RU, RD
    near = dist.min(1) <= EXIT_BALL_DEG
    idx[near] = 4 + dist.argmin(1)[near]
    return idx


def regions_matrix(d, kinds=("dwell", "binary")):
    """One row per cache trial: per-region dwell share and / or visited flag."""
    n = len(d["session"])
    rows = np.asarray(d["fix_row"], dtype=int)
    xy = np.asarray(d["fix_xy_d"], dtype=float)
    ms = np.asarray(d["fix_valid_ms"], dtype=float)
    h = np.asarray(d["h"], dtype=float)[rows]
    reg = region_index(xy, h)
    tot = np.bincount(rows, weights=ms, minlength=n)
    X, names = [], []
    for kind in kinds:
        for k, name in enumerate(REGION_NAMES):
            sel = reg == k
            if kind == "dwell":
                X.append(np.divide(np.bincount(rows[sel], weights=ms[sel], minlength=n), tot,
                                   out=np.zeros(n), where=tot > 0))
            elif kind == "binary":
                X.append((np.bincount(rows[sel], minlength=n) > 0).astype(float))
            else:
                raise ValueError(kind)
            names.append(f"{kind}: {name}")
    return np.stack(X, 1), names


# ---- assembly ----------------------------------------------------------------


def dataset(monkey, branch, *, labels="dendro", bin_ms=RAW_BIN_MS, refresh=False):
    """``(X, names, meta)`` over the labelled trials of `branch` in the
    `labels` scope (its sessions, label stem and mazes)."""
    scope = LABEL_SCOPES[labels]
    if branch == "raw":
        d = load_raw(monkey, labels=labels, bin_ms=bin_ms, refresh=refresh)
        X, names = raw_matrix(d)
    elif branch == "engineered":
        d = load_fix(monkey, labels=labels, refresh=refresh)
        X, names = engineered_matrix(d)
    elif branch in REGIONS_KINDS:
        d = load_fix(monkey, labels=labels, refresh=refresh)
        X, names = regions_matrix(d, REGIONS_KINDS[branch])
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
