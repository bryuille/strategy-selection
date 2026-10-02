"""One logistic regression: strategy ~ maze offsets + gaze predictors.

``log-odds(S) = offset[maze] + sum over states s and predictors k of beta[s,k] * x[s,k]``

* ``offset[maze]``: one free number per maze (sessions pooled, no global
  intercept). It absorbs everything constant within a maze: how often that
  maze is labelled S, and anything about its geometry that draws the eye the
  same way on every trial. Predictors are centred, so the offset is the
  log-odds of S in that maze for a trial with average gaze.
* ``x[s,k]``, for each codebook state s (origin, LU, LD, RU, RD):

  ``occupied``  1 if state s received any dwell, centred (beta = visited vs not)
  ``visits``    z(visit count), 0 where unvisited
  ``duration``  z(log mean fixation duration), 0 where unvisited

  The three z-scores are taken over *visited* trials only, so beta is the
  change in log-odds of S per 1 SD among visited trials, and ``occupied``
  separately carries visited-vs-not.

Label = 1 is sequential (S), 0 is hierarchical (H): beta > 0 means "more S".
Fitting is unpenalized Newton-IRLS; intervals are Wald (inverse Hessian).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit
from scipy.stats import norm

from regression.config import (
    ALPHA,
    K,
    MAX_R2,
    MIN_OFF_MODE,
    MIN_PER_LABEL,
    PREDICTORS,
    STATE_NAMES,
    VISIT_RATE_RANGE,
    ZERO_SD,
)


# ---- rows, mazes, states -----------------------------------------------------------


def select_mazes(mazes, y, min_per_label=MIN_PER_LABEL):
    """``(kept maze ids, table rows)``; a maze needs `min_per_label` of each label.
    Nearly pure mazes would be absorbed by their own offset (separation)."""
    table, kept = [], []
    for m in sorted(set(int(v) for v in mazes)):
        sel = mazes == m
        n_h, n_s = int((y[sel] == 0).sum()), int((y[sel] == 1).sum())
        ok = n_h >= min_per_label and n_s >= min_per_label
        table.append({"maze": m, "n_H": n_h, "n_S": n_s, "kept": ok})
        if ok:
            kept.append(m)
    return kept, table


def select_states(occ, rate_range=VISIT_RATE_RANGE):
    """``(kept state indices, visit rate per state)``. Label-free."""
    rate = occ.mean(axis=0) if occ.size else np.zeros(K)
    lo, hi = rate_range
    kept = [s for s in range(occ.shape[1]) if lo <= rate[s] <= hi]
    return kept, rate


# ---- design ----------------------------------------------------------------------


@dataclass
class Design:
    X: np.ndarray
    names: list  # column names: "offset[m2]" or "<predictor>[<state>]"
    n_offsets: int
    n_lead: int  # offsets + window column: everything before the gaze columns
    dropped: list = field(default_factory=list)  # (state, predictor, reason)


def _z_visited(raw, visited):
    """z-score `raw` over visited rows, 0 elsewhere. Returns (column, sd)."""
    ref = raw[visited]
    if ref.size < 2:
        return np.zeros(raw.size), 0.0
    sd = ref.std()
    if sd < ZERO_SD:
        return np.zeros(raw.size), sd
    return np.where(visited, (raw - ref.mean()) / sd, 0.0), sd


def _state_columns(meas, s):
    """The four candidate columns for state `s`, as ``{key: (column, reason)}``;
    a column that cannot be built has ``column = None`` and a reason."""
    occ = meas["occ"][:, s].astype(float)
    visited = occ > 0
    out = {}
    if occ.std() < ZERO_SD:
        out["occupied"] = (None, "zero variance")
    else:
        out["occupied"] = (occ - occ.mean(), None)

    with np.errstate(invalid="ignore", divide="ignore"):
        log_dur = np.where(visited, np.log(meas["dur_ms"][:, s]), np.nan)
    visits = np.where(visited, meas["visits"][:, s], np.nan)

    ref = visits[visited]
    if ref.size:
        _, counts = np.unique(ref, return_counts=True)
        off_mode = int(ref.size - counts.max())
    else:
        off_mode = 0
    for key, raw in (("visits", visits), ("duration", log_dur)):
        if key == "visits" and off_mode < MIN_OFF_MODE:
            out[key] = (None, f"only {off_mode} trials off the modal count")
            continue
        col, sd = _z_visited(raw, visited)
        out[key] = (None, "zero variance") if sd < ZERO_SD else (col, None)
    return out


def build_design(meas, mazes, maze_names, states, exclude=None):
    """Design matrix: one offset per maze, then each kept state's predictors.

    A predictor is dropped (and recorded with its reason) when it cannot vary,
    when visits are almost always the modal count, or when more than `MAX_R2`
    of it is explained by the earlier predictors of the same state.
    `exclude` maps ``(state name, predictor)`` to a reason: those columns are
    dropped up front (used when a fit fails to converge).
    """
    n = mazes.size
    cols = [(mazes == m).astype(float) for m in maze_names]
    names = [f"offset[m{m}]" for m in maze_names]
    dropped = []
    if "window_ms" in meas:  # variable-length window: control for its length
        w, sd = _z_visited(np.log(np.asarray(meas["window_ms"], float)), np.ones(n, bool))
        if sd >= ZERO_SD:
            cols.append(w)
            names.append("window")
    n_lead = len(cols)
    for s in states:
        kept = []  # (key, column)
        for key, _ in PREDICTORS:
            col, reason = _state_columns(meas, s)[key]
            if (STATE_NAMES[s], key) in (exclude or {}):
                col, reason = None, exclude[(STATE_NAMES[s], key)]
            if col is not None and kept:
                P = np.column_stack([c for _, c in kept])
                resid = col - P @ np.linalg.lstsq(P, col, rcond=None)[0]
                r2 = 1.0 - float(resid @ resid) / float(col @ col)
                # participation ratio: how many trials the leftover variation is spread over
                n_eff = float((resid @ resid) ** 2 / max(float(np.sum(resid**4)), 1e-300))
                parents = " + ".join(k for k, _ in kept)
                if r2 > MAX_R2:
                    col, reason = None, f"{r2:.0%} explained by {parents}"
                elif n_eff < MIN_OFF_MODE:
                    col, reason = None, f"differs from {parents} on only ~{n_eff:.0f} trials"
            if col is None:
                dropped.append((STATE_NAMES[s], key, reason))
                continue
            kept.append((key, col))
            cols.append(col)
            names.append(f"{key}[{STATE_NAMES[s]}]")
    return Design(np.column_stack(cols), names, len(maze_names), n_lead, dropped)


def vif(X, n_lead):
    """Variance inflation of each gaze column after the maze offsets are
    removed: how many times wider its interval is than if it were uncorrelated
    with the other gaze columns. ~1 is clean; > 5 means its beta is hard to
    separate from the others'."""
    G = X[:, n_lead:]
    if G.shape[1] == 0:
        return np.empty(0)
    O = X[:, :n_lead]
    G = G - O @ np.linalg.lstsq(O, G, rcond=None)[0]
    C = np.corrcoef(G, rowvar=False).reshape(G.shape[1], G.shape[1])
    return np.diag(np.linalg.inv(C))


# ---- fitting -------------------------------------------------------------------


@dataclass
class Fit:
    beta: np.ndarray
    loglik: float
    cov: np.ndarray | None
    converged: bool
    n_iter: int
    cond: float


def _loglik(X, y, beta):
    eta = X @ beta
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def fit_logit(X, y, *, max_iter=100, tol=1e-8):
    """Newton-IRLS with step halving. A fit is non-converged when it hits
    `max_iter` or any |beta| > 30 (quasi-separation: the MLE is running off
    to infinity)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    p = X.shape[1]
    pen = np.zeros(p)
    beta = np.zeros(p)

    def objective(b):
        return _loglik(X, y, b) - 0.5 * float(np.sum(pen * b * b))

    obj = objective(beta)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        mu = expit(X @ beta)
        w = mu * (1.0 - mu)
        grad = X.T @ (y - mu) - pen * beta
        H = (X * w[:, None]).T @ X + np.diag(pen)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        t = 1.0
        while t > 1e-6:
            cand = beta + t * step
            new = objective(cand)
            if new >= obj - 1e-12:
                break
            t *= 0.5
        beta, delta, obj = cand, new - obj, new
        if abs(delta) < tol and np.max(np.abs(t * step)) < 1e-6:
            converged = True
            break
    if np.any(np.abs(beta) > 30):
        converged = False

    mu = expit(X @ beta)
    w = mu * (1.0 - mu)
    H = (X * w[:, None]).T @ X + np.diag(pen)
    try:
        cond = float(np.linalg.cond(H))
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cond, cov = float("inf"), None
    return Fit(beta, _loglik(X, y, beta), cov, converged, it, cond)


def wald_table(fit, names):
    se = np.sqrt(np.clip(np.diag(fit.cov), 0, None)) if fit.cov is not None else np.full(len(names), np.nan)
    z = fit.beta / np.where(se > 0, se, np.nan)
    p = 2 * norm.sf(np.abs(z))
    return se, z, p


def holm(pvals):
    p = np.asarray(pvals, float)
    out = np.full(p.size, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    m, run = order.size, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        out[i] = run
    return out


# ---- one monkey ---------------------------------------------------------------------


def analyse(meas, mazes, y, maze_names, states):
    """Fit the model; return ``(coefficient rows, fit summary)``.

    One row per offset and per (state, predictor), including predictors that
    were dropped (``status`` says why; their numbers are NaN). `p_holm` is the
    Holm adjustment over the fitted gaze coefficients only.
    """
    # A fit that does not converge (an estimate running off to infinity: the
    # labels are separated by some combination of columns) is repaired by
    # dropping the gaze column with the largest |beta| and refitting, until it
    # converges. Each dropped column is reported with this reason.
    exclude = {}
    while True:
        d = build_design(meas, mazes, maze_names, states, exclude)
        f = fit_logit(d.X, y)
        gaze_j = [j for j in range(d.n_lead, len(d.names))]
        if f.converged or not gaze_j:
            break
        j = max(gaze_j, key=lambda j: abs(f.beta[j]))
        key, _, st = d.names[j].partition("[")
        exclude[(st.rstrip("]"), key)] = "separates the labels (estimate ran off to infinity)"
    se = np.sqrt(np.clip(np.diag(f.cov), 0, None)) if f.cov is not None else np.full(len(d.names), np.nan)
    z = f.beta / np.where(se > 0, se, np.nan)
    p = 2 * norm.sf(np.abs(z))
    crit = norm.ppf(1 - ALPHA / 2)
    v = vif(d.X, d.n_lead)

    rows = []
    for j, name in enumerate(d.names):
        kind, _, rest = name.partition("[")
        rest = rest.rstrip("]")
        is_off = j < d.n_offsets
        is_win = name == "window"
        rows.append({
            "kind": "offset" if is_off else "window" if is_win else "gaze",
            "maze": int(rest[1:]) if is_off else "",
            "state": "" if is_off or is_win else rest,
            "predictor": "" if is_off or is_win else kind,
            "beta": f.beta[j], "se": se[j],
            "ci_lo": f.beta[j] - crit * se[j], "ci_hi": f.beta[j] + crit * se[j],
            "z": z[j], "p": p[j], "p_holm": np.nan,
            "vif": v[j - d.n_lead] if j >= d.n_lead else np.nan,
            "status": "fit",
        })
    gaze = [r for r in rows if r["kind"] == "gaze"]
    for r, ph in zip(gaze, holm([r["p"] for r in gaze])):
        r["p_holm"] = ph
    for state, key, reason in d.dropped:
        rows.append({"kind": "gaze", "maze": "", "state": state, "predictor": key,
                     **{k: np.nan for k in ("beta", "se", "ci_lo", "ci_hi", "z", "p", "p_holm", "vif")},
                     "status": f"not fit: {reason}"})
    summary = {
        "n_rows": int(y.size), "n_H": int((y == 0).sum()), "n_S": int((y == 1).sum()),
        "n_params": int(d.X.shape[1]), "loglik": f.loglik,
        "converged": f.converged, "cond": f.cond,
    }
    return rows, summary
