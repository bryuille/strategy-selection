"""Models, cross-validation, the within-maze pairwise AUC and its shuffle null.

Every model sees the features plus a maze one-hot. Folds are stratified on
(session, maze, label) so each test fold mirrors the pool. The score is the
**within-maze pairwise AUC**: over all (H, S) pairs of trials from the same
maze, the fraction where the H trial got the lower S-probability (ties count
half). Maze identity cannot move it. Its null is the same CV repeated on
labels shuffled within (session, maze).

With `ORDER_CONTROL` (build --order-control) linear + quadratic trial order
join the design and the null becomes a conditional permutation: labels and
order stay fixed, each feature keeps its within-session order trend, and only
its residual is shuffled within (session, maze). Order earns the same credit
in the fit and in the null, so p asks whether gaze adds beyond drift; the
AUC itself (and the held-out AUC) includes order's share, so read it against
the null mean and the `order` branch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from classifier.config import MAZES, N_FOLDS, N_PERM, N_REPEATS, SEED


def make_model(kind, seed=SEED):
    if kind == "logreg":
        # standardise over the finite values, then NaN -> 0 (the column mean)
        return make_pipeline(StandardScaler(), SimpleImputer(strategy="constant", fill_value=0.0),
                             LogisticRegression(C=0.1, max_iter=2000))
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                              l2_regularization=1.0, random_state=seed)
    raise ValueError(kind)


INTERACTIONS = False  # set by build --interactions: gaze x maze terms


def with_maze(X, maze):
    """Append the maze one-hot; with `INTERACTIONS`, also one copy of the
    features per maze (zero outside that maze), so each maze gets its own
    weights and the shared block carries what is common. L2 then shrinks the
    per-maze deviations toward the shared weights."""
    onehot = (maze[:, None] == np.asarray(MAZES)[None]).astype(float)
    blocks = [X, onehot]
    if INTERACTIONS:
        blocks += [X * onehot[:, [k]] for k in range(len(MAZES))]
    return np.hstack(blocks)


ORDER_CONTROL = False  # set by build --order-control: order covariates + conditional permutation null


def order_rank(meta):
    """Trial position within its session, 0 (first scored trial) .. 1 (last)."""
    r = np.zeros(len(meta["session"]))
    for s in np.unique(meta["session"]):
        idx = np.flatnonzero(meta["session"] == s)
        ranks = np.argsort(np.argsort(meta["trial"][idx], kind="stable"), kind="stable")
        r[idx] = ranks / max(idx.size - 1, 1)
    return r


def order_terms(meta):
    """Linear and quadratic trial order: the drift a gaze model must beat."""
    r = order_rank(meta)
    return np.stack([r, r ** 2], 1)


def design(X, meta):
    """`with_maze`, plus the order terms under `ORDER_CONTROL`."""
    Xm = with_maze(X, meta["maze"])
    return np.hstack([Xm, order_terms(meta)]) if ORDER_CONTROL else Xm


def order_split(X, meta):
    """``(fit, resid)``: each feature regressed on (1, order, order^2) within
    session over its finite values; NaN stays NaN in `resid`."""
    T = np.column_stack([np.ones(len(X)), order_terms(meta)])
    fit = np.zeros_like(X, dtype=float)
    for s in np.unique(meta["session"]):
        rows = np.flatnonzero(meta["session"] == s)
        for j in range(X.shape[1]):
            v = X[rows, j]
            ok = np.isfinite(v)
            if ok.sum() > T.shape[1]:
                beta = np.linalg.lstsq(T[rows[ok]], v[ok], rcond=None)[0]
                fit[rows, j] = T[rows] @ beta
            elif ok.any():
                fit[rows, j] = v[ok].mean()
    return fit, X - fit


def conditional_permute(fit, resid, groups, rng):
    """Order-predicted part kept, residual rows permuted within `groups`:
    a copy of X that keeps every feature's drift but breaks any link to the
    label beyond it."""
    out = resid.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        out[idx] = resid[rng.permutation(idx)]
    return fit + out


def within_maze_auc(p, y, maze):
    """Pairwise AUC restricted to same-group (H, S) pairs; also per group.
    `maze` is the maze id, or any group key (e.g. ``"session|maze"``)."""
    num = den = 0.0
    per = {}
    as_key = lambda m: int(m) if np.issubdtype(np.asarray(m).dtype, np.integer) else str(m)
    for m in np.unique(maze):
        sel = maze == m
        ph, ps = p[sel & (y == 0)], p[sel & (y == 1)]
        if ph.size == 0 or ps.size == 0:
            continue
        diff = ps[:, None] - ph[None, :]
        c = (diff > 0).sum() + 0.5 * (diff == 0).sum()
        per[as_key(m)] = (c / diff.size, ph.size, ps.size)
        num += c
        den += diff.size
    return (num / den if den else np.nan), per


def strata(meta, y):
    return np.asarray([f"{s}|{m}|{int(v)}" for s, m, v in zip(meta["session"], meta["maze"], y)])


def oof_probabilities(X, y, meta, kind, *, seed=SEED, n_folds=N_FOLDS, n_repeats=N_REPEATS):
    """Mean out-of-fold S-probability over `n_repeats` stratified K-folds."""
    Xm = design(X, meta)
    st = strata(meta, y)
    # a stratum with < n_folds members cannot be split evenly; sklearn warns and
    # distributes it, which is what we want
    p = np.zeros(len(y))
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed * 1000 + r)
        for tr, te in skf.split(Xm, st):
            model = make_model(kind, seed=seed * 1000 + r)
            model.fit(Xm[tr], y[tr])
            p[te] += model.predict_proba(Xm[te])[:, 1]
    return p / n_repeats


def shuffle_within(y, groups, rng):
    out = y.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        out[idx] = rng.permutation(out[idx])
    return out


@dataclass
class Evaluation:
    auc: float
    per_maze: dict
    null: np.ndarray
    z: float
    p: float
    oof: np.ndarray
    pooled_auc: float  # ordinary AUC over all trials: inflated by maze, shown for contrast
    session_holdout: dict  # held-out session -> within-maze AUC (trained on the others)
    holdout_auc: float  # within-maze AUC over all held-out predictions pooled


def evaluate(X, y, meta, kind, *, n_perm=N_PERM, seed=SEED, n_repeats=N_REPEATS):
    oof = oof_probabilities(X, y, meta, kind, seed=seed, n_repeats=n_repeats)
    auc, per = within_maze_auc(oof, y, meta["maze"])
    pooled = within_maze_auc(oof, y, np.zeros_like(meta["maze"]))[0]

    groups = np.asarray([f"{s}|{m}" for s, m in zip(meta["session"], meta["maze"])])
    rng = np.random.default_rng((seed, 7))
    null = np.empty(n_perm)
    if ORDER_CONTROL:
        # labels and order fixed; only gaze beyond its own drift is shuffled
        fit, resid = order_split(X, meta)
    for i in range(n_perm):
        if ORDER_CONTROL:
            Xp, yp = conditional_permute(fit, resid, groups, rng), y
        else:
            Xp, yp = X, shuffle_within(y, groups, rng)
        pp = oof_probabilities(Xp, yp, meta, kind, seed=seed + 1 + i, n_repeats=1)
        null[i] = within_maze_auc(pp, yp, meta["maze"])[0]
    fin = null[np.isfinite(null)]
    z = (auc - fin.mean()) / fin.std(ddof=1) if fin.size > 1 and fin.std() > 0 else np.nan
    p = (1 + np.sum(fin >= auc)) / (fin.size + 1)  # one-sided: better than chance

    holdout, holdout_auc = session_holdout(X, y, meta, kind, seed=seed)
    return Evaluation(auc, per, null, z, p, oof, pooled, holdout, holdout_auc)


def session_holdout(X, y, meta, kind, *, seed=SEED):
    """Leave one session out: train on the other sessions, score the held-out
    one. Returns ``({session: within-maze AUC}, pooled AUC)``; the pooled
    value counts only same-(session, maze) pairs, since each session's
    probabilities come from a different model and are not comparable across
    sessions. With two sessions this is train-A-test-B both ways."""
    Xm = design(X, meta)
    sessions = np.unique(meta["session"])
    if sessions.size < 2:
        return {}, np.nan
    groups = np.asarray([f"{s}|{m}" for s, m in zip(meta["session"], meta["maze"])])
    per, p_all = {}, np.full(len(y), np.nan)
    for s in sessions:
        te = meta["session"] == s
        tr = ~te
        if np.unique(y[tr]).size < 2 or np.unique(y[te]).size < 2:
            continue
        model = make_model(kind, seed=seed)
        model.fit(Xm[tr], y[tr])
        p_all[te] = model.predict_proba(Xm[te])[:, 1]
        per[str(s)] = within_maze_auc(p_all[te], y[te], meta["maze"][te])[0]
    ok = np.isfinite(p_all)
    pooled = within_maze_auc(p_all[ok], y[ok], groups[ok])[0] if ok.any() else np.nan
    return per, pooled


def coefficients(X, y, meta, names, *, seed=SEED):
    """Standardised logistic coefficients on the whole pool (features only;
    the maze one-hot columns are dropped from the report)."""
    Xm = design(X, meta)
    model = make_model("logreg", seed=seed).fit(Xm, y)
    coef = model[-1].coef_.ravel()[: len(names)]
    return dict(zip(names, coef))


def gbm_importance(X, y, meta, names, *, seed=SEED, n_repeats=5):
    """Permutation importance of each feature for the within-maze AUC, fitted
    on the whole pool (optimistic, for ranking only)."""
    Xm = design(X, meta)
    model = make_model("gbm", seed=seed).fit(Xm, y)
    base = within_maze_auc(model.predict_proba(Xm)[:, 1], y, meta["maze"])[0]
    rng = np.random.default_rng(seed)
    imp = {}
    for j, name in enumerate(names):
        drop = []
        for _ in range(n_repeats):
            Xp = Xm.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drop.append(base - within_maze_auc(model.predict_proba(Xp)[:, 1], y, meta["maze"])[0])
        imp[name] = float(np.mean(drop))
    return imp
