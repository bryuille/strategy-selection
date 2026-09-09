"""Feature transforms that test whether central fixation crushes the range.

The existing 6x6 occupancy matrices sit compressed against 1.0 (`similarity.md`):
every cell shares a dominant common gaze profile -- central fixation -- so all
correlations pile up at 0.81-0.99, and a within-maze strategy effect must be
smaller than the resulting between-maze effect (as little as 0.04 for Faure
occ-bin at k=12). Two variants strip that common profile so the CV engine has
somewhere to show a real effect if one exists; `full` is the baseline both are
judged against. One is structural (`no_origin`), one fitted (`mean_removed`).

``full``        the cached feature vectors, unmodified. Directly comparable to
                the existing 6x6 figures and to `similarity.md`.
``no_origin``   drop the codebook state nearest the screen origin (0, 0) --
                central fixation, almost always the state absorbing the bulk of
                every trial's dwell time. `occ_ms`/`occ_bin` lose one
                dimension; `bigram` loses every ordered pair touching it.
``mean_removed`` subtract the pooled grand-mean profile -- the mean feature
                vector over trials -- fit once per (monkey, feature, k) over
                the widest scope, not per session or per cell. Same discipline
                as the retired `pc1_removed`, and for the same reason: a
                profile that differed between sessions would put each
                session's matrix in a different subspace, which is exactly the
                cross-session comparability problem this analysis has already
                been burned by (see `similarity.md`'s "comparable core"
                writeup for the deleted `heatmap_labels`).

`mean_removed` replaced `pc1_removed` (`git log`) because projecting out PC1
was the wrong operation for the stated intent. Pearson already centres each
vector across its own dimensions, so a per-trial offset is handled for free
and never needed removing; what was missing is subtraction of the mean profile
*across trials*, which is what `mean_removed` does. PC1 is something else
again -- `sklearn`'s PCA centres before it decomposes, so PC1 is a direction
of variance *around* the grand mean rather than the grand mean itself, and
removing it could as easily have deleted the strategy effect as the common
profile. `fit_pc1` is kept as a diagnostic to measure which of those it did;
see `pc1_diagnostics`.

Subtracting a common profile makes the mean off-diagonal r go negative:
centred vectors sum to (near) zero across cells, so that is arithmetic, not a
finding. Absolute r stops being interpretable under `mean_removed`; the reads
this package relies on -- a maze's same-maze `(m, H)` vs `(m, S)` cell against
that maze's own split-half reliability -- are relative, so they survive.

`no_origin` is the more defensible statement of the same intent: dropping the
origin state is a structural claim about which dimension central fixation
occupies, not a quantity fitted from the data, so it cannot absorb strategy
variance by accident. Read the two together. If `mean_removed` and `no_origin`
agree, the finding is robust; if only the fitted one shows an effect, distrust
it.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from data.attractor import MAZE_ORIGIN

VARIANTS = ("full", "no_origin", "mean_removed")


def origin_state(codebook_xy):
    """Index of the codebook state nearest the screen origin (0, 0).

    Works identically in ``unith`` (`data.attractor`'s k-means codebook) and
    ``deg`` (the per-monkey degree-space codebook `classifier.features` fits)
    spaces, since `MAZE_ORIGIN` is ``(0.0, 0.0)`` in both -- the maze's stem
    starts at screen center regardless of the coordinate warp.
    """
    codebook_xy = np.asarray(codebook_xy, dtype=float)
    ox, oy = MAZE_ORIGIN[1]
    dist = np.hypot(codebook_xy[:, 0] - ox, codebook_xy[:, 1] - oy)
    return int(np.argmin(dist))


def no_origin_dims(feature, k, origin):
    """Dimension indices to KEEP for `feature` at codebook size `k` and origin state `origin`."""
    if feature in ("occ_ms", "occ_bin"):
        return [d for d in range(k) if d != origin]
    if feature == "bigram":
        return [
            i * k + j for i in range(k) for j in range(k) if i != origin and j != origin
        ]
    raise ValueError(f"unknown feature {feature!r}")


def apply_no_origin(X, feature, k, origin):
    dims = no_origin_dims(feature, k, origin)
    return X[:, dims], dims


def fit_pc1(X):
    """PC1 loading (unit vector) and its explained-variance fraction, over `X`.

    No longer applied to any variant -- kept to compute `pc1_diagnostics`,
    which is how the retired `pc1_removed` gets audited.
    """
    finite = np.isfinite(X).all(axis=1)
    Xf = X[finite]
    if Xf.shape[0] < 2 or Xf.shape[1] < 1:
        return np.zeros(X.shape[1]), 0.0
    pca = PCA(n_components=1, random_state=0).fit(Xf)
    return pca.components_[0], float(pca.explained_variance_ratio_[0])


def fit_grand_mean(X):
    """Pooled grand-mean profile of `X` -- the mean feature vector over trials.

    Finite-row handling mirrors `fit_pc1`, so the two are fitted over the same
    rows and their diagnostics are comparable.
    """
    X = np.asarray(X, dtype=float)
    finite = np.isfinite(X).all(axis=1)
    Xf = X[finite]
    if Xf.shape[0] == 0:
        return np.zeros(X.shape[1])
    return Xf.mean(axis=0)


def apply_mean_removed(X, mean_profile):
    """Subtract the pooled grand-mean profile from every row of `X`."""
    return np.asarray(X, dtype=float) - np.asarray(mean_profile, dtype=float)


def _pearson(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _informative_groups(groups, y):
    """Mask of rows in groups that hold both labels -- the only strata that can
    speak to a within-group label effect.

    A group where every trial carries the same label has zero label variance,
    so its within-group label residuals are all 0 while its score residuals are
    not. Keeping such rows does not make the estimate undefined, it drags it
    toward 0 -- so they are dropped rather than centred, and the count of what
    survived is reported alongside the correlation.
    """
    keep = np.zeros(groups.size, dtype=bool)
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= 2 and np.unique(y[m]).size >= 2:
            keep |= m
    return keep


def _within_group_centred(v, groups):
    """`v` with each group's own mean subtracted -- residuals after group effects."""
    v = np.asarray(v, dtype=float)
    out = np.empty_like(v)
    for g in np.unique(groups):
        m = groups == g
        out[m] = v[m] - v[m].mean()
    return out


def pc1_diagnostics(X, *, pc1, explained, mean_profile, sessions, mazes, y):
    """Did `pc1_removed` remove the common profile, or the effect?

    Scalars for `examples_dims`' ``__pc1__`` row.

    ``pc1_mean_cos`` / ``pc1_mean_cos_abs`` -- cosine between the PC1 loading
    and the L2-normalised grand-mean profile. Near 1 means the retired variant
    roughly did what `mean_removed` now does explicitly; near 0 confirms it did
    not. PCA fixes no sign convention, so read the absolute value. The
    mechanism cuts both ways: PCA centres first, so PC1 cannot *be* the mean
    profile by construction, yet on these features the state that absorbs most
    dwell time also tends to carry the most variance, so the two can still run
    close. Worth measuring rather than predicting.

    ``pc1_label_pointbiserial`` -- Pearson r of each trial's PC1 score
    (``X @ pc1``) against its 0/1 strategy label, pooled. If this is
    non-trivial, PC1 was partly a strategy direction and `pc1_removed` was
    deleting the very effect it was meant to expose; every result from that
    variant needs rereading.

    ``pc1_label_pointbiserial_within`` -- the same after centring both score
    and label inside each (session, maze) group: a partial correlation holding
    constant what the pooled version confounds. Strategy is close to a
    deterministic function of (session, maze) here (`corr.md` section 1), so
    the pooled number partly measures which session and maze a trial came
    from. This is the honest read of the same question, and the two can
    disagree sharply -- a large pooled value beside a near-zero within value
    means PC1 tracked recording day and geometry, not strategy.

    ``pc1_label_within_n_groups`` / ``pc1_label_within_n_trials`` -- how many
    (session, maze) groups held both labels, and how many trials those groups
    contained. This is the denominator the within figure rests on, and it has
    to be read with it: that near-determinism is a matter of degree, and in the
    limit where no group is mixed the within correlation is undefined (NaN with
    zero groups), not zero. A within value resting on three groups is not the
    same evidence as one resting on forty.

    `X` must be the *untransformed* feature block, over the same rows `pc1`
    and `mean_profile` were fitted on.
    """
    X = np.asarray(X, dtype=float)
    pc1 = np.asarray(pc1, dtype=float)
    mean_profile = np.asarray(mean_profile, dtype=float)
    y = np.asarray(y, dtype=float)

    n_pc1, n_mu = float(np.linalg.norm(pc1)), float(np.linalg.norm(mean_profile))
    cos = float(pc1 @ mean_profile / (n_pc1 * n_mu)) if n_pc1 and n_mu else float("nan")

    scores = X @ pc1
    sess = np.asarray(sessions).astype(str)
    groups = np.char.add(np.char.add(sess, "|"), np.asarray(mazes, dtype=int).astype(str))
    finite = np.isfinite(scores) & np.isfinite(y)
    scores, y, groups = scores[finite], y[finite], groups[finite]

    mixed = _informative_groups(groups, y)
    s_w, y_w, g_w = scores[mixed], y[mixed], groups[mixed]
    return {
        "pc1_explained_var": float(explained),
        "pc1_mean_cos": cos,
        "pc1_mean_cos_abs": abs(cos) if np.isfinite(cos) else float("nan"),
        "grand_mean_norm": n_mu,
        "pc1_label_pointbiserial": _pearson(scores, y),
        "pc1_label_pointbiserial_within": _pearson(
            _within_group_centred(s_w, g_w), _within_group_centred(y_w, g_w)
        ),
        "pc1_label_within_n_groups": int(np.unique(g_w).size),
        "pc1_label_within_n_trials": int(g_w.size),
    }


def apply_variant(X, variant, *, feature=None, k=None, origin=None, mean_profile=None):
    """Apply one named variant to a (n, d) feature block.

    Returns ``(X_variant, dim_labels)``, where ``dim_labels`` are the original
    dimension indices retained (identity for `full` and `mean_removed`, a
    strict subset for `no_origin`) -- carried through so `examples_dims_k<K>.csv`
    can report a real dimension label rather than a post-drop position.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    if variant == "full":
        return X, list(range(X.shape[1]))
    if variant == "no_origin":
        if feature is None or k is None or origin is None:
            raise ValueError("no_origin needs feature, k and origin")
        X_v, dims = apply_no_origin(X, feature, k, origin)
        return X_v, dims
    # mean_removed
    if mean_profile is None:
        raise ValueError("mean_removed needs a precomputed grand-mean profile")
    return apply_mean_removed(X, mean_profile), list(range(X.shape[1]))
