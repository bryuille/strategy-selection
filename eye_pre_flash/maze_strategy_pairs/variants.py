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
profile. The `fit_pc1` / `pc1_diagnostics` pair that measured which of those
it did went with the estimator rewrite; recover from git if ever needed again.

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

VARIANTS = ("full", "no_origin", "mean_removed")

# Screen centre in unit H. The maze's stem starts here by construction, so the
# origin is (0, 0) whatever the trial's arm lengths.
MAZE_ORIGIN_XY = (0.0, 0.0)


def origin_state(codebook_xy):
    """Index of the codebook state nearest the screen origin (0, 0)."""
    codebook_xy = np.asarray(codebook_xy, dtype=float)
    ox, oy = MAZE_ORIGIN_XY
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
