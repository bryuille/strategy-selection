"""Feature transforms that test whether central fixation crushes the range.

The existing 6x6 occupancy matrices sit compressed against 1.0 (`similarity.md`):
every cell shares a dominant common gaze profile -- central fixation -- so all
correlations pile up at 0.81-0.99, and a within-maze strategy effect must be
smaller than the resulting between-maze effect (as little as 0.04 for Faure
occ-bin at k=12). Two variants strip that common profile so the CV engine has
somewhere to show a real effect if one exists; `full` is the baseline both are
judged against.

``full``        the cached feature vectors, unmodified. Directly comparable to
                the existing 6x6 figures and to `similarity.md`.
``no_origin``   drop the codebook state nearest the screen origin (0, 0) --
                central fixation, almost always the state absorbing the bulk of
                every trial's dwell time. `occ_ms`/`occ_bin` lose one
                dimension; `bigram` loses every ordered pair touching it.
``pc1_removed`` project out the leading principal component of the pooled
                trial matrix, fit once per (monkey, feature, k, space) over the
                widest scope, not per session or per cell -- a direction that
                differed between sessions would put each session's matrix in a
                different subspace, which is exactly the cross-session
                comparability problem this analysis has already been burned by
                (see `similarity.md`'s "comparable core" writeup for the
                deleted `heatmap_labels`).

A removed direction makes the mean off-diagonal r go negative: centred vectors
sum to (near) zero across cells, so that is arithmetic, not a finding.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from data.attractor import MAZE_ORIGIN

VARIANTS = ("full", "no_origin", "pc1_removed")


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
    """PC1 loading (unit vector) and its explained-variance fraction, over `X`."""
    finite = np.isfinite(X).all(axis=1)
    Xf = X[finite]
    if Xf.shape[0] < 2 or Xf.shape[1] < 1:
        return np.zeros(X.shape[1]), 0.0
    pca = PCA(n_components=1, random_state=0).fit(Xf)
    return pca.components_[0], float(pca.explained_variance_ratio_[0])


def apply_pc1_removed(X, pc1):
    """Project `pc1` out of every row of `X`. `pc1` need not be unit norm."""
    pc1 = np.asarray(pc1, dtype=float)
    norm2 = float(pc1 @ pc1)
    if norm2 == 0:
        return X.copy()
    proj = (X @ pc1) / norm2
    return X - np.outer(proj, pc1)


def apply_variant(X, variant, *, feature=None, k=None, origin=None, pc1=None):
    """Apply one named variant to a (n, d) feature block.

    Returns ``(X_variant, dim_labels)``, where ``dim_labels`` are the original
    dimension indices retained (identity for `full` and `pc1_removed`, a
    strict subset for `no_origin`) -- carried through so `examples_dims.csv`
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
    # pc1_removed
    if pc1 is None:
        raise ValueError("pc1_removed needs a precomputed pc1 loading")
    return apply_pc1_removed(X, pc1), list(range(X.shape[1]))
