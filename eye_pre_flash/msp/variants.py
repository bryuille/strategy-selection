"""Feature transforms applied to the (n, 5) binary-occupancy block.

``full``          the cached vectors, unmodified.
``no_origin``     drop the origin state (column `ORIGIN_STATE`, always 0 here
                  because the codebook is fixed), leaving the four exits.
                  Central fixation is the profile every trial shares; this is
                  the structural way of removing it.
``mean_removed``  subtract a grand-mean profile -- the mean vector over a set
                  of trials -- fitted by the caller. `build` fits it per
                  (monkey, maze) over exactly the rows the estimator pools.

Subtracting a common profile makes the mean off-diagonal r go negative by
arithmetic (centred vectors sum to ~0), so absolute r is uninterpretable under
`mean_removed`; the diagonal-vs-off-diagonal gap, and `z`, survive. Read
`no_origin` alongside it: agreement means the finding is robust, and an effect
only the fitted variant shows should be distrusted.

Reduced from `eye_pre_flash/maze_strategy_pairs/variants.py` to the one block
this package uses; the origin state is no longer looked up because it is state
0 by construction (`codebook.py`).
"""

from __future__ import annotations

import numpy as np

from eye_pre_flash.msp.codebook import K, ORIGIN_STATE

VARIANTS = ("full", "no_origin", "mean_removed")


def no_origin_dims(k=K, origin=ORIGIN_STATE):
    """Dimension indices to KEEP under `no_origin`."""
    return [d for d in range(k) if d != origin]


def apply_no_origin(X, k=K, origin=ORIGIN_STATE):
    dims = no_origin_dims(k, origin)
    return np.asarray(X, dtype=float)[:, dims], dims


def fit_grand_mean(X):
    """Mean feature vector over the finite rows of `X`."""
    X = np.asarray(X, dtype=float)
    finite = np.isfinite(X).all(axis=1)
    Xf = X[finite]
    if Xf.shape[0] == 0:
        return np.zeros(X.shape[1])
    return Xf.mean(axis=0)


def apply_mean_removed(X, mean_profile):
    return np.asarray(X, dtype=float) - np.asarray(mean_profile, dtype=float)


def apply_variant(X, variant, *, mean_profile=None):
    """Apply one named variant to an (n, K) block.

    Returns ``(X_variant, dims)`` where `dims` are the original dimension
    indices retained (identity except under `no_origin`).
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    X = np.asarray(X, dtype=float)
    if variant == "full":
        return X, list(range(X.shape[1]))
    if variant == "no_origin":
        return apply_no_origin(X, X.shape[1], ORIGIN_STATE)
    if mean_profile is None:
        raise ValueError("mean_removed needs a precomputed grand-mean profile")
    return apply_mean_removed(X, mean_profile), list(range(X.shape[1]))
