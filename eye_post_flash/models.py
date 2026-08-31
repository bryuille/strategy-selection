"""The manuscript's decision models as posteriors over the four exits.

Every model maps a pair of measured intervals `tm = (tm1, tm2)` to a
distribution over the four exits. The single-interval conditional is scalar
timing throughout -- `p(tm | t) = N(tm; t, wm * t)` -- so `wm` is the one
parameter every model carries.

    optimal       p(e) ~ p(tm1|t_e1) p(tm2|t_e2)                (4AFC)
    optimal_lapse (1 - lam) p_optimal(e) + lam/4                (4AFC, +lam)
    total_time    p(e) ~ p(tm1+tm2 | t_e1+t_e2), the convolution
                  (sd = wm sqrt(t_e1^2 + t_e2^2))               (4AFC)
    only_first    p(e) ~ p(tm1|t_e1) -- ignores the second interval, so the
                  two exits on a branch are exactly tied
    only_second   p(e) ~ p(tm2|t_e2) -- ignores the first interval
    hierarchical  p(e) = p(side|tm1) p(vert|tm2, side)           (2x2AFC)
    postdictive   as hierarchical, but the side marginalizes the vertical arm:
                  p(side) ~ sum_v p(tm1|t_side) p(tm2|t_side,v)  (2x2AFC)
    revision      hierarchical, but the side is re-decided when confidence is
                  below `theta`; revision is lossy, so the re-read intervals
                  carry extra noise N(beta, alpha * tm)   (+theta, alpha, beta)

`only_first` / `only_second` are not manuscript models; they are the two
single-interval floors that say how much of a fit is carried by one flash
alone. `CONFIDENCE_VARIANTS` are the four formulations in the manuscript's
Table 1.

Every function takes `tm` of shape (M, 2) and `intervals` of shape (4, 2) and
returns (M, 4) posteriors, so a whole quadrature grid is evaluated at once.
"""

from __future__ import annotations

import numpy as np


CONFIDENCE_VARIANTS = ("XUD", "Xmax", "XLR_UD", "XLR_max")
_LEFT = (0, 1)
_RIGHT = (2, 3)
_UP = (0, 2)


def _norm_pdf(x, mu, sd):
    z = (x - mu) / sd
    return np.exp(-0.5 * z**2) / (sd * np.sqrt(2.0 * np.pi))


def _normalize(p):
    total = p.sum(axis=-1, keepdims=True)
    return np.divide(p, total, out=np.zeros_like(p), where=total > 0)


def _marginals(tm, intervals, wm):
    """(M, 4) likelihood of each interval separately: p(tm_j | t_ej)."""
    t1, t2 = intervals[:, 0][None, :], intervals[:, 1][None, :]
    l1 = _norm_pdf(tm[:, 0][:, None], t1, wm * t1)
    l2 = _norm_pdf(tm[:, 1][:, None], t2, wm * t2)
    return l1, l2


def optimal(tm, intervals, wm):
    l1, l2 = _marginals(tm, intervals, wm)
    return _normalize(l1 * l2)


def optimal_lapse(tm, intervals, wm, lam):
    return (1.0 - lam) * optimal(tm, intervals, wm) + lam / 4.0


def total_time(tm, intervals, wm):
    mu = intervals.sum(axis=1)[None, :]
    sd = wm * np.sqrt((intervals**2).sum(axis=1))[None, :]
    return _normalize(_norm_pdf(tm.sum(axis=1)[:, None], mu, sd))


def only_first(tm, intervals, wm):
    l1, _ = _marginals(tm, intervals, wm)
    return _normalize(l1)


def only_second(tm, intervals, wm):
    _, l2 = _marginals(tm, intervals, wm)
    return _normalize(l2)


def _side_posterior_hierarchical(l1):
    """p(left|tm1) from the first interval alone; both left exits share t_e1."""
    left = l1[:, 0]
    right = l1[:, 2]
    return _normalize(np.stack([left, right], axis=1))


def _side_posterior_postdictive(l1, l2):
    """p(left|tm1,tm2), marginalizing the vertical arm within each side."""
    joint = l1 * l2
    left = joint[:, list(_LEFT)].sum(axis=1)
    right = joint[:, list(_RIGHT)].sum(axis=1)
    return _normalize(np.stack([left, right], axis=1))


def _vertical_posterior(l2):
    """p(vert | tm2, side) for each side, as (M, 4) with each side summing 1."""
    out = np.zeros_like(l2)
    for side in (_LEFT, _RIGHT):
        cols = list(side)
        out[:, cols] = _normalize(l2[:, cols])
    return out


def _two_stage(side_post, vert_post):
    out = np.zeros_like(vert_post)
    out[:, list(_LEFT)] = side_post[:, 0][:, None] * vert_post[:, list(_LEFT)]
    out[:, list(_RIGHT)] = side_post[:, 1][:, None] * vert_post[:, list(_RIGHT)]
    return out


def hierarchical(tm, intervals, wm):
    l1, l2 = _marginals(tm, intervals, wm)
    return _two_stage(_side_posterior_hierarchical(l1), _vertical_posterior(l2))


def postdictive(tm, intervals, wm):
    l1, l2 = _marginals(tm, intervals, wm)
    return _two_stage(_side_posterior_postdictive(l1, l2), _vertical_posterior(l2))


def _confidence(variant, side_post, l2):
    """Table 1 confidence in the initial left-right decision, per sample.

    `p(U or D | tm2)` is normalized across *all four* vertical arms, not
    within the committed side: within the side it is identically 1 and the
    model could never revise. Globally normalized it is the informative
    quantity -- how much of the second interval's evidence is consistent with
    either arm of the side just chosen.
    """
    chosen_side = np.argmax(side_post, axis=1)
    p_side = side_post[np.arange(len(side_post)), chosen_side]
    across = _normalize(l2)
    cols = np.where(chosen_side[:, None] == 0, np.array(_LEFT), np.array(_RIGHT))
    p_vert = np.take_along_axis(across, cols, axis=1)
    if variant in ("XUD", "XLR_UD"):
        base = p_vert.sum(axis=1)
    else:
        base = p_vert.max(axis=1)
    if variant in ("XLR_UD", "XLR_max"):
        return base * p_side
    return base


def revision(
    tm, intervals, wm, theta, alpha, beta, variant="XUD", inner_nodes=7
):
    """Hierarchical, re-deciding the side when confidence falls below theta.

    A revision re-reads both intervals from a lossy working-memory trace:
    `tm' = tm + N(beta, alpha * tm)`. The re-decision then evaluates all four
    exits (that is what makes it a revision rather than a repeat), so the
    revised posterior is the optimal one computed on `tm'`, marginalized over
    the loss. With `theta = 0` nothing is ever revised and the model reduces
    to `hierarchical`; with `theta = 1` everything is, and it approaches the
    optimal model degraded by the extra noise.
    """
    l1, l2 = _marginals(tm, intervals, wm)
    side_post = _side_posterior_hierarchical(l1)
    vert_post = _vertical_posterior(l2)
    hier = _two_stage(side_post, vert_post)
    revise = _confidence(variant, side_post, l2) < theta
    if not revise.any():
        return hier
    x, w = np.polynomial.hermite_e.hermegauss(inner_nodes)
    w = w / w.sum()
    sub = tm[revise]
    revised = np.zeros((len(sub), 4))
    for i1, xi in enumerate(x):
        for i2, xj in enumerate(x):
            noisy = np.stack(
                [
                    sub[:, 0] + beta + alpha * sub[:, 0] * xi,
                    sub[:, 1] + beta + alpha * sub[:, 1] * xj,
                ],
                axis=1,
            )
            revised += w[i1] * w[i2] * optimal(
                np.abs(noisy) + 1e-9, intervals, wm
            )
    out = hier.copy()
    out[revise] = revised
    return out


# name -> (function, tuple of free parameter names beyond wm)
MODELS = {
    "optimal": (optimal, ()),
    "optimal_lapse": (optimal_lapse, ("lam",)),
    "total_time": (total_time, ()),
    "only_first": (only_first, ()),
    "only_second": (only_second, ()),
    "hierarchical": (hierarchical, ()),
    "postdictive": (postdictive, ()),
    "revision": (revision, ("theta", "alpha", "beta")),
}
