"""Scalar-noise measurements, the decision rules and the beliefs of each
behavioral model.

Measurement: ``tm = t * (1 + w z)``, z ~ N(0, 1), floored at machine eps,
so tm ~ N(t, (w t)^2). Likelihood of an arm of duration a given tm:
``p(tm | a) = N(tm; a, (w a)^2)`` (`D_likelihood` of the Figure One MATLAB).

Arrays are batched: measurements (n, K), `dur` (n, 6) arm durations
(h1..h6 / vel). Exits are ordered LU LD RU RD; the side of an exit is left
(h1) for LU, LD and right (h4) for RU, RD; its vertical arm is h2, h3, h5,
h6. `terms` returns the log likelihoods every model uses:

``ll1``  (n, K, 2)  p(tm1 | horizontal arm), left / right
``ll2``  (n, K, 4)  p(tm2 | vertical arm), per exit
``llT``  (n, K, 4)  p(total | horizontal + vertical arm), `total` a separate
                    noisy measurement of the whole travel time (Total Time)
``llC``  (n, K, 4)  p(tm1 + tm2 | arm sum), sd w * sqrt(a1^2 + a2^2): the
                    convolution of the two interval likelihoods (Total Time,
                    convolution)

Models (rule: which exit is chosen on a draw; belief: the model's posterior
over the four exits on that draw, flat prior):

``H``  hierarchical    rule: side with the larger p(tm1 | side), then the arm
                       on that side with the larger p(tm2 | arm).
                       belief: p(side | tm1) * p(arm | tm2, that side).
``S``  second-only     rule: argmax over the four arms of p(tm2 | arm).
                       belief: p(arm | tm2) over the four arms.
``O``  optimal         rule: argmax p(tm1 | side) p(tm2 | arm).
                       belief: that product, normalised.
``P``  postdictive     rule: side with the larger p(tm1 | side) * sum of
                       p(tm2 | arm) over its two arms, then the arm on that
                       side with the larger p(tm2 | arm).
                       belief: the joint posterior (as optimal).
``T``  total time      rule: argmax p(total | arm sum).  belief: that, normalised.
``C``  total time (convolution)  as T on tm1 + tm2 with `llC`.
``F``  first-only      rule: side with the larger p(tm1 | side), then a fair
                       coin between its two arms.
                       belief: p(side | tm1) * 1/2.

Exact ties (equal arm lengths) split evenly, which is the fair coin of the
MATLAB helpers.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

SIDE_OF_EXIT = np.array([0, 0, 1, 1])
SIDE_ARM = np.array([0, 3])  # columns of dur: h1, h4
VERT_ARM = np.array([1, 2, 4, 5])  # h2, h3, h5, h6
SIDES = ((0, [0, 1]), (1, [2, 3]))


def measure(t, z, w):
    return np.maximum(t[:, None] * (1.0 + w * z), np.finfo(float).eps)


def _logpdf(tm, mean, sd):
    """log N(tm; mean, sd^2); tm (n, K), mean and sd (n, j) -> (n, K, j)."""
    m, s = mean[:, None, :], sd[:, None, :]
    return -0.5 * ((tm[..., None] - m) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)


def terms(tm1, tm2, total, dur, w):
    side, vert = dur[:, SIDE_ARM], dur[:, VERT_ARM]
    side_of = side[:, SIDE_OF_EXIT]
    return {
        "ll1": _logpdf(tm1, side, w * side),
        "ll2": _logpdf(tm2, vert, w * vert),
        "llT": _logpdf(total, side_of + vert, w * (side_of + vert)),
        "llC": _logpdf(tm1 + tm2, side_of + vert, w * np.sqrt(side_of ** 2 + vert ** 2)),
    }


def _norm(x):
    return x - logsumexp(x, axis=-1, keepdims=True)


def _side_posterior(L):
    return _norm(L["ll1"])[..., SIDE_OF_EXIT]


def log_beliefs(model, L):
    """(n, K, 4) log belief over exits."""
    if model == "S":
        return _norm(L["ll2"])
    if model == "H":
        within = np.empty_like(L["ll2"])
        for _, cols in SIDES:
            within[..., cols] = _norm(L["ll2"][..., cols])
        return _side_posterior(L) + within
    if model in ("O", "P"):
        return _norm(L["ll1"][..., SIDE_OF_EXIT] + L["ll2"])
    if model == "T":
        return _norm(L["llT"])
    if model == "C":
        return _norm(L["llC"])
    if model == "F":
        return _side_posterior(L) + np.log(0.5)
    raise ValueError(model)


def _argmax_share(x):
    """Per-draw pick probability of each column: 1/k on the k tied maxima."""
    top = np.isclose(x, x.max(axis=-1, keepdims=True), rtol=0.0, atol=1e-12)
    return top / top.sum(axis=-1, keepdims=True)


def _side_then_arm(side_score, L, within):
    """Pick the side by `side_score` (n, K, 2), then within it by `within`
    ('best': larger p(tm2 | arm); 'coin': 1/2 each)."""
    side = _argmax_share(side_score)
    out = np.empty_like(L["ll2"])
    for s, cols in SIDES:
        arm = _argmax_share(L["ll2"][..., cols]) if within == "best" else np.full(out[..., cols].shape, 0.5)
        out[..., cols] = side[..., [s]] * arm
    return out


def pick_probs(model, L):
    """(n, K, 4) probability that `model`'s rule picks each exit, per draw."""
    if model == "S":
        return _argmax_share(L["ll2"])
    if model == "H":
        return _side_then_arm(L["ll1"], L, "best")
    if model == "O":
        return _argmax_share(L["ll1"][..., SIDE_OF_EXIT] + L["ll2"])
    if model == "P":
        marg = np.stack([logsumexp(L["ll2"][..., cols], axis=-1) for _, cols in SIDES], axis=-1)
        return _side_then_arm(L["ll1"] + marg, L, "best")
    if model == "T":
        return _argmax_share(L["llT"])
    if model == "C":
        return _argmax_share(L["llC"])
    if model == "F":
        return _side_then_arm(L["ll1"], L, "coin")
    raise ValueError(model)
