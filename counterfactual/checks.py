"""Synthetic checks, no data needed.

    uv run python -m counterfactual.checks
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from counterfactual.compare import plackett_luce
from counterfactual.config import ALL_MODELS
from counterfactual.noise import log_beliefs, measure, pick_probs, terms
from counterfactual.task import chosen_exit, relation, true_exit, unchosen


def check_decoding():
    # (branch, LR, LR2) -> chosen exit; branch 1 correct/correct ... 4 wrong/wrong
    for lr in (-1, 1):
        for lr2 in (-1, 1):
            t = true_exit(lr, lr2)
            for branch in (1, 2, 3, 4):
                flags = [1.0 if k == branch else 0.0 for k in (1, 2, 3, 4)]
                c = chosen_exit(flags, lr, lr2)
                same_side = (c <= 2) == (t <= 2)
                same_level = (c in (1, 3)) == (t in (1, 3))
                assert same_side == (branch in (1, 2)) and same_level == (branch in (1, 3)), (lr, lr2, branch)
    assert chosen_exit([1, 1, 0, 0], 1, 1) is None
    assert relation(1, 2) == "vertical" and relation(1, 3) == "horizontal" and relation(1, 4) == "diagonal"
    print("decoding: 16 cases ok")


def check_beliefs():
    rng = np.random.default_rng(0)
    dur = np.array([[1.0, 0.6, 0.9, 1.4, 0.7, 0.5], [1.0, 0.6, 0.6, 1.0, 0.8, 0.6]])
    tm1, tm2, tot = rng.uniform(0.4, 1.6, (3, 2, 50))
    L = terms(tm1, tm2, tot + 1.0, dur, 0.15)
    for m in ALL_MODELS:
        assert np.allclose(np.exp(log_beliefs(m, L)).sum(-1), 1), m
        pp = pick_probs(m, L)
        assert np.allclose(pp.sum(-1), 1), m
    # optimal and postdictive share the joint posterior; first-only splits each side 50/50
    assert np.allclose(log_beliefs("O", L), log_beliefs("P", L))
    bf = np.exp(log_beliefs("F", L))
    assert np.allclose(bf[..., 0], bf[..., 1]) and np.allclose(pick_probs("F", L)[..., 0], pick_probs("F", L)[..., 1])
    # trial 2: h1 == h4 so H commits by coin; h2 == h3 and h6 == h2 ties for S
    ph = pick_probs("H", L)[1]
    assert np.allclose(ph[:, :2].sum(-1), 0.5)
    ps = pick_probs("S", L)[1]
    assert np.allclose(ps[:, 0], ps[:, 1])
    # H belief factorises: side posterior times within-side posterior
    lb = np.exp(log_beliefs("H", L))[0, 0]
    side = norm.pdf(tm1[0, 0], [1.0, 1.4], [0.15, 0.21])
    side /= side.sum()
    v = norm.pdf(tm2[0, 0], [0.6, 0.9, 0.7, 0.5], 0.15 * np.array([0.6, 0.9, 0.7, 0.5]))
    assert np.allclose(lb, [side[0] * v[0] / v[:2].sum(), side[0] * v[1] / v[:2].sum(),
                            side[1] * v[2] / v[2:].sum(), side[1] * v[3] / v[2:].sum()])
    print("beliefs and decision rules: ok")


def check_acceptance():
    """MC acceptance of S's rule vs the analytic probability on a 2-arm toy:
    S picks LU when N(tm2; .6, .6w) > N(tm2; .9, .9w) for the other arms."""
    w, t2 = 0.15, 0.75
    dur = np.array([[1.0, 0.6, 0.9, 1.0, 5.0, 5.0]])
    z = np.random.default_rng(1).standard_normal((1, 200_000))
    tm2 = measure(np.array([t2]), z, w)
    mc = pick_probs("S", terms(tm2, tm2, tm2, dur, w))[0, :, 0].mean()
    grid = np.linspace(0.2, 2.0, 200_001)
    dens = norm.pdf(grid, t2, w * t2)
    wins = norm.pdf(grid, 0.6, 0.6 * w) > np.maximum(norm.pdf(grid, 0.9, 0.9 * w), norm.pdf(grid, 5.0, 5.0 * w))
    exact = np.trapezoid(dens * wins, grid)
    assert abs(mc - exact) < 0.005, (mc, exact)
    print(f"acceptance: MC {mc:.4f} vs exact {exact:.4f}")


def check_plackett_luce():
    q = np.array([0.0, 0.5, 0.3, 0.2])  # chosen LU zeroed
    assert np.isclose(plackett_luce(q, [2]), np.log(0.5))
    assert np.isclose(plackett_luce(q, [2, 4]), np.log(0.5) + np.log(0.2 / 0.5))
    # the last remaining exit is forced (log 1); revisits add nothing
    assert np.isclose(plackett_luce(q, [2, 4, 3]), plackett_luce(q, [2, 4]))
    assert np.isclose(plackett_luce(q, [2, 2, 4, 2]), plackett_luce(q, [2, 4]))
    u = np.array([0.0, 1, 1, 1])
    assert np.isclose(plackett_luce(u, [3]), np.log(1 / 3)) and np.isclose(plackett_luce(u, [3, 2, 4]), np.log(1 / 6))
    print("plackett-luce: ok")


def check_slots():
    for c in (1, 2, 3, 4):
        assert len(unchosen(c)) == 3 and c not in unchosen(c)
    print("slots: ok")


def main():
    check_decoding()
    check_slots()
    check_beliefs()
    check_acceptance()
    check_plackett_luce()
    print("all counterfactual checks passed")


if __name__ == "__main__":
    main()
