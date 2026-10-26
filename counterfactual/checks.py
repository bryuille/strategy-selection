"""Synthetic checks, no data: the exit logic and the choice models.

    uv run python -m counterfactual.checks
"""

from __future__ import annotations

import numpy as np

from counterfactual.exits import assign_states, chosen_exit, correct_exit, exit_sequence

# The archived decode table (zarchive/MR_AG_MJ: answer branch -> (LR, LR2) -> exit).
# `chosen_exit` must reproduce it.
MONKEY_EXIT_MAP = {
    1: {(-1, 1): 1, (-1, -1): 2, (1, 1): 3, (1, -1): 4},
    2: {(-1, 1): 2, (-1, -1): 1, (1, 1): 4, (1, -1): 3},
    3: {(-1, 1): 3, (-1, -1): 4, (1, 1): 1, (1, -1): 2},
    4: {(-1, 1): 4, (-1, -1): 3, (1, 1): 2, (1, -1): 1},
}


def check_chosen_exit():
    for branch, table in MONKEY_EXIT_MAP.items():
        flags = tuple(1.0 if k == branch else 0.0 for k in range(1, 5))
        for (lr, lr2), want in table.items():
            assert chosen_exit(flags, lr, lr2) == want, (branch, lr, lr2)
    assert chosen_exit((0.0, 0.0, 0.0, 0.0), 1, 1) is None
    assert chosen_exit((1.0, 1.0, 0.0, 0.0), 1, 1) is None
    assert chosen_exit((1.0, 0.0, 0.0, 0.0), np.nan, 1) is None
    assert correct_exit(1) == 1 and correct_exit(4) == 4 and correct_exit(5) == 1 and correct_exit(24) == 4
    assert correct_exit(-99) is None and correct_exit(np.nan) is None
    print("chosen_exit reproduces the archived 16-entry decode table; correct_exit cycles 1-4")


def check_assign_states():
    h = (8.0, 6.0, 4.0, 10.0, 5.0, 7.0)
    pts = np.array([[-8, 6], [-8, -4], [10, 5], [10, -7], [0.5, -0.5], [-8.5, 7.5], [0, 20], [-8, 1]])
    got = assign_states(pts, h).tolist()
    assert got == [1, 2, 3, 4, 0, 1, -1, -1], got
    assert assign_states(np.empty((0, 2)), h).size == 0
    print("assign_states: exits 1-4 within 3 deg, origin ball, elsewhere")


def check_exit_sequence():
    assert exit_sequence(3, [3, 3, 0, 1, -1, 2, 3]) == [3, 3, 3, 1, 2, 3]
    assert exit_sequence(2, []) == [2]
    print("exit_sequence keeps repeats and drops origin / elsewhere")


def _matlab_loop(h_sec, lr, lr2, scalar, n_samples, rng):
    """Direct transcription of simulate_hs_models + the two MATLAB helpers,
    one draw at a time, to cross-check the vectorised port."""
    from scipy.stats import norm

    h1, h2, h3, h4, h5, h6 = h_sec
    t1 = h4 if lr == 1 else h1
    t2 = {(-1, 1): h2, (-1, -1): h3, (1, 1): h5, (1, -1): h6}[(lr, lr2)]
    counts_h = np.zeros(4)
    counts_s = np.zeros(4)
    for _ in range(n_samples):
        one = max(t1 + rng.standard_normal() * scalar * t1, np.finfo(float).eps)
        two = max(t2 + rng.standard_normal() * scalar * t2, np.finfo(float).eps)
        # Hierarchical_Model_Physiology
        i_h1 = norm.pdf(one, h1, h1 * scalar)
        i_h2 = norm.pdf(one, h4, h4 * scalar)
        if h1 == h4:
            i_h1, i_h2 = (1, 0) if rng.random() >= 0.5 else (0, 1)
        alts = np.zeros(4)
        if i_h1 > i_h2:
            alts[0], alts[1] = norm.pdf(two, h2, h2 * scalar), norm.pdf(two, h3, h3 * scalar)
        else:
            alts[2], alts[3] = norm.pdf(two, h5, h5 * scalar), norm.pdf(two, h6, h6 * scalar)
        tied = np.flatnonzero(alts == alts.max())
        counts_h[rng.choice(tied)] += 1
        # Sequential_Model_Physiology
        alts = np.array([norm.pdf(two, hk, hk * scalar) for hk in (h2, h3, h5, h6)])
        tied = np.flatnonzero(alts == alts.max())
        counts_s[rng.choice(tied)] += 1
    return counts_h / n_samples, counts_s / n_samples


def check_models():
    from counterfactual.models import counterfactual_dist, simulate_models, true_intervals

    def hs(*a, **k):
        prob = simulate_models(*a, **k)
        return prob["h"], prob["s"]

    h = np.array([[0.5, 0.9, 1.1, 0.75, 0.7, 0.5]])  # seconds (arm deg / 10 deg/s)
    t1, t2 = true_intervals(h, [-1], [1])
    assert t1[0] == 0.5 and t2[0] == 0.9
    t1, t2 = true_intervals(h, [1], [-1])
    assert t1[0] == 0.75 and t2[0] == 0.5
    # Vanishing noise: every model picks the true exit with probability 1.
    prob = simulate_models(h, [-1], [1], 1e-6, n_samples=200)
    assert all(prob[m][0, 0] == 1.0 for m in prob), prob
    # h1 == h4: H splits the side 50/50 (vertical arms distinct per side).
    hh = np.array([[0.5, 0.9, 0.5, 0.5, 0.9, 0.5]])
    ph, _ = hs(hh, [-1], [1], 1e-6, n_samples=4000)
    assert abs(ph[0, :2].sum() - 0.5) < 0.05, ph
    # S ignores the horizontal arms.
    _, ps1 = hs(h, [-1], [1], 0.16, n_samples=4000, seed=1)
    h_alt = h.copy()
    h_alt[0, [0, 3]] = [0.9, 0.6]
    _, ps2 = hs(h_alt, [-1], [1], 0.16, n_samples=4000, seed=1)
    assert np.allclose(ps1, ps2), (ps1, ps2)
    # H puts no mass on the far side when the horizontal arms are far apart;
    # Optimal without commitment then equals H.
    hf = np.array([[0.3, 0.9, 1.1, 1.5, 0.7, 0.5]])
    prob = simulate_models(hf, [-1], [1], 0.05, n_samples=2000)
    assert prob["h"][0, 2:].sum() == 0.0, prob["h"]
    assert np.allclose(prob["optimal"][0], prob["h"][0], atol=0.05), (prob["optimal"], prob["h"])
    # Total Time cannot tell exits with equal arm sums apart.
    h_eq = np.array([[0.5, 0.9, 1.1, 0.7, 0.7, 0.9]])  # h1+h2 == h4+h5 == 1.4
    prob = simulate_models(h_eq, [-1], [1], 1e-6, n_samples=2000)
    assert abs(prob["total"][0, 0] - 0.5) < 0.05 and abs(prob["total"][0, 2] - 0.5) < 0.05, prob["total"]
    # Vectorised port vs a direct MATLAB-loop transcription (Monte Carlo tolerance).
    rng = np.random.default_rng(7)
    ph_loop, ps_loop = _matlab_loop(h[0], -1, 1, 0.165, 6000, rng)
    ph_vec, ps_vec = hs(h, [-1], [1], 0.165, n_samples=6000, seed=3)
    assert np.abs(ph_loop - ph_vec[0]).max() < 0.03, (ph_loop, ph_vec)
    assert np.abs(ps_loop - ps_vec[0]).max() < 0.03, (ps_loop, ps_vec)
    # Counterfactual distribution.
    p = np.array([[0.6, 0.3, 0.1, 0.0]])
    q = counterfactual_dist(p, [1], eps=0.0)
    assert q[0, 0] == 0.0 and np.allclose(q[0, 1:], [0.75, 0.25, 0.0])
    q = counterfactual_dist(p, [1], eps=1e-3)
    assert q[0, 3] > 0 and np.isclose(q.sum(), 1.0)
    print("models: true intervals, noiseless limit, h1==h4 coin, S ignores h1/h4, "
          "H never crosses a clear side, vectorised == MATLAB loop within MC error, "
          "Total Time / Optimal limits, counterfactual q")


def check_plackett_luce():
    from counterfactual.compare import counterfactual_items, plackett_luce

    q = np.array([0.0, 0.5, 0.3, 0.2])
    assert np.isclose(plackett_luce(q, [2]), np.log(0.5))
    assert np.isclose(plackett_luce(q, [2, 3]), np.log(0.5) + np.log(0.3 / 0.5))
    assert np.isclose(plackett_luce(q, [2, 2, 3]), plackett_luce(q, [2, 3]))  # a return adds nothing
    assert counterfactual_items([3, 3, 1, 2, 3]) == [1, 2]
    print("plackett_luce: sequential draws without replacement; returns to a visited exit ignored")


def main():
    check_chosen_exit()
    check_assign_states()
    check_exit_sequence()
    check_models()
    check_plackett_luce()
    print("all checks passed")


if __name__ == "__main__":
    main()
