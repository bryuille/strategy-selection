"""Synthetic checks, no data: the exit logic and the fixation detector.

    uv run python -m counterfactual.checks
"""

from __future__ import annotations

import numpy as np

from counterfactual.exits import assign_states, chosen_exit, correct_exit, exit_sequence

# The archived decode table (zarchive/plotting/movie/qc.py MONKEY_EXIT_MAP):
# answer branch -> (LR, LR2) -> exit. `chosen_exit` must reproduce it.
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


def check_detector():
    import pymovements as pm

    from data.builder import EYE_SAMPLING_RATE_HZ, FIXATION_EVENT, trial_events

    exp = pm.Experiment(sampling_rate=EYE_SAMPLING_RATE_HZ)
    rng = np.random.default_rng(0)
    n = 2000
    t = np.arange(n) / 1000.0

    def events(x, y):
        frame = trial_events(t, x, y, np.empty((0, 2)), exp)
        rows = [] if frame is None else frame.select("name", "onset", "offset").rows()
        fix = sorted((int(o), int(f)) for nm, o, f in rows if nm == FIXATION_EVENT)
        sac = sorted((int(o), int(f)) for nm, o, f in rows if nm == "saccade")
        return fix, sac

    noise = lambda: rng.normal(0, 0.05, n)
    fix, sac = events(np.linspace(0, 0.3, n) + noise(), noise())
    assert len(fix) == 1 and fix[0] == (0, n - 1) and not sac, (fix, sac)
    fix, sac = events(np.linspace(0, 1.5, n) + noise(), noise())
    assert len(fix) == 1, fix
    ramp = np.clip((np.arange(n) - 1000) / 30.0, 0, 1)
    x = np.linspace(0, 0.3, n) + noise() + 5.0 * (3 * ramp**2 - 2 * ramp**3)
    fix, sac = events(x, noise())
    assert len(sac) == 1 and len(fix) == 2, (fix, sac)
    assert fix[0][1] < sac[0][0] and fix[1][0] > sac[0][1], (fix, sac)
    fix, sac = events(np.linspace(0, 3.0, n) + noise(), noise())
    assert len(fix) == 2 and not sac and fix[0][1] + 1 == fix[1][0], (fix, sac)
    assert 1200 <= fix[0][1] <= 1450, fix  # 2 deg from the first-100-ms anchor at ~1333
    fix, sac = events(np.zeros(n), np.zeros(n))
    assert len(fix) == 1 and not sac, (fix, sac)
    print("detector: one fixation under drift+noise, split by a saccade or by 2 deg from the anchor, no other cap")


def main():
    check_chosen_exit()
    check_assign_states()
    check_exit_sequence()
    check_detector()
    print("all checks passed")


if __name__ == "__main__":
    main()
