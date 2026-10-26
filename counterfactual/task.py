"""Pure task functions: chosen / correct exit, exit positions, look
assignment, true intervals. No I/O."""

from __future__ import annotations

import numpy as np

from counterfactual.config import ELSEWHERE, EXIT_RADIUS_DEG, ORIGIN, ORIGIN_RADIUS_DEG

# (chosen horizontal, chosen vertical) -> exit; -1 left / +1 right, -1 down / +1 up.
_EXIT = {(-1, 1): 1, (-1, -1): 2, (1, 1): 3, (1, -1): 4}
SIDE = np.array([0, 0, 0, 1, 1])  # by exit code: 0 left, 1 right (index 0 unused)
LEVEL = np.array([0, 1, 0, 1, 0])  # by exit code: 1 up, 0 down


def chosen_exit(flags, lr, lr2):
    """Exit chosen (1-4) or None. DATA_DICTIONARY.md: `trial_answer1..4` are
    correct/correct, correct/wrong, wrong/correct, wrong/wrong on the
    (horizontal, vertical) decisions relative to the ground truth `LR`, `LR2`."""
    active = [k + 1 for k, f in enumerate(flags) if np.isfinite(f) and f != 0]
    if len(active) != 1 or lr not in (-1, 1) or lr2 not in (-1, 1):
        return None
    branch = active[0]
    horiz = int(lr) if branch in (1, 2) else -int(lr)
    vert = int(lr2) if branch in (1, 3) else -int(lr2)
    return _EXIT[(horiz, vert)]


def true_exit(lr, lr2):
    return _EXIT[(int(lr), int(lr2))]


def correct_exit(path_type):
    """Rewarded exit of a fixed-geometry trial: `path_type` cycles LU LD RU RD."""
    return int((int(path_type) - 1) % 4 + 1)


def exit_xy(h):
    """(4, 2) exit positions in degrees (LU LD RU RD), fixation point at the origin."""
    h1, h2, h3, h4, h5, h6 = (float(v) for v in h)
    return np.array([[-h1, h2], [-h1, -h3], [h4, h5], [h4, -h6]])


def assign_exit(xy, h, radius=EXIT_RADIUS_DEG, origin_radius=ORIGIN_RADIUS_DEG):
    """State per row of `xy` (deg): nearest exit within `radius` (1-4), else
    origin within `origin_radius` (0), else -1."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    state = np.full(len(xy), ELSEWHERE, dtype=int)
    if not len(xy):
        return state
    d = np.linalg.norm(xy[:, None, :] - exit_xy(h)[None], axis=-1)
    near = d.min(axis=1) <= radius
    state[near] = d.argmin(axis=1)[near] + 1
    state[~near & (np.linalg.norm(xy, axis=1) <= origin_radius)] = ORIGIN
    return state


def arm_durations(h, vel):
    """(n, 6) arm durations in seconds: h1..h6 / vel."""
    return np.asarray(h, dtype=float).reshape(-1, 6) / np.asarray(vel, dtype=float).reshape(-1, 1)


def true_intervals(dur, lr, lr2):
    """(t1, t2) of the true path: horizontal arm by `lr`, vertical arm by (`lr`, `lr2`)."""
    lr = np.asarray(lr, dtype=int)
    lr2 = np.asarray(lr2, dtype=int)
    t1 = np.where(lr == 1, dur[:, 3], dur[:, 0])
    t2 = np.where(lr == -1, np.where(lr2 == 1, dur[:, 1], dur[:, 2]),
                  np.where(lr2 == 1, dur[:, 4], dur[:, 5]))
    return t1, t2


def unchosen(c):
    """The three unchosen exit codes, ascending: the slot order used everywhere."""
    return [e for e in (1, 2, 3, 4) if e != c]


def relation(a, b):
    """Direction from exit `a` to exit `b`: 'vertical' (same side, the sibling),
    'horizontal' (same up/down level, other side) or 'diagonal'."""
    if SIDE[a] == SIDE[b]:
        return "vertical"
    return "horizontal" if LEVEL[a] == LEVEL[b] else "diagonal"
