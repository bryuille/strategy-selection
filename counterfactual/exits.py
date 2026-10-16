"""Pure functions: chosen / correct exit from the behavioral record, state
assignment of fixation centroids, and the exit sequence. No I/O; what
`counterfactual.checks` exercises."""

from __future__ import annotations

import numpy as np

from counterfactual.config import ELSEWHERE, ORIGIN, EXIT_RADIUS_DEG, ORIGIN_RADIUS_DEG, LU, LD, RU, RD

# (horizontal, vertical) of the chosen exit, -1 left / +1 right, -1 down / +1 up.
EXIT_FROM_LR = {(-1, 1): LU, (-1, -1): LD, (1, 1): RU, (1, -1): RD}


def answer_branch(flags):
    """Which `trial_answer{1..4}` flag is set (1-4), or None unless exactly one."""
    active = [k + 1 for k, f in enumerate(flags) if np.isfinite(f) and f != 0]
    return active[0] if len(active) == 1 else None


def chosen_exit(flags, lr, lr2):
    """Exit the monkey chose (1-4), or None.

    DATA_DICTIONARY.md: `trial_answer1` correct on both decisions,
    `trial_answer2` correct horizontal / wrong vertical, `trial_answer3` wrong
    horizontal / correct vertical, `trial_answer4` wrong on both; `LR` and
    `LR2` are the ground-truth horizontal (+1 right) and vertical (+1 up)
    decisions. Flipping the wrong ones gives the chosen pair.
    """
    branch = answer_branch(flags)
    if branch is None:
        return None
    if not (np.isfinite(lr) and np.isfinite(lr2)) or lr not in (-1, 1) or lr2 not in (-1, 1):
        return None
    lr_chosen = int(lr) if branch in (1, 2) else -int(lr)
    ud_chosen = int(lr2) if branch in (1, 3) else -int(lr2)
    return EXIT_FROM_LR[(lr_chosen, ud_chosen)]


def correct_exit(path_type):
    """Rewarded exit (1-4) of a fixed-geometry trial, else None."""
    if not np.isfinite(path_type) or int(path_type) == -99:
        return None
    return int((int(path_type) - 1) % 4 + 1)


def exit_landmarks_deg(h):
    """(4, 2) exit positions in degrees, LU LD RU RD, fixation point at the origin."""
    h1, h2, h3, h4, h5, h6 = (float(v) for v in h)
    return np.array([[-h1, h2], [-h1, -h3], [h4, h5], [h4, -h6]], dtype=float)


def assign_states(xy, h, *, exit_radius=EXIT_RADIUS_DEG, origin_radius=ORIGIN_RADIUS_DEG):
    """State per centroid row of `xy` (degrees): nearest exit within
    `exit_radius` (1-4), else origin within `origin_radius` (0), else -1."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    state = np.full(xy.shape[0], ELSEWHERE, dtype=np.int16)
    if xy.shape[0] == 0:
        return state
    dist = np.linalg.norm(xy[:, None, :] - exit_landmarks_deg(h)[None, :, :], axis=-1)
    near = dist.min(axis=1) <= exit_radius
    state[near] = dist.argmin(axis=1)[near] + 1
    at_origin = ~near & (np.linalg.norm(xy, axis=1) <= origin_radius)
    state[at_origin] = ORIGIN
    return state


def exit_sequence(chosen, states):
    """``[chosen] + exits fixated afterwards``, in order, repeats kept; origin
    and elsewhere fixations are dropped."""
    return [int(chosen)] + [int(s) for s in states if 1 <= int(s) <= 4]
