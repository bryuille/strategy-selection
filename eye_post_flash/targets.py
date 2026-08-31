"""Exit geometry, chosen/correct exits, and the most-likely unchosen alternative.

Exit order is the published convention (`path_type` offset within a maze):
index 0 LU, 1 LD, 2 RU, 3 RD. `LR` is -1 left / +1 right; `LR2` is
-1 down / +1 up. The four `trial_answer` flags are relative to ground truth
(see DATA_DICTIONARY.md), so the chosen exit is recovered by flipping `LR` /
`LR2` where the corresponding decision was wrong.

The ball's path timing is fully determined by the geometry: interval 1
(flash 1 -> 2) is the horizontal arm, interval 2 (flash 2 -> 3) the vertical
arm, each at `h/vel`. The *most likely unchosen alternative* on a trial is the
unchosen exit whose expected intervals are most confusable with the evidence
the animal actually received (the correct path's intervals) under scalar
timing, where timing noise grows with the timed duration:

    d(exit) = sum_j |t_obs_j - t_exit_j| / t_exit_j

Normalizing by the *alternative's* duration is what scalar timing prescribes
(the likelihood of observing `t_obs` under path `exit` has sd proportional to
`t_exit_j`), and it breaks ties between alternatives equidistant in raw ms:
the longer alternative is the more confusable one.
"""

from __future__ import annotations

import numpy as np

EXIT_NAMES = ("LU", "LD", "RU", "RD")
_EXIT_INDEX = {(-1, 1): 0, (-1, -1): 1, (1, 1): 2, (1, -1): 3}


def exit_index(lr, lr2):
    """0 LU, 1 LD, 2 RU, 3 RD from signed horizontal/vertical decisions."""
    return _EXIT_INDEX[(int(np.sign(lr)), int(np.sign(lr2)))]


def exit_positions_deg(h1, h2, h3, h4, h5, h6):
    """(4, 2) screen-degree exit coordinates in EXIT_NAMES order."""
    return np.array(
        [[-h1, h2], [-h1, -h3], [h4, h5], [h4, -h6]], dtype=float
    )


def exit_intervals_ms(h1, h2, h3, h4, h5, h6, vel):
    """(4, 2) expected (flash1->2, flash2->3) durations per exit, ms."""
    arms = np.array([[h1, h2], [h1, h3], [h4, h5], [h4, h6]], dtype=float)
    return arms / float(vel) * 1000.0


def chosen_exit_index(lr, lr2, answer1, answer2, answer3, answer4):
    """The exit the animal chose, from ground truth and the answer category."""
    if not (bool(answer1) or bool(answer2) or bool(answer3) or bool(answer4)):
        return None
    horizontal_correct = bool(answer1) or bool(answer2)
    vertical_correct = bool(answer1) or bool(answer3)
    chose_lr = lr if horizontal_correct else -lr
    chose_lr2 = lr2 if vertical_correct else -lr2
    return exit_index(chose_lr, chose_lr2)


def timing_distances(intervals_ms, evidence_ms):
    """Scalar-timing distance of every exit's intervals from the evidence."""
    expected = np.asarray(intervals_ms, dtype=float)
    observed = np.asarray(evidence_ms, dtype=float)
    return (np.abs(observed[None, :] - expected) / expected).sum(axis=1)


def ranked_alternatives(intervals_ms, correct_idx, chosen_idx):
    """Unchosen exits sorted most- to least-confusable with the true path.

    Evidence is the correct path's expected intervals; pass measured flash
    intervals instead if per-trial timing is preferred (QC keeps them within
    a frame of expected, so the ranking is the same).
    """
    distances = timing_distances(intervals_ms, np.asarray(intervals_ms)[correct_idx])
    unchosen = [e for e in range(4) if e != chosen_idx]
    return sorted(unchosen, key=lambda e: (distances[e], e))


def most_likely_alternative(intervals_ms, correct_idx, chosen_idx):
    """The unchosen exit most confusable with the evidence received."""
    return ranked_alternatives(intervals_ms, correct_idx, chosen_idx)[0]
