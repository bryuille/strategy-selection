"""Row lookups and run/n-gram helpers shared by the feature blocks.

The feature blocks themselves live in `eye_pre_flash.classifier.features`, which
is the only place occupancy and bigrams are computed. This module holds the
pieces that are pure functions of a state sequence or a behavioral table.
"""

import numpy as np


def behavioral_lookup(behavioral):
    return {
        (str(session), int(trial_id)): i
        for session, trial_id, i in zip(
            behavioral["session"],
            behavioral["trial_indices_all"],
            range(len(behavioral["session"])),
        )
    }


def fix_start_ms(behavioral, beh_i):
    return (
        behavioral["fix_start"][beh_i] - behavioral["geo_present"][beh_i]
    ) * 1000.0


def collapse_state_runs(states):
    """Consecutive identical states collapsed to one entry each.

    Called on the *assigned-only* subsequence, so unassigned samples (saccades,
    blinks, gaze outside every ball) never appear and cannot split a run. That
    is what makes a merge step unnecessary: two consecutive fixation fragments
    that landed on the same cluster are already one run here.
    """
    runs = []
    for state in states:
        sid = int(state)
        if not runs or runs[-1] != sid:
            runs.append(sid)
    return runs


def ngram_proportions(runs, k, order):
    """Proportion of length-`order` state-run n-grams; zeros if too few runs."""
    counts = np.zeros((k,) * order, dtype=np.float64)
    n = len(runs) - order + 1
    if n <= 0:
        return counts
    for i in range(n):
        counts[tuple(runs[i : i + order])] += 1.0
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts
