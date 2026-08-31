"""Exit geometry, chosen/correct exits, and model-implied unchosen alternatives.

Exit order is the published convention (`path_type` offset within a maze):
index 0 LU, 1 LD, 2 RU, 3 RD. `LR` is -1 left / +1 right; `LR2` is
-1 down / +1 up. The four `trial_answer` flags are relative to ground truth
(see DATA_DICTIONARY.md), so the chosen exit is recovered by flipping `LR` /
`LR2` where the corresponding decision was wrong.

The ball's path timing is fully determined by the geometry: interval 1
(flash 1 -> 2) is the horizontal arm, interval 2 (flash 2 -> 3) the vertical
arm, each at `h/vel`. Which unchosen exit is the *most likely alternative*
depends on how the animal evaluates that timing, so the ranking is computed
per decision model (`ALT_MODELS`), mirroring the manuscript's model set:

Every ranking is a real likelihood under scalar timing: the single-interval
conditional is p(tm | t) = N(tm; t, wm * t) -- Gaussian, standard deviation
proportional to the timed duration, `wm` the Weber fraction (`DEFAULT_WM`,
not fitted here; see the constant).

  optimal       4AFC optimal model. log p(tm1, tm2 | t_e1, t_e2) summed over
                the two conditionally independent marginals.
  total_time    4AFC total-time model. Ignores the flash-two split and scores
                the sum: log p(tm1 + tm2 | t_e1 + t_e2), whose conditional is
                the convolution of the marginals, so sd = wm * sqrt(t1^2 +
                t2^2) -- not wm * (t1 + t2).
  hierarchical  2x2AFC hierarchical model. The left-right decision is
                committed first, so the alternative the second stage
                evaluated is the *other vertical arm on the chosen branch*.
                The sibling exit ranks first -- a structural prediction, so
                the only wm-free one of the three -- with the far branch
                behind it by joint likelihood.

Because the evidence used is the correct path's *expected* intervals, the
evidence is noiseless and the ranking is a deterministic function of (path
type, chosen exit): every trial of a path type gets the identical ranking,
and no trial-to-trial timing variability enters.

The models dissociate on real geometries. Maze 5 (equal horizontal arms),
truth LU: hierarchical says LD, total-time says RU. Maze 3, truth LU:
hierarchical says LD, total-time says RD. Trials where the implied
alternatives differ are what `eye_post_flash.counterfactual` uses to ask
whether counterfactual looks track the inferred strategy. The manuscript's
lapse, postdictive and revision variants are not separately operationalized
(lapses do not change the ranking; postdictive ranks like the optimal model;
revision predictions depend on fitted confidence thresholds).
"""

from __future__ import annotations

import numpy as np

EXIT_NAMES = ("LU", "LD", "RU", "RD")
_EXIT_INDEX = {(-1, 1): 0, (-1, -1): 1, (1, 1): 2, (1, -1): 3}
ALT_MODELS = ("optimal", "total_time", "hierarchical")

# Weber fraction for interval timing. The manuscript estimates wm from the
# separate T-maze experiment, whose data are not in this repo, so this is a
# stand-in at the middle of the usual range for monkey interval timing rather
# than a fitted value. The rankings are NOT uniformly robust to it -- a few
# path types reorder between 0.10 and 0.25 -- so `--wm` is exposed and the
# sensitivity is reported. Replace with the fitted value when it is available.
DEFAULT_WM = 0.15


def exit_index(lr, lr2):
    """0 LU, 1 LD, 2 RU, 3 RD from signed horizontal/vertical decisions."""
    return _EXIT_INDEX[(int(np.sign(lr)), int(np.sign(lr2)))]


def sibling_exit(exit_idx):
    """The other vertical arm on the same left/right branch (0<->1, 2<->3)."""
    return exit_idx ^ 1


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


def interval_loglik(observed_ms, expected_ms, wm=DEFAULT_WM):
    """log p(tm | t) for scalar timing: Gaussian, sd = wm * t.

    The `- log(sd)` normalizer is what makes a *longer* expected interval the
    more confusable one at equal absolute error, so it must be kept: dropping
    it (as a bare z-score or an absolute-error distance does) changes the
    ranking, not just its scale.
    """
    expected = np.asarray(expected_ms, dtype=float)
    sd = wm * expected
    z = (np.asarray(observed_ms, dtype=float) - expected) / sd
    return -0.5 * z**2 - np.log(sd) - 0.5 * np.log(2.0 * np.pi)


def joint_loglik(intervals_ms, evidence_ms, wm=DEFAULT_WM):
    """4AFC optimal model: log p(tm1, tm2 | t_e1, t_e2), one value per exit.

    The two measurements are conditionally independent, so the joint is the
    product of the marginals and the log-joint their sum.
    """
    expected = np.asarray(intervals_ms, dtype=float)
    observed = np.asarray(evidence_ms, dtype=float)
    return interval_loglik(observed[None, :], expected, wm).sum(axis=1)


def total_time_loglik(intervals_ms, evidence_ms, wm=DEFAULT_WM):
    """4AFC total-time model: log p(tm1 + tm2 | t_e1 + t_e2), per exit.

    The conditional for the sum is the convolution of the two marginals, so
    the variances add: sd = wm * sqrt(t_e1**2 + t_e2**2). Normalizing by the
    *sum* instead -- wm * (t_e1 + t_e2) -- is a different distribution and
    reorders real path types, so the convolution is computed explicitly.
    """
    expected = np.asarray(intervals_ms, dtype=float)
    mu = expected.sum(axis=1)
    sd = wm * np.sqrt((expected**2).sum(axis=1))
    z = (float(np.sum(evidence_ms)) - mu) / sd
    return -0.5 * z**2 - np.log(sd) - 0.5 * np.log(2.0 * np.pi)


def ranked_alternatives(
    intervals_ms, correct_idx, chosen_idx, model="optimal", wm=DEFAULT_WM
):
    """Unchosen exits, most- to least-likely under `model`.

    Evidence is the correct path's expected intervals, so the evidence is
    noiseless and the ranking is a deterministic function of (path type,
    chosen exit) -- every trial of a path type gets the same ranking. Pass
    measured flash intervals as `intervals_ms[correct_idx]` substitutes if
    per-trial timing is wanted.
    """
    if model not in ALT_MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {ALT_MODELS}")
    evidence = np.asarray(intervals_ms, dtype=float)[correct_idx]
    joint = joint_loglik(intervals_ms, evidence, wm)
    if model == "hierarchical":
        # The left-right decision is already committed, so the alternative the
        # second stage actually evaluated is the sibling -- a structural claim
        # that does not depend on wm. The far branch is ordered behind it by
        # the joint likelihood.
        sibling = sibling_exit(chosen_idx)
        rest = [e for e in range(4) if e not in (chosen_idx, sibling)]
        return [sibling, *sorted(rest, key=lambda e: (-joint[e], e))]
    scores = (
        total_time_loglik(intervals_ms, evidence, wm)
        if model == "total_time"
        else joint
    )
    unchosen = [e for e in range(4) if e != chosen_idx]
    return sorted(unchosen, key=lambda e: (-scores[e], e))


def most_likely_alternative(
    intervals_ms, correct_idx, chosen_idx, model="optimal", wm=DEFAULT_WM
):
    """The unchosen exit `model` says was most confusable with the evidence."""
    return ranked_alternatives(
        intervals_ms, correct_idx, chosen_idx, model, wm
    )[0]
