"""Invariant checks for the K-selection metric, on synthetic data.

``uv run python -m eye_pre_flash.maze_strategy_pairs.elbow_checks``

No feature cache, no attractor cache and no labels needed -- everything here is
generated. This is also the only part of the elbow analysis that runs on a
laptop: the real pipeline needs the full attractor cache, which lives on the
cluster.

`checks.py` covers the estimator. This covers the thing the estimator is now
downstream of: the criterion that picks K. The point is not to reproduce a
particular K but to pin the properties the selection would be meaningless
without -- above all that the metric tracks the *categorical* assignment
`occ_ms` depends on, rather than the continuous distance it does not.
"""

from __future__ import annotations

import numpy as np

from data.attractor import ASSIGN_RADIUS, assign_states_inference
from eye_pre_flash.maze_strategy_pairs.elbow_select import (
    EV_EPS,
    TrialPool,
    _trial_seed,
    fit_codebook,
    gated_ev,
    kneedle,
    marginal_gain_k,
    max_curvature_k,
    split_trials,
)

FAILURES = []

# Comfortably past any distance in these synthetic fixtures, so the radius gate
# is inert wherever a check is about something other than the gate.
WIDE = 1e6


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def blobs(rng, *, n_blobs=5, per_blob=200, spread=0.06, radius=2.0):
    """Well-separated 2-D clusters on a ring, plus their generating centres."""
    angles = np.linspace(0, 2 * np.pi, n_blobs, endpoint=False)
    centres = radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    pts = np.concatenate(
        [c + spread * rng.normal(size=(per_blob, 2)) for c in centres]
    )
    return pts.astype(np.float32), centres.astype(np.float32)


def check_ev_one_prototype_is_zero():
    """EV == 0 exactly when the single prototype sits at the test mean.

    This is what the test-mean denominator buys: TSS is then the exact K = 1
    case of SSE, so the curve starts at a true zero and every marginal gain is
    measured against a fixed origin. With a train-mean denominator this would
    come out at some small nonzero value that drifts with the split.
    """
    rng = np.random.default_rng(0)
    c, _centres = blobs(rng)
    ev, frac, _sse = gated_ev(c, c.mean(axis=0, keepdims=True), radius=WIDE)
    report(
        "EV(1) is exactly 0 at the test mean",
        abs(ev) < 1e-12 and frac == 1.0,
        f"ev={ev:.3e}",
    )


def check_ev_rises_to_the_true_k():
    """EV is non-decreasing up to the number of generating blobs."""
    rng = np.random.default_rng(1)
    n_blobs = 5
    train, _c = blobs(rng, n_blobs=n_blobs)
    test, _c2 = blobs(rng, n_blobs=n_blobs)
    evs = []
    for k in range(1, n_blobs + 1):
        codebook = fit_codebook(train, k, seed=0)
        evs.append(gated_ev(test, codebook, radius=WIDE)[0])
    diffs = np.diff(evs)
    report(
        "EV non-decreasing up to the true blob count",
        bool((diffs > -1e-6).all()) and evs[-1] > 0.95,
        f"ev(1..{n_blobs}) = " + ", ".join(f"{e:.3f}" for e in evs),
    )


def check_gate_drops_far_centroids():
    """Nothing in range means EV == 0, not merely a large finite SSE.

    An unassigned centroid is dropped from `occ_ms` entirely, so it must
    contribute its full deviation rather than its distance to a prototype it
    was never assigned to. Without this, a codebook that reaches nothing would
    still post a respectable EV.
    """
    rng = np.random.default_rng(2)
    c, _centres = blobs(rng, radius=0.2)
    far = np.array([[500.0, 500.0], [-500.0, -500.0]], dtype=np.float32)
    ev, frac, _sse = gated_ev(c, far, radius=ASSIGN_RADIUS)
    report(
        "unreachable codebook gives EV 0 and assigned fraction 0",
        abs(ev) < 1e-12 and frac == 0.0,
        f"ev={ev:.3e} frac={frac:.2f}",
    )


def check_gate_is_inert_when_wide():
    """With the radius past every distance, gated EV equals ungated EV."""
    rng = np.random.default_rng(3)
    train, _c = blobs(rng)
    test, _c2 = blobs(rng)
    codebook = fit_codebook(train, 5, seed=0)

    ev_gated, frac, _s = gated_ev(test, codebook, radius=WIDE)
    d = np.linalg.norm(test[:, None, :] - codebook[None, :, :], axis=-1)
    sse = float((d.min(axis=1) ** 2).sum())
    tss = float(((test - test.mean(axis=0)) ** 2).sum())
    report(
        "gated EV == plain nearest-prototype EV at a wide radius",
        abs(ev_gated - (1.0 - sse / tss)) < 1e-9 and frac == 1.0,
    )


def check_categorical_insensitivity():
    """Moving an assigned centroid nearer its prototype changes EV, not the state.

    This is the trap the whole metric design exists to bound: `occ_ms` reads
    the assignment, which is categorical, while a sum of squared distances
    reads the distance, which is continuous. Pinned so a future refactor cannot
    quietly swap the gated metric back for a plain SSE without a failing check.
    """
    codebook = np.array([[0.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    near = np.array([[0.40, 0.0], [3.0, 0.0]], dtype=np.float32)
    nearer = np.array([[0.05, 0.0], [3.0, 0.0]], dtype=np.float32)

    state_a = assign_states_inference(near, codebook, radius=ASSIGN_RADIUS)
    state_b = assign_states_inference(nearer, codebook, radius=ASSIGN_RADIUS)
    ev_a = gated_ev(near, codebook, radius=ASSIGN_RADIUS)[0]
    ev_b = gated_ev(nearer, codebook, radius=ASSIGN_RADIUS)[0]

    report(
        "distance shrinkage moves EV but not the assignment",
        bool((state_a == state_b).all()) and ev_b > ev_a,
        f"state unchanged {state_a.tolist()}, ev {ev_a:.4f} -> {ev_b:.4f}",
    )


def check_kneedle_finds_a_known_knee():
    """A curve that rises hard to K=5 then flattens has its knee at 5."""
    ks = np.arange(2, 21)
    ev = np.where(ks <= 5, 0.2 * (ks - 1), 0.8 + 0.002 * (ks - 5)).astype(float)
    report("kneedle recovers a planted knee", kneedle(ks, ev) == 5, f"got {kneedle(ks, ev)}")


def check_kneedle_refuses_a_flat_curve():
    """A flat curve has no knee, and must return None rather than an argmax."""
    ks = np.arange(2, 21)
    flat = np.full(ks.size, 0.5) + EV_EPS * 1e-3 * np.arange(ks.size)
    report("kneedle returns None on a flat curve", kneedle(ks, flat) is None)


def check_kneedle_ignores_a_noise_dip():
    """A single downward blip must not create a spurious knee on the recovery.

    The running maximum is what makes this hold: without it the dip lowers the
    normalisation floor and the steep climb out of the dip can beat the real
    bend.
    """
    ks = np.arange(2, 21)
    ev = np.where(ks <= 5, 0.2 * (ks - 1), 0.8 + 0.002 * (ks - 5)).astype(float)
    dipped = ev.copy()
    dipped[10] -= 0.05
    report(
        "a split-noise dip does not move the knee",
        kneedle(ks, dipped) == kneedle(ks, ev) == 5,
        f"clean {kneedle(ks, ev)}, dipped {kneedle(ks, dipped)}",
    )


def check_other_rules_agree_on_a_clean_curve():
    """On a planted knee the threshold and curvature rules land next to kneedle.

    They are not required to agree in general -- that is the whole reason all
    three are reported -- but on a curve with one unambiguous bend they should,
    and if they do not the implementations are wrong rather than the data hard.
    """
    ks = np.arange(2, 21)
    ev = np.where(ks <= 5, 0.2 * (ks - 1), 0.8 + 0.002 * (ks - 5)).astype(float)
    gain = marginal_gain_k(ks, ev)
    curv = max_curvature_k(ks, ev)
    report(
        "gain and curvature rules land within 1 of the knee",
        gain is not None and curv is not None
        and abs(gain - 5) <= 1 and abs(curv - 5) <= 1,
        f"kneedle 5, gain {gain}, curvature {curv}",
    )


def check_split_is_disjoint_and_complete():
    """Train and test partition the trials: no overlap, nothing dropped."""
    ok = True
    detail = ""
    for n in (20, 37, 100, 413):
        train, test = split_trials(n, np.random.default_rng(n))
        if set(train.tolist()) & set(test.tolist()):
            ok, detail = False, f"overlap at n={n}"
            break
        if sorted(train.tolist() + test.tolist()) != list(range(n)):
            ok, detail = False, f"not a partition at n={n}"
            break
        if test.size == 0 or train.size == 0:
            ok, detail = False, f"empty side at n={n}"
            break
    report("trial split is a partition with both sides non-empty", ok, detail)


def check_split_is_at_trial_level():
    """Every sample of a trial lands on one side of the split.

    The property that makes the held-out score meaningful. Samples within one
    trial are heavily dependent, so a sample-level split would put near
    duplicates on both sides and inflate EV at every K.
    """
    rng = np.random.default_rng(7)
    pools = [
        TrialPool(
            session="s0", trial_id=i,
            samples=np.full((5, 2), float(i), dtype=np.float32),
            centroids=np.full((2, 2), float(i), dtype=np.float32),
        )
        for i in range(40)
    ]
    train, test = split_trials(len(pools), rng)
    train_ids = {float(pools[i].samples[0, 0]) for i in train}
    test_ids = {float(pools[i].samples[0, 0]) for i in test}
    report(
        "no trial contributes samples to both sides",
        not (train_ids & test_ids) and len(train_ids) + len(test_ids) == len(pools),
    )


def check_trial_seed_is_order_invariant():
    """The per-trial seed depends on identity only, not iteration position.

    `comp_pooled_features` shares one RNG across its pooling loop, so trial
    *i*'s 50-sample draw depends on how many trials preceded it. Restricting
    the pool to a session subset would then change every draw, and the pool
    would not reproduce. Also checks the seed is stable across processes, which
    `hash()` on a string is not.
    """
    pairs = [("june_24_g0", 3), ("Nov_6_g0", 11), ("april_8_g0", 7)]
    first = [_trial_seed(s, t) for s, t in pairs]
    second = [_trial_seed(s, t) for s, t in reversed(pairs)][::-1]
    distinct = len(set(first)) == len(first)
    report(
        "per-trial seed is identity-based, stable and distinct",
        first == second and distinct and _trial_seed("june_24_g0", 3) == 1638830442,
        f"seeds {first}",
    )


def check_fit_refuses_less_data_than_k():
    """A pool smaller than K cannot be fitted and must say so, not crash."""
    # Distinct points, so the K = 2 case is a real fit rather than a
    # degenerate one that only warns its way to an answer.
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    report(
        "fit_codebook returns None below K samples",
        fit_codebook(xy, 6) is None and fit_codebook(xy, 2) is not None,
    )


def main():
    print("K-selection invariant checks (synthetic data, no cache needed)\n")

    print("The EV definition")
    check_ev_one_prototype_is_zero()
    check_ev_rises_to_the_true_k()

    print("\nThe radius gate")
    check_gate_drops_far_centroids()
    check_gate_is_inert_when_wide()
    check_categorical_insensitivity()

    print("\nThe elbow rules")
    check_kneedle_finds_a_known_knee()
    check_kneedle_refuses_a_flat_curve()
    check_kneedle_ignores_a_noise_dip()
    check_other_rules_agree_on_a_clean_curve()

    print("\nThe split")
    check_split_is_disjoint_and_complete()
    check_split_is_at_trial_level()
    check_trial_seed_is_order_invariant()
    check_fit_refuses_less_data_than_k()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {', '.join(FAILURES)}")
        raise SystemExit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
