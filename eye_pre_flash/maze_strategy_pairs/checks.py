"""Invariant checks for the estimator, on synthetic data.

``uv run python -m eye_pre_flash.maze_strategy_pairs.checks``

No feature cache and no labels needed -- everything here is generated. The
point is not to reproduce any particular number but to pin the properties the
2x2 would be meaningless without.
"""

from __future__ import annotations

import numpy as np

from eye_pre_flash.maze_strategy_pairs.estimator import (
    BLOCK_MEANS,
    H,
    S,
    coverage,
    delta,
    estimate,
    permute_within_session,
    quadrant_block_means,
    session_blocks,
)

FAILURES = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def synth(rng, *, n_h, n_s, d=12, n_sessions=3, sep=0.0, session_sd=0.0):
    """Trials for one maze, with optional H/S separation and session offsets.

    `sep` scales a shared H-only pattern, so `sep = 0` means H and S differ in
    nothing but noise. `session_sd` scales a per-session offset shared by every
    trial of that session, which is the confound the within-session shuffle
    exists to calibrate.
    """
    n = n_h + n_s
    y = np.r_[np.zeros(n_h, int), np.ones(n_s, int)]
    sessions = np.array([f"s{i % n_sessions}" for i in range(n)])
    X = rng.normal(size=(n, d))
    if sep:
        X[y == H] += sep * rng.normal(size=d)
    if session_sd:
        for name in np.unique(sessions):
            X[sessions == name] += session_sd * rng.normal(size=d)
    return X, y, sessions


def check_groups_disjoint_and_sized():
    """Four groups of exactly `m`, no trial in two of them.

    This is what keeps a trial from ever being correlated with itself. If the
    two groups of a strategy overlapped, the shared trials would contribute
    r = 1 self-pairs and lift that diagonal cell for no reason at all.
    """
    from eye_pre_flash.maze_strategy_pairs.estimator import _group_indicators

    rng = np.random.default_rng(0)
    idx = np.arange(5, 32)  # deliberately not starting at 0
    m, n_rounds, n = 9, 50, 40
    u_a, u_b = _group_indicators(idx, m, n, n_rounds, rng)

    sizes_ok = np.all(u_a.sum(1) == m) and np.all(u_b.sum(1) == m)
    disjoint = not np.any((u_a > 0) & (u_b > 0))
    in_range = not np.any((u_a + u_b)[:, np.setdiff1d(np.arange(n), idx)])
    report("groups are exactly m", sizes_ok)
    report("groups a and b are disjoint", disjoint)
    report("groups only draw from the strategy's own trials", in_range)


def check_no_self_pairs():
    """The diagonal must not be pinned near 1 when trials are pure noise.

    A self-pair leak is the failure this catches: with independent trials the
    true within-strategy similarity is ~0, but any overlap between the two
    groups would drag `r_HH` up toward 1.
    """
    rng = np.random.default_rng(1)
    X, y, _ = synth(rng, n_h=60, n_s=60, d=20)
    q = quadrant_block_means(X, y, 30, rng, n_rounds=200)
    worst = float(np.max(np.abs(q)))
    # One correlation of two noisy means per round, so the cell is not driven
    # as close to 0 as a mean over m*m trial pairs would be. A self-pair leak
    # pins it near 1; anything well below that is sampling noise.
    report("independent trials give a flat 2x2", worst < 0.25, f"max|r| = {worst:.4f}")


def check_relabel_symmetry():
    """Swapping which strategy is called H must not move delta systematically.

    The load-bearing one: if the estimator preferred the diagonal over the
    off-diagonal, or one cell over the other, it would show up here.

    Two parts, because an *exact* equality is not available.
    `quadrant_block_means` draws its partitions strategy by strategy, so
    relabelling swaps which trial block consumes which random numbers; the
    two runs therefore use genuinely different groupings and differ by Monte
    Carlo noise even when the estimator is perfectly symmetric. Seeding cannot
    be aligned around that without reaching into the sampler purely for a test.

    So: the statistic is checked for exact symmetry on a fixed matrix, and the
    sampler is checked by whether the relabelling gap *shrinks with rounds*.
    A structural asymmetry is a fixed offset and would not shrink; sampling
    noise falls as 1/sqrt(n_rounds).
    """
    rng = np.random.default_rng(2)
    for _ in range(200):
        q = rng.normal(size=(2, 2))
        if abs(delta(q) - delta(q[::-1, ::-1])) > 1e-15:
            report("delta is exactly symmetric in H and S", False)
            break
    else:
        report("delta is exactly symmetric in H and S", True)

    X, y, _sessions = synth(rng, n_h=40, n_s=34, d=12, sep=0.8)
    m = min((y == H).sum(), (y == S).sum()) // 2

    def mean_gap(n_rounds, n_seeds=8):
        gaps = []
        for seed in range(n_seeds):
            a = delta(quadrant_block_means(X, y, m, np.random.default_rng(seed), n_rounds=n_rounds))
            b = delta(quadrant_block_means(X, 1 - y, m, np.random.default_rng(seed), n_rounds=n_rounds))
            gaps.append(abs(a - b))
        return float(np.mean(gaps))

    coarse, fine = mean_gap(100), mean_gap(1600)
    # 16x the rounds should cut sampling noise ~4x. Require 2x, which a fixed
    # bias could not achieve, while tolerating the spread of only 8 seeds.
    report("relabelling gap shrinks with rounds (noise, not bias)", fine < coarse / 2,
           f"{coarse:.2e} at 100 rounds -> {fine:.2e} at 1600")
    report("relabelling gap is small at 1600 rounds", fine < 1e-2,
           f"mean |diff| = {fine:.2e}")


def check_shuffle_preserves_session_counts():
    """The null must not move a trial between sessions, only labels within one.

    If it did, the shuffled pool would carry a different mix of same-session
    trial pairs than the data, and `z` would be measured against the wrong
    floor -- see the first entry in CAVEATS.md.
    """
    rng = np.random.default_rng(3)
    _X, y, sessions = synth(rng, n_h=30, n_s=25, n_sessions=4)
    blocks = session_blocks(sessions)
    before = {s: int((y[sessions == s] == H).sum()) for s in np.unique(sessions)}

    ok, changed = True, False
    for _ in range(50):
        y_perm = permute_within_session(y, blocks, rng)
        after = {s: int((y_perm[sessions == s] == H).sum()) for s in np.unique(sessions)}
        ok &= after == before
        changed |= bool(np.any(y_perm != y))
    report("shuffle preserves each session's H/S counts", ok)
    report("shuffle actually moves labels", changed)


def check_null_is_centred():
    """With no real H/S difference, delta must sit inside its own null.

    Not that delta is zero -- the estimator has a floor at finite `m` and
    `d`, which is exactly why `z` is measured against `null_mean` rather than
    against zero -- but that `z` is unremarkable. `session_sd` puts a real
    per-session offset in the data, so this also checks that the
    within-session shuffle absorbs it rather than reporting it as an effect.
    """
    rng = np.random.default_rng(4)
    X, y, sessions = synth(rng, n_h=45, n_s=38, d=12, sep=0.0, session_sd=1.5)
    result, reason = estimate(
        X, y, sessions, method=BLOCK_MEANS, n_rounds=50, n_perm=300, seed=0
    )
    ok = result is not None and abs(result.z) < 3
    report("no true effect gives |z| < 3 despite session offsets", ok,
           f"z = {result.z:+.2f}, delta = {result.delta:+.4f}" if result else reason)


def check_real_effect_is_found():
    """A genuine H/S difference must clear the null."""
    rng = np.random.default_rng(5)
    X, y, sessions = synth(rng, n_h=45, n_s=38, d=12, sep=1.2)
    result, reason = estimate(
        X, y, sessions, method=BLOCK_MEANS, n_rounds=50, n_perm=300, seed=0
    )
    ok = result is not None and result.delta > 0 and result.z > 5
    report("a real H/S difference is detected", ok,
           f"z = {result.z:+.2f}, delta = {result.delta:+.4f}" if result else reason)


def check_block_means_rises_with_m():
    """The diagonal must climb as the groups get larger.

    Averaging `m` trials suppresses their independent noise before the
    correlation is taken, so raw cells are not comparable across panels with
    different group sizes. If the diagonal stopped rising with `m`, that
    warning would be wrong.
    """
    rng = np.random.default_rng(8)
    X, y, _sessions = synth(rng, n_h=80, n_s=80, d=12, sep=0.8)
    diags = []
    for m in (5, 40):
        q = quadrant_block_means(X, y, m, np.random.default_rng(1), n_rounds=200)
        diags.append(q[H, H])
    report("block_means diagonal rises with group size",
           diags[1] - diags[0] > 0.05,
           f"m=5: {diags[0]:.3f}; m=40: {diags[1]:.3f} "
           f"(+{diags[1] - diags[0]:.3f})")


def check_coverage_rule():
    """The 10-trial rule admits and rejects exactly what it should."""
    cases = [
        (10, 10, True), (10, 9, False), (9, 10, False),
        (40, 10, True), (3, 80, False),
    ]
    ok = True
    for n_h, n_s, expected in cases:
        y = np.r_[np.zeros(n_h, int), np.ones(n_s, int)]
        _h, _s, m, reason = coverage(y, min_trials=10)
        admitted = reason == ""
        ok &= admitted == expected
        if admitted:
            ok &= m == min(n_h, n_s) // 2
    report("coverage rule admits exactly n_H >= 10 and n_S >= 10", ok)


def check_degenerate_trials_kept_not_dropped():
    """A zero-variance trial stays in the pool and still enters the group mean.

    The contract that keeps every panel on the same dataset. Excluding such a
    trial would make the trial set depend on K and on the variant, and those
    trials are strategy-biased, so excluding them would remove a slice of the
    data correlated with the thing being measured.
    """
    rng = np.random.default_rng(6)
    X, y, sessions = synth(rng, n_h=30, n_s=30, d=12)
    X[0] = 0.0  # no gaze at all -> every feature equal
    X[45] = 7.0  # constant nonzero -> also no variance
    result, reason = estimate(X, y, sessions, n_rounds=20, n_perm=50, seed=0)

    report("zero-variance trials are counted", result is not None and result.n_degenerate == 2,
           f"n_degenerate = {result.n_degenerate}" if result else reason)
    report("zero-variance trials stay in n_H / n_S",
           result is not None and result.n_H == 30 and result.n_S == 30,
           f"n_H = {result.n_H}, n_S = {result.n_S}" if result else reason)
    report("the 2x2 is still finite", result is not None and np.isfinite(result.q).all())


def check_block_means_uses_all_trials():
    """An all-zero trial still contributes to its group's mean.

    That mean has variance, so the correlation is defined and the trial is
    not dropped from the pool.
    """
    rng = np.random.default_rng(9)
    X, y, sessions = synth(rng, n_h=30, n_s=30, d=12)
    X[0] = 0.0
    result, reason = estimate(
        X, y, sessions, method=BLOCK_MEANS, n_rounds=50, n_perm=50, seed=0
    )
    ok = result is not None and np.isfinite(result.q).all() and result.n_H == 30
    report("block_means handles zero-variance trials without dropping them", ok,
           f"n_H = {result.n_H}, q finite = {np.isfinite(result.q).all()}"
           if result else reason)


def main():
    print("Estimator invariant checks (synthetic data, no cache needed)\n")
    for check in (
        check_groups_disjoint_and_sized,
        check_no_self_pairs,
        check_relabel_symmetry,
        check_shuffle_preserves_session_counts,
        check_null_is_centred,
        check_real_effect_is_found,
        check_block_means_rises_with_m,
        check_coverage_rule,
        check_degenerate_trials_kept_not_dropped,
        check_block_means_uses_all_trials,
    ):
        print(f"\n{check.__name__}:")
        print(f"    {(check.__doc__ or '').strip().splitlines()[0]}")
        check()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        raise SystemExit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
