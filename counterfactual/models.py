"""Choice models ported from Figure Five: Hierarchical, Sequential, and the
two experimental comparators Total Time and Optimal.

Source: `zarchive/MR_AG_MJ/Figure_Five/03_Likelihood_Analysis/`, the per-trial
simulation `simulate_hs_models` in
`Run_Canonical_Neural_Strategy_vs_Trialwise_Opposite.m` and the helpers
`Model_Helpers/{Hierarchical,Sequential,Total_Time,Optimal}_Model_Physiology.m`,
`Model_Helpers/D_likelihood_Physiology.m`.

Per trial the monkey is assumed to measure the two interval durations of the
*true* path (first arm, second arm; durations = arm length / velocity in
seconds) with Weber noise: ``one = max(t1 + N(0, 1) * scalar * t1, eps)`` and
likewise ``two``. Each model then scores arms with the Gaussian
``normpdf(measured, arm, scalar * arm)``:

``H`` (hierarchical)  compare ``one`` against h1 vs h4 and commit left/right
                      (fair coin if h1 == h4), then compare ``two`` against the
                      two vertical arms on that side; exact ties uniform.
``S`` (sequential)    the paper's "second-only" heuristic: compare ``two``
                      against all four vertical arms h2, h3, h5, h6; ties
                      uniform. h1 and h4 are never consulted.
``total`` (Total Time) an independent noisy measurement of the whole travel
                      time, ``total = (t1 + t2)(1 + scalar * N)``, compared
                      against the four arm sums h1+h2, h1+h3, h4+h5, h4+h6.
``optimal``           the two intervals scored independently and multiplied,
                      ``L(one | horizontal arm) * L(two | vertical arm)`` for
                      each exit; like H without the initial commitment.

`N_SAMPLES` noise draws per trial give the choice probability of each exit
(LU, LD, RU, RD = codes 1-4). `scalar` is the published mean Weber fraction
per monkey (`WEBER_SCALAR`, from `MR_AG_MJ/load_timing_scalar.m` reading
`Data/Pre_Physiology/Model_Results/Behavioral_Psychometrics/timing_scalars.mat`).

The counterfactual distribution of a trial is the model's probability
renormalised over the three *unchosen* exits (`counterfactual_dist`).

Cache: ``<PROCESSED_ROOT>/<Monkey>_cf_models.npz`` (``prob_h``, ``prob_s``,
``prob_total``, ``prob_optimal``).

    uv run python -m counterfactual.models [--monkey ...] [--refresh]
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import norm

from data.builder import behavioral_lookup, trial_qc_ok
from data.config import processed_npz
from data.loader import load_eye_behavioral_data, savez_atomic
from data.mat import load_npz
from counterfactual.config import (
    MONKEYS,
    MODEL_EPS,
    MODEL_SEED,
    N_SAMPLES,
    SESSIONS,
    WEBER_SCALAR,
)
from counterfactual.exits import chosen_exit, correct_exit

MODELS_SCHEMA = 2
MODEL_NAMES = ("h", "s", "total", "optimal")

def _gauss(measured, arm, scalar):
    """`D_likelihood_Physiology`: normpdf(measured, arm, arm * scalar)."""
    return norm.pdf(measured, loc=arm, scale=arm * scalar)


def _argmax_random_ties(scores, rng):
    """Index of the maximum along the last axis; exact ties uniform at random
    (`resolve_ties` / `randomMaxIndex`)."""
    top = scores == scores.max(axis=-1, keepdims=True)
    # random tie-break: add a random key on the tied entries only
    key = np.where(top, rng.random(scores.shape), -1.0)
    return key.argmax(axis=-1)


def true_intervals(h_sec, lr, lr2):
    """``(t1, t2)`` seconds of the true path for each trial: horizontal arm by
    `lr` (-1 left = h1, +1 right = h4), vertical arm by `(lr, lr2)`."""
    h_sec = np.asarray(h_sec, dtype=float).reshape(-1, 6)
    lr = np.asarray(lr, dtype=int)
    lr2 = np.asarray(lr2, dtype=int)
    t1 = np.where(lr == 1, h_sec[:, 3], h_sec[:, 0])
    t2 = np.where(lr == -1, np.where(lr2 == 1, h_sec[:, 1], h_sec[:, 2]),
                  np.where(lr2 == 1, h_sec[:, 4], h_sec[:, 5]))
    return t1, t2


def simulate_models(h_sec, lr, lr2, scalar, *, n_samples=N_SAMPLES, seed=MODEL_SEED, batch=500):
    """``{"h", "s", "total", "optimal"}`` -> (n_trials, 4) choice probabilities
    over LU, LD, RU, RD. Vectorised port of `simulate_hs_models` extended with
    the Total Time and Optimal helpers; `n_samples` noise draws per trial,
    processed in batches of `batch` draws, all four models on the same draws.
    """
    h_sec = np.asarray(h_sec, dtype=float).reshape(-1, 6)
    n = h_sec.shape[0]
    if n == 0:
        return {k: np.empty((0, 4)) for k in MODEL_NAMES}
    if not np.all(np.isfinite(h_sec)) or np.any(h_sec <= 0):
        raise ValueError("arm durations must be finite and positive")
    rng = np.random.default_rng(seed)
    t1, t2 = true_intervals(h_sec, lr, lr2)
    t_total = t1 + t2
    counts = {k: np.zeros((n, 4)) for k in MODEL_NAMES}
    h1, h2, h3, h4, h5, h6 = (h_sec[:, k][None, :] for k in range(6))
    horizontal = (h1, h1, h4, h4)
    vertical_arms = (h2, h3, h5, h6)
    same_side = (h_sec[:, 0] == h_sec[:, 3])[None, :]
    eps = np.finfo(float).eps
    done = 0
    while done < n_samples:
        m = min(batch, n_samples - done)
        one = np.maximum(t1[None, :] + rng.standard_normal((m, n)) * scalar * t1[None, :], eps)
        two = np.maximum(t2[None, :] + rng.standard_normal((m, n)) * scalar * t2[None, :], eps)
        total = np.maximum(t_total[None, :] + rng.standard_normal((m, n)) * scalar * t_total[None, :], eps)
        vertical = np.stack([_gauss(two, hv, scalar) for hv in vertical_arms], axis=-1)  # (m, n, 4)
        # ---- Hierarchical: side first, then the two vertical arms on that side.
        go_left = _gauss(one, h1, scalar) > _gauss(one, h4, scalar)
        coin = rng.random((m, n)) >= 0.5  # `compute_likelihoods`: fair coin when h1 == h4
        go_left = np.where(same_side, coin, go_left)
        masked = vertical.copy()
        masked[..., 2:][go_left] = 0.0
        masked[..., :2][~go_left] = 0.0
        picks = {"h": _argmax_random_ties(masked, rng)}
        # ---- Sequential: all four vertical arms on the second interval.
        picks["s"] = _argmax_random_ties(vertical, rng)
        # ---- Total Time: the whole travel time against the four arm sums.
        sums = np.stack([_gauss(total, hh + hv, scalar) for hh, hv in zip(horizontal, vertical_arms)], axis=-1)
        picks["total"] = _argmax_random_ties(sums, rng)
        # ---- Optimal: both intervals, independently, no commitment.
        joint = np.stack([_gauss(one, hh, scalar) for hh in horizontal], axis=-1) * vertical
        picks["optimal"] = _argmax_random_ties(joint, rng)
        for name, pick in picks.items():
            for k in range(4):
                counts[name][:, k] += (pick == k).sum(axis=0)
        done += m
    prob = {k: c / n_samples for k, c in counts.items()}
    for p in prob.values():
        assert np.all(np.isfinite(p)) and np.all(p >= 0) and np.allclose(p.sum(axis=1), 1.0)
    return prob


def counterfactual_dist(prob, chosen, *, eps=MODEL_EPS):
    """``q`` over the four exit codes with the chosen exit at 0 and the three
    unchosen exits renormalised: ``q(e) = (p(e) + eps) / sum_{e' != chosen}(p(e') + eps)``."""
    prob = np.asarray(prob, dtype=float).reshape(-1, 4)
    chosen = np.asarray(chosen, dtype=int).reshape(-1)
    q = prob + eps
    q[np.arange(len(chosen)), chosen - 1] = 0.0
    return q / q.sum(axis=1, keepdims=True)


# ---- cache ---------------------------------------------------------------------


def cache_path(monkey):
    return processed_npz(f"{monkey}_cf_models")


def cache_ok(data, sessions, scalar):
    try:
        return (
            int(data["schema"]) == MODELS_SCHEMA
            and int(data["n_samples"]) == N_SAMPLES
            and int(data["seed"]) == MODEL_SEED
            and float(data["scalar"]) == float(scalar)
            and tuple(np.asarray(data["sessions"]).astype(str)) == tuple(sessions)
        )
    except KeyError:
        return False


def extract(monkey, sessions, *, scalar=None, n_samples=N_SAMPLES, seed=MODEL_SEED):
    """Per QC trial of `sessions` with a decodable chosen exit: the two
    probability vectors. Arm durations are `h_k / vel` with the session's
    single `vel` (asserted, as the MATLAB's `unique(vel)` assumes)."""
    scalar = WEBER_SCALAR[monkey] if scalar is None else float(scalar)
    sessions = tuple(sessions)
    behavioral = load_eye_behavioral_data(monkey)
    beh = behavioral_lookup(behavioral)
    out = {k: [] for k in ("session", "trial", "maze", "path", "h_sec", "lr", "lr2", "chosen", "correct")}
    for session in sessions:
        rows = [i for (s, t), i in beh.items() if s == session]
        rows.sort(key=lambda i: int(behavioral["trial_indices_all"][i]))
        vel = np.unique(behavioral["vel"][rows].astype(float))
        vel = vel[np.isfinite(vel)]
        assert vel.size == 1 and vel[0] > 0, f"{monkey} {session}: expected one velocity, got {vel}"
        for i in rows:
            if not trial_qc_ok(behavioral, i):
                continue
            flags = tuple(float(behavioral[f"trial_answer{k}"][i]) for k in range(1, 5))
            lr, lr2 = float(behavioral["LR"][i]), float(behavioral["LR2"][i])
            chosen = chosen_exit(flags, lr, lr2)
            if chosen is None:
                continue
            out["session"].append(session)
            out["trial"].append(int(behavioral["trial_indices_all"][i]))
            out["maze"].append(int(behavioral["geo_type"][i]))
            out["path"].append(int(behavioral["path_type"][i]))
            out["h_sec"].append([float(behavioral[f"h{k}"][i]) / vel[0] for k in range(1, 7)])
            out["lr"].append(int(lr))
            out["lr2"].append(int(lr2))
            out["chosen"].append(chosen)
            out["correct"].append(correct_exit(float(behavioral["path_type"][i])) or -1)
    h_sec = np.asarray(out["h_sec"], dtype=float).reshape(-1, 6)
    prob = simulate_models(h_sec, out["lr"], out["lr2"], scalar, n_samples=n_samples, seed=seed)
    return {
        "session": np.asarray(out["session"], dtype=str),
        "trial_indices_all": np.asarray(out["trial"], dtype=int),
        "maze_id": np.asarray(out["maze"], dtype=int),
        "path_type": np.asarray(out["path"], dtype=int),
        "h_sec": h_sec.astype(np.float32),
        "lr": np.asarray(out["lr"], dtype=int),
        "lr2": np.asarray(out["lr2"], dtype=int),
        "chosen_exit": np.asarray(out["chosen"], dtype=int),
        "correct_exit": np.asarray(out["correct"], dtype=int),
        "prob_h": prob["h"].astype(np.float32),
        "prob_s": prob["s"].astype(np.float32),
        "prob_total": prob["total"].astype(np.float32),
        "prob_optimal": prob["optimal"].astype(np.float32),
        "scalar": np.float64(scalar),
        "n_samples": np.int64(n_samples),
        "seed": np.int64(seed),
        "sessions": np.asarray(sessions, dtype=str),
        "schema": np.int64(MODELS_SCHEMA),
    }


def load(monkey, *, sessions=None, refresh=False):
    sessions = tuple(SESSIONS[monkey] if sessions is None else sessions)
    scalar = WEBER_SCALAR[monkey]
    path = cache_path(monkey)
    if path.exists() and not refresh:
        data = load_npz(path)
        if cache_ok(data, sessions, scalar):
            return data
        print(f"  {path.name}: schema / sessions / scalar differ; rebuilding")
    print(f"Simulating {monkey} H/S models ({', '.join(sessions)}; scalar {scalar}, "
          f"{N_SAMPLES} draws) ...", flush=True)
    data = extract(monkey, sessions)
    savez_atomic(path, **data)
    print(f"Saved {path} ({len(data['session'])} trials)")
    return data


def main():
    parser = argparse.ArgumentParser(description="Per-trial exit probabilities under the choice models (Figure Five port).")
    parser.add_argument("--monkey", nargs="*", default=list(MONKEYS), choices=list(MONKEYS))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    for monkey in args.monkey:
        d = load(monkey, refresh=args.refresh)
        ch = d["chosen_exit"]
        idx = np.arange(len(ch))
        parts = [f"p(chosen | {m}) {d[f'prob_{m}'].astype(float)[idx, ch - 1].mean():.3f}" for m in MODEL_NAMES]
        print(f"{monkey}: {len(ch)} trials; mean " + ", ".join(parts))


if __name__ == "__main__":
    main()
