"""Choice-conditioned beliefs over the unchosen exits, one cache per monkey.

q depends only on the maze and the chosen exit c: it is row c of the maze's
4 x 4 table (chosen exit x exit looked at), built from the path that ends at
c and applied to every trial that chose c, error trials included.

Per trial (README.md, "Draws" to "Beliefs"):

1. `N_DRAWS` measurement pairs (tm1, tm2) are drawn around the two
   intervals of the path that ends at the chosen exit (the trial's own path
   when it is correct) with the monkey's Weber fraction `WEBER`, plus a separate
   noisy measurement of the whole travel time t1 + t2 for Total Time (its
   own random stream, so the (tm1, tm2) draws do not depend on it).
2. For each strategy, a draw is kept with weight equal to the probability
   that the strategy's rule picks the chosen exit on it (1 or 0; 1/k on an
   exact k-way tie). If no draw picks the chosen exit, every draw is kept.
3. The strategy's belief over the four exits is averaged over the kept
   draws (weighted), the chosen exit is zeroed, `EPS` is added to the three
   unchosen exits and they are renormalised: q_M.

Every model in `ALL_MODELS` (`counterfactual.noise`) uses the same draws. Also stored: the acceptance rate (weighted share
of draws kept) and the unconditioned choice frequency of each rule on the
same path (the ``uncond`` q of `counterfactual.compare`; on correct trials it is
the per-trial choice probability of v1, commit `09e6fea`).

Cache: ``<PROCESSED_ROOT>/<Monkey>_cfv2_q.npz``.

    uv run python -m counterfactual.beliefs [--monkey ...] [--refresh]
"""

from __future__ import annotations

import argparse

import numpy as np

from data.loader import savez_atomic
from data.mat import load_npz
from counterfactual import extract
from counterfactual.config import ALL_MODELS, EPS, MONKEYS, N_DRAWS, SCHEMA, SEED, WEBER, q_cache
from counterfactual.noise import log_beliefs, measure, pick_probs, terms
from counterfactual.task import arm_durations, true_intervals

# chosen exit code -> (horizontal, vertical) of the path ending there: -1 left / +1 right, -1 down / +1 up
PATH_OF_EXIT = {1: (-1, 1), 2: (-1, -1), 3: (1, 1), 4: (1, -1)}

CHUNK = 6


def conditioned(tr, w, **kw):
    """Per trial, the row of its (geometry, chosen exit) cell: each distinct
    cell is simulated once (`_simulate`) and assigned to all of its trials,
    so trials in a cell share q exactly."""
    key = np.c_[arm_durations(tr["h"], tr["vel"]), tr["chosen"]]
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    cells = {k: v[first] for k, v in tr.items() if np.ndim(v) and len(v) == len(key)}
    out = _simulate(cells, w, **kw)
    return {k: v[inverse.ravel()] for k, v in out.items()}


def _simulate(tr, w, *, n_draws=N_DRAWS, seed=SEED, eps=EPS):
    dur = arm_durations(tr["h"], tr["vel"])
    lr, lr2 = np.array([PATH_OF_EXIT[int(e)] for e in tr["chosen"]]).T
    t1, t2 = true_intervals(dur, lr, lr2)  # the chosen exit's path
    c = tr["chosen"].astype(int) - 1
    n = len(c)
    rng, rng_total = np.random.default_rng(seed), np.random.default_rng([seed, 3])
    out = {f"{k}_{m}": np.zeros((n, 4)) for k in ("q", "choice") for m in ALL_MODELS}
    out.update({f"accept_{m}": np.zeros(n) for m in ALL_MODELS})
    for a in range(0, n, CHUNK):
        i = np.arange(a, min(a + CHUNK, n))
        z1, z2 = rng.standard_normal((2, len(i), n_draws))
        z3 = rng_total.standard_normal((len(i), n_draws))
        L = terms(measure(t1[i], z1, w), measure(t2[i], z2, w), measure(t1[i] + t2[i], z3, w), dur[i], w)
        for m in ALL_MODELS:
            pick = pick_probs(m, L)  # (n, K, 4)
            weight = pick[np.arange(len(i)), :, c[i]]  # (n, K): rule picks the chosen exit
            none = weight.sum(axis=1) == 0
            weight[none] = 1.0
            belief = np.exp(log_beliefs(m, L))
            q = np.einsum("nk,nke->ne", weight, belief) / weight.sum(axis=1, keepdims=True)
            q = q + eps
            q[np.arange(len(i)), c[i]] = 0.0
            out[f"q_{m}"][i] = q / q.sum(axis=1, keepdims=True)
            out[f"accept_{m}"][i] = np.where(none, 0.0, weight.mean(axis=1))
            out[f"choice_{m}"][i] = pick.mean(axis=1)
    return out


def load(monkey, *, refresh=False):
    tr = extract.load(monkey)
    path = q_cache(monkey)
    if path.exists() and not refresh:
        d = load_npz(path)
        if int(d["schema"]) == SCHEMA and float(d["w"]) == WEBER[monkey] and len(d["q_H"]) == len(tr["chosen"]):
            return d
    print(f"Conditioning {monkey} beliefs ({N_DRAWS} draws per (maze, chosen exit) cell, w = {WEBER[monkey]}) ...", flush=True)
    d = conditioned(tr, WEBER[monkey])
    d.update(w=np.float64(WEBER[monkey]), schema=np.int64(SCHEMA))
    savez_atomic(path, **d)
    for m in ALL_MODELS:
        acc = d[f"accept_{m}"]
        print(f"  {m}: acceptance median {np.median(acc):.2f}, {int(np.sum(acc == 0))} trials with no matching draw")
    return load_npz(path)


def main():
    p = argparse.ArgumentParser(description="Choice-conditioned beliefs per trial.")
    p.add_argument("--monkey", nargs="*", default=list(MONKEYS), choices=MONKEYS)
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()
    for m in args.monkey:
        load(m, refresh=args.refresh)


if __name__ == "__main__":
    main()
