"""Descriptive models of the first counterfactual look (see DESCRIPTIVE.md).

Which simple model explains where the monkey looks first after choosing?
Each model is a multinomial logit over the three unchosen exits with a few
features per exit, fitted to the looks (small L2 penalty) and scored by
10-fold cross-validated gain over chance per trial, log P(look) − log(1/3).
All trials with a counterfactual look are used (labels only split the
scores). The saturated reference is the add-one-smoothed frequency of each
target within its (maze, chosen exit) cell, scored the same way.

Features per unchosen exit e (chosen exit c):
``rel``    relation to c: horizontal, diagonal (sibling = reference)
``exit``   screen identity of e: LD, RU, RD (LU = reference)
``dvert``  |vertical arm of e − vertical arm of c| / 10 (S-like similarity)
``dtot``   |total path of e − total path of c| / 10 (Total Time-like)
``lq``     log q_H(e), log q_S(e) (`counterfactual.beliefs`)
A ``_maze`` suffix gives each maze its own copy of the feature.

Output: ``counterfactual/out/<Monkey>/descriptive_results.csv``.

    uv run python -m counterfactual.descriptive [--monkey ...]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from counterfactual import beliefs, extract
from counterfactual.config import MONKEYS, out_dir
from counterfactual.task import relation, unchosen

RELATIONS = ("horizontal", "diagonal")
N_FOLDS = 10
L2 = 0.1
MODELS = {
    "relation": ["rel"],
    "screen exit": ["exit"],
    "relation + screen exit": ["rel", "exit"],
    "relation + screen exit, per maze": ["rel_maze", "exit_maze"],
    "vertical-arm similarity": ["dvert"],
    "total-path similarity": ["dtot"],
    "strategy beliefs (free weights)": ["lq"],
    "relation + screen exit + strategy beliefs": ["rel", "exit", "lq"],
}


def build(monkey):
    tr, q = extract.load(monkey), beliefs.load(monkey)
    X = []
    for r in range(len(tr["session"])):
        items = extract.counterfactual_items(tr, r)
        if not items:
            continue
        c, h = int(tr["chosen"][r]), tr["h"][r]
        slots = unchosen(c)
        vert = {1: h[1], 2: h[2], 3: h[4], 4: h[5]}
        tot = {e: (h[0] if e <= 2 else h[3]) + vert[e] for e in (1, 2, 3, 4)}
        x = dict(
            maze=int(tr["maze"][r]), chosen=c, label=tr["label"][r], target=slots.index(items[0]),
            rel=np.array([[relation(c, e) == k for k in RELATIONS] for e in slots], float),
            exit=np.array([[e == k for k in (2, 3, 4)] for e in slots], float),
            dvert=np.array([[abs(vert[e] - vert[c]) / 10] for e in slots]),
            dtot=np.array([[abs(tot[e] - tot[c]) / 10] for e in slots]),
            lq=np.log(np.array([[q["q_H"][r][e - 1], q["q_S"][r][e - 1]] for e in slots])),
        )
        for f in ("rel", "exit"):
            z = np.zeros((3, x[f].shape[1] * 6))
            k = (x["maze"] - 1) * x[f].shape[1]
            z[:, k:k + x[f].shape[1]] = x[f]
            x[f"{f}_maze"] = z
        X.append(x)
    return X


def _logp(F, b):
    u = F @ b
    u = u - u.max(axis=1, keepdims=True)
    return u - np.log(np.exp(u).sum(axis=1, keepdims=True))


def fit(F, y):
    obj = lambda b: -_logp(F, b)[np.arange(len(y)), y].sum() + L2 * b @ b
    return minimize(obj, np.zeros(F.shape[2]), method="L-BFGS-B").x


def cv_gain(X, feats=None, seed=0):
    """Per-trial held-out gain over chance; ``feats=None`` is the saturated table."""
    fold = np.random.default_rng(seed).permutation(len(X)) % N_FOLDS
    y = np.array([x["target"] for x in X])
    out = np.zeros(len(X))
    if feats is None:
        key = np.array([x["maze"] * 10 + x["chosen"] for x in X])
        for i in range(len(X)):
            train = (fold != fold[i]) & (key == key[i])
            cnt = 1.0 + np.bincount(y[train], minlength=3)
            out[i] = np.log(cnt[y[i]] / cnt.sum() * 3)
        return out
    F = np.stack([np.concatenate([x[f] for f in feats], axis=1) for x in X])
    for f in range(N_FOLDS):
        te = fold == f
        b = fit(F[~te], y[~te])
        out[te] = _logp(F[te], b)[np.arange(te.sum()), y[te]] + np.log(3)
    return out


def run(monkey):
    X = build(monkey)
    lab = np.array([x["label"] for x in X])
    maze = np.array([x["maze"] for x in X])
    rows = []
    for name, feats in list(MODELS.items()) + [("saturated (maze, chosen) table", None)]:
        g = cv_gain(X, feats)
        rows.append(dict(model=name, all=g.mean(), H_labelled=g[lab == 0].mean(), S_labelled=g[lab == 1].mean(),
                         **{f"maze{k}": g[maze == k].mean() for k in range(1, 7)}))
    res = pd.DataFrame(rows)
    res.to_csv(out_dir(monkey) / "descriptive_results.csv", index=False)
    F = np.stack([np.concatenate([x[f] for f in ("rel", "exit")], axis=1) for x in X])
    b = fit(F, np.array([x["target"] for x in X]))
    print(f"\n{monkey} (n {len(X)}): 10-fold CV gain over chance per trial, first look")
    print(res.round(2).to_string(index=False))
    print("relation + screen exit coefficients:", dict(zip(("horizontal", "diagonal", "LD", "RU", "RD"), b.round(2))))
    return res


def main():
    p = argparse.ArgumentParser(description="Descriptive models of the first counterfactual look.")
    p.add_argument("--monkey", nargs="*", default=list(MONKEYS), choices=MONKEYS)
    for m in p.parse_args().monkey:
        run(m)


if __name__ == "__main__":
    main()
