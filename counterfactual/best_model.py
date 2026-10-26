"""Best behavioral model per neural label, ranked by how well its
choice-conditioned belief scores the counterfactual looks.

v1's best-model table (the earlier `counterfactual/` package, commit
`09e6fea`) ranks the models by how well they
predict the *choice* (held-out choice NLL). Here every model in
`ALL_MODELS` (`counterfactual.noise`; revision and lapse variants left out) is given
the same treatment as H and S in `counterfactual.compare` / `counterfactual.logp`: draws from the
correct path, kept when the model's rule picks the chosen exit, belief over
the unchosen exits averaged over the kept draws (`counterfactual.beliefs`). Each model
is scored by the mean gain over chance of the counterfactual looks,
``log P_M(looks) - log P_uniform(looks)`` (Plackett-Luce), per label group.
Nothing is fitted, so no cross-validation is needed.

Per label group: models ranked by mean gain (higher is better), with the
difference from the best model and its 95% paired bootstrap CI over trials
(`N_BOOT` resamples). Then the top four per session.

Output under ``counterfactual/out/<Monkey>/``: ``best_model_results.csv``,
``best_model_summary.md``.

    uv run python -m counterfactual.best_model [--monkey ...]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from counterfactual import beliefs
from counterfactual.compare import trial_table
from counterfactual.config import ALL_MODELS, H_LABEL, MODEL_NAMES, MONKEYS, N_BOOT, S_LABEL, out_dir

LABELS = {H_LABEL: "H", S_LABEL: "S"}


def rank(df, rng):
    """Rows per model for one group of trials, best first."""
    gain = {m: (df[f"logp_{m}"] - df.logp_uniform).to_numpy() for m in ALL_MODELS}
    order = sorted(ALL_MODELS, key=lambda m: -gain[m].mean())
    best = order[0]
    idx = rng.integers(0, len(df), (N_BOOT, len(df)))
    rows = []
    for m in order:
        diff = gain[m] - gain[best]
        bs = diff[idx].mean(axis=1)
        rows.append(dict(model=m, name=MODEL_NAMES[m], n=len(df), mean_gain=gain[m].mean(),
                         d_vs_best=diff.mean(), ci_lo=np.quantile(bs, 0.025), ci_hi=np.quantile(bs, 0.975)))
    return rows


def run(monkey, seed=0):
    rng = np.random.default_rng(seed)
    df = trial_table(monkey, ALL_MODELS)
    q = beliefs.load(monkey)
    rows = []
    for label, lname in LABELS.items():
        g = df[df.label == label]
        rows += [dict(monkey=monkey, label=lname, unit=monkey, **r) for r in rank(g, rng)]
        for s, gs in g.groupby("session"):
            rows += [dict(monkey=monkey, label=lname, unit=s, **r) for r in rank(gs, rng)]
    res = pd.DataFrame(rows)
    d = out_dir(monkey)
    res.to_csv(d / "best_model_results.csv", index=False)
    (d / "best_model_summary.md").write_text(summary_md(monkey, res, q))
    return res


def summary_md(monkey, res, q):
    L = [f"# {monkey}: best behavioral model per neural strategy label, by counterfactual looks", "",
         "Dendrogram labels, top-two sessions. Each model's belief over the unchosen exits is conditioned on "
         "the chosen exit (10,000 draws from the correct path, published Weber fraction, draws kept when the "
         "model's rule picks the chosen exit; counterfactual/README.md). Score: mean gain over chance of the counterfactual "
         "looks per trial, log P(model) − log P(uniform), Plackett–Luce; 0 = chance, higher is better. "
         "Δ vs best with a 95% paired bootstrap CI over trials. Nothing is fitted.", ""]
    for lname, text in (("H", "H-labelled"), ("S", "S-labelled")):
        r = res[(res.label == lname) & (res.unit == monkey)]
        L += [f"## {text} trials (n = {r.n.iloc[0]})", "",
              "| Model | Gain over chance / trial | Δ vs best [95% CI] | Acceptance (median) |",
              "|---|---|---|---|"]
        for x in r.itertuples():
            delta = "best" if x.d_vs_best == 0 else f"{x.d_vs_best:+.3f} [{x.ci_lo:+.3f}, {x.ci_hi:+.3f}]"
            L.append(f"| {x.name} | {x.mean_gain:+.3f} | {delta} | {np.median(q[f'accept_{x.model}']):.2f} |")
        L += ["", "Top four by session (gain over chance / trial):", ""]
        for s in sorted(set(res.unit) - {monkey}):
            t = res[(res.label == lname) & (res.unit == s)].head(4)
            if len(t):
                L.append(f"- {s} (n={t.n.iloc[0]}): " + ", ".join(f"{x.name} {x.mean_gain:+.3f}" for x in t.itertuples()))
        L.append("")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(description="Rank behavioral models by the counterfactual looks, per label.")
    p.add_argument("--monkey", nargs="*", default=list(MONKEYS), choices=MONKEYS)
    args = p.parse_args()
    for m in args.monkey:
        run(m)
        print((out_dir(m) / "best_model_summary.md").read_text())


if __name__ == "__main__":
    main()
