"""Which decision model best explains where the counterfactual saccade goes?

A catch-all model comparison, replacing the pairwise "model A vs model B"
discrimination tests. Every model in `eye_post_flash.models`, plus three
reference models, is fitted to the same observable and ranked by AIC/BIC.

The observable
--------------
On each correct trial with a post-feedback look at an unchosen exit, *which*
of the three unchosen exits was fixated first. Geometry is exactly constant
per path type, so trials collapse into 24 cells (one per path type, chosen =
correct there) each holding a 3-way count. That is a multinomial likelihood
with thousands of trials against at most five parameters.

Turning a model into a prediction
---------------------------------
A model maps measured intervals `tm` to a posterior over the four exits; the
predicted saccade target is the exit it ranks highest once the chosen exit is
removed. `tm` is never observed -- only the true stimulus intervals are -- so
it is integrated out against the model's own noise, `tm ~ N(t_true, wm *
t_true)`, on a Gauss-Hermite grid. That marginalization is what turns a
deterministic argmax into a distribution over the three exits, and it is why
the comparison has any power: models differ in *how much* of the mass they
put on the sibling versus the far branch.

Every model also carries `eps`, a uniform-over-unchosen mixing rate absorbing
looks the decision model does not explain at all (curiosity, residual
stimulus capture). Without it a single look at a zero-probability exit would
send the log-likelihood to -inf and the comparison would be decided by
outliers.

Reference models
----------------
    uniform     1/3 each. The floor: 0 parameters.
    proximity   p(e) ~ exp(-d(chosen, e) / scale), d in degrees of visual
                angle. THE control that matters -- the sibling exit shares
                the horizontal arm with the chosen exit and is usually the
                nearer candidate, so a purely spatial account has to be
                priced before any decision model is credited.
    sibling     the committed-hierarchical *readout*: the counterfactual
                is always the other vertical arm of the chosen branch.
                1 parameter (eps).
    saturated   per-cell empirical proportions. The ceiling: no decision
                model can beat it, and the gap to it is how much structure
                is left unexplained.

Two readouts of the hierarchical model
--------------------------------------
`models.hierarchical` scores exits by the two-stage posterior p(side|tm1) *
p(vert|tm2, side) and takes the argmax among unchosen. That does NOT reliably
pick the sibling: p(sibling) = p(chosen side) * p(*wrong* vertical arm), a
large number times a small one, which the far branch can beat. The separate
`sibling` model is the other readout -- once the side is committed, the only
alternative the second stage ever evaluated is the other arm on that branch,
so that is where the counterfactual look must go. These make genuinely
different predictions and are fitted separately; `eye_post_flash.targets`
implements the committed readout, so `sibling` is the model its
`hierarchical` ranking corresponds to.

`optimal_lapse` is expected to be near-unidentifiable here: a lapse makes the
model's ranking random, which is exactly what `eps` already does, so `lam` and
`eps` trade off. It is fitted anyway to show that.

Usage:
    uv run python -m eye_post_flash.model_comparison --align response
    uv run python -m eye_post_flash.model_comparison --monkey Faure --nodes 25

Output under `eye_post_flash/out/model_comparison/<align>/<monkey>/`:
`summary.md` (ranking, fitted parameters, per-cell fit of the winner),
`fits.csv`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from data.labeler import MAZE_GROUPS
from data.loader import load_eye_behavioral_data
from eye_post_flash import models as M
from eye_post_flash.counterfactual import ALIGNMENTS, build_trial_rows
from eye_post_flash.targets import (
    EXIT_NAMES,
    exit_index,
    exit_intervals_ms,
    exit_positions_deg,
)

OUT_ROOT = Path(__file__).resolve().parent / "out"
DEFAULT_NODES = 21
REVISION_NODES = 15
REVISION_INNER = 5
# Above this, `wm` is a flattening device rather than a timing parameter.
PLAUSIBLE_WM = 0.40

BOUNDS = {
    "wm": (0.02, 2.00),
    "eps": (1e-3, 0.95),
    "lam": (0.0, 1.0),
    "theta": (0.0, 1.0),
    "alpha": (0.0, 1.0),
    "beta": (-600.0, 600.0),
    "scale": (0.5, 200.0),
}


def geometry_by_path_type(monkey):
    """path_type -> (intervals (4,2) ms, exit positions (4,2) deg, correct).

    Geometry is constant within a path type (asserted), so a path type is a
    complete description of the trial's timing and layout.
    """
    b = load_eye_behavioral_data(monkey)
    out = {}
    for i in range(len(b["path_type"])):
        pt = b["path_type"][i]
        if not np.isfinite(pt) or pt == -99 or b["photodiode_qc_bad"][i]:
            continue
        h = tuple(float(b[f"h{j}"][i]) for j in range(1, 7))
        vel, lr, lr2 = b["vel"][i], b["LR"][i], b["LR2"][i]
        if not (np.all(np.isfinite(h)) and np.isfinite(vel) and vel > 0):
            continue
        if not (np.isfinite(lr) and np.isfinite(lr2)):
            continue
        key = int(pt)
        entry = (
            exit_intervals_ms(*h, vel),
            exit_positions_deg(*h),
            exit_index(lr, lr2),
        )
        if key in out:
            assert np.allclose(out[key][0], entry[0]), (
                f"{monkey} path_type {key} has inconsistent geometry"
            )
        else:
            out[key] = entry
    return out


def build_cells(monkey, *, align, window_ms, radius_deg):
    """[(path_type, chosen, correct, intervals, positions, counts(4,))]."""
    rows, _ = build_trial_rows(
        monkey, align=align, window_ms=window_ms, radius_deg=radius_deg
    )
    geom = geometry_by_path_type(monkey)
    tally = {}
    for r in rows:
        if not r["correct"] or not r["first_unchosen_look"]:
            continue
        chosen = EXIT_NAMES.index(r["chosen"])
        key = (r["path_type"], chosen)
        counts = tally.setdefault(key, np.zeros(4))
        counts[EXIT_NAMES.index(r["first_unchosen_look"])] += 1
    cells = []
    for (pt, chosen), counts in sorted(tally.items()):
        intervals, positions, correct = geom[pt]
        cells.append((pt, chosen, correct, intervals, positions, counts))
    return cells


def _grid(nodes):
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()
    xi, xj = np.meshgrid(x, x, indexing="ij")
    wi, wj = np.meshgrid(w, w, indexing="ij")
    return np.stack([xi.ravel(), xj.ravel()], axis=1), (wi * wj).ravel()


def model_prediction(name, params, cell, grid, weights, inner=REVISION_INNER):
    """(4,) predicted P(first-looked unchosen exit) under `name`."""
    _pt, chosen, correct, intervals, _pos, _counts = cell
    wm = params["wm"]
    t_true = intervals[correct]
    tm = t_true[None, :] * (1.0 + wm * grid)
    tm = np.maximum(tm, 1.0)

    fn, extra = M.MODELS[name]
    kwargs = {k: params[k] for k in extra}
    if name == "revision":
        kwargs["variant"] = params["variant"]
        kwargs["inner_nodes"] = inner
    posterior = fn(tm, intervals, wm, **kwargs)

    masked = posterior.copy()
    masked[:, chosen] = -np.inf
    top = np.argmax(masked, axis=1)
    p = np.zeros(4)
    np.add.at(p, top, weights)
    return p


def _mix(p, chosen, eps):
    """Blend a model prediction with uniform-over-unchosen at rate eps."""
    unchosen = [e for e in range(4) if e != chosen]
    out = np.zeros(4)
    out[unchosen] = (1.0 - eps) * p[unchosen] + eps / 3.0
    return out


def _loglik(cells, predict):
    """Multinomial log-likelihood of the observed counts under `predict`."""
    total = 0.0
    for cell in cells:
        counts = cell[5]
        p = predict(cell)
        mask = counts > 0
        if np.any(p[mask] <= 0):
            return -np.inf
        total += float(np.sum(counts[mask] * np.log(p[mask])))
    return total


def _param_names(name):
    if name in ("uniform",):
        return ()
    if name == "proximity":
        return ("scale",)
    if name == "sibling":
        return ("eps",)
    if name == "saturated":
        return ()
    return ("wm", "eps") + M.MODELS[name][1]


def _reference_predict(name, params):
    if name == "uniform":
        def predict(cell):
            return _mix(np.zeros(4), cell[1], 1.0)
    elif name == "proximity":
        def predict(cell):
            _pt, chosen, _c, _iv, pos, _n = cell
            d = np.hypot(*(pos - pos[chosen]).T)
            w = np.exp(-d / params["scale"])
            w[chosen] = 0.0
            return _mix(w / w.sum(), chosen, 1e-9)
    elif name == "sibling":
        def predict(cell):
            chosen = cell[1]
            p = np.zeros(4)
            p[chosen ^ 1] = 1.0
            return _mix(p, chosen, params["eps"])
    elif name == "saturated":
        def predict(cell):
            counts = cell[5]
            return counts / counts.sum()
    else:
        raise ValueError(name)
    return predict


def fit_model(name, cells, *, nodes=DEFAULT_NODES, seed=0, variant=None):
    """MLE fit; returns (params dict, loglik, n_params)."""
    if name in ("uniform", "saturated"):
        params = {}
        ll = _loglik(cells, _reference_predict(name, params))
        k = 0 if name == "uniform" else 2 * len(cells)
        return params, ll, k

    names = _param_names(name)
    if name in ("proximity", "sibling"):
        def objective(v):
            return -_loglik(
                cells, _reference_predict(name, dict(zip(names, v)))
            )
    else:
        n = REVISION_NODES if name == "revision" else nodes
        grid, weights = _grid(n)

        def objective(v):
            params = dict(zip(names, v))
            if variant is not None:
                params["variant"] = variant
            eps = params["eps"]
            try:
                return -_loglik(
                    cells,
                    lambda cell: _mix(
                        model_prediction(name, params, cell, grid, weights),
                        cell[1],
                        eps,
                    ),
                )
            except (FloatingPointError, ValueError):
                return np.inf

    bounds = [BOUNDS[p] for p in names]
    maxiter = 60
    popsize = 15
    result = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-4,
        polish=False,
    )
    params = dict(zip(names, result.x))
    if variant is not None:
        params["variant"] = variant
    return params, -float(result.fun), len(names)


def _predict_fn(name, params, nodes):
    if name in ("uniform", "proximity", "sibling", "saturated"):
        return _reference_predict(name, params)
    n = REVISION_NODES if name == "revision" else nodes
    grid, weights = _grid(n)
    eps = params["eps"]
    return lambda cell: _mix(
        model_prediction(name, params, cell, grid, weights), cell[1], eps
    )


def hit_rate(cells, predict):
    """Fraction of saccades whose target the model names correctly.

    For each cell the model has a single best guess -- the unchosen exit it
    gives the most probability to. The hit rate is the share of trials whose
    first look actually went there. Chance is 1/3 (three unchosen exits);
    the ceiling is `saturated`, which guesses each cell's most common target
    and so cannot be beaten by any model.
    """
    hits = total = 0.0
    for cell in cells:
        counts = cell[5]
        p = predict(cell)
        # Ties must be split, not broken by index. `uniform` puts equal mass
        # on all three unchosen exits; np.argmax would hand it the lowest
        # index every time, quietly turning the chance floor into whichever
        # exit that happens to be (the sibling) and scoring it far above
        # chance. Averaging the tied exits gives the expected accuracy.
        top = np.flatnonzero(p >= p.max() - 1e-12)
        hits += counts[top].sum() / len(top)
        total += counts.sum()
    return hits / total if total else np.nan


def _fmt_params(params):
    return " ".join(
        f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in params.items()
    )


FIT_ORDER = (
    "uniform",
    "proximity",
    "sibling",
    "only_first",
    "only_second",
    "total_time",
    "optimal",
    "optimal_lapse",
    "hierarchical",
    "postdictive",
)


def run(monkey, *, align, window_ms, radius_deg, nodes, seed):
    cells = build_cells(
        monkey, align=align, window_ms=window_ms, radius_deg=radius_deg
    )
    n_trials = int(sum(c[5].sum() for c in cells))
    print(f"{monkey}: {len(cells)} cells, {n_trials} informative trials")

    fits = []
    for name in FIT_ORDER:
        params, ll, k = fit_model(name, cells, nodes=nodes, seed=seed)
        fits.append((name, name, params, ll, k))
        print(f"  {name:<16} logL={ll:10.2f}  k={k}  {_fmt_params(params)}")

    best_rev = None
    for variant in M.CONFIDENCE_VARIANTS:
        params, ll, k = fit_model(
            "revision", cells, nodes=nodes, seed=seed, variant=variant
        )
        print(
            f"  revision[{variant:<7}] logL={ll:10.2f}  k={k}  "
            f"{_fmt_params(params)}"
        )
        if best_rev is None or ll > best_rev[3]:
            best_rev = (f"revision[{variant}]", "revision", params, ll, k)
    fits.append(best_rev)

    params, ll, k = fit_model("saturated", cells)
    fits.append(("saturated", "saturated", params, ll, k))
    print(f"  {'saturated':<16} logL={ll:10.2f}  k={k}")

    return _report(
        monkey, align, cells, n_trials, fits, nodes, window_ms, radius_deg
    )


def diagnose(params):
    """(params resting on a bound, implausible Weber fractions) as text.

    A fit that comes to rest on a bound is not an estimate, and a `wm` far
    above any real Weber fraction is a flattening device rather than a timing
    parameter -- either one means the model's rank is not evidence for its
    mechanism, so the table flags it.
    """
    hit_bound, bad_wm = [], []
    for pname, v in params.items():
        if pname not in BOUNDS or not isinstance(v, float):
            continue
        lo, hi = BOUNDS[pname]
        span = hi - lo
        if v - lo < 0.02 * span or hi - v < 0.02 * span:
            hit_bound.append(f"{pname}={v:.3g} at limit")
        elif pname == "wm" and v > PLAUSIBLE_WM:
            bad_wm.append(f"wm={v:.2f} implausible")
    return hit_bound, bad_wm


def _records(cells, n_trials, fits, nodes):
    """One dict per fitted model, with the interpretable metrics attached."""
    chance = 1.0 / 3.0
    rates = {}
    for label, base, params, ll, _k in fits:
        rates[label] = hit_rate(cells, _predict_fn(base, params, nodes))
    ceiling = rates["saturated"]
    floor_ll = next(ll for lbl, _b, _p, ll, _k in fits if lbl == "uniform")
    ceil_ll = next(ll for lbl, _b, _p, ll, _k in fits if lbl == "saturated")

    records = []
    for label, base, params, ll, k in fits:
        hit = rates[label]
        share = (
            (hit - chance) / (ceiling - chance)
            if ceiling > chance
            else np.nan
        )
        hit_bound, bad_wm = diagnose(params)
        records.append(
            {
                "model": label,
                "base": base,
                "hit_rate": hit,
                "share_of_ceiling": share,
                "loglik": ll,
                "loglik_share": (ll - floor_ll) / (ceil_ll - floor_ll),
                "k": k,
                "aic": 2 * k - 2 * ll,
                "bic": k * np.log(n_trials) - 2 * ll,
                "params": params,
                "flags": hit_bound + bad_wm,
            }
        )
    return records, chance, ceiling


def _report(
    monkey, align, cells, n_trials, fits, nodes, window_ms, radius_deg
):
    out_dir = OUT_ROOT / "model_comparison" / align / monkey
    out_dir.mkdir(parents=True, exist_ok=True)
    records, chance, ceiling = _records(cells, n_trials, fits, nodes)

    with open(out_dir / "fits.csv", "w", newline="") as fh:
        flat = []
        for r in records:
            row = {
                k: v for k, v in r.items() if k not in ("params", "flags")
            }
            row["flags"] = ";".join(r["flags"])
            for pname, v in r["params"].items():
                row[pname] = round(v, 4) if isinstance(v, float) else v
            flat.append(row)
        fields = sorted(
            {k for r in flat for k in r},
            key=lambda s: (
                [
                    "model", "base", "hit_rate", "share_of_ceiling",
                    "loglik", "loglik_share", "k", "aic", "bic", "flags",
                ].index(s)
                if s in (
                    "model", "base", "hit_rate", "share_of_ceiling",
                    "loglik", "loglik_share", "k", "aic", "bic", "flags",
                )
                else 99,
                s,
            ),
        )
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in flat:
            w.writerow(
                {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in row.items()
                }
            )

    decision = [r for r in records if r["model"] != "saturated"]
    winner = max(decision, key=lambda r: r["hit_rate"])

    out = [
        f"# Which model explains the counterfactual saccade? -- {monkey}",
        "",
        (
            f"align = {align}, window = {window_ms:.0f} ms, "
            f"radius = {radius_deg:.1f} deg, quadrature nodes = {nodes}"
        ),
        (
            "**The question.** After feedback, the animal sometimes looks at "
            "an exit it did not choose. There are three such exits. Each "
            "model says which one it should be. How often is it right?"
        ),
        (
            f"{n_trials} trials, grouped into {len(cells)} cells (one per "
            f"path type -- the maze geometry is identical within a path type, "
            f"so all its trials make the same prediction)."
        ),
        "",
        "## How to read the numbers",
        "",
        "| column | plain meaning |",
        "| ------ | ------------- |",
        (
            "| **correct** | Share of saccades that went to the exit the "
            "model named. This is the headline: bigger is better. |"
        ),
        (
            f"| **vs chance** | Same number minus {chance:.1%}, the rate you "
            f"get by guessing among three exits. Zero means the model knows "
            f"nothing. |"
        ),
        (
            f"| **of achievable** | Where the model sits between guessing "
            f"({chance:.1%} = 0%) and the best any model could do "
            f"({ceiling:.1%} = 100%). The ceiling is set by always naming "
            f"each cell's most common target. |"
        ),
        (
            "| **whole distribution** | The same 0-100% scale, but scoring "
            "the model on the *proportions* it predicts across all three "
            "exits, not just on whether its single best guess is right. "
            "This is the stricter number: naming the most common target is "
            "easy when a cell is lopsided, so a model can score well on "
            "`of achievable` while getting the split badly wrong. Where the "
            "two columns disagree, believe this one. |"
        ),
        (
            "| **params** | Free parameters the model was allowed to fit. "
            "Two models with equal `correct` are not equal if one used five "
            "parameters and the other one. |"
        ),
        (
            "| **flags** | A fitted parameter that hit its allowed limit, or "
            "a timing-noise value too large to be real. Either means the "
            "number was not estimated -- the fit was pushed into a "
            "degenerate shape to buy accuracy, so its rank is not evidence "
            "for that model. |"
        ),
        "",
        "## Ranking",
        "",
        "| model | correct | vs chance | of achievable "
        "| whole distribution | params | flags |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(records, key=lambda r: -r["hit_rate"]):
        flag = ", ".join(r["flags"]) if r["flags"] else ""
        note = " *(ceiling)*" if r["model"] == "saturated" else ""
        out.append(
            f"| {r['model']}{note} | {r['hit_rate']:.1%} "
            f"| {r['hit_rate'] - chance:+.1%} "
            f"| {r['share_of_ceiling']:.0%} "
            f"| {r['loglik_share']:.0%} | {r['k']} | {flag} |"
        )

    out.append(
        "\n`uniform` is the guess-at-random floor and `saturated` the "
        "per-cell ceiling; every other model sits between them. "
        "Log-likelihood, AIC and BIC are in `fits.csv` -- `whole "
        "distribution` is the log-likelihood on a 0-100% scale, and AIC's "
        "penalty for extra parameters is what the `params` column stands "
        "in for."
    )

    best_dist = max(decision, key=lambda r: r["loglik_share"])
    if best_dist["model"] != winner["model"]:
        out.append(
            f"\n**The two columns disagree here.** `{winner['model']}` names "
            f"the right exit most often ({winner['hit_rate']:.1%}), but "
            f"`{best_dist['model']}` predicts the three-way split best "
            f"({best_dist['loglik_share']:.0%} vs "
            f"{winner['loglik_share']:.0%} on the whole-distribution scale). "
            f"When a cell is lopsided, naming its most common target is easy "
            f"and says little; treat the accuracy ranking as the readable "
            f"summary and the whole-distribution one as the verdict."
        )

    prox = next(r for r in records if r["model"] == "proximity")
    if prox["hit_rate"] < chance:
        out.append(
            f"\n**The favoured exit is not the nearest one.** `proximity` "
            f"names whichever unchosen exit sits closest to the chosen one, "
            f"and it is right only {prox['hit_rate']:.1%} of the time -- "
            f"*below* the {chance:.1%} you get by guessing. So these saccades "
            f"are not simply going to the shortest hop, and the "
            f"sibling-is-usually-nearer worry does not explain them."
        )

    out.append(
        f"\n**Best model: {winner['model']}** -- names the right exit on "
        f"{winner['hit_rate']:.1%} of trials "
        f"({winner['hit_rate'] - chance:+.1%} over chance, "
        f"{winner['share_of_ceiling']:.0%} of what is achievable), using "
        f"{winner['k']} parameter(s)"
        + (
            f". Flagged: {', '.join(winner['flags'])} -- read the caveat in "
            f"the table above."
            if winner["flags"]
            else "."
        )
    )

    out.append("\n## Per-cell detail\n")
    out.append(
        "One row per path type. `looked at most` is where the saccades "
        "actually went most often; `model says` is the exit the best model "
        "named."
    )
    out.append("")
    out.append(
        "| path type | chosen exit | trials | looked at most (share) "
        "| model says | match |"
    )
    out.append("|---|---|---|---|---|---|")
    predict = _predict_fn(winner["base"], winner["params"], nodes)
    for cell in cells:
        pt, chosen, _c, _iv, _pos, counts = cell
        n = counts.sum()
        obs = int(np.argmax(counts))
        pred = int(np.argmax(predict(cell)))
        out.append(
            f"| {pt} | {EXIT_NAMES[chosen]} | {int(n)} "
            f"| {EXIT_NAMES[obs]} ({counts[obs] / n:.0%}) "
            f"| {EXIT_NAMES[pred]} | {'yes' if obs == pred else 'NO'} |"
        )

    (out_dir / "summary.md").write_text("\n".join(out) + "\n")
    print(f"Saved {out_dir / 'summary.md'}")
    return {
        "monkey": monkey,
        "records": records,
        "chance": chance,
        "ceiling": ceiling,
        "cells": cells,
        "nodes": nodes,
    }


def replay(monkey, *, align, window_ms, radius_deg, nodes):
    """Rebuild a monkey's report and figures from its saved `fits.csv`.

    The fits are the expensive part and they are deterministic given the
    seed, so re-deriving the metrics from stored parameters is exact -- not
    an approximation of a rerun.
    """
    out_dir = OUT_ROOT / "model_comparison" / align / monkey
    cells = build_cells(
        monkey, align=align, window_ms=window_ms, radius_deg=radius_deg
    )
    n_trials = int(sum(c[5].sum() for c in cells))
    fits = []
    with open(out_dir / "fits.csv") as fh:
        for row in csv.DictReader(fh):
            base = row["base"]
            params = {}
            for pname in ("wm", "eps", "lam", "theta", "alpha", "beta",
                          "scale"):
                if row.get(pname):
                    params[pname] = float(row[pname])
            if row.get("variant"):
                params["variant"] = row["variant"]
            fits.append(
                (row["model"], base, params, float(row["loglik"]),
                 int(row["k"]))
            )
    print(f"{monkey}: replayed {len(fits)} fits from fits.csv")
    return _report(
        monkey, align, cells, n_trials, fits, nodes, window_ms, radius_deg
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--monkey", choices=tuple(MAZE_GROUPS), default=None)
    parser.add_argument("--align", choices=ALIGNMENTS, default="response")
    parser.add_argument("--window-ms", type=float, default=1500.0)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--nodes", type=int, default=DEFAULT_NODES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="rebuild summaries and figures from the saved fits.csv "
        "without refitting (seconds instead of minutes)",
    )
    args = parser.parse_args(argv)

    results = []
    for monkey in (args.monkey,) if args.monkey else tuple(MAZE_GROUPS):
        if args.plots_only:
            results.append(
                replay(
                    monkey,
                    align=args.align,
                    window_ms=args.window_ms,
                    radius_deg=args.radius,
                    nodes=args.nodes,
                )
            )
        else:
            results.append(
                run(
                    monkey,
                    align=args.align,
                    window_ms=args.window_ms,
                    radius_deg=args.radius,
                    nodes=args.nodes,
                    seed=args.seed,
                )
            )
    from eye_post_flash.plot_model_comparison import plot_all

    plot_all(results, OUT_ROOT / "model_comparison" / args.align)


if __name__ == "__main__":
    main()
