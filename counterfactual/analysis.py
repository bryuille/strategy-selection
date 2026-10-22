"""Do post-choice counterfactual saccades follow the neurally labelled strategy?

Per trial (publication sessions, QC, dendrogram label present) the observed
quantity is the ordered list of *unchosen* exits the monkey looked at after
the choice (`counterfactual.extract`); the predictions are the Hierarchical
and Sequential choice probabilities (`counterfactual.models`), each
renormalised over the three unchosen exits (`q_H`, `q_S`).

Primary statistic. For the first counterfactual exit ``e1``:
``LL_m = log q_m(e1)`` and ``delta = LL_matched - LL_mismatched`` with matched
= the trial's neural label (H -> q_H, S -> q_S). Per monkey: mean delta, a
bootstrap CI over trials, and a permutation p from label shuffles within
session x maze (preserves label counts per stratum, mirrors Figure Five's
label-shuffle control). Also split by label so a result cannot be carried
by one label alone.

Secondary. (i) Top-counterfactual accuracy: does ``e1`` equal each model's
highest-ranked unchosen exit, by label, on the *disagreement* trials where
the two models rank a different exit first. (ii) Sequence: Plackett-Luce log
likelihood of the ordered distinct unchosen exits (sampling without
replacement from ``q_m``), same delta and shuffle test.

Controls. Uniform ``q = 1/3``; a spatial-proximity model ``q ~ exp(-d / tau)``
(H predicts the same-side exit, which is also the nearest one); the same-side
rate of ``e1`` by label; correct vs error trials; sensitivity to the
smoothing ``eps`` and to the Weber scalar (x0.7, x1.3, fewer draws).
"""

from __future__ import annotations

import numpy as np

from counterfactual.config import EXIT_CODES, MODEL_EPS, WEBER_SCALAR
from counterfactual.exits import exit_landmarks_deg
from counterfactual.models import counterfactual_dist, simulate_hs

PROXIMITY_TAU_DEG = 5.0
N_BOOT = 2000
H, S = 0, 1  # dendrogram label codes


# ---- per-trial table ----------------------------------------------------------------


def side(code):
    return "L" if int(code) in (1, 2) else "R"


def counterfactual_items(exit_seq):
    """Post-choice exit items with repeats of the chosen exit removed, and the
    ordered distinct unchosen exits."""
    chosen = int(exit_seq[0])
    items = [int(e) for e in exit_seq[1:] if int(e) != chosen]
    distinct = list(dict.fromkeys(items))
    return items, distinct


def plackett_luce_ll(q, order):
    """log P(order) under sampling without replacement from q (codes 1-4)."""
    q = np.asarray(q, dtype=float).copy()
    ll = 0.0
    for e in order:
        denom = q.sum()
        if denom <= 0 or q[e - 1] <= 0:
            return -np.inf
        ll += np.log(q[e - 1] / denom)
        q[e - 1] = 0.0
    return ll


def proximity_dist(h, chosen, *, tau=PROXIMITY_TAU_DEG):
    ex = exit_landmarks_deg(h)
    d = np.linalg.norm(ex - ex[chosen - 1], axis=1)
    q = np.exp(-d / tau)
    q[chosen - 1] = 0.0
    return q / q.sum()


def join(exits, models):
    """Row pairs ``(exits_row, models_row)`` on (session, trial)."""
    key = {(str(s), int(t)): i for i, (s, t) in enumerate(zip(models["session"], models["trial_indices_all"]))}
    pairs = []
    for i, (s, t) in enumerate(zip(exits["session"], exits["trial_indices_all"])):
        j = key.get((str(s), int(t)))
        if j is not None:
            pairs.append((i, j))
    return pairs


def trial_table(exits, models, *, eps=MODEL_EPS, prob_h=None, prob_s=None):
    """One record per labelled trial with a decodable chosen exit. `prob_h` /
    `prob_s` override the cached probabilities (sensitivity runs)."""
    prob_h = models["prob_h"].astype(float) if prob_h is None else np.asarray(prob_h, dtype=float)
    prob_s = models["prob_s"].astype(float) if prob_s is None else np.asarray(prob_s, dtype=float)
    rows = []
    for i, j in join(exits, models):
        label = float(exits["strategy"][i])
        if not np.isfinite(label):
            continue
        chosen = int(exits["chosen_exit"][i])
        assert chosen == int(models["chosen_exit"][j])
        items, distinct = counterfactual_items(exits["exit_seq"][i])
        q_h = counterfactual_dist(prob_h[j], [chosen], eps=eps)[0]
        q_s = counterfactual_dist(prob_s[j], [chosen], eps=eps)[0]
        q_p = proximity_dist(exits["h"][i].astype(float), chosen)
        top_h = int(np.argmax(q_h)) + 1
        top_s = int(np.argmax(q_s)) + 1
        rec = dict(
            session=str(exits["session"][i]), trial=int(exits["trial_indices_all"][i]),
            maze=int(exits["maze_id"][i]), label=int(label), chosen=chosen,
            correct=int(exits["correct_exit"][i]), is_correct=int(exits["correct_exit"][i]) == chosen,
            n_cf=len(items), e1=items[0] if items else 0, items=items, distinct=distinct,
            q_h=q_h, q_s=q_s, q_prox=q_p, top_h=top_h, top_s=top_s, disagree=top_h != top_s,
        )
        if items:
            e1 = items[0]
            rec.update(
                ll_h=np.log(q_h[e1 - 1]), ll_s=np.log(q_s[e1 - 1]),
                ll_prox=np.log(q_p[e1 - 1]), ll_unif=np.log(1 / 3),
                same_side=side(e1) == side(chosen),
                hit_h=e1 == top_h, hit_s=e1 == top_s,
                seq_ll_h=plackett_luce_ll(q_h, distinct), seq_ll_s=plackett_luce_ll(q_s, distinct),
            )
        rows.append(rec)
    return rows


# ---- statistics ------------------------------------------------------------------------


def _delta(rows, key_h, key_s):
    """Matched minus mismatched per trial: +(H - S) on H trials, -(H - S) on S."""
    diff = np.array([r[key_h] - r[key_s] for r in rows])
    sign = np.array([1.0 if r["label"] == H else -1.0 for r in rows])
    return diff, sign


def permutation_test(diff, sign, strata, *, n_perm, rng):
    """Mean of sign*diff against label shuffles within strata (sign permuted
    within each stratum). Returns (observed, null means, one-sided p for
    observed > null, two-sided p)."""
    observed = float(np.mean(sign * diff))
    strata = np.asarray(strata)
    groups = [np.flatnonzero(strata == g) for g in np.unique(strata)]
    null = np.empty(n_perm)
    perm_sign = sign.copy()
    for k in range(n_perm):
        for g in groups:
            perm_sign[g] = sign[g][rng.permutation(g.size)]
        null[k] = np.mean(perm_sign * diff)
    p_one = float((np.sum(null >= observed) + 1) / (n_perm + 1))
    p_two = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1))
    return observed, null, p_one, p_two


def bootstrap_ci(values, *, rng, n_boot=N_BOOT, level=0.95):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan
    idx = rng.integers(0, values.size, (n_boot, values.size))
    means = values[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


MIN_PER_LABEL = 10  # a session x maze stratum is "mixed" with this many of each label


def mixed_strata(rows, *, min_per_label=MIN_PER_LABEL):
    """Session x maze strata holding at least `min_per_label` trials of each
    label. Labels are nearly a function of maze (maze 1 all H, maze 6 all S),
    so only these strata can tell a label effect from a maze effect; the
    within-stratum permutation is informative only there."""
    counts = {}
    for r in rows:
        k = (r["session"], r["maze"])
        counts.setdefault(k, [0, 0])[r["label"]] += 1
    return {k for k, (nh, ns) in counts.items() if nh >= min_per_label and ns >= min_per_label}


def summarize(rows, *, n_perm=10_000, seed=0):
    """Statistics for one monkey; `rows` from `trial_table`. Returns a dict of
    plain values plus the permutation null for the primary statistic."""
    rng = np.random.default_rng(seed)
    out = {}
    out["mixed_strata"] = sorted(mixed_strata(rows))
    out["n_labelled"] = len(rows)
    for lab, name in ((H, "H"), (S, "S")):
        sub = [r for r in rows if r["label"] == lab]
        out[f"n_{name}"] = len(sub)
        out[f"cf_rate_{name}"] = float(np.mean([r["n_cf"] > 0 for r in sub])) if sub else np.nan
    cf = [r for r in rows if r["n_cf"] > 0]
    out["n_cf"] = len(cf)
    if not cf:
        return out
    strata = [f"{r['session']}:{r['maze']}" for r in cf]

    # Primary: first counterfactual exit.
    diff, sign = _delta(cf, "ll_h", "ll_s")
    obs, null, p1, p2 = permutation_test(diff, sign, strata, n_perm=n_perm, rng=rng)
    lo, hi = bootstrap_ci(sign * diff, rng=rng)
    out.update(delta_mean=obs, delta_ci=(lo, hi), delta_p_one=p1, delta_p_two=p2, delta_null=null,
               delta_null_sd=float(null.std()), delta_null_mean=float(null.mean()))
    # Mixed strata only: the label effect separable from the maze effect.
    mixed = out["mixed_strata"]
    mk = [k for k, r in enumerate(cf) if (r["session"], r["maze"]) in set(mixed)]
    out["n_mixed"] = len(mk)
    if mk:
        mobs, mnull, mp1, mp2 = permutation_test(diff[mk], sign[mk], [strata[k] for k in mk], n_perm=n_perm, rng=rng)
        out.update(mixed_delta_mean=mobs, mixed_delta_ci=bootstrap_ci((sign * diff)[mk], rng=rng),
                   mixed_delta_p_one=mp1, mixed_delta_p_two=mp2, mixed_null_mean=float(mnull.mean()),
                   mixed_null_sd=float(mnull.std()))
        for lab, name in ((H, "H"), (S, "S")):
            sub = [k for k in mk if cf[k]["label"] == lab]
            v = (sign * diff)[sub]
            out[f"mixed_delta_{name}_mean"] = float(v.mean()) if v.size else np.nan
            out[f"mixed_delta_{name}_n"] = int(v.size)
            dsub = [k for k in sub if cf[k]["disagree"]]
            out[f"mixed_top_acc_h_{name}"] = float(np.mean([cf[k]["hit_h"] for k in dsub])) if dsub else np.nan
            out[f"mixed_top_acc_s_{name}"] = float(np.mean([cf[k]["hit_s"] for k in dsub])) if dsub else np.nan
            out[f"mixed_n_disagree_{name}"] = len(dsub)
    for lab, name in ((H, "H"), (S, "S")):
        sub = [k for k, r in enumerate(cf) if r["label"] == lab]
        v = (sign * diff)[sub]
        out[f"delta_{name}_mean"] = float(v.mean()) if v.size else np.nan
        out[f"delta_{name}_ci"] = bootstrap_ci(v, rng=rng)
        out[f"delta_{name}_n"] = int(v.size)
        for m in ("h", "s", "prox", "unif"):
            out[f"ll_{m}_{name}"] = float(np.mean([cf[k][f"ll_{m}"] for k in sub])) if sub else np.nan
        out[f"same_side_{name}"] = float(np.mean([cf[k]["same_side"] for k in sub])) if sub else np.nan

    # Top-counterfactual accuracy on disagreement trials.
    dis = [r for r in cf if r["disagree"]]
    out["n_disagree"] = len(dis)
    for lab, name in ((H, "H"), (S, "S")):
        sub = [r for r in dis if r["label"] == lab]
        out[f"top_acc_h_{name}"] = float(np.mean([r["hit_h"] for r in sub])) if sub else np.nan
        out[f"top_acc_s_{name}"] = float(np.mean([r["hit_s"] for r in sub])) if sub else np.nan
        out[f"n_disagree_{name}"] = len(sub)

    # Sequence statistic (Plackett-Luce over the ordered distinct unchosen exits).
    seq_ok = [k for k, r in enumerate(cf) if np.isfinite(r["seq_ll_h"]) and np.isfinite(r["seq_ll_s"])]
    sdiff = np.array([cf[k]["seq_ll_h"] - cf[k]["seq_ll_s"] for k in seq_ok])
    ssign = sign[seq_ok]
    sobs, snull, sp1, sp2 = permutation_test(sdiff, ssign, [strata[k] for k in seq_ok], n_perm=n_perm, rng=rng)
    out.update(seq_delta_mean=sobs, seq_delta_ci=bootstrap_ci(ssign * sdiff, rng=rng), seq_delta_p_one=sp1,
               seq_delta_p_two=sp2, n_multi=int(np.sum([len(cf[k]["distinct"]) > 1 for k in seq_ok])))

    # Correct vs error trials.
    for flag, name in ((True, "correct"), (False, "error")):
        sub = [k for k, r in enumerate(cf) if r["is_correct"] == flag]
        v = (sign * diff)[sub]
        out[f"delta_{name}_mean"] = float(v.mean()) if v.size else np.nan
        out[f"delta_{name}_ci"] = bootstrap_ci(v, rng=rng)
        out[f"delta_{name}_n"] = int(v.size)
    return out


def sensitivity(exits, models, monkey, *, eps_values=(1e-4, 1e-2), scalar_scales=(0.7, 1.3),
                n_samples=2000, seed=0):
    """Mean primary delta under other smoothing values and Weber scalars."""
    rng = np.random.default_rng(seed)
    out = {}
    for eps in eps_values:
        rows = [r for r in trial_table(exits, models, eps=eps) if r["n_cf"] > 0]
        diff, sign = _delta(rows, "ll_h", "ll_s")
        out[f"delta_eps_{eps:g}"] = float(np.mean(sign * diff)) if rows else np.nan
    for scale in scalar_scales:
        scalar = WEBER_SCALAR[monkey] * scale
        ph, ps = simulate_hs(models["h_sec"].astype(float), models["lr"], models["lr2"], scalar,
                             n_samples=n_samples, seed=int(rng.integers(1 << 30)))
        rows = [r for r in trial_table(exits, models, prob_h=ph, prob_s=ps) if r["n_cf"] > 0]
        diff, sign = _delta(rows, "ll_h", "ll_s")
        out[f"delta_scalar_x{scale:g}"] = float(np.mean(sign * diff)) if rows else np.nan
    return out


def results_rows(monkey, stats, sens):
    """Long-format rows for results.csv."""
    rows = []

    def add(stat, value, lo=np.nan, hi=np.nan, p=np.nan, n=np.nan):
        rows.append(dict(monkey=monkey, statistic=stat, value=value, ci_lo=lo, ci_hi=hi, p=p, n=n))

    add("n_labelled", stats["n_labelled"])
    for name in ("H", "S"):
        add(f"n_{name}", stats[f"n_{name}"])
        add(f"cf_rate_{name}", stats[f"cf_rate_{name}"], n=stats[f"n_{name}"])
    if "delta_mean" not in stats:
        return rows
    add("delta_first_cf", stats["delta_mean"], *stats["delta_ci"], stats["delta_p_one"], stats["n_cf"])
    add("delta_first_cf_p_two_sided", stats["delta_p_two"], n=stats["n_cf"])
    add("delta_null_sd", stats["delta_null_sd"])
    add("delta_null_mean", stats["delta_null_mean"])
    add("n_mixed_strata", len(stats["mixed_strata"]))
    if stats.get("n_mixed"):
        add("mixed_delta_first_cf", stats["mixed_delta_mean"], *stats["mixed_delta_ci"], stats["mixed_delta_p_one"], stats["n_mixed"])
        add("mixed_delta_null_mean", stats["mixed_null_mean"])
        for name in ("H", "S"):
            add(f"mixed_delta_first_cf_{name}", stats[f"mixed_delta_{name}_mean"], n=stats[f"mixed_delta_{name}_n"])
            add(f"mixed_top_acc_H_model_{name}", stats[f"mixed_top_acc_h_{name}"], n=stats[f"mixed_n_disagree_{name}"])
            add(f"mixed_top_acc_S_model_{name}", stats[f"mixed_top_acc_s_{name}"], n=stats[f"mixed_n_disagree_{name}"])
    for name in ("H", "S"):
        add(f"delta_first_cf_{name}", stats[f"delta_{name}_mean"], *stats[f"delta_{name}_ci"], n=stats[f"delta_{name}_n"])
        for m in ("h", "s", "prox", "unif"):
            add(f"mean_ll_{m}_{name}", stats[f"ll_{m}_{name}"], n=stats[f"delta_{name}_n"])
        add(f"same_side_rate_{name}", stats[f"same_side_{name}"], n=stats[f"delta_{name}_n"])
        add(f"top_acc_H_model_{name}", stats[f"top_acc_h_{name}"], n=stats[f"n_disagree_{name}"])
        add(f"top_acc_S_model_{name}", stats[f"top_acc_s_{name}"], n=stats[f"n_disagree_{name}"])
    add("n_disagree", stats["n_disagree"])
    add("delta_sequence", stats["seq_delta_mean"], *stats["seq_delta_ci"], stats["seq_delta_p_one"], stats["n_multi"])
    for name in ("correct", "error"):
        add(f"delta_first_cf_{name}", stats[f"delta_{name}_mean"], *stats[f"delta_{name}_ci"], n=stats[f"delta_{name}_n"])
    for k, v in sens.items():
        add(k, v)
    return rows


def summary_text(monkey, stats, sens):
    L = [f"{monkey}: counterfactual saccades vs the H / S exit rankings", ""]
    L.append(f"labelled trials {stats['n_labelled']} (H {stats['n_H']}, S {stats['n_S']}); "
             f"with >= 1 counterfactual exit look: H {stats['cf_rate_H']:.0%}, S {stats['cf_rate_S']:.0%}")
    if "delta_mean" not in stats:
        return "\n".join(L) + "\n"
    lo, hi = stats["delta_ci"]
    L += ["",
          "PRIMARY  delta = log q_matched(e1) - log q_mismatched(e1), first counterfactual exit",
          f"  mean {stats['delta_mean']:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  n {stats['n_cf']}  "
          f"permutation (labels shuffled within session x maze): one-sided p {stats['delta_p_one']:.4f}, "
          f"two-sided p {stats['delta_p_two']:.4f}, null mean {stats['delta_null_mean']:+.3f}, "
          f"null sd {stats['delta_null_sd']:.3f}",
          "  (labels are nearly a function of maze, so the within-maze null sits near the observed value:",
          "   the CI tests delta against 0 across all trials; the permutation tests the label effect beyond maze)"]
    for name, pos, neg in (("H", "H", "S"), ("S", "S", "H")):
        lo, hi = stats[f"delta_{name}_ci"]
        L.append(f"  {name}-labelled trials (n {stats[f'delta_{name}_n']}): log q_{pos} - log q_{neg} = "
                 f"{stats[f'delta_{name}_mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]; "
                 f"mean log q: H {stats[f'll_h_{name}']:+.3f}, S {stats[f'll_s_{name}']:+.3f}, "
                 f"proximity {stats[f'll_prox_{name}']:+.3f}, uniform {stats[f'll_unif_{name}']:+.3f}; "
                 f"e1 on the chosen side {stats[f'same_side_{name}']:.0%}")
    mixed = stats["mixed_strata"]
    L += ["", f"MIXED STRATA (session x maze with >= {MIN_PER_LABEL} trials of each label): "
          + (", ".join(f"{s_}:{m}" for s_, m in mixed) if mixed else "none")]
    if stats.get("n_mixed"):
        lo, hi = stats["mixed_delta_ci"]
        L.append(f"  delta {stats['mixed_delta_mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]  n {stats['n_mixed']}  "
                 f"permutation one-sided p {stats['mixed_delta_p_one']:.4f} (null mean {stats['mixed_null_mean']:+.3f}, "
                 f"sd {stats['mixed_null_sd']:.3f}); by label: H {stats['mixed_delta_H_mean']:+.3f} (n {stats['mixed_delta_H_n']}), "
                 f"S {stats['mixed_delta_S_mean']:+.3f} (n {stats['mixed_delta_S_n']})")
        L.append(f"  top counterfactual on disagreement trials: H-labelled (n {stats['mixed_n_disagree_H']}) "
                 f"e1 == H's top {stats['mixed_top_acc_h_H']:.0%}, == S's top {stats['mixed_top_acc_s_H']:.0%}; "
                 f"S-labelled (n {stats['mixed_n_disagree_S']}) e1 == H's top {stats['mixed_top_acc_h_S']:.0%}, "
                 f"== S's top {stats['mixed_top_acc_s_S']:.0%}")
    L += ["",
          f"TOP COUNTERFACTUAL on the {stats['n_disagree']} trials where H and S rank a different unchosen exit first",
          f"  H-labelled (n {stats['n_disagree_H']}): e1 == H's top {stats['top_acc_h_H']:.0%}, == S's top {stats['top_acc_s_H']:.0%}",
          f"  S-labelled (n {stats['n_disagree_S']}): e1 == H's top {stats['top_acc_h_S']:.0%}, == S's top {stats['top_acc_s_S']:.0%}",
          "",
          f"SEQUENCE (Plackett-Luce over the ordered distinct unchosen exits; {stats['n_multi']} trials with >= 2): "
          f"delta {stats['seq_delta_mean']:+.3f} [{stats['seq_delta_ci'][0]:+.3f}, {stats['seq_delta_ci'][1]:+.3f}], "
          f"one-sided p {stats['seq_delta_p_one']:.4f}",
          "",
          f"CORRECT vs ERROR: delta {stats['delta_correct_mean']:+.3f} (n {stats['delta_correct_n']}) vs "
          f"{stats['delta_error_mean']:+.3f} (n {stats['delta_error_n']})",
          "SENSITIVITY: " + ", ".join(f"{k} {v:+.3f}" for k, v in sens.items()),
          "",
          "A positive delta means the matched strategy's ranking explains the first counterfactual saccade better.",
          "Caveat: H predicts the same-side exit, which is also the nearest; compare with the proximity row."]
    return "\n".join(L) + "\n"
