"""Printed markdown summary of one figure's three variants.

Kept to `eye_pre_flash.corr.report`'s discipline: printed to the run log, not
written beside the png -- the `out/` tree carries figures and CSVs only.

Where corr's report is a six-maze table, this one is a three-variant table for
a single maze, and it carries two things corr's does not:

* **Both half sizes side by side.** This package's maze-restricted shared half
  next to what corr's unrestricted 12-cell rule would have chosen on the same
  session. The two figures show the same cells with different numbers, and
  that difference is a precision gain, not a result. Printing both is what
  stops it being read as one.
* **The permutation null's own limits.** ``n_strata`` (sessions that actually
  had both labels and so could be shuffled) and ``log10`` of the number of
  distinct label assignments. With a single maze the latter can bind, and then
  ``p = 1/(n_perm + 1)`` overstates the resolution -- unlike in corr, where
  pooling six mazes makes it astronomically large and safe to ignore.
"""

from __future__ import annotations

import numpy as np

from eye_pre_flash.mazestratpair import pair as pairmod
from eye_pre_flash.plotting.plot_io import window_label


def _f(value, digits=3):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(
    results, nulls, dims, *, monkey, maze, source, feature, scope, k, space,
    n_splits, min_trials, min_stable, half_rule, n_perm, variants,
):
    names = pairmod.cell_labels(maze)
    lines = [
        f"## {monkey} maze {maze} — {feature} / {source} / {scope} / K={k}",
        "",
        f"space={space}, {window_label()}, {n_splits} splits, "
        f"min-trials={min_trials}, min-stable={min_stable}, half-rule={half_rule}, "
        f"{n_perm} permutation resamples",
        "",
    ]

    anchor = next((results[v] for v in variants if results.get(v) is not None), None)
    if anchor is None or not anchor.n_sessions:
        lines += [
            f"No usable session: a session must have at least {min_trials} trials of "
            f"**both** strategies on maze {maze}. With two cells a single cell yields "
            "no correlation at all, so such a session is dropped rather than marked "
            "thin.",
            "",
        ]
        print("\n".join(lines))
        return "\n".join(lines)

    thin = pairmod.thin_mask(anchor)
    lines += [
        "### Census",
        "",
        "| | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
        "| trials (pooled) | " + " | ".join(str(int(c)) for c in anchor.census) + " |",
        "| sessions | " + " | ".join(str(int(u)) for u in anchor.used) + " |",
        "| thin (*) | " + " | ".join("yes" if t else "no" for t in thin) + " |",
        "",
        f"{anchor.n_sessions} of {anchor.n_sessions_in_scope} sessions in scope carried "
        f"both strategies on maze {maze}.",
        "",
        "### Half sizes: this package vs corr's 12-cell rule",
        "",
        "Restricting the shared half size to these two cells is the whole reason "
        "these numbers differ from corr's same-maze cells. Every `r` here is "
        "computed from more trials per half; that is precision, not an effect.",
        "",
        "| Session | n(H) | n(S) | half(H) | half(S) | corr half(H) | corr half(S) | equal n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, session in enumerate(anchor.session_names):
        counts = anchor.cell_counts[i]
        halves = anchor.half_sizes[i]
        corr_h = anchor.half_corr_rule[i]
        lines.append(
            f"| {session} | {counts.get(0, 0)} | {counts.get(1, 0)} | "
            f"{halves.get(0, 0)} | {halves.get(1, 0)} | "
            f"{corr_h.get(0, 0) or '—'} | {corr_h.get(1, 0) or '—'} | "
            f"{'yes' if halves.get(0) == halves.get(1) else 'NO'} |"
        )

    lines += [
        "",
        "### The comparison: H vs S within maze " + str(maze),
        "",
        "| Variant | d | Sessions | r(H,H) | r(S,S) | r(H,S) | Δ | null mean | "
        "null SD | z | p (2-sided) | p (1-sided) | null 2.5-97.5% |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for variant in variants:
        result, null = results.get(variant), nulls.get(variant) or {}
        if result is None or not result.n_sessions:
            lines.append(f"| {variant} | — | 0 |" + " — |" * 10)
            continue
        lines.append(
            f"| {variant} | {dims.get(variant, 0)} | {null.get('n_sessions_both', 0)} | "
            f"{_f(null.get('r_HH'))} | {_f(null.get('r_SS'))} | {_f(null.get('r_HS'))} | "
            f"{_f(null.get('delta_obs'))} | {_f(null.get('null_mean'))} | "
            f"{_f(null.get('null_sd'))} | {_f(null.get('z_perm'), 2)} | "
            f"{_f(null.get('p_two_sided'), 4)} | {_f(null.get('p_one_sided_greater'), 4)} | "
            f"[{_f(null.get('null_p025'))}, {_f(null.get('null_p975'))}] |"
        )

    lines += [
        "",
        "Δ = 0.5·(r_HH + r_SS) − r_HS. Its reference is **null mean, not zero**: the "
        "estimator carries a small floor of its own at d as low as 6, which is why "
        "the null is measured rather than assumed. Δ is not comparable across "
        "variants (`full` sits in a compressed high-r band, `mean_removed` is "
        "mean-centred); `z` is.",
        "",
        "### Per-session Δ",
        "",
        "| Session | " + " | ".join(variants) + " |",
        "|---|" + "---|" * len(variants),
    ]
    for i, session in enumerate(anchor.session_names):
        row = []
        for variant in variants:
            result = results.get(variant)
            if result is None or session not in result.session_names:
                row.append("—")
                continue
            row.append(_f(result.session_deltas[result.session_names.index(session)]))
        lines.append(f"| {session} | " + " | ".join(row) + " |")

    positives = []
    for variant in variants:
        result = results.get(variant)
        if result is None or not result.session_deltas:
            continue
        deltas = np.asarray(result.session_deltas, dtype=float)
        finite = deltas[np.isfinite(deltas)]
        if finite.size:
            positives.append(
                f"{variant} {int((finite > 0).sum())}/{finite.size}"
            )

    lines += [
        "",
        ("Sessions with Δ > 0: " + ", ".join(positives) + ".") if positives else "",
        "",
        "The per-session Δ counts are descriptive only. A formal session-level test "
        "is not reported: at these session counts (2 for `publication`, 4 for "
        "`top_four`) it would not be evidence, and the permutation null already "
        "conditions on exactly these recording days.",
        "",
        "### The null's own limits",
        "",
        "| Variant | strata (shufflable sessions) | log10 distinct assignments | "
        "unexchangeable sessions | finite draws |",
        "|---|---|---|---|---|",
    ]
    for variant in variants:
        null = nulls.get(variant) or {}
        if not null:
            continue
        lines.append(
            f"| {variant} | {null.get('n_strata', 0)} | "
            f"{_f(null.get('n_perm_distinct_log10'), 1)} | "
            f"{null.get('n_sessions_unexchangeable', 0)} | {null.get('n_perm', 0)} |"
        )
    lines += [
        "",
        "`log10 distinct assignments` is the p-value's true floor. If it drops near "
        "or below log10(n_perm) the reported p is limited by the strata, not by "
        "`--n-perm`. `unexchangeable sessions` are sessions whose maze trials were "
        "all one strategy — invariant under every permutation, so they would dilute "
        "the null toward the observed; `pair.build_pools` drops them.",
        "",
    ]

    if any(
        (results.get(v) is not None and results[v].n_degenerate) for v in variants
    ):
        lines += [
            "Some split-halves were degenerate: both cells had data but the Pearson "
            "came back NaN anyway, from a zero-variance half-mean. Those splits are "
            "excluded from the average rather than counted as zero, so an em-dash "
            "from zero variance is distinguishable from an em-dash for no coverage.",
            "",
        ]
    if "mean_removed" in variants and results.get("mean_removed") is not None:
        lines += [
            "`mean_removed`: the grand-mean profile is fitted once per (monkey, "
            "feature, K) over the widest scope's labelled trials across **all** "
            "mazes, exactly as corr fits it — not on this maze. Subtracting a common "
            "profile makes the mean off-diagonal r negative by arithmetic, so "
            "absolute r is not interpretable in that panel; only Δ against that "
            "panel's own ceiling is.",
            "",
        ]

    text = "\n".join(lines)
    print(text)
    return text
