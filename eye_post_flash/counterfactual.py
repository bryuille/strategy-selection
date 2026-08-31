"""Post-feedback counterfactual saccades by strategy regime.

Quantifies the probability that, after feedback, the animal makes a voluntary
saccade toward the *most likely unchosen alternative* exit, and compares that
probability between mazes solved in the hierarchical vs the sequential regime
(`data.labeler.MAZE_GROUPS`), and between per-trial neural strategy labels
where those exist.

Which exit counts as the most likely alternative depends on the decision
model (`eye_post_flash.targets.ALT_MODELS`): the hierarchical model implies
the sibling exit on the chosen branch, the total-time and optimal models can
imply an opposite-branch exit. `--alt-model` picks the model for the headline
rate tables; the *strategy discrimination* tables then restrict to trials
where two models disagree and ask whether the first-looked implied exit
tracks the inferred strategy -- the direct test that counterfactual
evaluation follows the strategy rather than a fixed evidence model.

A "look at exit e" is a clean fixation event (either pymovements detector)
whose centroid lies within `--radius` degrees of e and whose onset falls in
the post-feedback window. The chosen exit is excluded throughout: gaze is
expected there (that is where the answer saccade landed and where the outcome
is revealed), so only looks at *unchosen* exits are counterfactual. Correct
trials are the primary claim -- on error trials the correct exit is where the
ball actually emerged, so a look there is stimulus-driven, not counterfactual.

Two alignments for the window start:

  feedback  (default) `feedback_time`. Needs behavioral npz converted after
            `feedback_time` entered `data.convert.BEHAVIORAL_FIELDS`; the
            script says exactly what to re-run if they predate it.
  response  the onset of the answer fixation itself (first post-flash-three
            fixation at the chosen exit), detected from the eye data alone.
            Runs on any checkout, but starts slightly *before* feedback, so
            treat its numbers as provisional.

Usage:
    uv run python -m eye_post_flash.counterfactual --monkey Faure
    uv run python -m eye_post_flash.counterfactual --monkey Nielsen --align response
    uv run python -m eye_post_flash.counterfactual --align response --window-ms 2000

Output under `eye_post_flash/out/counterfactual/<align>/<monkey>/`:
`summary.md` (regime and neural-label tables, stats), `trials.csv` (one row
per analysed trial), `regime_rates.png`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, chi2 as chi2_dist, fisher_exact

from data.labeler import MAZE_GROUPS
from data.loader import load_clean_eye_data, load_eye_behavioral_data
from eye_post_flash.targets import (
    ALT_MODELS,
    DEFAULT_WM,
    EXIT_NAMES,
    chosen_exit_index,
    exit_index,
    exit_intervals_ms,
    exit_positions_deg,
    most_likely_alternative,
)
from utils import maze_for

OUT_ROOT = Path(__file__).resolve().parent / "out"
ALIGNMENTS = ("feedback", "response")
DEFAULT_WINDOW_MS = 1500.0
DEFAULT_RADIUS_DEG = 3.0

RECONVERT_HELP = (
    "feedback_time is missing or all-NaN in the eye-behavioral cache. The "
    "behavioral npz predate its addition to data.convert.BEHAVIORAL_FIELDS. "
    "Where data/mat/ lives (Engaging -- see cloud.md), run\n"
    "    uv run python -m data.convert behavioral --overwrite\n"
    "then delete data/processed/<Monkey>_eye_behavioral.npz and rerun (or "
    "submit slurm/run_post_flash.sbatch, which does all of this). "
    "`--align response` runs on the present caches in the meantime."
)


def _fixations_by_trial(clean):
    """(session, trial) -> (onset, x, y) arrays of fixation events, by onset."""
    names = np.asarray(clean["name"]).astype(str)
    is_fix = np.char.find(names, "fixation") >= 0
    onset = np.asarray(clean["onset"], dtype=float)[is_fix]
    x = np.asarray(clean["location_x"], dtype=float)[is_fix]
    y = np.asarray(clean["location_y"], dtype=float)[is_fix]
    sessions = np.asarray(clean["session"]).astype(str)[is_fix]
    trials = np.asarray(clean["trial_indices_all"], dtype=int)[is_fix]

    table = {}
    order = np.lexsort((onset, trials, sessions))
    for i in order:
        table.setdefault((sessions[i], int(trials[i])), []).append(
            (onset[i], x[i], y[i])
        )
    return {
        key: tuple(np.asarray(col) for col in zip(*rows))
        for key, rows in table.items()
    }


def _nearest_exit(x, y, exits_deg, radius_deg):
    """Index of the exit within `radius_deg` of (x, y), else -1."""
    d = np.hypot(exits_deg[:, 0] - x, exits_deg[:, 1] - y)
    nearest = int(np.argmin(d))
    return nearest if d[nearest] <= radius_deg else -1


def build_trial_rows(
    monkey,
    *,
    align="feedback",
    alt_model="optimal",
    wm=DEFAULT_WM,
    window_ms=DEFAULT_WINDOW_MS,
    radius_deg=DEFAULT_RADIUS_DEG,
    include_faded=False,
):
    """One record per analysable trial; see `trials.csv` columns in the doc."""
    behavioral = load_eye_behavioral_data(monkey)
    fixations = _fixations_by_trial(load_clean_eye_data(monkey))

    feedback = np.asarray(behavioral.get("feedback_time", np.nan), dtype=float)
    if align == "feedback" and not np.isfinite(feedback).any():
        raise SystemExit(RECONVERT_HELP)

    hier_mazes = set(MAZE_GROUPS[monkey][0])
    n = len(behavioral["path_type"])
    rows = []
    skipped = {"no_response_fixation": 0, "no_events": 0, "no_feedback": 0}
    for i in range(n):
        path_type = behavioral["path_type"][i]
        if not np.isfinite(path_type) or path_type == -99:
            continue
        if behavioral["photodiode_qc_bad"][i]:
            continue
        if not include_faded and behavioral["trial_fade"][i] != 0:
            continue
        h = [behavioral[f"h{j}"][i] for j in range(1, 7)]
        vel = behavioral["vel"][i]
        lr, lr2 = behavioral["LR"][i], behavioral["LR2"][i]
        if not (np.all(np.isfinite(h)) and np.isfinite(vel) and vel > 0):
            continue
        if not (np.isfinite(lr) and np.isfinite(lr2)):
            continue
        answers = [behavioral[f"trial_answer{j}"][i] for j in range(1, 5)]
        if not all(np.isfinite(answers)):
            continue
        chosen = chosen_exit_index(lr, lr2, *answers)
        if chosen is None:
            continue

        correct_idx = exit_index(lr, lr2)
        exits_deg = exit_positions_deg(*h)
        intervals = exit_intervals_ms(*h, vel)
        alts = {
            m: most_likely_alternative(
                intervals, correct_idx, chosen, m, wm
            )
            for m in ALT_MODELS
        }
        ml_alt = alts[alt_model]
        alt_dist = {
            m: float(np.hypot(*(exits_deg[alts[m]] - exits_deg[chosen])))
            for m in ALT_MODELS
        }

        session = str(behavioral["session"][i])
        trial = int(behavioral["trial_indices_all"][i])
        events = fixations.get((session, trial))
        if events is None:
            skipped["no_events"] += 1
            continue
        onset, ex, ey = events
        fix_exit = np.array(
            [_nearest_exit(x, y, exits_deg, radius_deg) for x, y in zip(ex, ey)]
        )

        geo_present = behavioral["geo_present"][i]
        if align == "feedback":
            if not np.isfinite(feedback[i]):
                skipped["no_feedback"] += 1
                continue
            start_ms = (feedback[i] - geo_present) * 1000.0
        else:
            flash3_ms = (behavioral["flash_three"][i] - geo_present) * 1000.0
            response = np.flatnonzero((onset > flash3_ms) & (fix_exit == chosen))
            if response.size == 0:
                skipped["no_response_fixation"] += 1
                continue
            start_ms = float(onset[response[0]])

        in_window = (onset > start_ms) & (onset <= start_ms + window_ms)
        unchosen = [e for e in range(4) if e != chosen]
        looked = {e: bool(np.any(in_window & (fix_exit == e))) for e in unchosen}
        window_unchosen = np.flatnonzero(in_window & np.isin(fix_exit, unchosen))
        look_seq = [EXIT_NAMES[int(fix_exit[j])] for j in window_unchosen]
        first_look = (
            int(fix_exit[window_unchosen[0]]) if window_unchosen.size else -1
        )

        rows.append(
            {
                "session": session,
                "trial": trial,
                "path_type": int(path_type),
                "maze": maze_for(path_type),
                "regime": "hierarchical"
                if maze_for(path_type) in hier_mazes
                else "sequential",
                "correct": chosen == correct_idx,
                "chosen": EXIT_NAMES[chosen],
                "correct_exit": EXIT_NAMES[correct_idx],
                "ml_alt": EXIT_NAMES[ml_alt],
                **{f"alt_{m}": EXIT_NAMES[alts[m]] for m in ALT_MODELS},
                **{
                    f"alt_dist_{m}": round(alt_dist[m], 3)
                    for m in ALT_MODELS
                },
                "unchosen_look_seq": "|".join(look_seq),
                "looked_ml_alt": looked[ml_alt],
                "n_other_alt_looked": sum(
                    looked[e] for e in unchosen if e != ml_alt
                ),
                "looked_any_unchosen": any(looked.values()),
                "first_unchosen_look": EXIT_NAMES[first_look]
                if first_look >= 0
                else "",
                "first_look_is_ml": (first_look == ml_alt)
                if first_look >= 0
                else None,
                "window_start_ms": start_ms,
                "n_fixations_in_window": int(np.count_nonzero(in_window)),
            }
        )
    return rows, skipped


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (center - half, center + half)


def _rate_line(rows):
    """(n, k_ml, p_ml, ci, p_other_per_exit, k_first_ml, n_first)."""
    n = len(rows)
    k_ml = sum(r["looked_ml_alt"] for r in rows)
    p_other = (
        np.mean([r["n_other_alt_looked"] / 2.0 for r in rows]) if n else np.nan
    )
    firsts = [r for r in rows if r["first_look_is_ml"] is not None]
    k_first = sum(r["first_look_is_ml"] for r in firsts)
    return n, k_ml, (k_ml / n if n else np.nan), wilson_ci(k_ml, n), p_other, k_first, len(firsts)


def _regime_table(rows, title, out):
    out.append(f"\n### {title}\n")
    out.append(
        "| regime | trials | P(look ML alt) | 95% CI | P(look per other alt) "
        "| first-look = ML (chance 1/3) |"
    )
    out.append("|---|---|---|---|---|---|")
    counts = {}
    for regime in ("hierarchical", "sequential"):
        sub = [r for r in rows if r["regime"] == regime]
        n, k, p, ci, p_other, k_first, n_first = _rate_line(sub)
        counts[regime] = (k, n)
        first_txt = (
            f"{k_first}/{n_first} = {k_first / n_first:.3f}"
            if n_first
            else "--"
        )
        binom_txt = ""
        if n_first:
            binom_txt = f" (p={binomtest(k_first, n_first, 1 / 3).pvalue:.2g})"
        out.append(
            f"| {regime} | {n} | {k}/{n} = {p:.3f} | "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | {p_other:.3f} | {first_txt}{binom_txt} |"
        )
    (kh, nh), (ks, ns) = counts["hierarchical"], counts["sequential"]
    if nh and ns:
        _, p_fisher = fisher_exact([[kh, nh - kh], [ks, ns - ks]])
        out.append(
            f"\nhierarchical vs sequential P(look ML alt), Fisher exact: "
            f"p = {p_fisher:.3g}"
        )
    return counts


def _first_between(row, name_a, name_b):
    """Which of two exit names was looked at first in the window, if either."""
    if not row["unchosen_look_seq"]:
        return None
    for name in row["unchosen_look_seq"].split("|"):
        if name in (name_a, name_b):
            return name
    return None


def mantel_haenszel(tables):
    """Common odds ratio and continuity-corrected p across 2x2 strata.

    `tables` are (a, b, c, d) counts per stratum. Strata that carry no
    contrast contribute nothing, so a stratifying variable that is a
    deterministic function of the maze yields no test at all -- which is
    exactly why the maze-regime comparison cannot be geometry-controlled
    this way and the per-trial neural labels can.
    """
    num = den = observed = expected = variance = 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        if n < 2:
            continue
        r1, r2, c1, c2 = a + b, c + d, a + c, b + d
        num += a * d / n
        den += b * c / n
        observed += a
        expected += r1 * c1 / n
        variance += r1 * r2 * c1 * c2 / (n * n * (n - 1))
    if variance <= 0:
        return np.nan, np.nan
    chi2 = (abs(observed - expected) - 0.5) ** 2 / variance
    return (num / den if den else np.inf), float(chi2_dist.sf(chi2, 1))


def _directed_counts(rows, key_a, key_b):
    """(n toward model_a's exit, n toward model_b's exit) by first look."""
    k_a = k_b = 0
    for r in rows:
        first = _first_between(r, r[key_a], r[key_b])
        if first == r[key_a]:
            k_a += 1
        elif first == r[key_b]:
            k_b += 1
    return k_a, k_b


def _stratified_discrimination(rows, model_a, model_b, title, out, by="maze"):
    """Regime effect on first-looked alternative, stratified by `by`.

    The unstratified contrast confounds strategy with geometry: the
    hierarchical model's alternative is the sibling exit, which shares the
    horizontal arm with the chosen exit and is therefore usually the nearer
    of the two candidates (see `alt_dist_*` in trials.csv). Because maze
    geometry differs across regimes, a raw regime difference can be pure
    saccade amplitude. Stratifying by `by` holds geometry fixed and asks
    whether the regime still moves the first look within a stratum.
    """
    keys = (by,) if isinstance(by, str) else tuple(by)
    name = " x ".join(keys)
    key_a, key_b = f"alt_{model_a}", f"alt_{model_b}"
    disc = [r for r in rows if r[key_a] != r[key_b]]
    out.append(f"\n### {title}\n")
    out.append(
        f"| {name} | hier informative | frac {model_a}-directed "
        f"| seq informative | frac {model_a}-directed |"
    )
    out.append("|---|---|---|---|---|")

    def stratum_of(row):
        return tuple(row[k] for k in keys)

    tables = []
    for stratum in sorted({stratum_of(r) for r in disc}):
        cells = {}
        for regime in ("hierarchical", "sequential"):
            sub = [
                r
                for r in disc
                if stratum_of(r) == stratum and r["regime"] == regime
            ]
            cells[regime] = _directed_counts(sub, key_a, key_b)
        (a, b), (c, d) = cells["hierarchical"], cells["sequential"]
        tables.append((a, b, c, d))
        if a + b == 0 and c + d == 0:
            continue
        f_h = f"{a / (a + b):.3f}" if a + b else "--"
        f_s = f"{c / (c + d):.3f}" if c + d else "--"
        label = " / ".join(str(v) for v in stratum)
        out.append(f"| {label} | {a + b} | {f_h} | {c + d} | {f_s} |")
    odds, p = mantel_haenszel(tables)
    if np.isfinite(p):
        if p < 0.05:
            gloss = (
                f"hierarchical-regime trials are "
                f"{'more' if odds > 1 else 'LESS'} {model_a}-directed than "
                f"sequential-regime trials within the same {name}"
            )
        else:
            gloss = (
                f"no {name}-controlled regime effect (OR > 1 would mean "
                f"hierarchical-regime trials are more {model_a}-directed)"
            )
        out.append(
            f"\nMantel-Haenszel ({name}-stratified) common OR = {odds:.3f}, "
            f"p = {p:.4g} -- {gloss}."
        )
    else:
        out.append(
            f"\n(no within-{name} contrast: the strata carry no regime "
            f"variation, so this test is undefined)"
        )
    return disc


def _proximity_control(disc, model_a, model_b, out):
    """Split the discriminating trials by which candidate exit is nearer.

    If the first look merely goes to whichever alternative is closest to the
    chosen exit, `frac model_a-directed` collapses to ~1 in the
    `model_a nearer` stratum and ~0 in the other, with no regime effect
    surviving inside either.
    """
    key_a, key_b = f"alt_{model_a}", f"alt_{model_b}"
    out.append(
        f"\n| {model_a} alt nearer? | regime | informative "
        f"| frac {model_a}-directed |"
    )
    out.append("|---|---|---|---|")
    for nearer in (True, False):
        for regime in ("hierarchical", "sequential"):
            sub = [
                r
                for r in disc
                if r["regime"] == regime
                and (r[f"alt_dist_{model_a}"] < r[f"alt_dist_{model_b}"])
                is nearer
            ]
            k_a, k_b = _directed_counts(sub, key_a, key_b)
            n = k_a + k_b
            frac = f"{k_a / n:.3f}" if n else "--"
            out.append(
                f"| {'yes' if nearer else 'no'} | {regime} | {n} | {frac} |"
            )


def _discrimination_table(rows, model_a, model_b, title, out):
    """Do looks track the strategy where two models imply different exits?

    Restricted to trials on which `model_a` and `model_b` disagree about the
    most likely alternative. 'informative' trials looked at at least one of
    the two implied exits; 'directed' is whichever was looked at first. If
    counterfactual evaluation follows the inferred strategy, the
    hierarchical-regime rows should be more `model_a`-directed than the
    sequential-regime rows (for model_a = hierarchical).
    """
    key_a, key_b = f"alt_{model_a}", f"alt_{model_b}"
    disc = [r for r in rows if r[key_a] != r[key_b]]
    out.append(f"\n### {title}\n")
    out.append(
        f"| regime | discriminating trials | informative | -> {model_a} alt "
        f"| -> {model_b} alt | frac {model_a}-directed |"
    )
    out.append("|---|---|---|---|---|---|")
    counts = {}
    for regime in ("hierarchical", "sequential"):
        sub = [r for r in disc if r["regime"] == regime]
        k_a, k_b = _directed_counts(sub, key_a, key_b)
        counts[regime] = (k_a, k_b)
        n_informative = k_a + k_b
        frac = f"{k_a / n_informative:.3f}" if n_informative else "--"
        out.append(
            f"| {regime} | {len(sub)} | {n_informative} | {k_a} | {k_b} "
            f"| {frac} |"
        )
    (kah, kbh), (kas, kbs) = counts["hierarchical"], counts["sequential"]
    if (kah + kbh) and (kas + kbs):
        _, p = fisher_exact([[kah, kbh], [kas, kbs]])
        out.append(
            f"\nregime x first-looked alternative, Fisher exact: p = {p:.3g}"
        )
    return disc


def _discrimination_by_maze(disc, model_a, model_b, out):
    key_a, key_b = f"alt_{model_a}", f"alt_{model_b}"
    out.append(
        f"\n| maze (regime) | discriminating | informative "
        f"| frac {model_a}-directed |"
    )
    out.append("|---|---|---|---|")
    for maze in sorted({r["maze"] for r in disc}):
        sub = [r for r in disc if r["maze"] == maze]
        k_a, k_b = _directed_counts(sub, key_a, key_b)
        n_informative = k_a + k_b
        frac = f"{k_a / n_informative:.3f}" if n_informative else "--"
        regimes = "/".join(sorted({r["regime"] for r in sub}))
        out.append(
            f"| {maze} ({regimes}) | {len(sub)} | {n_informative} | {frac} |"
        )


def _session_table(rows, out):
    out.append("\n### Per session (correct trials)\n")
    out.append("| session | regime(s) | trials | P(look ML alt) |")
    out.append("|---|---|---|---|")
    for session in sorted({r["session"] for r in rows}):
        sub = [r for r in rows if r["session"] == session]
        regimes = "/".join(sorted({r["regime"] for r in sub}))
        n, k, p, *_ = _rate_line(sub)
        out.append(f"| {session} | {regimes} | {n} | {k}/{n} = {p:.3f} |")


def _neural_label_rows(rows, monkey):
    """Attach per-trial neural strategy labels; only labelled rows returned."""
    from eye_pre_flash.classifier.labels import (
        available_label_sessions,
        strategy_label_lookup,
    )

    lookup = strategy_label_lookup(available_label_sessions())
    labelled = []
    for r in rows:
        label = lookup.get((r["session"], r["trial"]))
        if label is not None:
            labelled.append(
                {**r, "regime": "hierarchical" if label == 0.0 else "sequential"}
            )
    return labelled


def _figure(correct_rows, monkey, align, out_dir):
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    xs, heights, errs, labels = [], [], [], []
    for i, regime in enumerate(("hierarchical", "sequential")):
        sub = [r for r in correct_rows if r["regime"] == regime]
        n, _, p, ci, p_other, *_ = _rate_line(sub)
        xs += [i * 2.0, i * 2.0 + 0.8]
        heights += [p, p_other]
        errs += [
            (np.nan, np.nan) if n == 0 else (p - ci[0], ci[1] - p),
            (0.0, 0.0),
        ]
        labels += [f"{regime}\nML alt", f"{regime}\nother alt"]
    err_arr = np.array(errs, dtype=float).T
    colors = ["#3b6fb6", "#a8c4e6", "#b6533b", "#e6b3a8"]
    ax.bar(xs, heights, width=0.7, yerr=err_arr, capsize=3, color=colors)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("P(post-feedback look)")
    ax.set_title(f"{monkey} -- correct trials, align={align}")
    fig.tight_layout()
    path = out_dir / "regime_rates.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def run(
    monkey, *, align, alt_model, wm, window_ms, radius_deg, include_faded
):
    rows, skipped = build_trial_rows(
        monkey,
        align=align,
        alt_model=alt_model,
        wm=wm,
        window_ms=window_ms,
        radius_deg=radius_deg,
        include_faded=include_faded,
    )
    out_dir = OUT_ROOT / "counterfactual" / align / monkey
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "trials.csv"
    if rows:
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {csv_path}")

    correct_rows = [r for r in rows if r["correct"]]
    error_rows = [r for r in rows if not r["correct"]]

    out = [
        f"# Post-feedback counterfactual looks -- {monkey}",
        "",
        (
            f"align = {align}, ML-alt model = {alt_model}, wm = {wm:.3f}, "
            f"window = {window_ms:.0f} ms, "
            f"radius = {radius_deg:.1f} deg, trial_fade "
            f"{'included' if include_faded else '== 0'}"
        ),
        (
            f"trials analysed: {len(rows)} "
            f"({len(correct_rows)} correct, {len(error_rows)} error); "
            f"skipped: {skipped}"
        ),
    ]
    _regime_table(
        correct_rows, "By maze regime -- correct trials (primary)", out
    )
    disc = _discrimination_table(
        correct_rows,
        "hierarchical",
        "total_time",
        "Strategy discrimination -- hierarchical vs total-time alternative "
        "(correct trials)",
        out,
    )
    _discrimination_by_maze(disc, "hierarchical", "total_time", out)
    out.append(
        "\nThe sibling exit the hierarchical model implies shares the "
        "horizontal arm with the chosen exit, so it is usually the nearer of "
        "the two candidates (maze 6 is the exception -- there the sibling is "
        "farther, and its frac drops accordingly). Because regime is a "
        "function of the maze, the regime contrast above is confounded with "
        "geometry; the proximity split and the neural-label strata below are "
        "the controlled versions."
    )
    _proximity_control(disc, "hierarchical", "total_time", out)
    _discrimination_table(
        correct_rows,
        "hierarchical",
        "optimal",
        "Strategy discrimination -- hierarchical vs optimal alternative "
        "(correct trials)",
        out,
    )
    _regime_table(error_rows, "By maze regime -- error trials", out)
    _session_table(correct_rows, out)

    labelled = _neural_label_rows(correct_rows, monkey)
    if labelled:
        _regime_table(
            labelled,
            f"By per-trial neural strategy label -- correct trials "
            f"({len(labelled)} labelled)",
            out,
        )
        _discrimination_table(
            labelled,
            "hierarchical",
            "total_time",
            "Strategy discrimination by neural label -- hierarchical vs "
            "total-time (correct trials)",
            out,
        )
        labelled_disc = _stratified_discrimination(
            labelled,
            "hierarchical",
            "total_time",
            "Strategy discrimination by neural label, maze-stratified -- "
            "hierarchical vs total-time (correct trials)",
            out,
        )
        _proximity_control(
            labelled_disc, "hierarchical", "total_time", out
        )
        _stratified_discrimination(
            labelled,
            "hierarchical",
            "total_time",
            "Strategy discrimination by neural label, session-stratified -- "
            "hierarchical vs total-time (correct trials)",
            out,
            by="session",
        )
        _stratified_discrimination(
            labelled,
            "hierarchical",
            "total_time",
            "Strategy discrimination by neural label, maze x session "
            "stratified -- hierarchical vs total-time (correct trials). "
            "Both nuisances held fixed at once; this is the decisive test.",
            out,
            by=("maze", "session"),
        )
        _stratified_discrimination(
            labelled,
            "hierarchical",
            "optimal",
            "Strategy discrimination by neural label, maze-stratified -- "
            "hierarchical vs optimal (correct trials)",
            out,
        )
    else:
        out.append("\n(no per-trial neural strategy labels available)")

    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(out) + "\n")
    print(f"Saved {md_path}")
    print("\n".join(out))

    _figure(correct_rows, monkey, align, out_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--monkey", choices=tuple(MAZE_GROUPS), default=None)
    parser.add_argument("--align", choices=ALIGNMENTS, default="feedback")
    parser.add_argument("--alt-model", choices=ALT_MODELS, default="optimal")
    parser.add_argument(
        "--wm",
        type=float,
        default=DEFAULT_WM,
        help="Weber fraction for scalar timing (not fitted here)",
    )
    parser.add_argument("--window-ms", type=float, default=DEFAULT_WINDOW_MS)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS_DEG)
    parser.add_argument("--include-faded", action="store_true")
    args = parser.parse_args(argv)

    monkeys = (args.monkey,) if args.monkey else tuple(MAZE_GROUPS)
    for monkey in monkeys:
        run(
            monkey,
            align=args.align,
            alt_model=args.alt_model,
            wm=args.wm,
            window_ms=args.window_ms,
            radius_deg=args.radius,
            include_faded=args.include_faded,
        )


if __name__ == "__main__":
    main()
