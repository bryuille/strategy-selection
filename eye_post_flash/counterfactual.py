"""Post-feedback counterfactual saccades by strategy regime.

Quantifies the probability that, after feedback, the animal makes a voluntary
saccade toward the *most likely unchosen alternative* exit (see
`eye_post_flash.targets`), and compares that probability between mazes solved
in the hierarchical vs the sequential regime (`data.labeler.MAZE_GROUPS`), and
between per-trial neural strategy labels where those exist.

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
from scipy.stats import binomtest, fisher_exact

from data.labeler import MAZE_GROUPS
from data.loader import load_clean_eye_data, load_eye_behavioral_data
from eye_post_flash.targets import (
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
        ml_alt = most_likely_alternative(intervals, correct_idx, chosen)

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
        first_look = (
            int(fix_exit[window_unchosen[0]]) if window_unchosen.size else -1
        )

        rows.append(
            {
                "session": session,
                "trial": trial,
                "maze": maze_for(path_type),
                "regime": "hierarchical"
                if maze_for(path_type) in hier_mazes
                else "sequential",
                "correct": chosen == correct_idx,
                "chosen": EXIT_NAMES[chosen],
                "correct_exit": EXIT_NAMES[correct_idx],
                "ml_alt": EXIT_NAMES[ml_alt],
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


def run(monkey, *, align, window_ms, radius_deg, include_faded):
    rows, skipped = build_trial_rows(
        monkey,
        align=align,
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
            f"align = {align}, window = {window_ms:.0f} ms, "
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
    parser.add_argument("--window-ms", type=float, default=DEFAULT_WINDOW_MS)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS_DEG)
    parser.add_argument("--include-faded", action="store_true")
    args = parser.parse_args(argv)

    monkeys = (args.monkey,) if args.monkey else tuple(MAZE_GROUPS)
    for monkey in monkeys:
        run(
            monkey,
            align=args.align,
            window_ms=args.window_ms,
            radius_deg=args.radius,
            include_faded=args.include_faded,
        )


if __name__ == "__main__":
    main()
