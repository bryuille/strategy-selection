"""Write the per-trial exit sequences and a summary for each monkey.

Output: ``counterfactual/out/<Monkey>/exit_sequences.csv`` (one row per
trial) and ``summary.txt``. No statistics: the likelihood of a sequence under
a strategy is not defined yet.

    uv run python -m counterfactual.build [--monkey ...] [--dry-run] [--refresh] [--out-root DIR]
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from counterfactual.config import EXIT_NAMES, MONKEYS, OUT_ROOT, SESSIONS, STATE_NAMES
from counterfactual.extract import load


def exit_name(code):
    return EXIT_NAMES[int(code) - 1] if 1 <= int(code) <= 4 else STATE_NAMES.get(int(code), "?")


def summarize(monkey, data):
    seqs = [list(q) for q in data["exit_seq"]]
    chosen = data["chosen_exit"].astype(int)
    correct = data["correct_exit"].astype(int)
    lengths = np.array([len(q) for q in seqs])
    with_fix = [k for k, q in enumerate(seqs) if len(q) > 1]
    landing = np.asarray(data["landing_on_chosen"], dtype=bool)
    lines = [f"{monkey}: post-choice exit sequences", ""]
    sessions = data["sessions"].astype(str).tolist()
    dropped = np.asarray(data["dropped"], dtype=int)
    reasons = data["drop_reasons"].astype(str).tolist()
    for j, s in enumerate(sessions):
        n = int((data["session"].astype(str) == s).sum())
        drops = ", ".join(f"{r}={dropped[j, i]}" for i, r in enumerate(reasons))
        lines.append(f"  {s}: {n} trials kept; dropped {drops}; pupil={data['pupil_source'][j]}")
    lines += [
        "",
        f"trials: {len(seqs)}; fixations in window: {len(data['fix_row'])}",
        f"exit-list length (incl. chosen): "
        + ", ".join(f"{k}: {int((lengths == k).sum())}" for k in sorted(set(lengths.tolist()))),
        f"trials with >= 1 post-choice exit fixation: {len(with_fix)} ({len(with_fix) / max(len(seqs), 1):.0%})",
        f"choice-saccade landing fixation on the decoded chosen exit: {int(landing.sum())}/{len(seqs)} "
        f"({landing.mean() if landing.size else 0:.0%})  <- sanity check of the LR/LR2 decoding; "
        "that fixation is item 0, not repeated",
        f"trials whose list visits an exit other than the chosen one: "
        f"{sum(1 for q in seqs if any(e != q[0] for e in q[1:]))}",
        f"trials whose list visits the correct exit after choosing wrongly: "
        f"{sum(1 for q, c in zip(seqs, correct) if c > 0 and q[0] != c and c in q[1:])}",
        f"chosen == correct: {int((chosen == correct).sum())}/{len(seqs)}",
        "",
        "post-choice fixation states: "
        + ", ".join(f"{STATE_NAMES[int(k)]}={v}" for k, v in sorted(Counter(data["fix_state"].astype(int).tolist()).items())),
        "strategy label present: "
        + f"{int(np.isfinite(data['strategy']).sum())}/{len(seqs)} (0 = H, 1 = S; from <session>_strategy_choices)",
    ]
    return "\n".join(lines) + "\n"


def write_csv(path, data):
    seqs = [list(q) for q in data["exit_seq"]]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "trial", "maze", "chosen", "correct", "strategy",
                    "window_lo_ms", "window_hi_ms", "n_fix", "landing_on_chosen",
                    "exit_seq", "exit_seq_names", "state_seq"])
        for k in range(len(seqs)):
            strat = data["strategy"][k]
            w.writerow([
                data["session"][k], int(data["trial_indices_all"][k]), int(data["maze_id"][k]),
                int(data["chosen_exit"][k]), int(data["correct_exit"][k]),
                "" if not np.isfinite(strat) else int(strat),
                f"{float(data['window_lo_ms'][k]):.0f}", f"{float(data['window_hi_ms'][k]):.0f}",
                int(data["n_fix"][k]), int(data["landing_on_chosen"][k]),
                " ".join(str(e) for e in seqs[k]),
                " ".join(exit_name(e) for e in seqs[k]),
                " ".join(str(s) for s in data["state_seq"][k]),
            ])


def main():
    parser = argparse.ArgumentParser(description="Per-trial post-choice exit sequences.")
    parser.add_argument("--monkey", nargs="*", default=list(MONKEYS), choices=list(MONKEYS))
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the summary, write nothing")
    args = parser.parse_args()
    for monkey in args.monkey:
        data = load(monkey, refresh=args.refresh)
        text = summarize(monkey, data)
        print(text)
        if args.dry_run:
            continue
        out = args.out_root / monkey
        out.mkdir(parents=True, exist_ok=True)
        write_csv(out / "exit_sequences.csv", data)
        (out / "summary.txt").write_text(text)
        print(f"wrote {out / 'exit_sequences.csv'} and summary.txt ({len(data['session'])} trials, "
              f"sessions {', '.join(SESSIONS[monkey])})")


if __name__ == "__main__":
    main()
