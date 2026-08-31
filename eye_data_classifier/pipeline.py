"""One command from `data/mat/` to every eye_data_classifier table.

Each stage runs as its own subprocess. That is not cosmetic: the intermediate
caches run to tens of gigabytes, and a fresh process per stage hands the memory
back instead of carrying it into the next one.

Stages are ordered by dependency and every one is a cache warm -- rerunning is
cheap because each underlying loader skips work whose npz already exists. So the
same command serves a cold cloud checkout and a re-run after a code change; use
`--from` to resume partway rather than trimming the list by hand.

    convert     eye, behavioral and single-trial mat -> npz (all sessions)
    labels      strategy labels: convert, label and free one session at a time
    features    per-monkey codebooks (k=6 and k=12, unit-H and degree space)
                and the cached feature blocks
    decode-pub  decoding tables, the 4 publication clustering sessions
    decode-all  the same over every snr_auc >= 0.95 session (label-vetted)
    decode-allplus  the same without the label vetting, as a robustness check

The decoding runs in **two scopes**. `publication` is the four sessions the
published clustering was defined on -- the strictest audience-facing set. `all`
is the 23 sessions clearing `snr_auc >= 0.95`, which is where the power is.
Reporting both is the point: an effect present in `all` and absent in
`publication` is an effect that needs the wider pool to see, and saying so is
more honest than picking whichever scope reads better.

Usage:
    uv run python -m eye_data_classifier.pipeline
    uv run python -m eye_data_classifier.pipeline --from decode-pub
    uv run python -m eye_data_classifier.pipeline --only decode-all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def stages():
    """``name -> argv`` in dependency order."""
    return {
        # Eye and behavioral data are pooled across every session to fit the
        # per-monkey codebooks, so those convert wholesale.
        "convert": ["data.convert", "eye", "behavioral", "single_trial"],
        # Neural data is never held for the whole pool. A label is ~5 KB but its
        # neural npz is 2-14 GB, and there are 23 eligible sessions against a
        # 67 GB budget, so `--sweep` converts one session, writes its label and
        # frees both intermediates before starting the next. Peak extra usage is
        # one session, and the sweep resumes where it stopped.
        "labels": ["data.labeler", "--type", "strategy", "--sweep"],
        "features": ["eye_data_classifier.features"],
        "decode-pub": ["eye_data_classifier.decoding", "--scope", "publication"],
        "decode-all": ["eye_data_classifier.decoding", "--scope", "all"],
        # Same sessions as decode-all but without the anchor-maze label
        # vetting -- the robustness check on what vetting removes.
        "decode-allplus": ["eye_data_classifier.decoding", "--scope", "allplus"],
    }


def run(name, argv, *, dry_run):
    cmd = [sys.executable, "-u", "-m", *argv]
    print(f"\n{'=' * 72}\n=== {name}\n=== {' '.join(cmd[3:])}\n{'=' * 72}", flush=True)
    if dry_run:
        return 0.0
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise SystemExit(
            f"\nstage '{name}' failed with exit code {result.returncode} "
            f"after {elapsed / 60:.1f} min.\nResume with: "
            f"python -m eye_data_classifier.pipeline --from {name}"
        )
    print(f"--- {name} done in {elapsed / 60:.1f} min", flush=True)
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--from", dest="start", help="resume at this stage")
    parser.add_argument("--only", nargs="+", help="run just these stages")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = stages()
    names = list(plan)
    for name in (args.only or []) + ([args.start] if args.start else []):
        if name not in plan:
            raise SystemExit(f"unknown stage {name!r}; choose from {names}")
    if args.only:
        names = [n for n in names if n in set(args.only)]
    elif args.start:
        names = names[names.index(args.start) :]

    print(f"pipeline: {' -> '.join(names)}", flush=True)
    t0 = time.time()
    for name in names:
        run(name, plan[name], dry_run=args.dry_run)
    print(
        f"\n{'=' * 72}\npipeline finished in {(time.time() - t0) / 60:.1f} min\n"
        f"Results under eye_data_classifier/out/decoding/\n{'=' * 72}",
        flush=True,
    )


if __name__ == "__main__":
    main()
