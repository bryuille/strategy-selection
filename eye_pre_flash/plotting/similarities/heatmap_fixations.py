"""Cross-validated unit-H heatmap similarities, fixation samples only.

Same split-half CV as ``similarities_heatmap``, but restricted to
samples labeled as a fixation (event name ``fixation_idt`` in the cleaned
event table from ``load_clean_eye_data``) rather than the full pre-fixation
gaze trace.

Usage:
    uv run python -m eye_pre_flash.plotting.similarities.heatmap_fixations --monkey Faure
    uv run python -m eye_pre_flash.plotting.similarities.heatmap_fixations --monkey Nielsen
    uv run python -m eye_pre_flash.plotting.similarities.heatmap_fixations --monkey Faure --bin-w 0.142857
"""

from __future__ import annotations

import argparse

from eye_pre_flash.plotting.similarities.heatmap_log import (
    COARSE_BIN,
    DEFAULT_BIN,
    MIN_TRIALS,
    N_SPLITS,
)
from eye_pre_flash.plotting.similarities.heatmap_log import (
    plot_similarity as _plot_similarity,
)

VISUALIZER = "similarities/heatmap_fixations"


def plot_similarity(monkey="Faure", **kwargs):
    kwargs.setdefault("use_log", False)
    kwargs.setdefault("event_kind", "fixation")
    kwargs.setdefault("visualizer", VISUALIZER)
    return _plot_similarity(monkey=monkey, **kwargs)


def plot_all(monkey="Faure", **kwargs):
    plot_similarity(monkey=monkey, bin_w=DEFAULT_BIN, **kwargs)
    plot_similarity(monkey=monkey, bin_w=COARSE_BIN, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monkey", default="Faure", choices=("Faure", "Nielsen"))
    parser.add_argument(
        "--bin-w",
        type=float,
        default=None,
        help=f"Bin width in unit H (default: generate both {DEFAULT_BIN:g} and {COARSE_BIN:g})",
    )
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--min-trials", type=int, default=MIN_TRIALS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    kwargs = dict(
        n_splits=args.n_splits, min_trials=args.min_trials, seed=args.seed
    )
    if args.bin_w is not None:
        plot_similarity(monkey=args.monkey, bin_w=args.bin_w, **kwargs)
    else:
        plot_all(monkey=args.monkey, **kwargs)


if __name__ == "__main__":
    main()
