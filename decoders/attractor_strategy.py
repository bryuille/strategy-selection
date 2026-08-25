"""Supervised strategy decoder from k-means pre-fixation occupancy.

Trains an L2 logistic regression (SGD) to predict neural strategy labels
(`build_strategy_choices`) from codebook time-occupancy features.
Labels are built per session wherever paired behavioral + neural npz exist.
Cross-validation uses session-blocked folds when multiple sessions are
available; otherwise stratified trial splits within the session.

Usage:
    uv run python -m decoders.attractor_strategy
    uv run python -m decoders.attractor_strategy --monkey Faure --k 12
"""

from __future__ import annotations

import argparse

import numpy as np

from data.labeler import attractor_labeled_occupancy
from data.loader import (
    load_all_session_strategy_choices,
    load_attractor_eye_data,
    load_eye_behavioral_data,
)
from decoders.common import N_CV_SPLITS, cross_validate_decoder_grouped


def prepare_attractor_decoder_data(monkey="Faure", k=12):
    attractor = load_attractor_eye_data(monkey, k=k)
    behavioral = load_eye_behavioral_data(monkey)
    strategy = load_all_session_strategy_choices(monkey)
    X, y, groups, trial_ids, mazes = attractor_labeled_occupancy(
        attractor, behavioral, strategy
    )
    return X, y, groups, trial_ids, mazes, attractor, strategy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monkey", default="Faure", choices=("Faure", "Nielsen"))
    parser.add_argument("--k", type=int, default=12, help="Attractor codebook size")
    args = parser.parse_args()

    print("Loading attractor occupancy + neural strategy labels...")
    X, y, groups, trial_ids, mazes, attractor, strategy = prepare_attractor_decoder_data(
        monkey=args.monkey, k=args.k
    )
    n_sessions = len(np.unique(groups))
    n_features = X.shape[1] if X.size else int(np.asarray(attractor["codebook_k"]))
    print(
        f"Trials: {len(y)}  sessions: {n_sessions}  "
        f"features: {n_features} occupancy dims (K={args.k})"
    )
    print(
        f"Label balance: 0={np.mean(y == 0):.1%}  1={np.mean(y == 1):.1%}  "
        f"(from {len(strategy['session'])} labeled neural trials)"
    )
    if len(y) < 20:
        raise ValueError("Too few labeled trials for cross-validation")

    print("Running cross-validation...")
    best_lambda, cv_acc = cross_validate_decoder_grouped(X, y, groups)
    split_desc = (
        f"{min(N_CV_SPLITS, n_sessions)}-fold session-blocked"
        if n_sessions > 1
        else f"{N_CV_SPLITS} stratified trial splits"
    )
    print(f"Best lambda: {best_lambda:.2e}")
    print(f"Cross-validated accuracy ({split_desc}): {cv_acc:.3f}")


if __name__ == "__main__":
    main()
