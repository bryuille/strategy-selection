"""Dendrogram strategy labels (0 = hierarchical, 1 = sequential).

Read straight from ``<session>_strategy_choices.npz`` (the published Ward
clustering), never through `data.loader.load_strategy_choices`: that loader
builds on a miss, and a build means converting the session's neural
recording, which must not start silently inside an analysis job. A missing
label file raises and names the session.

``strategy_choices[i]`` is the label of trial id ``i + 1``; NaN outside the
published single-trial range.
"""

from __future__ import annotations

import numpy as np

from data.config import processed_npz

LABEL_STEM = "strategy_choices"
LABEL_NAMES = ("hierarchical", "sequential")


def label_path(session):
    return processed_npz(f"{session}_{LABEL_STEM}")


def label_lookup(sessions):
    """``(session, trial_id) -> 0.0/1.0`` for every labelled trial in `sessions`."""
    missing = [s for s in sessions if not label_path(s).exists()]
    if missing:
        raise FileNotFoundError(
            "missing dendrogram label cache(s): "
            + ", ".join(str(label_path(s)) for s in missing)
            + " -- build them with `python -m data.labels --type strategy` first"
        )
    lookup = {}
    for session in sessions:
        with np.load(label_path(session), allow_pickle=True) as f:
            choices = np.asarray(f["strategy_choices"], dtype=float)
        for idx in np.flatnonzero(np.isfinite(choices)):
            lookup[(session, int(idx) + 1)] = float(choices[idx])
    return lookup


def labels_for_rows(sessions, trial_ids, lookup):
    """Label vector aligned to the given rows; NaN where the trial is unlabelled."""
    return np.asarray(
        [lookup.get((str(s), int(t)), np.nan) for s, t in zip(sessions, trial_ids)],
        dtype=float,
    )
