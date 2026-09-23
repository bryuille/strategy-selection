"""SVM strategy labels (0 = hierarchical, 1 = sequential) and the session scope.

This package uses one label source at one scope: the maze-1-vs-6 SVM labels
(`data.labeler.build_svm_choices`, cached as ``<session>_strategy_svm.npz``)
over the ten highest-`snr_auc` sessions per monkey drawn from
`data.labeler.CLUSTERING_SESSIONS`, with no anchor vetting. That is exactly
`eye_pre_flash.label_sources`' ``svm/top_ten``, reproduced here so nothing is
imported from another `eye_pre_flash` folder:

* pool = `CLUSTERING_SESSIONS` for the monkey (23 sessions total: 13 Faure,
  10 Nielsen);
* rank by `snr_auc` descending, session name ascending on ties -- the same
  ``sorted(sorted(ss), key=..., reverse=True)`` expression;
* keep the top ten. Nielsen has exactly ten, so nothing is ranked out; Faure
  drops june_17_g0, April_8_g0 and june_11_g0.

Membership does not depend on the labels themselves, so the ranking happens
first and only the chosen sessions' label files are read.

Label files are read directly, never through `data.loader.load_svm_choices`:
that loader builds on miss, and a build means converting the session's neural
recording, which must not start silently inside an analysis job. A missing
label file raises and names the session.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

from data.config import monkey_for_session, processed_npz
from data.labeler import CLUSTERING_SESSIONS, SNR_AUC

LABEL_NAMES = ("hierarchical", "sequential")
LABEL_STEM = "strategy_svm"
TOP_N = 10

_SNR_CSV = (
    Path(__file__).resolve().parents[2]
    / "zrefs/Dendogram_all/SNR_All_Sessions/all_session_snr_results.csv"
)


@lru_cache(maxsize=1)
def snr_auc_table():
    """``session -> snr_auc``; the reference CSV, else the transcribed dict."""
    try:
        with _SNR_CSV.open(newline="") as fh:
            return {r["session"]: float(r["snr_auc"]) for r in csv.DictReader(fh)}
    except OSError:
        return dict(SNR_AUC)


def _snr_key(session):
    return snr_auc_table().get(session, -1.0)


def top_ten_sessions(monkey, top_n=TOP_N):
    """The monkey's `CLUSTERING_SESSIONS`, ranked by `snr_auc`, truncated."""
    pool = [s for s in CLUSTERING_SESSIONS if monkey_for_session(s) == monkey]
    return tuple(sorted(sorted(pool), key=_snr_key, reverse=True)[:top_n])


def label_path(session):
    return processed_npz(f"{session}_{LABEL_STEM}")


def svm_lookup(sessions):
    """``(session, trial_id) -> 0.0/1.0`` for every labelled trial in `sessions`."""
    missing = [s for s in sessions if not label_path(s).exists()]
    if missing:
        raise FileNotFoundError(
            "missing SVM label cache(s): "
            + ", ".join(str(label_path(s)) for s in missing)
            + " -- build them with the label pipeline before running msp"
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
