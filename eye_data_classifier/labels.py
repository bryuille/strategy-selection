"""Strategy labels (0 = hierarchical, 1 = sequential) and session scopes.

Labels come from :func:`data.labeler.build_strategy_choices`, which needs the
session's neural recording, so the labelled pool is whichever of
`data.labeler.CLUSTERING_SESSIONS` have had their ~5 KB label built.
"""

from __future__ import annotations

import numpy as np

from data.config import monkey_for_session, processed_npz
from data.labeler import CLUSTERING_SESSIONS
from data.loader import load_strategy_choices

LABEL_NAMES = ("hierarchical", "sequential")
TRAIN_SESSIONS = tuple(CLUSTERING_SESSIONS)

# The four sessions the published clustering was defined on -- ranked #2-#5 of
# 61 by `snr_auc`. `all` is every session clearing snr_auc >= 0.95, widening
# the publication pool roughly sixfold while keeping its criterion.
PUBLICATION_SESSIONS = ("june_8_g0", "june_24_g0", "Nov_3_g0", "Oct_22_g0")
SCOPES = {
    "publication": PUBLICATION_SESSIONS,
    "all": tuple(CLUSTERING_SESSIONS),
    "allplus": tuple(CLUSTERING_SESSIONS),
}
DEFAULT_SCOPE = "all"

# Scopes that skip the anchor-maze label vetting: `allplus` analyses every
# SNR-eligible session that labels at all, as a robustness check on the
# concern that vetting also removes genuinely mixed (label-noisy but
# information-rich) sessions.
UNVETTED_SCOPES = ("allplus",)


# Label-quality vetting. The maze largely determines the strategy, and the
# anchor mazes are known: 1-3 are solved hierarchically, 5-6 sequentially.
# Maze 4 is the boundary maze and is left unscored -- it is genuinely mixed for
# Nielsen (both publication sessions included) and near-mixed for Faure. A
# session whose labels agree with the anchor map at barely above chance is a
# session whose neural labelling did not work, and its trials are label noise.
# The observed split is bimodal (0.76-1.00 vs 0.54-0.65), so the threshold sits
# in the gap.
ANCHOR_MAJORITY = {1: 0, 2: 0, 3: 0, 5: 1, 6: 1}
MIN_LABEL_AGREEMENT = 0.70

_MAZE_FEATURES = {}


def label_agreement(session):
    """Fraction of the session's anchor-maze labels matching the expected map."""
    from eye_data_classifier.features import load_features

    monkey = monkey_for_session(session)
    if monkey not in _MAZE_FEATURES:
        _MAZE_FEATURES[monkey] = load_features(monkey, k=6, space="unith")
    data = _MAZE_FEATURES[monkey]
    rows = np.asarray(data["session"]).astype(str)
    y = labels_for_rows(
        rows, data["trial_indices_all"], strategy_label_lookup((session,))
    )
    mazes = np.asarray(data["maze_id"], dtype=int)
    keep = (rows == session) & np.isfinite(y) & np.isin(mazes, list(ANCHOR_MAJORITY))
    if not keep.any():
        return 0.0
    expected = np.array([ANCHOR_MAJORITY[m] for m in mazes[keep]])
    return float((y[keep].astype(int) == expected).mean())


def scope_sessions(scope, *, vet=None):
    """``monkey -> labelled sessions`` inside `scope`, plus missing and dropped.

    Filtered through `available_label_sessions`, so a checkout partway through
    the `labels` sweep analyses what it actually has instead of failing on the
    first session whose neural npz was never converted. With ``vet`` (on by
    default except for `UNVETTED_SCOPES`), sessions whose anchor-maze label
    agreement falls below `MIN_LABEL_AGREEMENT` are returned in ``dropped``
    (session -> agreement) instead of the analysis pool.
    """
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}; choose from {sorted(SCOPES)}")
    if vet is None:
        vet = scope not in UNVETTED_SCOPES
    wanted = SCOPES[scope]
    have = set(available_label_sessions(wanted))
    by_monkey, dropped = {}, {}
    missing = tuple(s for s in wanted if s not in have)
    for s in wanted:
        if s not in have:
            continue
        if vet:
            agreement = label_agreement(s)
            if agreement < MIN_LABEL_AGREEMENT:
                dropped[s] = agreement
                continue
        by_monkey.setdefault(monkey_for_session(s), []).append(s)
    return {m: tuple(v) for m, v in by_monkey.items()}, missing, dropped

def available_label_sessions(sessions=TRAIN_SESSIONS):
    """The subset of `sessions` whose strategy labels are already cached.

    `CLUSTERING_SESSIONS` lists every session *eligible* for a label, but
    building one needs that session's multi-gigabyte neural npz. On a checkout
    that holds only some of them -- a laptop, or a cluster run partway through
    the `labels` stage -- this returns what can actually be read without
    triggering a conversion, so an analysis runs on what exists instead of
    failing on the first missing session.
    """
    return tuple(
        s for s in sessions if processed_npz(f"{s}_strategy_choices").exists()
    )


def strategy_label_lookup(sessions=TRAIN_SESSIONS):
    """``(session, trial_id) -> 0.0/1.0`` for every labelled trial."""
    lookup = {}
    for session in sessions:
        choices = np.asarray(load_strategy_choices(session), dtype=float)
        for idx in np.flatnonzero(np.isfinite(choices)):
            lookup[(session, int(idx) + 1)] = float(choices[idx])
    return lookup


def labels_for_rows(sessions, trial_ids, lookup):
    """Label vector aligned to the given rows; NaN where the trial is unlabelled."""
    return np.asarray(
        [
            lookup.get((str(s), int(t)), np.nan)
            for s, t in zip(sessions, trial_ids)
        ],
        dtype=float,
    )
