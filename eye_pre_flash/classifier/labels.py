"""Strategy labels (0 = hierarchical, 1 = sequential) and session scopes.

Labels come from :func:`data.labeler.build_strategy_choices`, which needs the
session's neural recording, so the labelled pool is whichever of
`data.labeler.CLUSTERING_SESSIONS` have had their ~5 KB label built.
"""

from __future__ import annotations

import numpy as np

from data.config import monkey_for_session
from data.labeler import CLUSTERING_SESSIONS

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


def label_agreement(session, *, lookup=None, anchors=None):
    """Fraction of the session's anchor-maze labels matching the expected map.

    ``lookup`` and ``anchors`` default to the dendrogram lookup and the full
    five-maze `ANCHOR_MAJORITY`; a caller vetting a different label source
    (e.g. `eye_pre_flash.label_sources`' SVM labels, which are defined by mazes 1 and 6
    and so cannot be honestly graded against them) passes its own lookup and a
    restricted anchor map instead of duplicating this scoring logic.
    """
    from eye_pre_flash.classifier.features import load_features

    if anchors is None:
        anchors = ANCHOR_MAJORITY
    if lookup is None:
        lookup = strategy_label_lookup((session,))

    monkey = monkey_for_session(session)
    if monkey not in _MAZE_FEATURES:
        _MAZE_FEATURES[monkey] = load_features(monkey, k=6)
    data = _MAZE_FEATURES[monkey]
    rows = np.asarray(data["session"]).astype(str)
    y = labels_for_rows(rows, data["trial_indices_all"], lookup)
    mazes = np.asarray(data["maze_id"], dtype=int)
    keep = (rows == session) & np.isfinite(y) & np.isin(mazes, list(anchors))
    if not keep.any():
        return 0.0
    expected = np.array([anchors[m] for m in mazes[keep]])
    return float((y[keep].astype(int) == expected).mean())


def scope_sessions(scope, *, vet=None, stem="strategy_choices", sessions=None):
    """``monkey -> labelled sessions`` inside `scope`, plus dropped.

    Labels are loaded through `strategy_label_lookup`, which is build-on-miss
    like every other cache in `data.loader` -- so a scope whose labels were
    never built, or were deleted with the rest of `processed/`, builds them
    here rather than quietly resolving to an empty pool. Building one costs
    that session's multi-gigabyte neural conversion, so this is not free; it is
    simply honest about what the analysis needs.

    A session whose label cannot be built (its neural npz was never converted,
    say) raises. A scope names the sessions the analysis claims to cover, so
    quietly covering fewer of them is worse than stopping. With ``vet`` (on by
    default except for `UNVETTED_SCOPES`),
    sessions whose anchor-maze label agreement falls below
    `MIN_LABEL_AGREEMENT` are returned in ``dropped`` (session -> agreement)
    instead of the analysis pool.

    ``stem`` selects which cached label a caller is scoping -- the dendrogram
    label by default (``<session>_strategy_choices.npz``), or e.g.
    ``"strategy_svm"`` for `eye_pre_flash.label_sources`' SVM labels
    (``<session>_strategy_svm.npz``). ``sessions`` overrides `SCOPES[scope]`
    with a caller-supplied session list -- `corr` needs this because its SVM
    ``all``/``allplus`` scopes are not the same session lists as the
    dendrogram's (see `eye_pre_flash.label_sources.SVM_SCOPES`), while still
    reusing this function's vetting.
    """
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}; choose from {sorted(SCOPES)}")
    if vet is None:
        vet = scope not in UNVETTED_SCOPES
    wanted = SCOPES[scope] if sessions is None else tuple(sessions)
    by_monkey, dropped = {}, {}
    for s in wanted:
        # Build-on-miss, and a session that cannot be labelled raises rather
        # than dropping out of the pool: a scope that silently analyses fewer
        # sessions than it names is the failure this is meant to prevent.
        lookup = strategy_label_lookup((s,), stem=stem)
        if vet:
            agreement = label_agreement(s, lookup=lookup)
            if agreement < MIN_LABEL_AGREEMENT:
                dropped[s] = agreement
                continue
        by_monkey.setdefault(monkey_for_session(s), []).append(s)
    return {m: tuple(v) for m, v in by_monkey.items()}, dropped


def strategy_label_lookup(sessions=TRAIN_SESSIONS, *, stem="strategy_choices"):
    """``(session, trial_id) -> 0.0/1.0`` for every labelled trial.

    ``stem`` reads a different cached label under the same 0.0/1.0/NaN,
    ``trial_id - 1`` convention -- e.g. ``"strategy_svm"`` for the SVM labels
    built by `data.labeler.build_svm_choices` / `data.loader.load_svm_choices`.
    Each stem's loader (build-on-miss, same as the default) is looked up by
    name rather than duplicated here.
    """
    from data.loader import load_strategy_choices, load_svm_choices

    loaders = {"strategy_choices": load_strategy_choices, "strategy_svm": load_svm_choices}
    if stem not in loaders:
        raise ValueError(f"unknown label stem {stem!r}; choose from {sorted(loaders)}")
    load = loaders[stem]

    lookup = {}
    for session in sessions:
        loaded = load(session)
        choices = loaded["strategy_choices"] if isinstance(loaded, dict) else loaded
        choices = np.asarray(choices, dtype=float)
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
