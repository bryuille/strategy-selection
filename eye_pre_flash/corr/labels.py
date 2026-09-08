"""Label-source registry: which sessions and which cells, per label source.

Two label sources, sharing one vetting path
(`eye_pre_flash.classifier.labels.scope_sessions`) through a `stem` keyword
rather than duplicating it:

``dendro``  the existing Ward-clustering labels
            (`data.labeler.build_strategy_choices`), anchor-agreement vetted
            against all five anchor mazes (`classifier.labels.ANCHOR_MAJORITY`).
``svm``     the maze-1-vs-6 linear SVM labels (`data.labeler.build_svm_choices`).
            Grading an SVM trained on mazes 1 and 6 by its agreement with a
            map that calls mazes 1 and 6 hierarchical/sequential is circular,
            so this source is not anchor-vetted at all -- its `all`/`allplus`
            scopes are session *lists* (below) rather than a quality gate, and
            its cells `1S`/`6H` (the anchor-minority cells) are skipped
            outright rather than computed thin, since the classifier's own
            training axis makes them near-empty by construction.
"""

from __future__ import annotations

from data.labeler import CLUSTERING_SESSIONS, PUBLICATION_SESSIONS, SNR_AUC
from eye_pre_flash.classifier.labels import scope_sessions, strategy_label_lookup
from eye_pre_flash.corr.cells import cell_index

SOURCES = ("dendro", "svm")
SOURCE_STEM = {"dendro": "strategy_choices", "svm": "strategy_svm"}

# Cells skipped outright for a source, rather than computed thin. Only the
# SVM source skips anything -- see the module docstring.
SOURCE_SKIP_CELLS = {
    "dendro": (),
    "svm": (cell_index(1, 1), cell_index(6, 0)),  # 1S, 6H
}

# Top 4 per monkey by `data.labeler.SNR_AUC`: a strict superset of the
# original-clustering 4 (june_24_g0, june_8_g0, Nov_3_g0, Oct_22_g0), extended
# by the same maze-1-vs-6 AUC criterion the publication set was picked by (all
# four extra sessions score >= 0.983, against a 0.95 eligibility floor).
#
# This exists because `data.labeler.CLUSTERING_SESSIONS` -- the dendrogram
# `all`/`allplus` pool -- was *defined* by `snr_auc >= 0.95`, so gating the SVM
# source's `all` scope on that same threshold would drop nobody and leave
# `all` and `allplus` byte-identical. A session-list `all` avoids that without
# inventing an arbitrary quality threshold.
SVM_ALL_SESSIONS = (
    "june_24_g0", "june_8_g0", "june_28_g0", "june_29_g0",  # Faure, >= 0.983
    "Nov_6_g0", "Oct_22_g0", "Nov_3_g0", "Nov_18_g0",  # Nielsen, >= 0.990
)

SVM_SCOPES = {
    "publication": PUBLICATION_SESSIONS,
    "all": SVM_ALL_SESSIONS,
    "allplus": tuple(CLUSTERING_SESSIONS),
}


def source_scope_sessions(source, scope):
    """``(by_monkey, missing, dropped)`` for `source` in `scope`.

    Reuses `classifier.labels.scope_sessions` for both sources: `dendro`
    passes nothing extra (its own `SCOPES`, its own anchor-agreement gate);
    `svm` passes its own session list via `sessions=` and ``vet=False`` (no
    gate at all, for the circularity reason above).
    """
    if source not in SOURCES:
        raise ValueError(f"unknown label source {source!r}; choose from {SOURCES}")
    stem = SOURCE_STEM[source]
    if source == "dendro":
        return scope_sessions(scope, stem=stem)
    if scope not in SVM_SCOPES:
        raise ValueError(f"unknown scope {scope!r}; choose from {sorted(SVM_SCOPES)}")
    return scope_sessions(scope, stem=stem, vet=False, sessions=SVM_SCOPES[scope])


def source_lookup(source, sessions):
    """``(session, trial_id) -> 0.0/1.0`` for `source`."""
    return strategy_label_lookup(sessions, stem=SOURCE_STEM[source])


def top_sessions(sessions, top_n):
    """`sessions` truncated to the `top_n` highest by `data.labeler.SNR_AUC`.

    ``top_n <= 0`` disables truncation. Opt-in and off by default: the three
    scopes already define the session sets, and truncating on top of vetting
    would let a session vetting will reject consume a slot meant for one that
    passes -- rank, then vet, then (optionally) truncate, in that order.
    """
    if top_n is None or top_n <= 0:
        return tuple(sessions)
    ranked = sorted(sessions, key=lambda s: SNR_AUC.get(s, -1.0), reverse=True)
    return tuple(ranked[:top_n])
