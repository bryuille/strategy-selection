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
            so this source is not anchor-vetted at all -- its one scope
            (`top_ten`) ranks by `snr_auc` rather than by any agreement with
            the dendrogram, and
            its cells `1S`/`6H` (the anchor-minority cells) are skipped
            outright rather than computed thin, since the classifier's own
            training axis makes them near-empty by construction.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from data.labeler import CLUSTERING_SESSIONS, SNR_AUC
from eye_pre_flash.classifier.labels import scope_sessions, strategy_label_lookup

SOURCES = ("dendro", "svm")
SOURCE_STEM = {"dendro": "strategy_choices", "svm": "strategy_svm"}

# Scopes, per source. Each source gets the scopes that make sense for the
# labels it produces, and nothing else -- there is no scope both sources share
# except `publication`, and only `dendro` uses it.
#
#   dendro/publication  the 4 original publication sessions (2 per monkey).
#   dendro/top_four     the 4 highest-`SNR_AUC` sessions per monkey, drawn
#                       from the vetted `CLUSTERING_SESSIONS` pool. A strict
#                       superset of `publication`.
#   svm/top_ten         the 10 highest-`snr_auc` sessions per monkey, from the
#                       unvetted pool.
#
# Both ranked scopes use the same metric, `snr_auc` (`snr_auc_table`), so
# there is one ordering in this package rather than one per source.
#
# Every scope is (pool, top_n per monkey). Truncation happens *after* vetting,
# never before: truncating first would let a session that vetting rejects
# consume a slot meant for one that passes.
#
# Measured, and worth knowing before reading `top_four` as a selection:
# anchor vetting drops 15 of the 23 `CLUSTERING_SESSIONS` (agreements 0.486-
# 0.653, all under `MIN_LABEL_AGREEMENT` = 0.70), leaving exactly 4 per monkey.
# So `top_four`'s cap currently binds on nothing -- it *is* the whole vetted
# pool, identical to the retired `all`/`allplus` dendro scopes, and the
# `SNR_AUC` ranking never gets to choose. The cap stays as a guard in case
# vetting ever passes a fifth. Two consequences:
#
#   * The dendro pair is really "publication (2/monkey)" vs "everything whose
#     labels vet (4/monkey)", not a quality gradient within a larger pool.
#   * Neural SNR does not predict dendrogram label quality. Faure's #3 and #4
#     by `SNR_AUC` (june_28_g0 at 0.9931, june_29_g0 at 0.9834) both fail
#     vetting at 0.542 and 0.486, while june_16_g0 (0.9779) and june_22_g0
#     (0.9712) pass and take those slots.
#
# `top_ten` is likewise only a real selection for one animal, and the reason is
# the data rather than the cap. Nielsen has 34 neural recordings in the
# reference CSV, but only 10 clear the 0.95 `snr_auc` gate that defines
# `CLUSTERING_SESSIONS`, so its `top_ten` is all 10 and nothing is ranked out.
# That gate is not arbitrary -- it sits just above a cliff in both animals:
#
#   Nielsen ranks  9-16 by snr_auc: .965 .958 | .948 | .883 .810 .792 .791 .786
#   Faure   ranks 11-18 by snr_auc: .971 .971 | .969 .962 .952 | .942 .892 .870
#
# Widening the Nielsen pool past ~11 means admitting sessions at snr_auc <= 0.88,
# where the neural clustering barely separates maze 1 from maze 6 and the
# strategy labels are mostly noise. So the asymmetry is a property of the
# recordings, not something a bigger pool fixes. For Faure the cap does bind,
# dropping 3 of 13 (june_17_g0, April_8_g0, june_11_g0).
SOURCE_SCOPES = {
    "dendro": ("publication", "top_four"),
    "svm": ("top_ten",),
}
SCOPES = ("publication", "top_four", "top_ten")

# The widest scope per source -- where the `mean_removed` grand mean and the
# PC1 diagnostic are fitted, and where `examples*` are written.
WIDEST_SCOPE = {"dendro": "top_four", "svm": "top_ten"}

_SCOPE_SPEC = {
    # scope -> (pool scope in `classifier.labels.SCOPES`, top_n per monkey)
    "publication": ("publication", 0),
    "top_four": ("all", 4),
    "top_ten": ("allplus", 10),
}

_SNR_CSV = (
    Path(__file__).resolve().parents[1]
    / "zrefs/Dendogram_all/SNR_All_Sessions/all_session_snr_results.csv"
)


@lru_cache(maxsize=1)
def snr_auc_table():
    """``session -> snr_auc`` for every reference session, read from `zrefs`.

    The maze-1-vs-6 neural separability figure, straight from
    `zrefs/Dendogram_all/SNR_All_Sessions/all_session_snr_results.csv` (61
    sessions, 27 Faure and 34 Nielsen, all ``status=passed``).

    Read from the file rather than from `data.labeler.SNR_AUC`, which is a
    transcription of the same CSV restricted to the 23 sessions clearing its
    0.95 gate -- verified identical to the file on all 23, so this is a strict
    superset and the ranking keeps working if the pool ever widens. Falls back
    to the transcribed dict if `zrefs` is not present (it is a repo file, not a
    `STRATEGY_DATA_ROOT` file, and only `*.mat` under it is rsync-excluded).
    """
    try:
        with _SNR_CSV.open(newline="") as fh:
            return {r["session"]: float(r["snr_auc"]) for r in csv.DictReader(fh)}
    except OSError:
        return dict(SNR_AUC)


def _snr_key(session):
    """Rank key: `snr_auc` descending, session name breaking ties."""
    return snr_auc_table().get(session, -1.0)


def source_scope_sessions(source, scope):
    """``(by_monkey, missing, dropped)`` for `source` in `scope`.

    Reuses `classifier.labels.scope_sessions` for the pool and the vetting,
    then truncates each monkey's survivors to the scope's `top_n`. `dendro`
    keeps its anchor-agreement gate; `svm` passes ``vet=False`` (no gate at
    all, for the circularity reason in the module docstring) over the full
    `CLUSTERING_SESSIONS` pool.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown label source {source!r}; choose from {SOURCES}")
    if scope not in SOURCE_SCOPES[source]:
        raise ValueError(
            f"scope {scope!r} is not defined for source {source!r}; "
            f"choose from {SOURCE_SCOPES[source]}"
        )
    stem = SOURCE_STEM[source]
    pool, top_n = _SCOPE_SPEC[scope]
    if source == "dendro":
        by_monkey, missing, dropped = scope_sessions(pool, stem=stem)
    else:
        by_monkey, missing, dropped = scope_sessions(
            pool, stem=stem, vet=False, sessions=CLUSTERING_SESSIONS
        )
    if top_n:
        by_monkey = {
            m: tuple(sorted(sorted(ss), key=_snr_key, reverse=True)[:top_n])
            for m, ss in by_monkey.items()
        }
    return by_monkey, missing, dropped


def source_lookup(source, sessions):
    """``(session, trial_id) -> 0.0/1.0`` for `source`."""
    return strategy_label_lookup(sessions, stem=SOURCE_STEM[source])


def scope_census():
    """``(source, scope, monkey) -> sessions``, for documenting what each scope is."""
    out = {}
    for source, scopes in SOURCE_SCOPES.items():
        for scope in scopes:
            by_monkey, _missing, _dropped = source_scope_sessions(source, scope)
            for monkey, sess in sorted(by_monkey.items()):
                out[(source, scope, monkey)] = sess
    return out
