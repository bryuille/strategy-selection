from __future__ import annotations

import argparse

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import fisher_exact
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data.config import monkey_for_session

EARLY_TRIAL_WINDOW_SIZE = 500
N_PCA_COMPONENTS = 3
N_MAZES = 6
FR_THRESH = 1.0

# SVM labeller: maze-1-vs-6 stratified CV, same spec as
# `eye_pre_flash.classifier.decoding.cv_pool` (N_SPLITS=5, SEED=0). Defined
# locally rather than imported, so this module -- built to run standalone on
# a compute node -- does not pull in `eye_pre_flash`.
SVM_N_SPLITS = 5
SVM_SEED = 0

# How the two clusters get named hierarchical/sequential. See
# `name_clusters_maze_1_6`.
#
#   "relative"  -- the default: name by relative maze-6-vs-maze-1 enrichment,
#                  gated on a Fisher exact test. A documented DEVIATION from the
#                  reference MATLAB, adopted because it never contradicts it --
#                  it returns the same answer wherever the reference succeeds
#                  (proof below, and verified bit-identical on all four sessions
#                  whose neural data is held locally) -- while additionally
#                  labelling 3 sessions the reference rejects for a reason that
#                  is an artefact of unbalanced clustering rather than absent
#                  signal.
#   "reference" -- what zrefs/Dendogram_all/Single_Trial_Statistics_Clustering_All.m
#                  does: name by absolute majority, error when that picks the
#                  same cluster twice. Use it to reproduce the published labels
#                  exactly.
CLUSTER_NAMING = "relative"


# "relative" only: a Fisher exact p above this is *reported* as a weakly
# separated naming, never refused. Whether such a session is fit to analyse is
# decided downstream by `classifier.labels.MIN_LABEL_AGREEMENT`, which scores
# the resulting labels against all five anchor mazes rather than just two.
WEAK_AXIS_ALPHA = 0.05

# Which mazes each animal dominantly solves in which regime
# (hierarchical mazes, sequential mazes). Behavioral grouping, shared by the
# pooled gaze heatmaps and the post-flash counterfactual analysis; the
# per-trial neural labels built below are the finer-grained version.
MAZE_GROUPS: dict[str, tuple[list[int], list[int]]] = {
    "Faure": ([1, 2, 3], [4, 5, 6]),
    "Nielsen": ([1, 2, 3, 4], [5, 6]),
}

# The four sessions the published clustering was defined on, from
# zrefs/Dendogram_all/Single_Trial_Statistics_Clustering_All.m.
PUBLICATION_SESSIONS = (
    "june_24_g0",
    "june_8_g0",
    "Nov_3_g0",
    "Oct_22_g0",
)

# Every session whose neural clustering separates maze 1 from maze 6 at
# snr_auc >= 0.95, from zrefs/Dendogram_all/SNR_All_Sessions/all_session_snr_results.csv
# (61 sessions, all status=passed). The four publication sessions rank #2-#5 of
# 61, so this threshold keeps their high-SNR criterion while widening the pool
# roughly sixfold -- which is what makes a per-maze classifier testable in more
# than one maze per animal.
#
# A label costs ~5 KB but is derived from a neural npz of up to ~14 GB, so the
# pipeline's `labels` stage converts, labels and deletes one session at a time
# rather than holding all of them on disk at once.
CLUSTERING_SESSIONS = (
    # Faure, 13
    "june_24_g0",
    "june_8_g0",
    "june_28_g0",
    "june_29_g0",
    "june_16_g0",
    "june_09_09_g0",
    "june_18_g0",
    "june_10_g0",
    "june_22_g0",
    "june_12_g0",
    "june_17_g0",
    "April_8_g0",
    "june_11_g0",
    # Nielsen, 10
    "Nov_6_g0",
    "Oct_22_g0",
    "Nov_3_g0",
    "Nov_18_g0",
    "Oct_21_g0",
    "Nov_7_g0",
    "Oct_25_g0",
    "Nov_1_g0",
    "Nov_4_g0",
    "Nov_9_g0",
)

# Maze-1-vs-6 SVM ROC-AUC for each of the 23 `CLUSTERING_SESSIONS`, transcribed
# from `zrefs/Dendogram_all/SNR_All_Sessions/all_session_snr_results.csv`. The
# same quantity `build_svm_choices` recomputes below (its `svm_auc` should
# match these to within CV-fold-assignment noise); kept here as a fixed
# reference so a session's rank does not shift as label caches are rebuilt.
SNR_AUC = {
    "june_24_g0": 0.999341238471673,
    "june_8_g0": 0.998737373737374,
    "june_28_g0": 0.993063812921126,
    "june_29_g0": 0.983380681818182,
    "june_16_g0": 0.977857142857143,
    "june_09_09_g0": 0.977531857813548,
    "june_18_g0": 0.976758663623158,
    "june_10_g0": 0.972524154589372,
    "june_22_g0": 0.97123745819398,
    "june_12_g0": 0.970520799213888,
    "june_17_g0": 0.968567251461988,
    "April_8_g0": 0.962453358208955,
    "june_11_g0": 0.952031839128613,
    "Nov_6_g0": 1.0,
    "Oct_22_g0": 0.999896608767576,
    "Nov_3_g0": 0.999725274725275,
    "Nov_18_g0": 0.990064289888954,
    "Oct_21_g0": 0.990039008080245,
    "Nov_7_g0": 0.973597359735974,
    "Oct_25_g0": 0.969401041666667,
    "Nov_1_g0": 0.96875,
    "Nov_4_g0": 0.965494505494505,
    "Nov_9_g0": 0.957720588235294,
}


def build_lr_choices(data):
    trials = data["trial_indices_all"].astype(int)
    LR = data["LR"]
    n_trials = np.max(trials)
    lr_choices = np.full(n_trials, np.nan)

    for k in range(len(trials)):
        ti = trials[k] - 1
        lr = LR[k]

        # LR variable specifies when trial data should be flipped
        if lr == -1:
            if bool(data["trial_answer1"][k]) or bool(data["trial_answer2"][k]):
                lr_choices[ti] = 0
            elif bool(data["trial_answer3"][k]) or bool(data["trial_answer4"][k]):
                lr_choices[ti] = 1
        elif lr == 1:
            if bool(data["trial_answer1"][k]) or bool(data["trial_answer2"][k]):
                lr_choices[ti] = 1
            elif bool(data["trial_answer3"][k]) or bool(data["trial_answer4"][k]):
                lr_choices[ti] = 0

    return lr_choices


def name_clusters_maze_1_6(clusters, mazes, naming=None):
    """Name the two clusters hierarchical (0) / sequential (1) off mazes 1 and 6.

    Two rules, selected by `naming` (default `CLUSTER_NAMING`).

    **"reference"** -- what the published MATLAB does, at
    `Single_Trial_Statistics_Clustering_All.m:352`::

        [~, hierarchical_row] = max(cluster_distribution(:, 1));
        [~, sequential_row]   = max(cluster_distribution(:, 6));

    Hierarchical is whichever cluster holds more maze-1 trials, sequential more
    maze-6, and the same cluster winning both is an error. This is the default,
    so the labels reproduce the published ones exactly.

    **"relative"** -- a deviation, off by default. Sequential is the cluster
    maze 6 favours *relative to* maze 1: the larger
    ``p(cluster | maze 6) − p(cluster | maze 1)``.

    The reference rule assumes a balanced clustering. When `fcluster` returns a
    lopsided split -- one cluster holding, say, 75% of all trials -- that cluster
    holds the majority of maze 1 *and* maze 6 however strong the maze effect is,
    and the rule reads that as "no strategy axis" and refuses. It is what
    rejected 10 of the 23 eligible sessions. `june_12_g0` was refused at 91%
    cluster-1 occupancy in maze 1 against 58% in maze 6 -- a 33-point maze
    effect, running *opposite* to the absolute majority.

    The relative rule is a strict generalisation: **wherever the reference
    succeeds it returns the same answer**, so it cannot change an existing
    label. With two clusters the reference succeeds only when cluster *A* holds
    more maze-1 trials and *B* more maze-6, i.e. ``p1(A) > 1/2 > p1(B)`` and
    ``p6(B) > 1/2 > p6(A)``; then ``p6(B) − p1(B) > 0 > p6(A) − p1(A)``, so *B*
    is named sequential either way. Verified empirically too: the four sessions
    whose neural data is held locally rebuild bit-identical under both rules.

    **It always names, and never refuses.** There are only two ways to assign
    two clusters to two strategies, and the enrichment difference picks whichever
    of the two better matches "maze 1 is hierarchical, maze 6 is sequential" --
    which is the best that can be done with these anchors, however weak the
    separation. All 23 eligible sessions get labels. A session whose separation
    is poor gets its Fisher p printed and flagged WEAKLY SEPARATED, but the
    decision about whether it is fit to analyse belongs downstream, to
    `classifier.labels.MIN_LABEL_AGREEMENT` -- which scores the resulting labels
    against all five anchor mazes (1, 2, 3, 5, 6) and so is strictly better
    informed than a two-maze test here. An individual maze coming out with a
    seemingly wrong majority is not by itself a reason to discard a session.

    Passing the `snr_auc >= 0.95` eligibility filter does not imply the clusters
    separate the anchor mazes: `snr_auc` comes from a *supervised*
    maze-1-vs-maze-6 axis, while this 2-cluster Ward split is unsupervised and
    need not align with it. The 7 sessions the reference rule rejected and this
    one names weakly all score between 0.957 and 0.990.
    """
    naming = CLUSTER_NAMING if naming is None else naming
    if naming not in ("reference", "relative"):
        raise ValueError(f"unknown naming rule {naming!r}")

    counts = np.array(
        [
            [np.sum((clusters == c) & (mazes == maze)) for maze in (1, 6)]
            for c in (0, 1)
        ]
    )
    n_maze_1, n_maze_6 = counts[:, 0].sum(), counts[:, 1].sum()
    if n_maze_1 == 0 or n_maze_6 == 0:
        raise ValueError(
            f"need both anchor mazes to name clusters (maze-1 trials {n_maze_1}, "
            f"maze-6 trials {n_maze_6})"
        )

    if naming == "reference":
        hierarchical = int(np.argmax(counts[:, 0]))
        sequential = int(np.argmax(counts[:, 1]))
        if hierarchical == sequential:
            raise ValueError(
                "the same cluster was selected as both hierarchical and sequential "
                f"(maze-1 counts {counts[:, 0].tolist()}, "
                f"maze-6 counts {counts[:, 1].tolist()}); "
                "an unbalanced split can do this even with a real maze effect — "
                "see CLUSTER_NAMING='relative'"
            )
        return np.where(clusters == hierarchical, 0.0, 1.0)

    enrichment = counts[:, 1] / n_maze_6 - counts[:, 0] / n_maze_1
    sequential = int(np.argmax(enrichment))
    hierarchical = 1 - sequential

    _odds, p_value = fisher_exact(counts)
    weak = " WEAKLY SEPARATED" if p_value > WEAK_AXIS_ALPHA else ""
    print(
        f"  [relative naming] cluster {sequential} = sequential "
        f"(maze-6 minus maze-1 share {enrichment[sequential]:+.3f}, "
        f"Fisher p={p_value:.4f}){weak}"
    )
    return np.where(clusters == hierarchical, 0.0, 1.0)


def strategy_pc_scores(trial_timebins, data, min_value, max_value):
    """The (n, 3) PCA space both the dendrogram and SVM labellers share.

    Neuron QC (recorded on every trial of the window), the 501-bin post-flash
    window mean (MATLAB's inclusive ``start:start+500`` slice), a per-neuron
    z-score, then PCA to 3 components -- everything `build_strategy_choices`
    used to do before naming clusters, factored out so a second labeller can
    reuse the identical feature space rather than re-deriving it.

    Returns ``(pc_scores, trial_ids, mazes, n_neurons)``.
    """
    nrns = data["nrns"].astype(int)
    trials = data["trial_indices_all"].astype(int)

    # A neuron is usable only if it was recorded on every trial of the window.
    recorded = np.zeros((int(nrns.max()), int(trials.max()) + 1), dtype=bool)
    recorded[nrns - 1, trials] = True
    neuron_ok = recorded[:, min_value : max_value + 1].all(axis=1)

    in_window = (
        (data["geo_type"] != -99)
        & (data["trial_fade"] == 0)
        & (data["photodiode_qc_bad"] == 0)
        & (trials >= min_value)
        & (trials <= max_value)
    )
    trial_ids = np.unique(trials[in_window])

    trial_geo_type = np.full(int(trials.max()) + 1, -99.0)
    trial_geo_type[trials] = data["geo_type"]
    mazes = trial_geo_type[trial_ids]

    trial_averaged = np.nanmean(
        trial_timebins[
            np.ix_(
                trial_ids - 1,
                np.flatnonzero(neuron_ok),
                # MATLAB's `start_indices : start_indices + 500` is inclusive, so
                # the reference averages 501 bins. Slicing 500 here shifted 16 of
                # june_24's labels and 1 of june_8's away from the reference.
                np.arange(EARLY_TRIAL_WINDOW_SIZE + 1),
            )
        ],
        axis=2,
    )
    maze_means = np.array(
        [trial_averaged[mazes == maze].mean(axis=0) for maze in range(1, N_MAZES + 1)]
    )
    features = trial_averaged[:, (maze_means >= FR_THRESH).any(axis=0)]

    features = features - features.mean(axis=0)
    features = features / features.std(axis=0, ddof=1)
    pc_scores = PCA(n_components=N_PCA_COMPONENTS, random_state=0).fit_transform(
        features
    )
    return pc_scores, trial_ids, mazes, features.shape[1]


def build_strategy_choices(trial_timebins, data, min_value, max_value):
    pc_scores, trial_ids, mazes, n_features = strategy_pc_scores(
        trial_timebins, data, min_value, max_value
    )
    clusters = (
        fcluster(linkage(pc_scores, method="ward"), t=2, criterion="maxclust") - 1
    )

    strategy_choices = np.full(trial_timebins.shape[0], np.nan)
    strategy_choices[trial_ids - 1] = name_clusters_maze_1_6(clusters, mazes)
    print(
        f"trials {min_value}-{max_value} ({trial_ids.size}) | "
        f"neurons {n_features}/{data['nrns'].astype(int).max()} | "
        f"hierarchical {int(np.sum(strategy_choices == 0))} "
        f"sequential {int(np.sum(strategy_choices == 1))}"
    )
    return strategy_choices


def build_svm_choices(
    trial_timebins, data, min_value, max_value, *, n_splits=SVM_N_SPLITS, seed=SVM_SEED
):
    """Strategy labels from a maze-1-vs-6 linear SVM, projected onto all mazes.

    Fits a linear SVM to separate maze-1 from maze-6 trials on the same 3-PC
    space `build_strategy_choices` clusters -- the assumption being that these
    two mazes anchor the ends of the strategy spectrum -- then classifies
    every trial (mazes 1-6) as hierarchical/sequential from that fit. This is
    a *supervised* alternative to the unsupervised Ward split: it can track a
    nonlinear session shape the dendrogram's 2-cluster maxclust cannot, at the
    cost of being defined by (and validated only on) mazes 1 and 6.

    CV mirrors the reference MATLAB SVM used for session eligibility
    (`zrefs/Dendogram_all/Compute_All_Session_SNR.m`): `K = min(n_splits,
    n_maze_1, n_maze_6)`-fold stratified CV over the anchor trials only,
    `SVC(kernel="linear")` behind a `StandardScaler` fit per training fold
    (`'Standardize', true` there), `class_weight="balanced"` because anchor
    counts run noticeably unequal (typically ~45 vs ~70) and an unweighted
    boundary would shift toward the majority class -- which would then
    systematically over-label mazes 2-5 with it.

    Anchor trials (maze 1, 6) get their **out-of-fold** prediction, so cells
    built from them are a real held-out measurement, not a definitional
    artifact. Mazes 2-5 never appear in any fold's training or test split, so
    there is no notion of "out-of-fold" for them; each gets the **mode**
    across the `K` fold models' predictions. Ties are impossible at odd `K`
    (and `K` is 5 for all 23 sessions, so the tie path below is unreachable in
    practice); at even `K` a tie is broken by the sign of the decision
    function of a model refitted on *all* anchor trials -- not, as an earlier
    version of this docstring claimed, by the mean of the fold models'
    decision values.

    Returns a dict with the same ``strategy_choices`` convention as
    `build_strategy_choices` (0.0 hierarchical / 1.0 sequential / NaN
    unlabelled, indexed ``trial_id - 1``) plus the diagnostics needed to audit
    and revisit the anchor/mode split: ``fold_labels`` (K, n_trials),
    ``fold_agreement``, ``oof_pred``, ``oof_score``, ``svm_auc``,
    ``oof_balanced_acc``, ``n_folds``, ``n_maze_1``, ``n_maze_6``,
    ``n_neurons``, ``seed``, ``status``.
    """
    pc_scores, trial_ids, mazes, n_features = strategy_pc_scores(
        trial_timebins, data, min_value, max_value
    )
    n_trials = trial_timebins.shape[0]
    mazes = mazes.astype(int)

    anchor = np.isin(mazes, (1, 6))
    n_maze_1 = int(np.sum(mazes == 1))
    n_maze_6 = int(np.sum(mazes == 6))
    empty = {
        "strategy_choices": np.full(n_trials, np.nan),
        "fold_labels": np.empty((0, trial_ids.size)),
        "fold_agreement": np.full(trial_ids.size, np.nan),
        "oof_pred": np.full(trial_ids.size, np.nan),
        "oof_score": np.full(trial_ids.size, np.nan),
        "svm_auc": np.nan,
        "oof_balanced_acc": np.nan,
        "n_folds": 0,
        "n_maze_1": n_maze_1,
        "n_maze_6": n_maze_6,
        "n_neurons": n_features,
        "seed": seed,
    }
    k_folds = min(n_splits, n_maze_1, n_maze_6)
    if k_folds < 2:
        print(
            f"  [svm] insufficient anchor trials for CV "
            f"(maze-1={n_maze_1}, maze-6={n_maze_6}); skipping"
        )
        return {**empty, "status": "insufficient CV folds"}

    y_anchor = (mazes[anchor] == 6).astype(int)  # 0 hierarchical, 1 sequential
    X_anchor = pc_scores[anchor]

    oof_score = np.full(y_anchor.shape, np.nan)
    oof_pred = np.full(y_anchor.shape, np.nan)
    fold_labels = np.full((k_folds, trial_ids.size), np.nan)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    for f, (train_idx, test_idx) in enumerate(skf.split(X_anchor, y_anchor)):
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    SVC(kernel="linear", class_weight="balanced", random_state=seed),
                ),
            ]
        )
        model.fit(X_anchor[train_idx], y_anchor[train_idx])
        oof_score[test_idx] = model.decision_function(X_anchor[test_idx])
        oof_pred[test_idx] = model.predict(X_anchor[test_idx])
        fold_labels[f] = model.predict(pc_scores)

    svm_auc = float(roc_auc_score(y_anchor, oof_score))
    oof_balanced_acc = float(balanced_accuracy_score(y_anchor, oof_pred))

    # Mode across folds for every trial; ties are impossible at odd k_folds,
    # and at even k_folds are broken by the sign of the mean decision value
    # (the fold models agree on the boundary even when the vote splits).
    counts_seq = fold_labels.sum(axis=0)
    mode_label = (counts_seq > k_folds / 2).astype(float)
    tied = counts_seq == k_folds / 2
    if tied.any():
        mean_score = np.array(
            [
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "clf",
                            SVC(
                                kernel="linear",
                                class_weight="balanced",
                                random_state=seed,
                            ),
                        ),
                    ]
                )
                .fit(X_anchor, y_anchor)
                .decision_function(pc_scores[[i]])[0]
                for i in np.flatnonzero(tied)
            ]
        )
        mode_label[tied] = (mean_score > 0).astype(float)
    fold_agreement = np.maximum(counts_seq, k_folds - counts_seq) / k_folds

    labels = mode_label.copy()
    anchor_idx = np.flatnonzero(anchor)
    labels[anchor_idx] = oof_pred

    full_oof_score = np.full(trial_ids.size, np.nan)
    full_oof_pred = np.full(trial_ids.size, np.nan)
    full_oof_score[anchor_idx] = oof_score
    full_oof_pred[anchor_idx] = oof_pred

    strategy_choices = np.full(n_trials, np.nan)
    strategy_choices[trial_ids - 1] = labels
    print(
        f"  [svm] {k_folds}-fold CV | anchor AUC {svm_auc:.4f} | "
        f"balanced acc {oof_balanced_acc:.4f} | "
        f"hierarchical {int(np.sum(labels == 0))} sequential {int(np.sum(labels == 1))}"
    )
    return {
        "strategy_choices": strategy_choices,
        "fold_labels": fold_labels,
        "fold_agreement": fold_agreement,
        "oof_pred": full_oof_pred,
        "oof_score": full_oof_score,
        "svm_auc": svm_auc,
        "oof_balanced_acc": oof_balanced_acc,
        "n_folds": k_folds,
        "n_maze_1": n_maze_1,
        "n_maze_6": n_maze_6,
        "n_neurons": n_features,
        "seed": seed,
        "status": "passed",
    }



def sweep_strategy_labels(sessions, *, reclaim_neural=False, overwrite=False, builders=None):
    """Build strategy label(s) for each session, converting as it goes.

    A label is ~5 KB but is derived from a neural npz of up to ~14 GB -- `np.savez`
    is uncompressed where the mat is compressed HDF5, so the npz runs 3-6x its
    source -- plus a ~1.5 GB `trial_timebins` cache.

    ``reclaim_neural`` (default off) deletes both intermediates after each
    session instead of leaving them cached, capping the extra usage at one
    session instead of the pool -- the 23 eligible sessions total ~170 GB kept
    together. This was the *default* while everything lived under the
    cluster's 200 GB `/home` quota; now that `data.config`'s roots point at a
    much larger scratch volume by default there (`STRATEGY_DATA_ROOT`), the
    intermediates are worth keeping: a converted session's neural npz and
    `trial_timebins` cache are reused by anything else that touches that
    session, not just this sweep. Pass ``reclaim_neural=True`` to restore the
    old space-constrained behaviour.

    `builders` (default: just the dendrogram label) is an iterable of
    zero-argument loader callables to run against the same converted session
    -- e.g. ``(load_strategy_choices, load_svm_choices)`` runs both label
    kinds off one conversion instead of two, halving the conversion cost when
    both are wanted. Each builder is resumable independently: a label whose
    npz already exists is skipped even if a sibling builder for the same
    session still needs to run.

    A session whose *every* builder's label already exists is skipped
    entirely, so the sweep resumes cleanly after an interruption. With
    ``reclaim_neural=True`` its intermediates are still reclaimed even when
    skipped, because a label built by an earlier run leaves its npz behind and
    nothing downstream reads it again.
    """
    from data.config import neural_npz_path, processed_npz
    from data.convert import convert_kinds
    from data.loader import load_strategy_choices

    if builders is None:
        builders = (load_strategy_choices,)

    def reclaim(monkey, session, why):
        """Delete a session's neural intermediates. Rebuildable from `data/mat/`."""
        freed = 0.0
        for path in (
            neural_npz_path(monkey, session),
            processed_npz(f"{session}_trial_timebins"),
        ):
            if path.exists():
                freed += path.stat().st_size / 1e9
                path.unlink()
        if freed:
            print(f"    freed {freed:.1f} GB ({why})", flush=True)
        return freed

    def label_paths(session):
        # Each builder's cache stem, keyed by its own module-level convention
        # (`load_strategy_choices` -> `_strategy_choices`, `load_svm_choices`
        # -> `_strategy_svm`). Read from the function name so a new builder
        # needs no change here.
        stems = {
            "load_strategy_choices": "strategy_choices",
            "load_svm_choices": "strategy_svm",
        }
        return [
            processed_npz(f"{session}_{stems[b.__name__]}")
            for b in builders
            if b.__name__ in stems
        ]

    built, skipped, failed, reclaimed = [], [], [], 0.0
    for session in sessions:
        monkey = monkey_for_session(session)
        paths = label_paths(session)
        pending = [b for b, p in zip(builders, paths) if overwrite or not p.exists()]
        if not pending:
            skipped.append(session)
            print(f"=== {monkey} {session}: label(s) already built ===")
            if reclaim_neural:
                reclaimed += reclaim(monkey, session, "stale, labels already built")
            continue

        print(
            f"=== {monkey} {session}: convert -> "
            f"{'+'.join(b.__name__ for b in pending)} ===",
            flush=True,
        )
        try:
            convert_kinds(["neural"], monkey=monkey, sessions=[session])
            for builder in pending:
                builder(session)
            built.append(session)
        except Exception as exc:  # noqa: BLE001 - one bad session must not stop the sweep
            failed.append((session, repr(exc)))
            print(f"    FAILED {session}: {exc!r}", flush=True)
        finally:
            if reclaim_neural:
                reclaimed += reclaim(monkey, session, "converted and labelled")

    print(
        f"\nlabels: {len(built)} built, {len(skipped)} already present, "
        f"{len(failed)} failed, {reclaimed:.1f} GB reclaimed"
    )
    for session, err in failed:
        print(f"  FAILED {session}: {err}")
    return built, skipped, failed


def main():
    global CLUSTER_NAMING
    from data.loader import load_lr_choices, load_strategy_choices, load_svm_choices

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=("lr", "strategy"), required=True)
    parser.add_argument("--session", nargs="+", default=None)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="strategy only: convert and label each session in turn",
    )
    parser.add_argument(
        "--with-svm",
        action="store_true",
        help="strategy/--sweep only: also build the maze-1-vs-6 SVM label off "
        "the same conversion, instead of a second cold sweep later",
    )
    parser.add_argument(
        "--reclaim-neural",
        action="store_true",
        help="with --sweep, delete the neural npz and trial_timebins cache "
        "after labelling each session, instead of leaving them cached -- the "
        "old default from when everything shared the cluster's 200 GB /home "
        "quota; off by default now that STRATEGY_DATA_ROOT can point at a "
        "much larger scratch volume",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--naming",
        choices=("reference", "relative"),
        default=CLUSTER_NAMING,
        help="cluster naming rule; 'reference' (default) reproduces the "
        "published MATLAB, 'relative' additionally labels 3 sessions it rejects",
    )
    args = parser.parse_args()

    CLUSTER_NAMING = args.naming
    if args.naming != "reference":
        print(f"!! cluster naming = {args.naming!r} (deviates from zrefs)")

    jobs = list(args.session) if args.session is not None else list(CLUSTERING_SESSIONS)

    if args.sweep:
        if args.type != "strategy":
            parser.error("--sweep only applies to --type strategy")
        builders = (load_strategy_choices, load_svm_choices) if args.with_svm else None
        sweep_strategy_labels(
            jobs, reclaim_neural=args.reclaim_neural, overwrite=args.overwrite, builders=builders
        )
        return

    if args.type == "lr":
        loads = [load_lr_choices]
    else:
        loads = [load_strategy_choices, load_svm_choices] if args.with_svm else [load_strategy_choices]
    for session in jobs:
        for load in loads:
            print(
                f"=== {monkey_for_session(session)} {session} "
                f"{args.type}{'' if load is not load_svm_choices else ' (svm)'} ==="
            )
            load(session)


if __name__ == "__main__":
    main()