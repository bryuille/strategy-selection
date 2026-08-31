from __future__ import annotations

import argparse

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA

from data.config import monkey_for_session

EARLY_TRIAL_WINDOW_SIZE = 500
N_PCA_COMPONENTS = 3
N_MAZES = 6
FR_THRESH = 1.0

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


def name_clusters_maze_1_6(clusters, mazes):
    """Hierarchical (0) is the cluster holding more maze-1 trials, sequential (1) maze-6."""
    counts = np.array(
        [
            [np.sum((clusters == c) & (mazes == maze)) for maze in (1, 6)]
            for c in (0, 1)
        ]
    )
    hierarchical = int(np.argmax(counts[:, 0]))
    sequential = int(np.argmax(counts[:, 1]))
    if hierarchical == sequential:
        raise ValueError(
            "the same cluster was selected as both hierarchical and sequential "
            f"(maze-1 counts {counts[:, 0].tolist()}, maze-6 counts {counts[:, 1].tolist()})"
        )
    return np.where(clusters == hierarchical, 0.0, 1.0)


def build_strategy_choices(trial_timebins, data, min_value, max_value):
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
    clusters = (
        fcluster(linkage(pc_scores, method="ward"), t=2, criterion="maxclust") - 1
    )

    strategy_choices = np.full(trial_timebins.shape[0], np.nan)
    strategy_choices[trial_ids - 1] = name_clusters_maze_1_6(clusters, mazes)
    print(
        f"trials {min_value}-{max_value} ({trial_ids.size}) | "
        f"neurons {features.shape[1]}/{nrns.max()} | "
        f"hierarchical {int(np.sum(strategy_choices == 0))} "
        f"sequential {int(np.sum(strategy_choices == 1))}"
    )
    return strategy_choices



def sweep_strategy_labels(sessions, *, keep_neural=False, overwrite=False):
    """Build a strategy label for each session, converting and freeing as it goes.

    A label is ~5 KB but is derived from a neural npz of up to ~14 GB -- `np.savez`
    is uncompressed where the mat is compressed HDF5, so the npz runs 3-6x its
    source -- plus a ~1.5 GB `trial_timebins` cache. The 23 eligible sessions
    would be ~170 GB held together, so this converts one session, writes its
    label, then deletes both intermediates before starting the next. Peak extra
    usage is one session, not the pool.

    A session whose label already exists is skipped, so the sweep resumes cleanly
    after an interruption -- but its intermediates are still reclaimed, because a
    label built by an earlier run leaves its npz behind and nothing downstream
    reads it again.
    """
    from data.config import neural_npz_path, processed_npz
    from data.convert import convert_kinds
    from data.loader import load_strategy_choices

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

    built, skipped, failed, reclaimed = [], [], [], 0.0
    for session in sessions:
        monkey = monkey_for_session(session)
        label_path = processed_npz(f"{session}_strategy_choices")
        if label_path.exists() and not overwrite:
            skipped.append(session)
            print(f"=== {monkey} {session}: label already built ===")
            if not keep_neural:
                reclaimed += reclaim(monkey, session, "stale, label already built")
            continue

        print(f"=== {monkey} {session}: convert -> label -> free ===", flush=True)
        try:
            convert_kinds(["neural"], monkey=monkey, sessions=[session])
            load_strategy_choices(session)
            built.append(session)
        except Exception as exc:  # noqa: BLE001 - one bad session must not stop the sweep
            failed.append((session, repr(exc)))
            print(f"    FAILED {session}: {exc!r}", flush=True)
        finally:
            if not keep_neural:
                reclaimed += reclaim(monkey, session, "converted and labelled")

    print(
        f"\nlabels: {len(built)} built, {len(skipped)} already present, "
        f"{len(failed)} failed, {reclaimed:.1f} GB reclaimed"
    )
    for session, err in failed:
        print(f"  FAILED {session}: {err}")
    return built, skipped, failed


def main():
    from data.loader import load_lr_choices, load_strategy_choices

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=("lr", "strategy"), required=True)
    parser.add_argument("--session", default=None)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="strategy only: convert, label and free each session in turn",
    )
    parser.add_argument(
        "--keep-neural",
        action="store_true",
        help="with --sweep, do not delete the neural npz after labelling",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    jobs = [args.session] if args.session is not None else list(CLUSTERING_SESSIONS)

    if args.sweep:
        if args.type != "strategy":
            parser.error("--sweep only applies to --type strategy")
        sweep_strategy_labels(
            jobs, keep_neural=args.keep_neural, overwrite=args.overwrite
        )
        return

    load = load_lr_choices if args.type == "lr" else load_strategy_choices
    for session in jobs:
        print(f"=== {monkey_for_session(session)} {session} {args.type} ===")
        load(session)


if __name__ == "__main__":
    main()