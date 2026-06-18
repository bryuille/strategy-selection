import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA

FLASH1_WINDOW_BINS = 300  # 300 ms following flash 1
N_PCA_COMPONENTS = 3
N_CLUSTERS = 2


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


########## Strategy logic


def build_strategy_choices(trial_timebins):
    window = trial_timebins[:, :, :FLASH1_WINDOW_BINS]
    trial_averaged = np.nanmean(window, axis=2)

    features = np.nan_to_num(trial_averaged, nan=0.0)
    features -= features.mean(axis=0)
    features /= features.std(axis=0) + 1e-8

    pc_scores = PCA(n_components=N_PCA_COMPONENTS, random_state=0).fit_transform(
        features
    )
    clusters = (
        fcluster(linkage(pc_scores, method="ward"), t=N_CLUSTERS, criterion="maxclust")
        - 1
    )
    return clusters.astype(float)
