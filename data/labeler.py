import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA

EARLY_TRIAL_WINDOW_SIZE = 500
N_PCA_COMPONENTS = 3


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


def build_strategy_choices(trial_timebins, path_type):
    n_trials = trial_timebins.shape[0]
    strategy_choices = np.full(n_trials, np.nan)

    valid_idx = np.where(path_type != -99)[0]
    trial_window = trial_timebins[valid_idx, :, :EARLY_TRIAL_WINDOW_SIZE]
    trial_averaged = np.nanmean(trial_window, axis=2)

    neuron_idx = np.where(~np.isnan(trial_averaged).any(axis=0))[0]
    features = trial_averaged[:, neuron_idx]
    features = features - features.mean(axis=0)
    features = features / (features.std(axis=0) + 1e-8)

    pc_scores = PCA(n_components=N_PCA_COMPONENTS, random_state=0).fit_transform(
        features
    )
    clusters = (
        fcluster(linkage(pc_scores, method="ward"), t=2, criterion="maxclust") - 1
    )

    strategy_choices[valid_idx] = (1 - clusters).astype(float)
    return strategy_choices
