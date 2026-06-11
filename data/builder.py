import h5py
import numpy as np

from utils import path_type_for, sigmoid_dv

############ Data Input

DATA_PATH = "./data/raw/june_24_g0_good_trials_concat.mat"
FR_PATH = "./data/raw/june_24_g0_Whole_Trial_FR_Causal.mat"

FIELDS = {
    # "Area",
    # "Grid_hole",
    "LR",
    # "LR2",
    # "all_synctimes",
    # "answer_time",
    # "cids",
    # "date_of_recording",
    # "feedback_time",
    # "fix_start",
    # "fixation_cue_present",
    # "fixation_off",
    "flash_one",
    "flash_three",
    "flash_two",
    "geo_present",
    # "geo_type",
    "h",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    # "labels",
    "nrns",
    # "old_nrns",
    # "one",
    # "one_hier",
    "path_type",
    # "probe_sync",
    # "probe_type",
    # "rand_geo",
    # "reward",
    # "saccade_init",
    # "spikes",
    # "spikes_all",
    # "sync_params",
    "trial_answer1",
    "trial_answer2",
    "trial_answer3",
    "trial_answer4",
    # "trial_end",
    # "trial_fade",
    "trial_indices_all",
    # "trial_time_to_fix",
    # "two",
    # "two_hier",
    # "two_hier_alt",
    # "two_hier_correct",
    # "two_hier_left",
    # "two_hier_right",
    # "vel",
    # "which_sync",
}


def import_data():
    data = {}

    with h5py.File(DATA_PATH, "r") as f:
        all_data = f["save_all_data"]

        for field in FIELDS:
            data[field] = convert_mat_field(f, all_data[field])

    with h5py.File(FR_PATH, "r") as f:
        data["FR_WH"] = convert_mat_field(f, f["smooth_session"])

    return data


def convert_mat_field(f, dataset):
    if dataset.dtype == h5py.ref_dtype:
        refs = dataset[()]
        return [np.array(f[r]).squeeze() for r in refs.flat]

    return np.array(dataset).squeeze()


########## Data Processing

# Per path-type timing (ms from flash 1). Indices 0-23 map to path_type 1-24.
FIRST_LEN_MS = [
    500,
    500,
    1000,
    1000,
    500,
    500,
    1000,
    1000,
    500,
    500,
    750,
    750,
    500,
    500,
    750,
    750,
    500,
    500,
    500,
    500,
    500,
    500,
    500,
    500,
]
SECOND_LEN_MS = [
    900,
    1100,
    700,
    500,
    700,
    1100,
    900,
    500,
    900,
    1100,
    700,
    500,
    700,
    1100,
    900,
    500,
    900,
    1100,
    700,
    500,
    700,
    1100,
    900,
    500,
]
STOP_LEN_MS = [
    1700,
    1900,
    2000,
    1800,
    1500,
    1900,
    2200,
    1800,
    1700,
    1900,
    1750,
    1550,
    1500,
    1900,
    1950,
    1550,
    1700,
    1900,
    1500,
    1300,
    1500,
    1900,
    1700,
    1300,
]


def build_timebins(data):
    trials = data["trial_indices_all"].astype(int)
    nrns = data["nrns"].astype(int)

    n_trials = np.max(trials)
    n_neurons = np.max(nrns)

    geo_present = data["geo_present"]
    flash_one = data["flash_one"]
    flash_three = data["flash_three"]
    fr_raw = data["FR_WH"].T

    start_bins = np.floor((flash_one - geo_present) * 1000).astype(int)
    end_bins = np.floor((flash_three - geo_present) * 1000).astype(int) + 300

    max_length = np.max(end_bins - start_bins)

    tensor = np.full((n_trials, n_neurons, max_length), np.nan)

    for k in range(len(nrns)):
        ti = trials[k] - 1  # matlab indexing correction
        ni = nrns[k] - 1
        segment = fr_raw[k, start_bins[k] : end_bins[k]]
        tensor[ti, ni, : len(segment)] = segment

    return tensor


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


def build_trial_metadata(data):
    trials = data["trial_indices_all"].astype(int)
    n_trials = np.max(trials)

    path_type = np.full(n_trials, np.nan)
    flash2_ms = np.full(n_trials, np.nan)
    flash3_ms = np.full(n_trials, np.nan)

    for k in range(len(trials)):
        ti = trials[k] - 1
        path_type[ti] = data["path_type"][k]
        flash2_ms[ti] = (data["flash_two"][k] - data["flash_one"][k]) * 1000
        flash3_ms[ti] = (data["flash_three"][k] - data["flash_one"][k]) * 1000

    trial_mask = path_type != -99
    return path_type, flash2_ms, flash3_ms, trial_mask


########## Custom Data

GAP_THRESHOLD = 0.3
TIME_THRESHOLD = 200


def build_trial_strategies():
    from decoders.common import compute_dv_traces
    from decoders.lr import prepare_decoder_data

    X, labels, trial_mask, path_type, _, flash3_ms = prepare_decoder_data()
    path_type = path_type[trial_mask]
    flash3_ms = flash3_ms[trial_mask]

    _, dv, _, _ = compute_dv_traces(X, labels, trial_mask)
    dv_prob = sigmoid_dv(dv)

    dv_means = np.array(
        [np.nanmean(dv_prob[path_type == p], axis=0) for p in range(1, 25)]
    )

    strategy = np.full(6, np.nan)

    for maze in range(1, 7):
        pt = path_type_for(maze, 1)
        cutoff = int(np.nanmedian(flash3_ms[path_type == pt]) - TIME_THRESHOLD)

        mean_lu = dv_means[path_type_for(maze, 1) - 1]
        mean_ru = dv_means[path_type_for(maze, 3) - 1]

        mean_ld = dv_means[path_type_for(maze, 2) - 1]
        mean_rd = dv_means[path_type_for(maze, 4) - 1]

        gap_u = np.abs(mean_lu[:cutoff] - mean_ru[:cutoff])
        gap_d = np.abs(mean_ld[:cutoff] - mean_rd[:cutoff])

        strategy[maze - 1] = int(
            np.nanmax(gap_u) <= GAP_THRESHOLD and np.nanmax(gap_d) <= GAP_THRESHOLD
        )

    trial_maze = ((path_type - 1) // 4).astype(int)
    trial_strategies = strategy[trial_maze].astype(int)
    return trial_strategies
