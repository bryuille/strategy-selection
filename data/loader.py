import h5py
import numpy as np

############ Data Input

DATA_PATH = "./data/june_24_g0_good_trials_concat.mat"
FR_PATH = "./data/june_24_g0_Whole_Trial_FR_Causal.mat"

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
    "geo_type",
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


def load_data():
    data = {}

    with h5py.File(DATA_PATH, "r") as f:
        all_data = f["save_all_data"]

        for field in FIELDS:
            data[field] = load_mat_field(f, all_data[field])

    with h5py.File(FR_PATH, "r") as f:
        data["FR_WH"] = load_mat_field(f, f["smooth_session"])

    return data


def load_mat_field(f, dataset):
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


def build_timebin_tensor(data):
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


def build_lr_choice(data):
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

    geo_type = np.full(n_trials, np.nan)
    path_type = np.full(n_trials, np.nan)
    flash2_ms = np.full(n_trials, np.nan)
    flash3_ms = np.full(n_trials, np.nan)

    for k in range(len(trials)):
        ti = trials[k] - 1
        geo_type[ti] = data["geo_type"][k]
        path_type[ti] = data["path_type"][k]
        flash2_ms[ti] = (data["flash_two"][k] - data["flash_one"][k]) * 1000
        flash3_ms[ti] = (data["flash_three"][k] - data["flash_one"][k]) * 1000

    trial_mask = path_type != -99
    return geo_type, path_type, flash2_ms, flash3_ms, trial_mask
