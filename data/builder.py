import h5py
import numpy as np

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
    "fix_start",
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
# (first length, second length, stop condition length)
PATH_TYPE_LEN_MS = [
    (500, 900, 1700),
    (500, 1100, 1900),
    (1000, 700, 2000),
    (1000, 500, 1800),
    (500, 700, 1500),
    (500, 1100, 1900),
    (1000, 900, 2200),
    (1000, 500, 1800),
    (500, 900, 1700),
    (500, 1100, 1900),
    (750, 700, 1750),
    (750, 500, 1550),
    (500, 700, 1500),
    (500, 1100, 1900),
    (750, 900, 1950),
    (750, 500, 1550),
    (500, 900, 1700),
    (500, 1100, 1900),
    (500, 700, 1500),
    (500, 500, 1300),
    (500, 700, 1500),
    (500, 1100, 1900),
    (500, 900, 1700),
    (500, 500, 1300),
]


def build_trial_timebins(data):
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


def build_pre_flash_timebins(data):
    trials = data["trial_indices_all"].astype(int)
    nrns = data["nrns"].astype(int)

    n_trials = np.max(trials)
    n_neurons = np.max(nrns)

    geo_present = data["geo_present"]
    fix_start = data["fix_start"]
    flash_one = data["flash_one"]
    fr_raw = data["FR_WH"].T

    start_bins = np.floor((fix_start - geo_present) * 1000).astype(int)
    end_bins = np.floor((flash_one - geo_present) * 1000).astype(int)

    max_length = np.max(end_bins - start_bins)

    tensor = np.full((n_trials, n_neurons, max_length), np.nan)

    for k in range(len(nrns)):
        ti = trials[k] - 1  # matlab indexing correction
        ni = nrns[k] - 1
        segment = fr_raw[k, start_bins[k] : end_bins[k]]
        tensor[ti, ni, : len(segment)] = segment

    return tensor


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


def build_pre_flash_metadata(data):
    trials = data["trial_indices_all"].astype(int)
    n_trials = np.max(trials)

    path_type = np.full(n_trials, np.nan)
    fix_start = data["fix_start"]
    flash1_ms = np.full(n_trials, np.nan)

    for k in range(len(trials)):
        ti = trials[k] - 1
        path_type[ti] = data["path_type"][k]
        flash1_ms[ti] = (data["flash_one"][k] - data["fix_start"][k]) * 1000

    trial_mask = path_type != -99
    return path_type, flash1_ms, trial_mask