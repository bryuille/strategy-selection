import os

import numpy as np

from data.builder import (
    build_lr_choices,
    build_timebins,
    build_trial_metadata,
    build_trial_strategies,
    import_data,
)

RAW_PATH = "./data/processed/raw.npz"
TIMEBINS_PATH = "./data/processed/timebins.npz"
LR_CHOICES_PATH = "./data/processed/lr_choices.npz"
METADATA_PATH = "./data/processed/trial_metadata.npz"
TRIAL_STRATEGIES_PATH = "./data/processed/trial_strategies.npz"


def load_data():
    if os.path.exists(RAW_PATH):
        with np.load(RAW_PATH, allow_pickle=True) as f:
            return {k: f[k] for k in f.files}
    data = import_data()
    np.savez(RAW_PATH, **data)
    return data


def load_timebins():
    if os.path.exists(TIMEBINS_PATH):
        return np.load(TIMEBINS_PATH)["timebins"]

    timebins = build_timebins(load_data())
    np.savez(TIMEBINS_PATH, timebins=timebins)
    return timebins


def load_lr_choices():
    if os.path.exists(LR_CHOICES_PATH):
        return np.load(LR_CHOICES_PATH)["lr_choices"]

    lr_choices = build_lr_choices(load_data())
    np.savez(LR_CHOICES_PATH, lr_choices=lr_choices)
    return lr_choices


def load_trial_metadata():
    if os.path.exists(METADATA_PATH):
        loaded = np.load(METADATA_PATH)
        return (
            loaded["path_type"],
            loaded["flash2_ms"],
            loaded["flash3_ms"],
            loaded["trial_mask"],
        )

    path_type, flash2_ms, flash3_ms, trial_mask = build_trial_metadata(load_data())
    np.savez(
        METADATA_PATH,
        path_type=path_type,
        flash2_ms=flash2_ms,
        flash3_ms=flash3_ms,
        trial_mask=trial_mask,
    )
    return path_type, flash2_ms, flash3_ms, trial_mask


########## Custom data


def load_trial_strategies():
    if os.path.exists(TRIAL_STRATEGIES_PATH):
        return np.load(TRIAL_STRATEGIES_PATH)["trial_strategies"]

    trial_strategies = build_trial_strategies(load_data())
    np.savez(TRIAL_STRATEGIES_PATH, trial_strategies=trial_strategies)
    return trial_strategies
