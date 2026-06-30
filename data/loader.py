import os

import numpy as np

from data.builder import (
    build_post_flash_metadata,
    build_post_flash_timebins,
    build_pre_flash_metadata,
    build_pre_flash_timebins,
    build_trial_metadata,
    build_trial_timebins,
    import_data,
)
from data.labeler import build_lr_choices, build_strategy_choices

RAW_PATH = "./data/processed/raw.npz"
TRIAL_TIMEBINS_PATH = "./data/processed/trial_timebins.npz"
TRIAL_METADATA_PATH = "./data/processed/trial_metadata.npz"
PRE_FLASH_TIMEBINS_PATH = "./data/processed/pre_flash_timebins.npz"
PRE_FLASH_METADATA_PATH = "./data/processed/pre_flash_metadata.npz"
POST_FLASH_TIMEBINS_PATH = "./data/processed/post_flash_timebins.npz"
POST_FLASH_METADATA_PATH = "./data/processed/post_flash_metadata.npz"
LR_CHOICES_PATH = "./data/processed/lr_choices.npz"
STRATEGY_CHOICES_PATH = "./data/processed/strategy_choices.npz"


def load_data():
    if os.path.exists(RAW_PATH):
        with np.load(RAW_PATH, allow_pickle=True) as f:
            return {k: f[k] for k in f.files}
    data = import_data()
    np.savez(RAW_PATH, **data)
    return data


def load_trial_timebins():
    if os.path.exists(TRIAL_TIMEBINS_PATH):
        return np.load(TRIAL_TIMEBINS_PATH)["trial_timebins"]

    trial_timebins = build_trial_timebins(load_data())
    np.savez(TRIAL_TIMEBINS_PATH, trial_timebins=trial_timebins)
    return trial_timebins


def load_trial_metadata():
    if os.path.exists(TRIAL_METADATA_PATH):
        loaded = np.load(TRIAL_METADATA_PATH)
        return (
            loaded["path_type"],
            loaded["flash2_ms"],
            loaded["flash3_ms"],
            loaded["trial_mask"],
        )

    path_type, flash2_ms, flash3_ms, trial_mask = build_trial_metadata(load_data())
    np.savez(
        TRIAL_METADATA_PATH,
        path_type=path_type,
        flash2_ms=flash2_ms,
        flash3_ms=flash3_ms,
        trial_mask=trial_mask,
    )
    return path_type, flash2_ms, flash3_ms, trial_mask


def load_pre_flash_timebins():
    if os.path.exists(PRE_FLASH_TIMEBINS_PATH):
        return np.load(PRE_FLASH_TIMEBINS_PATH)["pre_flash_timebins"]

    pre_flash_timebins = build_pre_flash_timebins(load_data())
    np.savez(PRE_FLASH_TIMEBINS_PATH, pre_flash_timebins=pre_flash_timebins)
    return pre_flash_timebins


def load_pre_flash_metadata():
    if os.path.exists(PRE_FLASH_METADATA_PATH):
        loaded = np.load(PRE_FLASH_METADATA_PATH)
        return loaded["path_type"], loaded["flash1_ms"], loaded["trial_mask"]

    path_type, flash1_ms, trial_mask = build_pre_flash_metadata(load_data())
    np.savez(
        PRE_FLASH_METADATA_PATH,
        path_type=path_type,
        flash1_ms=flash1_ms,
        trial_mask=trial_mask,
    )
    return path_type, flash1_ms, trial_mask


def load_post_flash_timebins():
    if os.path.exists(POST_FLASH_TIMEBINS_PATH):
        return np.load(POST_FLASH_TIMEBINS_PATH)["post_flash_timebins"]

    post_flash_timebins = build_post_flash_timebins(load_data())
    np.savez(POST_FLASH_TIMEBINS_PATH, post_flash_timebins=post_flash_timebins)
    return post_flash_timebins


def load_post_flash_metadata():
    if os.path.exists(POST_FLASH_METADATA_PATH):
        loaded = np.load(POST_FLASH_METADATA_PATH)
        return loaded["path_type"], loaded["trial_mask"]

    path_type, trial_mask = build_post_flash_metadata(load_data())
    np.savez(
        POST_FLASH_METADATA_PATH,
        path_type=path_type,
        trial_mask=trial_mask,
    )
    return path_type, trial_mask


########## Labels


def load_lr_choices():
    if os.path.exists(LR_CHOICES_PATH):
        return np.load(LR_CHOICES_PATH)["lr_choices"]

    lr_choices = build_lr_choices(load_data())
    np.savez(LR_CHOICES_PATH, lr_choices=lr_choices)
    return lr_choices


def load_strategy_choices():
    if os.path.exists(STRATEGY_CHOICES_PATH):
        return np.load(STRATEGY_CHOICES_PATH)["strategy_choices"]

    trial_timebins = load_trial_timebins()
    path_type, _, _, _ = load_trial_metadata()
    strategy_choices = build_strategy_choices(trial_timebins, path_type)
    np.savez(STRATEGY_CHOICES_PATH, strategy_choices=strategy_choices)
    return strategy_choices
