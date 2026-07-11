import numpy as np

from data.labeler import EARLY_TRIAL_WINDOW_SIZE
from data.loader import (
    load_post_flash_metadata,
    load_post_flash_timebins,
    load_pre_flash_metadata,
    load_pre_flash_timebins,
    load_strategy_choices,
    load_trial_metadata,
    load_trial_timebins,
)
from decoders.common import N_CV_SPLITS, cross_validate_decoder


def get_initial_mean(X):
    return np.nan_to_num(np.nanmean(X, axis=2), nan=0.0)


def zscore_per_neuron(X, trial_mask):
    mu = np.nanmean(X[trial_mask], axis=(0, 2))
    sd = np.nanstd(X[trial_mask], axis=(0, 2))
    return (X - mu[None, :, None]) / sd[None, :, None]


def prepare_decoder_data():
    X_raw = load_trial_timebins()[:, :, :EARLY_TRIAL_WINDOW_SIZE]
    y = load_strategy_choices()
    path_type, flash2_ms, flash3_ms, trial_mask = load_trial_metadata()
    X = zscore_per_neuron(X_raw, trial_mask)
    return X, y, trial_mask, path_type, flash2_ms, flash3_ms


def prepare_pre_flash_data():
    X_raw = load_pre_flash_timebins()
    path_type, flash1_ms, trial_mask = load_pre_flash_metadata()
    X = zscore_per_neuron(X_raw, trial_mask)
    return X, path_type, flash1_ms, trial_mask


def prepare_post_flash_data():
    X_raw = load_post_flash_timebins()
    path_type, trial_mask = load_post_flash_metadata()
    X = zscore_per_neuron(X_raw, trial_mask)
    return X, path_type, trial_mask


def main():
    print("Loading data...")
    X, y, trial_mask, *_ = prepare_decoder_data()
    n_valid = trial_mask.sum()
    print(f"Decoder trials: {n_valid} (omits random geometry trials)")

    print("Running cross-validation evaluation...")
    X_mean = get_initial_mean(X)
    best_lambda, cv_acc = cross_validate_decoder(X_mean, y, trial_mask)
    print(f"Cross-validated accuracy ({N_CV_SPLITS} iterations): {cv_acc:.3f}")


if __name__ == "__main__":
    main()
