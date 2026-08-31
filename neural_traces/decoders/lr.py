import numpy as np

from data.loader import (
    load_lr_choices,
    load_pre_flash_metadata,
    load_pre_flash_timebins,
    load_trial_metadata,
    load_trial_timebins,
)
from neural_traces.decoders.common import N_CV_SPLITS, cross_validate_decoder

ENDPOINT_WINDOW_SIZE = 301


def get_endpoint_mean(X, flash3_ms):
    t = np.arange(X.shape[-1])
    flash3_idx = np.floor(flash3_ms).astype(int)

    in_window = (
        (t[None, None, :] >= flash3_idx[:, None, None])
        & (t[None, None, :] < flash3_idx[:, None, None] + ENDPOINT_WINDOW_SIZE)
        & ~np.isnan(X)
    )

    window_sum = np.where(in_window, X, 0.0).sum(axis=-1)
    window_count = in_window.sum(axis=-1)
    endpoint = np.divide(
        window_sum,
        window_count,
        out=np.full_like(window_sum, np.nan),
        where=window_count > 0,
    )
    return np.nan_to_num(endpoint, nan=0.0)


def zscore_per_neuron(X_raw, trial_mask):
    mu = np.nanmean(X_raw[trial_mask], axis=(0, 2))
    sd = np.nanstd(X_raw[trial_mask], axis=(0, 2))
    return (X_raw - mu[None, :, None]) / sd[None, :, None]


def prepare_decoder_data():
    X_raw = load_trial_timebins()
    y = load_lr_choices()
    path_type, flash2_ms, flash3_ms, trial_mask = load_trial_metadata()
    X = zscore_per_neuron(X_raw, trial_mask)
    return X, y, trial_mask, path_type, flash2_ms, flash3_ms


def prepare_pre_flash_data():
    X_raw = load_pre_flash_timebins()
    path_type, flash1_ms, trial_mask = load_pre_flash_metadata()
    X = zscore_per_neuron(X_raw, trial_mask)
    return X, path_type, flash1_ms, trial_mask


def main():
    print("Loading data...")
    X, y, trial_mask, _, _, flash3_ms = prepare_decoder_data()
    n_valid = trial_mask.sum()
    print(f"Decoder trials: {n_valid} (omits random geometry trials)")

    print("Running cross-validation evaluation...")
    X_mean = get_endpoint_mean(X, flash3_ms)
    best_lambda, cv_acc = cross_validate_decoder(X_mean, y, trial_mask)
    print(f"Cross-validated accuracy ({N_CV_SPLITS} iterations): {cv_acc:.3f}")


if __name__ == "__main__":
    main()
