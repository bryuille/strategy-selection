import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from decoders.common import (
    N_CV_SPLITS,
    cross_validate_decoder,
    load_lr_choices,
    load_timebins,
    load_trial_metadata,
    zscore_per_neuron,
)


def prepare_decoder_data():
    X = zscore_per_neuron(load_timebins())
    y = load_lr_choices()
    path_type, flash2_ms, flash3_ms, trial_mask = load_trial_metadata()
    return X, y, trial_mask, path_type, flash2_ms, flash3_ms


def main():
    print("Loading data...")
    X, y, trial_mask, *_ = prepare_decoder_data()
    n_eligible = trial_mask.sum()
    print(f"Decoder trials: {n_eligible} (omits random geometry trials)")

    print("Running cross-validation evaluation...")
    best_lambda, cv_acc = cross_validate_decoder(X, y, trial_mask)
    print(f"Cross-validated accuracy ({N_CV_SPLITS} iterations): {cv_acc:.3f}")


if __name__ == "__main__":
    main()
