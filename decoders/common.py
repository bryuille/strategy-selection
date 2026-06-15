import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from data.loader import (
    load_data,
    load_lr_choices,
    load_strategy_choices,
    load_trial_metadata,
    load_trial_timebins,
)

LAMBDA_GRID = np.logspace(-6, -0.5, 11)
N_CV_SPLITS = 10


def make_sgd_classifier(alpha, random_state=0):
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        class_weight="balanced",
        learning_rate="optimal",
        random_state=random_state,
        max_iter=2000,
        tol=1e-4,
    )


def zscore_per_neuron(X):
    X_out = X.copy()
    for ni in range(X.shape[1]):
        vals = X[:, ni, :]
        X_out[:, ni, :] = (vals - np.nanmean(vals)) / np.nanstd(vals)
    return X_out


# average firing rate in 300 ms window after third flash
def get_endpoint_mean(X, window_bins=301):
    valid = ~np.isnan(X)
    cum_valid = valid.cumsum(axis=-1)
    total_valid = cum_valid[:, :, -1]

    in_window = cum_valid > (total_valid[:, :, None] - window_bins)
    in_window = in_window & valid

    window_sum = np.where(in_window, X, 0.0).sum(axis=-1)
    window_count = in_window.sum(axis=-1)
    endpoint = np.divide(
        window_sum,
        window_count,
        out=np.full_like(window_sum, np.nan),
        where=window_count > 0,
    )
    return np.nan_to_num(endpoint, nan=0.0)


def pick_lambda(X, y, n_splits=10, random_state=0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    ce = np.zeros(len(LAMBDA_GRID))
    coef_norm = np.zeros(len(LAMBDA_GRID))

    for i, lam in enumerate(LAMBDA_GRID):
        clf = make_sgd_classifier(lam, random_state)
        ce[i] = -cross_val_score(clf, X, y, cv=skf, scoring="neg_log_loss").mean()
        clf.fit(X, y)
        coef_norm[i] = np.sum(np.abs(clf.coef_))

    def normalize(x):
        x = np.log10(x + 1e-12)
        lo, hi = x.min(), x.max()
        return np.zeros_like(x) if hi == lo else (x - lo) / (hi - lo)

    score = (
        normalize(ce)
        + normalize(coef_norm)
        + np.abs(normalize(ce) - normalize(coef_norm))
    )
    return LAMBDA_GRID[int(np.argmin(score))]


# used for cases when class counts are not equal
def balance_train_indices(idx_train, y):
    classes, counts = np.unique(y[idx_train], return_counts=True)
    min_count = counts.min()
    balanced_idx = np.concatenate(
        [
            np.random.choice(idx_train[y[idx_train] == c], min_count, replace=False)
            for c in classes
        ]
    )
    np.random.shuffle(balanced_idx)
    return balanced_idx


def fit_decoder(X, y, trial_mask, random_state=0, alpha=None):
    eligible = np.where(trial_mask)[0]
    X_endpoint = get_endpoint_mean(X)
    X_train = X_endpoint[eligible]
    y_train = y[eligible].astype(int)

    best_lambda = (
        alpha
        if alpha is not None
        else pick_lambda(X_train, y_train, random_state=random_state)
    )

    balanced_idx = balance_train_indices(eligible, y)
    clf = make_sgd_classifier(best_lambda, random_state)
    clf.fit(X_endpoint[balanced_idx], y[balanced_idx].astype(int))
    return clf, trial_mask, best_lambda


def compute_dv_traces(clf, X, trial_mask):
    X_valid = np.nan_to_num(X[trial_mask], nan=0.0)
    return np.einsum("tnb,n->tb", X_valid, clf.coef_[0]) + clf.intercept_[0]


def compute_shuffle_dv_traces(
    X, y, trial_mask, random_state=0, alpha=None, n_shuffles=5
):
    eligible = np.where(trial_mask)[0]
    shuffled_traces = []
    rng = np.random.default_rng(random_state)

    for i in range(n_shuffles):
        y_shuf = y.copy()
        y_shuf[eligible] = rng.permutation(y[eligible])
        clf, _, _ = fit_decoder(
            X, y_shuf, trial_mask, random_state=random_state + i + 1, alpha=alpha
        )
        shuffled_traces.append(compute_dv_traces(clf, X, trial_mask))

    return np.nanmean(np.stack(shuffled_traces, axis=0), axis=0)


# for testing model performance
def cross_validate_decoder(X, y, trial_mask, n_splits=N_CV_SPLITS):
    eligible = np.where(trial_mask)[0]
    X_endpoint = get_endpoint_mean(X)

    best_lambda = pick_lambda(X_endpoint[eligible], y[eligible].astype(int))

    accs = []
    for i in range(n_splits):
        idx_train, idx_test = train_test_split(
            eligible,
            test_size=0.5,
            stratify=y[eligible],
            random_state=i,
        )
        balanced_idx = balance_train_indices(idx_train, y)
        clf = make_sgd_classifier(best_lambda, i)
        clf.fit(X_endpoint[balanced_idx], y[balanced_idx].astype(int))
        score = clf.score(X_endpoint[idx_test], y[idx_test].astype(int))
        accs.append(score)

    return best_lambda, np.mean(accs)
