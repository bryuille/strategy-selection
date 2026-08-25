import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score, train_test_split

LAMBDA_GRID = np.logspace(-6, -0.5, 11)
N_CV_SPLITS = 10
N_SHUFFLE_ITERS = 50 # Increase for less bias in shuffle DV


def zscore_features(X_train, X_eval):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-8
    return (X_train - mu) / sd, (X_eval - mu) / sd


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


def balance_train_indices(idx_train, y, random_state=0):
    np.random.seed(random_state)
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


def fit_decoder(X_mean, y, trial_mask, random_state=0, alpha=None):
    valid = np.where(trial_mask)[0]
    X_train = X_mean[valid]
    y_train = y[valid].astype(int)

    best_lambda = (
        alpha
        if alpha is not None
        else pick_lambda(X_train, y_train, random_state=random_state)
    )

    balanced_idx = balance_train_indices(valid, y, random_state)
    clf = make_sgd_classifier(best_lambda, random_state)
    clf.fit(X_mean[balanced_idx], y[balanced_idx].astype(int))
    return clf, trial_mask, best_lambda


def compute_dv_traces(clf, X, trial_mask):
    X_valid = np.nan_to_num(X[trial_mask], nan=0.0)
    return np.einsum("tnb,n->tb", X_valid, clf.coef_[0]) + clf.intercept_[0]


def compute_shuffle_dv_traces(
    X_mean,
    X,
    y,
    trial_mask,
    random_state=0,
    alpha=None,
    n_shuffles=N_SHUFFLE_ITERS,
):
    valid = np.where(trial_mask)[0]
    y_valid = y[valid]
    dv_sum = np.zeros((len(valid), X.shape[-1]))
    counts = np.zeros(len(valid))

    for i in range(n_shuffles):
        np.random.seed(random_state + i)
        idx_train, idx_test = train_test_split(
            valid,
            test_size=0.5,
            stratify=y_valid,
            random_state=random_state + i,
        )
        train_mask = np.zeros_like(trial_mask, dtype=bool)
        train_mask[idx_train] = True

        y_shuf = y.copy()
        y_shuf[idx_train] = np.random.permutation(y[idx_train])

        clf, _, _ = fit_decoder(
            X_mean, y_shuf, train_mask, random_state=random_state + i, alpha=alpha
        )
        dv = compute_dv_traces(clf, X, trial_mask)
        test_local = np.isin(valid, idx_test)
        dv_sum[test_local] += dv[test_local]
        counts[test_local] += 1

    return dv_sum / counts[:, None]


# for testing model performance
def cross_validate_decoder(X_mean, y, trial_mask, n_splits=N_CV_SPLITS):
    valid = np.where(trial_mask)[0]
    X_train = X_mean[valid]
    y_train = y[valid].astype(int)

    best_lambda = pick_lambda(X_train, y_train)

    accs = []
    for i in range(n_splits):
        idx_train, idx_test = train_test_split(
            valid,
            test_size=0.5,
            stratify=y_train,
            random_state=i,
        )
        balanced_idx = balance_train_indices(idx_train, y, random_state=i)
        clf = make_sgd_classifier(best_lambda, i)
        clf.fit(X_mean[balanced_idx], y[balanced_idx].astype(int))
        score = clf.score(X_mean[idx_test], y[idx_test].astype(int))
        accs.append(score)

    return best_lambda, np.mean(accs)


def cross_validate_decoder_grouped(
    X,
    y,
    groups,
    *,
    n_splits=N_CV_SPLITS,
    random_state=0,
):
    """Cross-validated accuracy with session-blocked or stratified trial splits."""
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))

    if n_groups > 1:
        n_splits = min(n_splits, n_groups)
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X, y, groups=groups)
        pick_idx = np.arange(len(y))
    else:
        valid = np.arange(len(y))
        pick_idx = valid
        best_lambda = pick_lambda(X, y, n_splits=min(n_splits, 5), random_state=random_state)
        accs = []
        for i in range(n_splits):
            idx_train, idx_test = train_test_split(
                valid,
                test_size=0.5,
                stratify=y,
                random_state=random_state + i,
            )
            balanced_idx = balance_train_indices(idx_train, y, random_state=random_state + i)
            X_tr, X_te = zscore_features(X[balanced_idx], X[idx_test])
            clf = make_sgd_classifier(best_lambda, random_state + i)
            clf.fit(X_tr, y[balanced_idx])
            accs.append(clf.score(X_te, y[idx_test]))
        return best_lambda, float(np.mean(accs))

    best_lambda = pick_lambda(
        X[pick_idx], y[pick_idx], n_splits=min(n_splits, 5), random_state=random_state
    )
    accs = []
    for fold, (idx_train, idx_test) in enumerate(split_iter):
        balanced_idx = balance_train_indices(idx_train, y, random_state=random_state + fold)
        X_tr, X_te = zscore_features(X[balanced_idx], X[idx_test])
        clf = make_sgd_classifier(best_lambda, random_state + fold)
        clf.fit(X_tr, y[balanced_idx])
        accs.append(clf.score(X_te, y[idx_test]))
    return best_lambda, float(np.mean(accs))

