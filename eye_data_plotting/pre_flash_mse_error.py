"""Held-out MSE (quantization error) for the gaze-position k-means codebook.

Measures how well a K-centroid codebook reconstructs *unseen gaze samples*.
The codebook (``data.attractor``) is fit by k-means on pooled in-maze
fixation ``(x, y)`` samples; for each candidate K, repeatedly split that
pool 80/20, fit centroids on the 80% train split (same call as production
codebook fitting), and score mean squared distance from the held-out 20% to
their nearest train centroid. Train-set MSE is reported alongside it so the
train/test gap is visible.

This does **not** select K. Held-out MSE for k-means falls monotonically in
K — a finer codebook always reconstructs held-out points better, right up to
one centroid per sample — so there is no minimum to read off. Use it only to
see the absolute reconstruction scale and to confirm the train/test gap stays
small. For choosing K see ``pre_flash_centroid_stability``, which asks
whether the centroids themselves land in the same places on independent
halves of the data.

Usage:
    uv run python -m eye_data_plotting.pre_flash_mse_error --monkey Faure
    uv run python -m eye_data_plotting.pre_flash_mse_error --monkey Nielsen --n-repeats 100
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from data.attractor import codebook_xy_pool, pack_trials, fit_code_xy_kmeans
from data.loader import load_eye_behavioral_data, load_eye_data
from eye_data_plotting.plot_io import save_figure

VISUALIZER = "pre_flash_mse_error"
K_SWEEP = (2, 3, 4, 5, 6, 8, 10, 12)
TRAIN_FRACTION = 0.8
N_REPEATS = 100
MAX_SAMPLES = 80000


def _mean_sq_dist(xy, centers):
    dist = np.linalg.norm(xy[:, None, :] - centers[None, :, :], axis=-1)
    return float(np.mean(dist.min(axis=1) ** 2))


def _fit_and_score(train_xy, test_xy, k, seed):
    """Fit a k-means codebook on the train split, score train and test MSE."""
    centers = fit_code_xy_kmeans(train_xy, k, seed=seed)
    return _mean_sq_dist(train_xy, centers), _mean_sq_dist(test_xy, centers)


def mse_error_curve(
    pool, *, ks=K_SWEEP, train_fraction=TRAIN_FRACTION, n_repeats=N_REPEATS, seed=0
):
    """Per-K (train mean, train sem, test mean, test sem) over repeated 80/20 splits."""
    rng = np.random.default_rng(seed)
    n = pool.shape[0]
    n_train = int(round(n * train_fraction))

    train_errs = {k: [] for k in ks}
    test_errs = {k: [] for k in ks}
    for rep in range(n_repeats):
        order = rng.permutation(n)
        train_xy, test_xy = pool[order[:n_train]], pool[order[n_train:]]
        for k in ks:
            fit_seed = int(rng.integers(1 << 31))
            train_mse, test_mse = _fit_and_score(train_xy, test_xy, k, fit_seed)
            train_errs[k].append(train_mse)
            test_errs[k].append(test_mse)
        if (rep + 1) % 10 == 0:
            print(f"  rep {rep + 1}/{n_repeats}")

    def _mean_sem(errs, k):
        vals = np.asarray(errs[k], dtype=float)
        return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(len(vals)))

    train_mean, train_sem, test_mean, test_sem = [], [], [], []
    for k in ks:
        m, s = _mean_sem(train_errs, k)
        train_mean.append(m)
        train_sem.append(s)
        m, s = _mean_sem(test_errs, k)
        test_mean.append(m)
        test_sem.append(s)

    return (
        np.array(train_mean),
        np.array(train_sem),
        np.array(test_mean),
        np.array(test_sem),
    )


def plot_mse_error(
    monkey="Faure",
    *,
    ks=K_SWEEP,
    train_fraction=TRAIN_FRACTION,
    n_repeats=N_REPEATS,
    seed=0,
    max_samples=MAX_SAMPLES,
):
    eye = load_eye_data(monkey)
    behavioral = load_eye_behavioral_data(monkey)
    packed, _ = pack_trials(eye, behavioral, monkey=monkey)
    pool = codebook_xy_pool(packed, max_samples=max_samples, seed=seed)
    print(f"Pooled {pool.shape[0]} in-maze fixation samples for {monkey}")

    train_mean, train_sem, test_mean, test_sem = mse_error_curve(
        pool, ks=ks, train_fraction=train_fraction, n_repeats=n_repeats, seed=seed
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
    ax.errorbar(
        ks, train_mean, yerr=train_sem, marker="o", capsize=3,
        color="#999999", label="train (in-sample)",
    )
    ax.errorbar(
        ks, test_mean, yerr=test_sem, marker="o", capsize=3,
        color="#2266aa", label="held-out (test)",
    )
    ax.set_xlabel("K (codebook size)")
    ax.set_ylabel("mean squared distance to nearest\ncentroid (maze units²)")
    ax.set_xticks(list(ks))
    ax.legend(frameon=False)
    train_pct = round(train_fraction * 100)
    ax.set_title(
        f"{monkey} k-means codebook held-out MSE\n"
        f"{train_pct}/{100 - train_pct} train/test | {n_repeats} repeats | "
        f"n={pool.shape[0]} fixation samples",
        fontsize=10,
    )

    save_figure(fig, "mse_error", rel_dir=f"{VISUALIZER}/{monkey}")
    plt.close(fig)
    return train_mean, train_sem, test_mean, test_sem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monkey", default="Faure", choices=("Faure", "Nielsen"))
    parser.add_argument("--ks", type=int, nargs="+", default=list(K_SWEEP))
    parser.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    args = parser.parse_args()
    plot_mse_error(
        monkey=args.monkey,
        ks=tuple(args.ks),
        train_fraction=args.train_fraction,
        n_repeats=args.n_repeats,
        seed=args.seed,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
