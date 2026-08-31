"""Split-half spatial stability of the k-means codebook centroids — picks K.

Held-out MSE cannot choose K for k-means: it falls monotonically as the
codebook gets finer (see ``pre_flash_mse_error``). The question that *does*
have an interior optimum is whether the centroids land in the same **places**
when fit to independent samples. Real attractors are properties of the
behaviour, so they reappear in any sufficiently large sample; surplus
centroids have nothing to lock onto and drift to wherever the particular
sample happens to be lumpy. Sample-dependent centres are the fingerprint of
too large a K.

Per repeat, per K:

1. Shuffle the pooled in-maze fixation ``(x, y)`` samples and cut them into
   two **disjoint** halves A and B (disjoint matters — overlapping subsamples
   share points, which correlates the two fits and inflates stability).
2. Fit a K-centroid codebook on A and on B with the production call
   (``fit_code_xy_kmeans``, k-means++ with ``n_init=10``). Both halves get the
   same k-means seed, so the intended difference between them is the sample.
3. Find the correspondence between the two centroid sets: the one-to-one
   matching that minimizes total squared distance, solved exactly with the
   Hungarian algorithm (``scipy.optimize.linear_sum_assignment``). This step
   is required — k-means state labels are arbitrary, so A's ``k1`` and B's
   ``k1`` are unrelated. Greedy nearest-neighbour matching is not used: it can
   double-book one centroid and then overstate drift when two centroids swap.
   Because the matching minimizes the same sum that gets reported, the score
   never blames the codebook for an arbitrary ordering of its states.
4. Score the matched pairs: mean squared distance between corresponding
   centroids, reported as its root (RMS drift, in maze units) so it reads on
   the same scale as the maze itself — the exits sit at ``(±1, ±1)`` and
   ``ASSIGN_RADIUS`` = 1.0 is the radius within which a fixation is assigned
   to a prototype. The worst-matched pair is plotted alongside the mean,
   because a codebook can look fine on average while one centroid is pure
   sample noise, and that is usually the first thing to go as K passes the
   true value.

The mean over matched pairs is deliberately unweighted, so a centroid holding
2% of the fixations counts as much as one holding 30%. That is the point: an
unneeded centroid parks in a sparse tail and wanders, and weighting by mass
would hide exactly the overfitting this is meant to detect.

**Reading it.** Drift stays low and flat while every centroid has a real
landmark to sit on, and climbs once K exceeds the number of landmarks there
are. It also spikes at *under*-parameterized K — too few centroids to cover
the real modes have no unique grouping either, so they flip between samples —
so the answer is not the global minimum (which is anyway trivially K=2: two
centroids have almost nothing to disagree about). It is the top of the
**stable band**: the longest contiguous run of Ks that all hold drift under
``DRIFT_STABLE``. Contiguity is what separates a genuine plateau — a range of
codebook sizes that all recover the same landmarks — from a lucky single K.

Centroid spacing is printed beside drift because it shrinks as K grows (the
centroids simply pack closer), which pulls raw drift down a little at large K.
Over this sweep spacing falls by under a factor of two while drift varies by
more than an order of magnitude, so the confound never changes which band
wins; the ``spacing`` column is there to keep that checkable.

Each fit sees half the pool, so this is a slightly conservative read of the
production codebook's stability, which is fit on all of it. Halving the sample
raises drift at every K; the position of the band is the robust part, not its
absolute height.

Usage:
    uv run python -m eye_data_plotting.pre_flash_centroid_stability --monkey Faure
    uv run python -m eye_data_plotting.pre_flash_centroid_stability --monkey Nielsen \
        --n-repeats 50
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from data.attractor import (
    ASSIGN_RADIUS,
    codebook_xy_pool,
    fit_code_xy_kmeans,
    pack_trials,
)
from data.config import MONKEYS
from data.loader import load_eye_behavioral_data, load_eye_data
from eye_data_plotting.plot_io import save_figure

VISUALIZER = "pre_flash_centroid_stability"
K_SWEEP = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20)
N_REPEATS = 30
MAX_SAMPLES = 80000
# RMS drift (maze units) under this counts as a codebook that reproduces
# itself: a tenth of ASSIGN_RADIUS, and a tenth of the distance from the maze
# centre to an exit coordinate.
DRIFT_STABLE = 0.10

SERIES_MEAN = "#2a78d6"
SERIES_WORST = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e6e5e1"


def sq_dist_matrix(a, b):
    """Pairwise squared Euclidean distances, shape (len(a), len(b))."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)


def match_centroids(centers_a, centers_b):
    """Squared displacements of corresponding centroids, optimally matched."""
    cost = sq_dist_matrix(centers_a, centers_b)
    rows, cols = linear_sum_assignment(cost)
    return cost[rows, cols]


def nn_spacing(centers):
    """Mean distance from each centroid to its nearest other centroid.

    Not part of the score — reported so drift can be read against how far
    apart the centroids sit at that K.
    """
    if len(centers) < 2:
        return np.nan
    dist = np.sqrt(sq_dist_matrix(centers, centers))
    np.fill_diagonal(dist, np.inf)
    return float(dist.min(axis=1).mean())


def split_half_drift(pool, k, *, rng):
    """One A/B split of `pool`: fit both halves, match centroids, measure drift."""
    n = pool.shape[0]
    half = n // 2
    order = rng.permutation(n)
    xy_a, xy_b = pool[order[:half]], pool[order[half : 2 * half]]

    fit_seed = int(rng.integers(1 << 31))
    centers_a = fit_code_xy_kmeans(xy_a, k, seed=fit_seed)
    centers_b = fit_code_xy_kmeans(xy_b, k, seed=fit_seed)

    sq = match_centroids(centers_a, centers_b)
    msd = float(sq.mean())
    return {
        "msd": msd,
        "rms": float(np.sqrt(msd)),
        "max": float(np.sqrt(sq.max())),
        "spacing": 0.5 * (nn_spacing(centers_a) + nn_spacing(centers_b)),
    }


def stable_band(ks, rms, threshold=DRIFT_STABLE):
    """Longest contiguous run of Ks whose RMS drift stays under `threshold`."""
    ok = np.asarray(rms) <= threshold
    best, run = [], []
    for i, hit in enumerate(ok):
        run = run + [ks[i]] if hit else []
        if len(run) > len(best):
            best = run
    return best


def _mean_sem(values):
    vals = np.asarray(values, dtype=float)
    sem = (
        float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else float("nan")
    )
    return float(vals.mean()), sem


def stability_curve(pool, *, ks=K_SWEEP, n_repeats=N_REPEATS, seed=0):
    """Per-K (mean, sem) of every drift score over repeated A/B splits."""
    rng = np.random.default_rng(seed)
    reps = {k: [] for k in ks}
    for rep in range(n_repeats):
        for k in ks:
            reps[k].append(split_half_drift(pool, k, rng=rng))
        print(f"  repeat {rep + 1}/{n_repeats}", flush=True)
    return {
        key: np.array([_mean_sem([r[key] for r in reps[k]]) for k in ks])
        for key in reps[ks[0]][0]
    }


def _line(ax, ks, stat, *, color, label, marker="o"):
    ax.errorbar(
        ks,
        stat[:, 0],
        yerr=stat[:, 1],
        marker=marker,
        markersize=5,
        linewidth=2,
        capsize=2.5,
        elinewidth=1,
        color=color,
        label=label,
    )


def _note(ax, y, text, *, color, ha="left"):
    """Note on a reference line, parked at whichever end the series is not."""
    ax.annotate(
        text,
        xy=(0.005 if ha == "left" else 0.995, y),
        xycoords=ax.get_yaxis_transform(),
        xytext=(0, 3),
        textcoords="offset points",
        ha=ha,
        va="bottom",
        fontsize=8.5,
        color=color,
    )


def plot_stability(curve, *, monkey, ks, n_pool, n_repeats):
    band = stable_band(ks, curve["rms"][:, 0])
    fig, ax = plt.subplots(figsize=(7.6, 5.4), layout="constrained")

    if len(band) > 1:
        ax.axvspan(band[0], band[-1], color=SERIES_MEAN, alpha=0.08, zorder=0)
        ax.annotate(
            f"stable band K={band[0]:g}–{band[-1]:g}",
            xy=(0.5 * (band[0] + band[-1]), 0.98),
            xycoords=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color=SERIES_MEAN,
        )
    ax.axhline(DRIFT_STABLE, color=SERIES_MEAN, linewidth=1, linestyle="--")
    _note(ax, DRIFT_STABLE, f"reproducible (< {DRIFT_STABLE:g})", color=SERIES_MEAN)
    ax.axhline(ASSIGN_RADIUS, color=INK_MUTED, linewidth=1, linestyle="--")
    _note(ax, ASSIGN_RADIUS, f"assign radius {ASSIGN_RADIUS:g}", color=INK_MUTED, ha="right")

    _line(ax, ks, curve["rms"], color=SERIES_MEAN, label="RMS over matched centroids")
    _line(
        ax, ks, curve["max"], color=SERIES_WORST, label="worst matched pair", marker="s"
    )

    ax.set_yscale("log")
    ax.set_xlabel("K (codebook size)", fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel(
        "distance between corresponding centroids\n"
        "of the two half-sample fits (maze units)",
        fontsize=9.5,
        color=INK_SECONDARY,
    )
    ax.set_xticks(list(ks))
    ax.tick_params(labelsize=8.5, colors=INK_SECONDARY, length=3)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="lower right")

    ax.set_title(
        f"{monkey} codebook centroid stability across disjoint halves "
        f"(finest reproducible codebook: {f'K={band[-1]:g}' if band else 'none'})\n"
        f"50/50 split | {n_repeats} repeats | n={n_pool} in-maze fixation samples | "
        f"Hungarian centroid correspondence",
        fontsize=10,
        color=INK_PRIMARY,
    )
    save_figure(fig, "centroid_stability", rel_dir=f"{VISUALIZER}/{monkey}")
    plt.close(fig)
    return band


def print_table(curve, ks, *, monkey):
    print(f"\n{monkey} centroid drift between half-sample fits (mean ± sem)")
    header = f"{'K':>3} {'msd':>9} {'rms':>13} {'worst':>13} {'spacing':>8}"
    print(header)
    print("-" * len(header))
    for i, k in enumerate(ks):
        print(
            f"{k:>3} {curve['msd'][i, 0]:>9.4f} "
            f"{curve['rms'][i, 0]:>6.3f}±{curve['rms'][i, 1]:<6.3f} "
            f"{curve['max'][i, 0]:>6.3f}±{curve['max'][i, 1]:<6.3f} "
            f"{curve['spacing'][i, 0]:>8.3f}"
        )
    band = stable_band(ks, curve["rms"][:, 0])
    if band:
        print(
            f"  stable band K={band[0]}–{band[-1]} (RMS drift < {DRIFT_STABLE:g}); "
            f"finest reproducible codebook K={band[-1]}"
        )
    else:
        print(f"  no K holds RMS drift under {DRIFT_STABLE:g}")
    return band


def plot_centroid_stability(
    monkey="Faure", *, ks=K_SWEEP, n_repeats=N_REPEATS, seed=0, max_samples=MAX_SAMPLES
):
    eye = load_eye_data(monkey)
    behavioral = load_eye_behavioral_data(monkey)
    packed, _ = pack_trials(eye, behavioral, monkey=monkey)
    pool = codebook_xy_pool(packed, max_samples=max_samples, seed=seed)
    print(f"Pooled {pool.shape[0]} in-maze fixation samples for {monkey}", flush=True)

    curve = stability_curve(pool, ks=ks, n_repeats=n_repeats, seed=seed)
    print_table(curve, ks, monkey=monkey)
    plot_stability(
        curve, monkey=monkey, ks=ks, n_pool=pool.shape[0], n_repeats=n_repeats
    )
    return curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monkey", default="Faure", choices=MONKEYS)
    parser.add_argument("--ks", type=int, nargs="+", default=list(K_SWEEP))
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    args = parser.parse_args()
    plot_centroid_stability(
        monkey=args.monkey,
        ks=tuple(args.ks),
        n_repeats=args.n_repeats,
        seed=args.seed,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
