"""K selection that scores reproducibility *and* explanatory power together.

``pre_flash_centroid_stability`` measures only whether the centroids land in
the same places on two disjoint halves of the data. That is the right question
about overfitting, but on its own it cannot choose K, because it is trivially
satisfied at the bottom of the sweep: two centroids have almost nothing to
disagree about, so K=2 always looks superb. The missing half of the criterion
is that K=2 explains almost nothing — a two-state codebook cannot represent
the maze. The two failure modes sit at opposite ends of K:

    small K -> reproducible but uninformative
    large K -> informative but not reproducible

So score both, on a common 0–1 scale, and take the product. Neither end can
win, and the optimum is an interior maximum rather than a threshold crossing.

Per repeat, per K: shuffle the pooled in-maze fixation ``(x, y)`` samples, cut
them into two **disjoint** halves A and B, and fit a codebook on each with the
production call (``fit_code_xy_kmeans``, k-means++ ``n_init=10``, both halves
on the same k-means seed so the intended difference is the sample). Then:

``coverage`` — explanatory power, held out
    Fraction of the held-out half's variance that the *other* half's codebook
    explains::

        coverage = 1 - E[ min_j |b - c_A,j|^2 ] / E[ |b - mean(B)|^2 ]

    The numerator is the quantization error of A's centroids applied to B —
    the same quantity ``pre_flash_mse_error`` plots — and the denominator is
    B's total variance, i.e. what a single centroid at the mean would leave
    unexplained. So this is the clustering R², computed across the split so
    it is never in-sample. 0 means the codebook is no better than the grand
    mean; 1 means it reconstructs held-out gaze exactly. It rises steeply
    while centroids are still picking up real landmarks and then saturates,
    which is exactly the penalty K=2 deserves and the reason MSE alone cannot
    select K: nothing about this term ever turns around.

``reproducibility`` — spatial stability of the centroids
    Corresponding centroids are found by the one-to-one matching that
    minimizes total squared distance (Hungarian algorithm; k-means labels are
    arbitrary, so A's ``k1`` and B's ``k1`` are unrelated). Then::

        reproducibility = 1 - RMS drift / mean centroid spacing

    clipped at 0. Dividing by the centroids' own mean nearest-neighbour
    spacing makes it dimensionless and removes the shrinking-scale confound:
    as K grows the centroids pack closer together, so a raw drift in maze
    units would drift down for free. 1 is a codebook that reproduces itself
    exactly; 0 means centroids move as far as their own spacing and no longer
    have identities that survive a resample.

``score = coverage × reproducibility``
    Read the argmax. The product is the choice being made explicit: it treats
    a codebook as worth having to the extent that it both explains held-out
    gaze and puts its centroids in the same places every time, weighting the
    two equally. That equal weighting is a convention, not a derivation — but
    it is a mild one, because the two curves turn over sharply in opposite
    directions and the argmax is set by where they cross rather than by the
    exact exchange rate. ``--exponent`` reweights reproducibility
    (``score = coverage × reproducibility ** e``) if you want to see how
    little it matters.

``k_eff`` (printed, not plotted) — how many centroids actually survived
    Number of matched pairs whose displacement is under ``--tolerance`` × that
    centroid's own nearest-neighbour spacing (default 0.5 — move less than
    halfway to your nearest sibling and the two fits are unambiguously
    describing the same landmark). Unlike the RMS, this says *how many* states
    are trustworthy rather than how bad the average is, so it reads as a count
    of real landmarks. It is a diagnostic here because it saturates rather
    than turning over: a K=12 codebook with 8 solid centroids and 4 wandering
    ones scores k_eff=8, which flatters a codebook whose remaining third is
    junk.

Each fit sees half the pool, so this is a conservative read of the production
codebook, which is fit on all of it: halving the sample lowers coverage and
raises drift at every K. The argmax is the robust part, not the score height.

Usage:
    uv run python -m eye_data_plotting.pre_flash_centroid_stability_better --monkey Faure
    uv run python -m eye_data_plotting.pre_flash_centroid_stability_better \
        --monkey Nielsen --n-repeats 50
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from data.attractor import codebook_xy_pool, fit_code_xy_kmeans, pack_trials
from data.config import MONKEYS
from data.loader import load_eye_behavioral_data, load_eye_data
from eye_data_plotting.plot_io import save_figure

VISUALIZER = "pre_flash_centroid_stability_better"
K_SWEEP = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20)
N_REPEATS = 30
MAX_SAMPLES = 80000
# A centroid counts as surviving if it moved less than this fraction of the way
# to its own nearest sibling centroid.
TOLERANCE = 0.5
# score = coverage * reproducibility ** EXPONENT
EXPONENT = 1.0

SERIES_SCORE = "#2a78d6"
SERIES_COVERAGE = "#eb6834"
SERIES_REPRO = "#4a3aa7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e6e5e1"


def sq_dist_matrix(a, b):
    """Pairwise squared Euclidean distances, shape (len(a), len(b))."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)


def matched_displacements(centers_a, centers_b):
    """Distances between corresponding centroids, optimally matched.

    The correspondence is the one-to-one assignment minimizing total squared
    distance, solved exactly. Greedy nearest-neighbour matching is not used:
    it can double-book a centroid and overstate drift when two centroids swap.
    """
    cost = sq_dist_matrix(centers_a, centers_b)
    rows, cols = linear_sum_assignment(cost)
    return np.sqrt(cost[rows, cols]), rows, cols


def nn_spacing(centers):
    """Distance from each centroid to its nearest other centroid."""
    dist = np.sqrt(sq_dist_matrix(centers, centers))
    np.fill_diagonal(dist, np.inf)
    return dist.min(axis=1)


def coverage(xy, centers):
    """Held-out explained-variance fraction of `centers` on `xy` (clustering R²)."""
    xy = np.asarray(xy, dtype=np.float64)
    within = float(sq_dist_matrix(xy, centers).min(axis=1).mean())
    total = float(((xy - xy.mean(axis=0)) ** 2).sum(axis=1).mean())
    return 1.0 - within / total if total > 0 else 0.0


def split_half_scores(pool, k, *, rng, tolerance=TOLERANCE, exponent=EXPONENT):
    """One A/B split: fit both halves, then score coverage and reproducibility."""
    n = pool.shape[0]
    half = n // 2
    order = rng.permutation(n)
    xy_a, xy_b = pool[order[:half]], pool[order[half : 2 * half]]

    fit_seed = int(rng.integers(1 << 31))
    centers_a = fit_code_xy_kmeans(xy_a, k, seed=fit_seed)
    centers_b = fit_code_xy_kmeans(xy_b, k, seed=fit_seed)

    # Each codebook scored on the half it did not see, then averaged.
    cover = 0.5 * (coverage(xy_b, centers_a) + coverage(xy_a, centers_b))

    disp, rows, _ = matched_displacements(centers_a, centers_b)
    spacing_a, spacing_b = nn_spacing(centers_a), nn_spacing(centers_b)
    spacing = 0.5 * (spacing_a.mean() + spacing_b.mean())
    rms = float(np.sqrt((disp**2).mean()))
    repro = max(0.0, 1.0 - rms / spacing)

    # How many centroids individually stayed put, in units of their own spacing.
    survived = int((disp < tolerance * spacing_a[rows]).sum())

    return {
        "coverage": cover,
        "repro": repro,
        "score": cover * repro**exponent,
        "rms": rms,
        "spacing": float(spacing),
        "k_eff": float(survived),
        "frac_eff": survived / k,
    }


def _mean_sem(values):
    vals = np.asarray(values, dtype=float)
    sem = (
        float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else float("nan")
    )
    return float(vals.mean()), sem


def score_curves(
    pool, *, ks=K_SWEEP, n_repeats=N_REPEATS, seed=0, tolerance=TOLERANCE,
    exponent=EXPONENT,
):
    """Per-K (mean, sem) of every score over repeated disjoint A/B splits."""
    rng = np.random.default_rng(seed)
    reps = {k: [] for k in ks}
    for rep in range(n_repeats):
        for k in ks:
            reps[k].append(
                split_half_scores(
                    pool, k, rng=rng, tolerance=tolerance, exponent=exponent
                )
            )
        print(f"  repeat {rep + 1}/{n_repeats}", flush=True)
    return {
        key: np.array([_mean_sem([r[key] for r in reps[k]]) for k in ks])
        for key in reps[ks[0]][0]
    }


def _line(ax, ks, stat, *, color, label, marker="o", linewidth=2, alpha=1.0):
    ax.errorbar(
        ks,
        stat[:, 0],
        yerr=stat[:, 1],
        marker=marker,
        markersize=5,
        linewidth=linewidth,
        capsize=2.5,
        elinewidth=1,
        color=color,
        alpha=alpha,
        label=label,
        zorder=3 if linewidth > 2 else 2,
    )
    # Direct label at the right end, so identity is never colour-alone.
    ax.annotate(
        label.split(" (")[0],
        xy=(ks[-1], stat[-1, 0]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=8.5,
        color=color,
    )


def plot_scores(curve, *, monkey, ks, n_pool, n_repeats, exponent):
    best_i = int(np.nanargmax(curve["score"][:, 0]))
    best_k = ks[best_i]
    fig, ax = plt.subplots(figsize=(8.0, 5.4), layout="constrained")

    ax.axvline(best_k, color=SERIES_SCORE, linewidth=1, linestyle=":", zorder=0)
    ax.annotate(
        f"K={best_k:g}",
        xy=(best_k, 0.99),
        xycoords=ax.get_xaxis_transform(),
        xytext=(4, 0),
        textcoords="offset points",
        va="top",
        fontsize=10,
        color=SERIES_SCORE,
    )
    _line(
        ax, ks, curve["coverage"], color=SERIES_COVERAGE,
        label="coverage (held-out variance explained)", marker="s", linewidth=1.6,
        alpha=0.85,
    )
    _line(
        ax, ks, curve["repro"], color=SERIES_REPRO,
        label="reproducibility (1 − drift/spacing)", marker="^", linewidth=1.6,
        alpha=0.85,
    )
    _line(
        ax, ks, curve["score"], color=SERIES_SCORE, label="score (product)",
        linewidth=3,
    )

    ax.set_ylim(0, 1.04)
    ax.set_xlim(ks[0] - 0.6, ks[-1] + 3.4)
    ax.set_xlabel("K (codebook size)", fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel("fraction (0–1)", fontsize=9.5, color=INK_SECONDARY)
    ax.set_xticks(list(ks))
    ax.tick_params(labelsize=8.5, colors=INK_SECONDARY, length=3)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="lower left")

    exp_note = "" if exponent == 1.0 else f" | reproducibility^{exponent:g}"
    ax.set_title(
        f"{monkey} codebook: reproducible × informative (best K={best_k:g})\n"
        f"disjoint 50/50 split | {n_repeats} repeats | n={n_pool} in-maze fixation "
        f"samples | Hungarian correspondence{exp_note}\n"
        f"small K is reproducible but explains little; large K explains more but "
        f"its centroids move",
        fontsize=10,
        color=INK_PRIMARY,
    )
    save_figure(fig, "centroid_stability_better", rel_dir=f"{VISUALIZER}/{monkey}")
    plt.close(fig)
    return best_k


def print_table(curve, ks, *, monkey):
    print(f"\n{monkey} reproducible × informative (mean ± sem over repeats)")
    header = (
        f"{'K':>3} {'coverage':>13} {'repro':>13} {'score':>13} "
        f"{'rms':>7} {'spacing':>8} {'k_eff':>6}"
    )
    print(header)
    print("-" * len(header))
    for i, k in enumerate(ks):
        print(
            f"{k:>3} "
            f"{curve['coverage'][i, 0]:>6.3f}±{curve['coverage'][i, 1]:<6.3f} "
            f"{curve['repro'][i, 0]:>6.3f}±{curve['repro'][i, 1]:<6.3f} "
            f"{curve['score'][i, 0]:>6.3f}±{curve['score'][i, 1]:<6.3f} "
            f"{curve['rms'][i, 0]:>7.3f} {curve['spacing'][i, 0]:>8.3f} "
            f"{curve['k_eff'][i, 0]:>6.1f}"
        )
    best = int(np.nanargmax(curve["score"][:, 0]))
    print(
        f"  best K={ks[best]} "
        f"(score {curve['score'][best, 0]:.3f} = coverage "
        f"{curve['coverage'][best, 0]:.3f} × repro {curve['repro'][best, 0]:.3f}); "
        f"k_eff {curve['k_eff'][best, 0]:.1f}/{ks[best]} centroids survived"
    )
    return ks[best]


def plot_centroid_stability_better(
    monkey="Faure",
    *,
    ks=K_SWEEP,
    n_repeats=N_REPEATS,
    seed=0,
    max_samples=MAX_SAMPLES,
    tolerance=TOLERANCE,
    exponent=EXPONENT,
):
    eye = load_eye_data(monkey)
    behavioral = load_eye_behavioral_data(monkey)
    packed, _ = pack_trials(eye, behavioral, monkey=monkey)
    pool = codebook_xy_pool(packed, max_samples=max_samples, seed=seed)
    print(f"Pooled {pool.shape[0]} in-maze fixation samples for {monkey}", flush=True)

    curve = score_curves(
        pool, ks=ks, n_repeats=n_repeats, seed=seed, tolerance=tolerance,
        exponent=exponent,
    )
    print_table(curve, ks, monkey=monkey)
    plot_scores(
        curve, monkey=monkey, ks=ks, n_pool=pool.shape[0], n_repeats=n_repeats,
        exponent=exponent,
    )
    return curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monkey", default="Faure", choices=MONKEYS)
    parser.add_argument("--ks", type=int, nargs="+", default=list(K_SWEEP))
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--exponent", type=float, default=EXPONENT)
    args = parser.parse_args()
    plot_centroid_stability_better(
        monkey=args.monkey,
        ks=tuple(args.ks),
        n_repeats=args.n_repeats,
        seed=args.seed,
        max_samples=args.max_samples,
        tolerance=args.tolerance,
        exponent=args.exponent,
    )


if __name__ == "__main__":
    main()
