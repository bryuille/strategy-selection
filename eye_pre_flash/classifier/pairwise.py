"""Pairwise maze-identity decoding from pre-fixation gaze.

One question, and it has nothing to do with strategy: **can a classifier tell
which maze was on the screen from where the animal looked?** Every trial's
maze id is known without any neural recording, so this analysis needs no
strategy label -- it runs on every trial of the scope's sessions.

Instead of one 6-way multiclass score (whose chance level, 0.167, is awkward to
read and whose single number hides which mazes are confusable), each of the
**15 maze pairs** gets its own binary classifier: maze *i* vs maze *j*, chance
0.500, the same metric as every table in `decoding.py`. The 6x6 matrix of those
accuracies is the output. It is symmetric -- *i* vs *j* is the same problem as
*j* vs *i*. Its **diagonal** carries the split-half null: the same classifier
asked to separate one maze's own trials into two random halves, which should
land at 0.500 and so measures the floor the off-diagonal cells are read
against (see `diagonal_null`).

Features are in unit-H, the one space the pipeline fits a codebook in. A flat
matrix is the warp working as intended -- `data.attractor.to_maze` exists to
remove maze geometry, so gaze that tracked only the geometry on screen should
no longer separate one maze pair from another. This is the control for the
across-maze regime in `decoding.py`.

Reading the matrix: the diagonal should sit at 0.500 -- if it does not, the
off-diagonal cells above it are inflated by whatever structure separates one
maze's trials from each other. High off-diagonal cells mean the animal's gaze
pattern
distinguishes those two mazes, i.e. it inspected them differently. Cells should
be highest for the geometric extremes (maze 1 vs 6) and lowest for neighbours.
That gradient, not the mean, is the result -- it says gaze tracks maze geometry
in a graded way rather than merely exceeding chance somewhere.

Accuracies are uncorrected over the 15 cells; treat a single marginal cell
accordingly.
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from eye_pre_flash.classifier.classifier_io import (
    out_dir,
    render_table_png,
    save_figure,
)
from eye_pre_flash.classifier.decoding import (
    FEATURE_ROWS,
    MIN_PER_CLASS,
    N_SPLITS,
    SEED,
    _fold_std,
    fmt,
    score_groups,
)
from eye_pre_flash.classifier.features import N_MAZES, load_features
from eye_pre_flash.classifier.labels import (
    DEFAULT_SCOPE,
    MIN_LABEL_AGREEMENT,
    SCOPES,
    labels_for_rows,
    scope_sessions,
    strategy_label_lookup,
)
from eye_pre_flash.classifier.models import make_models
from eye_pre_flash.plotting.plot_io import window_label
from eye_pre_flash.plotting.similarities.common import BLUE_YELLOW

# Every feature set gets its own matrix, drawn for this model alone. One
# model keeps each matrix a plain grid of numbers -- the forest's value in
# `decoding.py` is its train/CV gap as an overfitting readout, which a matrix
# has no room to show, and it still appears in the summary table and the raw
# csv alongside the logistic.
MATRIX_MODEL = "Logistic (L2)"

MAZES = tuple(range(1, N_MAZES + 1))
PAIRS = tuple(combinations(MAZES, 2))

# Independent random halvings averaged into each diagonal cell. The null's
# expected value is 0.5 whatever this is; repeats only tighten the estimate,
# and each one costs a full CV, so this trades precision against runtime.
DIAG_REPEATS = 5



def feature_slug(block, k):
    """Filename slug for a feature row: the block, plus the K parameterising it."""
    return f"{block}_k{int(k)}"


def load_mazes(monkey, sessions, *, k, labelled_only=False):
    """Feature blocks and maze ids for every trial of `sessions`.

    Unlike `decoding.load_labelled` this keeps trials that carry no strategy
    label: maze identity is known for all of them, so dropping them would
    throw away power for nothing. `labelled_only` restores the strategy
    tables' exact trial set when the two need to be compared cell for cell.
    """
    data = load_features(monkey, k=k)
    rows = np.asarray(data["session"]).astype(str)
    keep = np.isin(rows, list(sessions))
    if labelled_only:
        y = labels_for_rows(
            rows, data["trial_indices_all"], strategy_label_lookup(sessions)
        )
        keep &= np.isfinite(y)
    out = {}
    for key, value in data.items():
        arr = np.asarray(value)
        out[key] = arr[keep] if arr.ndim >= 1 and arr.shape[0] == keep.size else value
    return out


def diagonal_null(X, mazes, factory, *, n_splits=N_SPLITS, seed=SEED,
                  repeats=DIAG_REPEATS):
    """Per-maze split-half null for the matrix diagonal.

    A maze cannot be discriminated from itself, so the diagonal's analogue of
    an off-diagonal cell is the same classifier trained to separate one maze's
    trials into two **arbitrary halves** -- the decoding counterpart of the
    split-half diagonal in `similarity.md`. Its expected value is 0.500, which
    makes it the *measured* floor the off-diagonal cells are read against
    rather than the assumed one.

    Read it as a diagnostic. A diagonal cell near 0.5 says the CV and the
    balanced-accuracy metric are behaving on this feature set. A cell well
    above 0.5 says a random half of one maze's trials is separable from the
    other half, which can only come from structure the split does not
    control -- session or day effects, or drift across the recording -- and
    that same structure inflates the off-diagonal cells by an unknown amount.

    Averaged over `repeats` independent random halvings, since a single
    labelling is a noisy estimate of the floor.
    """
    out = np.full(N_MAZES, np.nan)
    for m in MAZES:
        keep = mazes == m
        n = int(keep.sum())
        if n < 2 * MIN_PER_CLASS:
            continue
        scores = []
        for r in range(repeats):
            # Seeded per (maze, repeat) so the null is reproducible and does
            # not reuse one halving across mazes.
            rng = np.random.default_rng((seed, m, r))
            y = np.zeros(n, dtype=int)
            y[rng.permutation(n)[: n // 2]] = 1
            got = score_groups(
                [("diag", X[keep], y)], factory, n_splits=n_splits, seed=seed
            )
            if got:
                scores.append(got["cv"])
        if scores:
            out[m - 1] = float(np.mean(scores))
    return out


def pair_matrices(X, mazes, factory, *, n_splits=N_SPLITS, seed=SEED):
    """Symmetric 6x6 train, CV and trial-count matrices over the maze pairs.

    Each pair is its own binary problem -- maze *a* is class 0, maze *b* class
    1 -- scored by balanced accuracy under stratified k-fold, so chance is
    exactly 0.500 however unequal the two mazes' trial counts are. The
    diagonal stays NaN: a maze cannot be told from itself.
    """
    train = np.full((N_MAZES, N_MAZES), np.nan)
    cv = np.full((N_MAZES, N_MAZES), np.nan)
    n = np.zeros((N_MAZES, N_MAZES), dtype=int)
    folds = {}
    for a, b in PAIRS:
        keep = (mazes == a) | (mazes == b)
        if not keep.any():
            continue
        y = (mazes[keep] == b).astype(int)
        scores = score_groups(
            [("pair", X[keep], y)], factory, n_splits=n_splits, seed=seed
        )
        if scores is None:
            continue
        i, j = a - 1, b - 1
        train[i, j] = train[j, i] = scores["train"]
        cv[i, j] = cv[j, i] = scores["cv"]
        n[i, j] = n[j, i] = scores["n_trials"]
        folds[(a, b)] = scores["per_group"]["pair"]
    return train, cv, n, folds


def plot_matrix(cv, counts, *, title, stem, rel_dir, cbar_label):
    """Annotated 6x6 accuracy matrix in the `similarity.md` house style.

    Deliberately the same furniture as
    `eye_pre_flash.plotting.similarities.common.plot_cv_grid` -- same figure
    size, parula colour map, colorbar geometry, plain 10-pt multi-line title
    carrying the metadata, and 2-decimal cell annotations switched black or
    white on cell luminance -- so a pairwise decoding matrix can sit beside a
    maze-similarity matrix without the eye having to re-learn the format.
    Note the two show different quantities; only the layout is shared.

    The colour scale autoscales, as it does there. No anchoring at chance is
    needed: the diagonal holds the split-half null, which pins the low end of
    the range at 0.5 on its own.

    The one addition is the trial count under each x tick, which the
    similarity plots carry in the title instead.
    """
    finite = cv[np.isfinite(cv)]
    if finite.size:
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmin == vmax:
            vmax = vmin + 1e-12
    else:
        vmin, vmax = 0.5, 1.0

    fig, ax = plt.subplots(figsize=(6.4, 5.6), layout="constrained")
    im = ax.imshow(
        cv, origin="upper", cmap=BLUE_YELLOW, vmin=vmin, vmax=vmax,
        aspect="equal",
    )
    ax.set_xticks(
        range(N_MAZES), labels=[f"{m}\n({c})" for m, c in zip(MAZES, counts)]
    )
    ax.set_yticks(range(N_MAZES), labels=MAZES)
    ax.set_xlabel("Maze (trials)")
    ax.set_ylabel("Maze")
    ax.set_title(title, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    for i in range(N_MAZES):
        for j in range(N_MAZES):
            val = cv[i, j]
            if not np.isfinite(val):
                ax.text(j, i, "—", ha="center", va="center", color="0.45")
                continue
            r, g, b, _ = im.cmap(im.norm(val))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="black" if lum > 0.55 else "white",
                fontsize=9,
            )
    path = save_figure(fig, stem, rel_dir=rel_dir)
    plt.close(fig)
    return path


def _offdiag_mean(mat):
    """Mean over the off-diagonal cells, so the diagonal's null is excluded.

    Each of the 15 pairs is counted twice, once per triangle, which leaves the
    mean unchanged.
    """
    vals = mat[~np.eye(N_MAZES, dtype=bool)]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else None


def _offdiag_range(mat):
    vals = mat[~np.eye(N_MAZES, dtype=bool)]
    vals = vals[np.isfinite(vals)]
    if not vals.size:
        return None, None
    return float(vals.min()), float(vals.max())


def run_monkey(monkey, sessions, *, scope, n_splits, seed, labelled_only):
    """Matrices and summary for one monkey; returns raw long-format rows."""
    models = make_models()
    model_names = list(models)
    rel_dir = f"pairwise/{scope}"
    raw = []

    # Census from the k=6 cache; the row set is identical across caches.
    base = load_mazes(
        monkey, sessions, k=6, labelled_only=labelled_only
    )
    mazes_all = np.asarray(base["maze_id"], dtype=int)
    counts = [int((mazes_all == m).sum()) for m in MAZES]
    census = [
        [str(m), str(c)] for m, c in zip(MAZES, counts)
    ] + [["total", str(int(mazes_all.size))]]
    render_table_png(
        ["Maze", "Trials"],
        census,
        stem=f"counts_{monkey}",
        rel_dir=rel_dir,
        title=(
            f"{monkey} -- trials per maze (scope: {scope}): "
            f"{len(sessions)} session(s) pooled"
        ),
        footnote=(
            f"{', '.join(sessions)}. Every trial of these sessions"
            + ("" if labelled_only else ", labelled or not")
            + ". Each pairwise cell pools the two mazes' trials."
        ),
    )

    summary = []
    matrices = []  # (label, block, k, cv matrix for MATRIX_MODEL)
    for label, block, k in FEATURE_ROWS:
        data = load_mazes(
            monkey, sessions, k=k, labelled_only=labelled_only
        )
        X = np.asarray(data[block], dtype=np.float64)
        mazes = np.asarray(data["maze_id"], dtype=int)
        row = [label, str(X.shape[1])]
        cv_by_model = {}
        for model_name, factory in models.items():
            train, cv, n, folds = pair_matrices(
                X, mazes, factory, n_splits=n_splits, seed=seed
            )
            # The diagonal is the split-half null, not a discrimination: it
            # measures the floor the off-diagonal cells are read against.
            # `_offdiag_mean`/`_offdiag_range` mask it out, so it never enters
            # a mean or a range.
            diag = diagonal_null(
                X, mazes, factory, n_splits=n_splits, seed=seed
            )
            cv[np.diag_indices(N_MAZES)] = diag
            cv_by_model[model_name] = cv
            for m in MAZES:
                if not np.isfinite(diag[m - 1]):
                    continue
                raw.append(
                    dict(
                        scope=scope,
                        monkey=monkey,
                        feature_set=label,
                        model=model_name,
                        maze_a=m,
                        maze_b=m,
                        d=X.shape[1],
                        n_trials=int((mazes == m).sum()),
                        train_bacc=np.nan,
                        cv_bacc=diag[m - 1],
                        train_bacc_std=np.nan,
                        cv_bacc_std=np.nan,
                    )
                )
            lo, hi = _offdiag_range(cv)
            row += [
                fmt(_offdiag_mean(train)),
                fmt(_offdiag_mean(cv)),
                f"{fmt(lo)}-{fmt(hi)}" if lo is not None else "--",
            ]
            for a, b in PAIRS:
                i, j = a - 1, b - 1
                if not np.isfinite(cv[i, j]):
                    continue
                g = folds[(a, b)]
                raw.append(
                    dict(
                        scope=scope,
                        monkey=monkey,
                        feature_set=label,
                        model=model_name,
                        maze_a=a,
                        maze_b=b,
                        d=X.shape[1],
                        n_trials=int(n[i, j]),
                        train_bacc=train[i, j],
                        cv_bacc=cv[i, j],
                        train_bacc_std=_fold_std(g["fold_train"]),
                        cv_bacc_std=_fold_std(g["fold_cv"]),
                    )
                )
            print(
                f"  {monkey} {label:32s} {model_name:14s} "
                f"mean {fmt(_offdiag_mean(cv))}  range {fmt(lo)}-{fmt(hi)}"
            )
        summary.append(row)
        matrices.append((label, block, k, cv_by_model[MATRIX_MODEL]))

    space_note = "unit-H"

    # Title in the similarity plots' idiom: what it is, what the number means,
    # then a metadata line. Every cell value is annotated in the figure, so the
    # matrix is its own table and no numeric render is emitted beside it.
    for label, block, k, cv in matrices:
        plot_matrix(
            cv,
            counts,
            title=(
                f"{monkey} pairwise maze decoding ({label}, CV)\n"
                f"balanced accuracy, maze i vs maze j | {MATRIX_MODEL} | "
                f"scope {scope}\n"
                f"{window_label()} | {space_note} | chance 0.500 | "
                f"diagonal = split-half null ({DIAG_REPEATS} halvings)"
            ),
            stem=f"pairs_{monkey}_{feature_slug(block, k)}",
            rel_dir=rel_dir,
            cbar_label="CV balanced accuracy",
        )

    # The eight matrices each answer their own question; this collapses all of
    # them to one number apiece so the feature sets can be ranked at a glance,
    # and is where the random forest appears -- its train/CV gap is the
    # overfitting readout a matrix has no room for.
    sum_cols = ["Feature set", "d"]
    for model_name in model_names:
        sum_cols += [f"{model_name} train", f"{model_name} CV", f"{model_name} range"]
    sum_title = (
        f"{monkey} -- pairwise maze decoding, mean over the 15 pairs "
        f"({space_note}, scope {scope})\nBalanced accuracy, chance 0.500"
    )
    sum_note = (
        "Mean and min-max of each feature set's 15 pairwise CV accuracies. "
        "Per-feature matrices: pairs_<Monkey>_<feature>."
    )
    sum_stem = f"summary_{monkey}"
    breaks = {i for i in range(2, len(summary), 2)}
    render_table_png(
        sum_cols, summary, stem=sum_stem, rel_dir=rel_dir, title=sum_title,
        group_breaks=breaks, footnote=sum_note,
    )
    return raw


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scope", choices=sorted(SCOPES), default=DEFAULT_SCOPE)
    parser.add_argument(
        "--labelled-only",
        action="store_true",
        help="restrict to strategy-labelled trials, matching the decoding "
             "tables' trial set exactly (this analysis does not need labels)",
    )
    parser.add_argument("--splits", type=int, default=N_SPLITS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    by_monkey, missing, dropped = scope_sessions(args.scope)
    if missing:
        print(
            f"scope {args.scope!r}: {len(missing)} session(s) not labelled yet, "
            f"skipped: {', '.join(missing)}"
        )
    if dropped:
        print(
            f"scope {args.scope!r}: {len(dropped)} session(s) dropped for "
            f"anchor-maze label agreement below {MIN_LABEL_AGREEMENT}: "
            + ", ".join(f"{s} ({a:.2f})" for s, a in sorted(dropped.items()))
        )
    if not by_monkey:
        raise SystemExit("no labelled sessions in scope; run the labels stage first")

    raw = []
    for monkey, sessions in sorted(by_monkey.items()):
        print(f"{monkey}: {len(sessions)} session(s) in scope {args.scope!r}")
        raw += run_monkey(
            monkey,
            sessions,
            scope=args.scope,
            n_splits=args.splits,
            seed=args.seed,
            labelled_only=args.labelled_only,
        )

    path = out_dir(f"pairwise/{args.scope}") / "results_raw.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw[0].keys()))
        writer.writeheader()
        writer.writerows(raw)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
