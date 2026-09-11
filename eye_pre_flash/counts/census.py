"""H/S trial census per (session, maze), as a heatmap, for either label source.

One cell per (session, maze): the text is ``H/S`` -- hierarchical and
sequential trial counts -- and the fill is the sequential share ``S / (H + S)``,
so blue is an all-hierarchical cell, orange an all-sequential one, and grey a
balanced one. A cell whose minority strategy falls below `MIN_MINORITY_SHARE`
is printed red and bold: it has counts, but not on both sides, so it cannot
support a within-cell H-vs-S contrast.

The trial pool is exactly `eye_pre_flash.classifier.decoding.load_labelled`'s:
rows of the windowed feature table (k=6, unit-H -- the row set is the same
across caches) whose trial carries a finite label from the chosen source. So
each column of this figure sums to that scope's row in
`classifier/out/decoding/<scope>/counts_<monkey>.png`, and the two figures
cannot disagree.

Two label sources, taken from `eye_pre_flash.label_sources.SOURCE_STEM` rather
than re-spelled here:

``dendro``  Ward-clustering labels (`data.labeler.build_strategy_choices`).
``svm``     maze-1-vs-6 linear SVM labels (`data.labeler.build_svm_choices`).

**The SVM's mazes 1 and 6 are near-single-strategy by construction.** Those two
mazes are the classifier's own training axis, so it labels them almost purely
hierarchical and sequential respectively -- `corr.labels.SOURCE_SKIP_CELLS`
drops the `1S`/`6H` cells outright for this reason. Expect the red-bold marking
to fill the maze-1 and maze-6 columns of the ``svm`` figure: that is the label
source's definition showing through, not a coverage problem, and it is why
`corr` gives the SVM no anchor-vetted scope.

Usage:
    uv run python -m eye_pre_flash.counts.census --source svm
    uv run python -m eye_pre_flash.counts.census --source dendro --monkey Nielsen
    uv run python -m eye_pre_flash.counts.census --source svm --scope all

Writes ``eye_pre_flash/counts/<source>/hs_<monkey>_<scope>.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from data.labeler import CLUSTERING_SESSIONS
from eye_pre_flash.classifier.features import load_features
from eye_pre_flash.classifier.labels import (
    PUBLICATION_SESSIONS,
    labels_for_rows,
    scope_sessions,
    strategy_label_lookup,
)
from eye_pre_flash.label_sources import SOURCE_STEM, SOURCES

OUT_ROOT = Path(__file__).resolve().parent
N_MAZES = 6

# The census cache. `decoding.run_monkey` reads the same one and says why: the
# feature row set is identical across k and space, so the counts do not depend
# on this choice.
CENSUS_K = 6
CENSUS_SPACE = "unith"

# A cell needs both strategies to support a within-cell contrast. Below this
# minority share it is marked rather than dropped -- the count is still real
# and still worth reading, it just cannot carry the comparison.
MIN_MINORITY_SHARE = 0.10

# Diverging blue -> light grey -> orange, matched to the figure this one
# reproduces. Deliberately not `BLUE_YELLOW` (the sequential parula the
# similarity heatmaps use): the quantity here is a *balance* around 0.5, not a
# magnitude, so it wants a midpoint the eye can find, and 0/1 are two named
# strategies rather than a low and a high.
SHARE_CMAP = LinearSegmentedColormap.from_list(
    "hs_share",
    ["#2c6fbb", "#8bb0d8", "#ececec", "#e8b17e", "#d9791f"],
    N=256,
)
NO_DATA_COLOR = "#ffffff"


def source_sessions(source, scope):
    """``monkey -> sessions`` carrying labels of `source` inside `scope`.

    Vetting is `classifier.labels.scope_sessions`'s, with one exception copied
    from `corr.labels.source_scope_sessions`: the SVM source is never
    anchor-vetted, because grading a classifier trained on mazes 1 and 6
    against a map that calls mazes 1 and 6 hierarchical/sequential is circular.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown label source {source!r}; choose from {SOURCES}")
    if source == "svm":
        return scope_sessions(
            scope, stem=SOURCE_STEM[source], vet=False, sessions=CLUSTERING_SESSIONS
        )
    return scope_sessions(scope, stem=SOURCE_STEM[source])


def census_counts(monkey, sessions, source):
    """``(sessions, counts)`` where counts is ``(n_sessions, N_MAZES, 2)``.

    Same pool as `decoding.load_labelled`: feature rows for `sessions` whose
    trial has a finite label from `source`. Last axis is (hierarchical,
    sequential).
    """
    sessions = tuple(sorted(sessions))
    data = load_features(monkey, k=CENSUS_K, space=CENSUS_SPACE)
    rows = np.asarray(data["session"]).astype(str)
    y = labels_for_rows(
        rows,
        data["trial_indices_all"],
        strategy_label_lookup(sessions, stem=SOURCE_STEM[source]),
    )
    mazes = np.asarray(data["maze_id"], dtype=int)
    keep = np.isin(rows, list(sessions)) & np.isfinite(y)
    rows, y, mazes = rows[keep], y[keep].astype(int), mazes[keep]

    counts = np.zeros((len(sessions), N_MAZES, 2), dtype=int)
    for i, session in enumerate(sessions):
        in_session = rows == session
        for maze in range(1, N_MAZES + 1):
            cell = in_session & (mazes == maze)
            counts[i, maze - 1, 0] = int((y[cell] == 0).sum())
            counts[i, maze - 1, 1] = int((y[cell] == 1).sum())
    return sessions, counts


def sequential_share(counts):
    """``S / (H + S)`` per cell, NaN where the cell has no trials at all."""
    total = counts.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(total > 0, counts[:, :, 1] / total, np.nan)
    return share, total


def thin_mask(counts):
    """True where the minority strategy is under `MIN_MINORITY_SHARE`.

    An empty cell is not "thin" -- it is absent, and is left unmarked so the
    marking keeps meaning "counts, but one-sided".
    """
    total = counts.sum(axis=2)
    minority = counts.min(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(total > 0, minority / total, np.nan)
    return (total > 0) & (frac < MIN_MINORITY_SHARE)


def plot_census(
    monkey, scope, source, sessions, counts, *, title=None, out_root=None, highlight_sessions=None
):
    share, total = sequential_share(counts)
    thin = thin_mask(counts)

    fig, ax = plt.subplots(
        figsize=(2.0 + 1.55 * N_MAZES, 2.6 + 0.62 * len(sessions))
    )
    cmap = SHARE_CMAP.copy()
    cmap.set_bad(NO_DATA_COLOR)
    im = ax.imshow(
        np.ma.masked_invalid(share),
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )

    for i in range(len(sessions)):
        for j in range(N_MAZES):
            h, s = counts[i, j]
            ax.text(
                j,
                i,
                f"{h}/{s}",
                ha="center",
                va="center",
                fontsize=12,
                color="#b00020" if thin[i, j] else "#111111",
                fontweight="bold" if thin[i, j] else "normal",
            )

    ax.set_xticks(range(N_MAZES), [str(m) for m in range(1, N_MAZES + 1)])
    ax.set_yticks(range(len(sessions)), list(sessions))
    ax.set_xlabel("Maze")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    # One full-width box per highlighted session's row -- not necessarily
    # contiguous, since `sessions` is alphabetical and the publication set
    # spans both `june_*` and `Nov_*`/`Oct_*` names. Black rather than red:
    # red already means "one-sided cell" (`thin_mask`) at the per-cell level,
    # and this is an unrelated, row-level distinction.
    if highlight_sessions:
        wanted = set(highlight_sessions)
        for i, session in enumerate(sessions):
            if session in wanted:
                ax.add_patch(Rectangle(
                    (-0.5, i - 0.5), N_MAZES, 1,
                    fill=False, edgecolor="#111111", linewidth=2.4, zorder=6,
                ))

    ax.set_title(
        title if title is not None else (
            f"{monkey} · scope {scope} · source {source} "
            "— H/S per session and maze\n"
            "blue = all hierarchical, orange = all sequential, grey = balanced\n"
            f"red bold = minority strategy below {MIN_MINORITY_SHARE:.0%}, so that "
            "(session, maze) cannot support the A contrast"
        ),
        fontsize=13,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("sequential share")
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    fig.tight_layout()
    out_dir = (out_root if out_root is not None else OUT_ROOT) / source
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"hs_{monkey}_{scope}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}  ({total.sum()} trials, {len(sessions)} session(s))")
    n_cells = thin.size
    print(
        f"  one-sided cells (minority < {MIN_MINORITY_SHARE:.0%}): "
        f"{int(thin.sum())}/{n_cells}"
    )
    per_maze = counts.sum(axis=0)
    for maze in range(1, N_MAZES + 1):
        h, s = per_maze[maze - 1]
        print(f"  maze {maze}: {h} H / {s} S  (sequential share {s / (h + s):.3f})")
    grand = counts.sum(axis=(0, 1))
    print(f"  total: {grand[0]} H / {grand[1]} S")
    return path


def run(monkey, scope, source):
    by_monkey, missing, dropped = source_sessions(source, scope)
    sessions = by_monkey.get(monkey, ())
    if not sessions:
        raise SystemExit(
            f"no {source} labels for {monkey} in scope {scope}"
            + (f"; missing labels: {', '.join(missing)}" if missing else "")
        )
    if missing:
        print(f"  unlabelled sessions skipped: {', '.join(sorted(missing))}")
    if dropped:
        print(
            "  vetting dropped: "
            + ", ".join(f"{s} ({a:.3f})" for s, a in sorted(dropped.items()))
        )
    sessions, counts = census_counts(monkey, sessions, source)
    return plot_census(monkey, scope, source, sessions, counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="svm", choices=SOURCES)
    parser.add_argument("--monkey", default="Faure", choices=("Faure", "Nielsen"))
    parser.add_argument(
        "--scope",
        default="allplus",
        choices=("publication", "all", "allplus"),
        help="`classifier.labels` scope; allplus is unvetted (default)",
    )
    args = parser.parse_args()
    run(args.monkey, args.scope, args.source)


if __name__ == "__main__":
    main()
