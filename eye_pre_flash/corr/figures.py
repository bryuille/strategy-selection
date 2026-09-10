"""The 12x12 (maze x strategy) heatmap.

Carries over the deleted `heatmap_labels.plot_labels`' furniture verbatim
(figure size, parula colormap, quadrant split, colorbar geometry,
luminance-switched annotations) -- see `similarity.md`'s
house style and `eye_pre_flash.classifier.pairwise.plot_matrix`'s docstring,
which both describe the same convention. What changed is listed in the
module-level comment on `plot_label_matrix`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from eye_pre_flash.corr import cells as cellmod
from eye_pre_flash.corr.corr_io import save_figure
from eye_pre_flash.plotting.plot_io import window_label
from eye_pre_flash.plotting.similarities.common import BLUE_YELLOW

FEATURE_LABEL = {
    "occupancy": "occupancy",
    "occupancy_bin": "binary occupancy",
    "bigram": "state-bigram",
}
# The unit a half-mean carries, per feature. These name the unit only -- the
# operation is identical for all three (`matrix.session_cv_matrix` takes an
# (n, d) block and has no feature branch), so the figure states the operation
# once and the unit in parentheses rather than describing three methods.
FEATURE_UNIT = {
    "occupancy": "seconds per state",
    "occupancy_bin": "visit fraction per state",
    "bigram": "bigram proportion",
}
SOURCE_LABEL = {"dendro": "dendrogram", "svm": "SVM"}


def cell_thin_mask(result):
    """Per-cell bool: True if the cell was thin (below `min_stable`) in most
    of the sessions it appeared in, or never appeared at all."""
    n = cellmod.N_CELLS
    stable_count = np.zeros(n)
    seen_count = np.zeros(n)
    for stable in result.stable:
        for idx, is_stable in stable.items():
            seen_count[idx] += 1
            stable_count[idx] += int(is_stable)
    thin = np.ones(n, dtype=bool)
    have = seen_count > 0
    thin[have] = (stable_count[have] / seen_count[have]) < 0.5
    return thin


def plot_label_matrix(
    result,
    *,
    monkey,
    scope,
    source,
    feature,
    variant,
    k,
    space,
    dim,
    n_splits,
    min_stable,
    stem,
    rel_dir,
):
    """The 12x12 figure. `result` is a `matrix.LabelSimilarityResult`.

    Changes from the deleted `heatmap_labels.plot_labels`:

    1. Title carries feature/source/variant/space instead of bin width, and
       prints ``d = {dim}`` explicitly -- a Pearson over 6-144 points is the
       headline caveat here and the figure has to admit it.
    2. Uses `plot_io.window_label` instead of re-deriving the end-of-window
       phrasing (the deleted module duplicated that conditional).
    3. The `*` mark means "thin" (below `--min-stable` in most of the
       sessions it appeared in), not "off the comparable core" -- that
       exhaustive-subset restriction was deliberately dropped from plotting
       (still reconstructable from `results_raw_k<K>.csv`).
    4. SVM source: the anchor-minority cells (1S, 6H) are blank by design, not
       for lack of coverage -- the footer says so.
    5. Always emitted, however thin the coverage -- no skip path.
    6. Saved at dpi=300 through `corr_io.save_figure` (this package mirrors
       the classifier tree's dpi, not the plotting tree's).
    7. No per-cell boxes on the same-maze `(m, H)` vs `(m, S)` pairs. Those
       are still the read of record (`corr.md`), but as one bold cell per
       maze they drew a second diagonal through the off-diagonal quadrants
       that read as structure in the data. Their addresses are fixed and
       given by `cells.cell_index(maze, strategy)`; the quadrant split lines
       are enough to locate them.
    """
    n = cellmod.N_CELLS
    names = cellmod.cell_labels()
    corr = result.mean
    thin = cell_thin_mask(result)
    star = thin[:, None] | thin[None, :]

    finite = corr[np.isfinite(corr)]
    if finite.size:
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmin == vmax:
            vmax = vmin + 1e-12
    else:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(8.6, 7.4), layout="constrained")
    im = ax.imshow(corr, origin="upper", cmap=BLUE_YELLOW, vmin=vmin, vmax=vmax, aspect="equal")
    tick_names = [f"{name}*" if thin[i] else name for i, name in enumerate(names)]
    ax.set_xticks(range(n), labels=tick_names, fontsize=8)
    ax.set_yticks(range(n), labels=tick_names, fontsize=8)
    ax.set_xlabel("Maze x decoded strategy")
    ax.set_ylabel("Maze x decoded strategy")

    split = len(cellmod.MAZES) - 0.5
    ax.axhline(split, color="black", lw=1.6)
    ax.axvline(split, color="black", lw=1.6)

    unit_label = FEATURE_UNIT.get(feature, feature)
    feature_label = FEATURE_LABEL.get(feature, feature)
    source_label = SOURCE_LABEL.get(source, source)

    footer_bits = [f"d = {dim}", f"{n_splits} splits"]
    if result.n_degenerate:
        footer_bits.append(f"{result.n_degenerate} degenerate (zero-variance) split-halves")
    skip_note = ""
    if source == "svm":
        skip_note = " | 1S, 6H blank: skipped as the SVM's anchor-minority cells, not for lack of coverage"

    ax.set_title(
        f"{monkey} pre-fixation {feature_label} similarity by {source_label} strategy "
        f"(K={k}, {space}, {variant}, CV, balanced n) | scope {scope}\n"
        f"split-half Pearson r of cell half-means ({unit_label}) | "
        f"{result.n_sessions} sessions{skip_note}\n"
        f"{window_label()} | {', '.join(footer_bits)}",
        fontsize=9,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("correlation coefficient")

    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            if not np.isfinite(val):
                ax.text(j, i, "—", ha="center", va="center", color="0.4", fontsize=6)
                continue
            r, g, b, _ = im.cmap(im.norm(val))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(
                j, i,
                f"{val:.2f}".lstrip("0") + ("*" if star[i, j] else ""),
                ha="center", va="center",
                color="black" if lum > 0.55 else "white",
                fontsize=6,
            )

    # `supxlabel`, not `fig.text`: constrained layout reserves vertical space
    # for a supxlabel but not for free-floating figure text, which put this
    # note on top of the x tick labels and the axis label.
    fig.supxlabel(
        f"* = fewer than {min_stable} trials in a cell in most sessions, high variance, "
        f"not comparable with unmarked entries",
        x=0.01, ha="left", fontsize=6, color="0.3",
    )

    save_figure(fig, stem, rel_dir=rel_dir)
    plt.close(fig)
    return fig
