"""Builders for `examples.csv` and `examples_dims.csv`.

Computed once per (label_source, feature, variant, k, space, monkey) at the
variant level, on the widest scope (`allplus`) -- input-vector skew is a
property of the feature and variant, not of which scope's sessions happen to
be kept, so writing these per scope would triple identical files.

Two files because the units differ: `examples.csv` is one row per (sampled
row, dimension) -- a long-format sample of what actual input vectors look
like; `examples_dims.csv` is one row per (cell, dimension) -- the aggregate
distribution that answers "is this heavily skewed to a few values". Forcing
both into one schema would make `dim` mean two different things and leave a
NaN-heavy column union.
"""

from __future__ import annotations

import numpy as np

from eye_pre_flash.corr import cells as cellmod

EXAMPLE_COLUMNS = (
    "label_source", "feature", "variant", "k", "space", "monkey", "scope_computed_on",
    "row_kind", "session", "trial_id", "maze", "strategy", "strategy_tag", "cell_name",
    "half", "split", "n_trials_in_row", "row_sum", "n_nonzero", "dim", "dim_label", "value",
)

DIM_COLUMNS = (
    "label_source", "feature", "variant", "k", "space", "monkey", "scope_computed_on",
    "cell_name", "n_trials", "dim", "dim_label", "frac_zero", "min", "p05", "p25",
    "median", "p75", "p95", "max", "mean", "std", "n_distinct", "top_value",
    "top_value_frac", "mean_n_nonzero_per_trial", "pc1_explained_var", "pc1_loading",
)


def _row(columns, meta, **fields):
    out = {c: "" for c in columns}
    out.update(meta)
    out.update(fields)
    return out


def _cell_masks(mazes, y):
    mazes = np.asarray(mazes, dtype=int)
    y = np.asarray(y, dtype=int)
    for maze in cellmod.MAZES:
        for strategy in cellmod.STRATEGIES:
            mask = (mazes == maze) & (y == strategy)
            if mask.any():
                yield maze, strategy, mask


def trial_example_rows(X, sessions, trials, mazes, y, *, dims, meta, n_per_cell=3, seed=0):
    """`row_kind="trial"` rows: `n_per_cell` raw trial vectors per (maze, strategy) cell."""
    rng = np.random.default_rng((seed, 1))  # tag 1: trial examples
    sessions = np.asarray(sessions).astype(str)
    trials = np.asarray(trials, dtype=int)
    rows = []
    for maze, strategy, mask in _cell_masks(mazes, y):
        idx_pool = np.flatnonzero(mask)
        chosen = rng.choice(idx_pool, size=min(n_per_cell, idx_pool.size), replace=False)
        cell_name = f"{maze}{cellmod.STRATEGY_TAG[strategy]}"
        for row_i in chosen:
            vec = X[row_i]
            row_sum = float(np.nansum(vec))
            n_nonzero = int(np.sum(vec != 0))
            for d, val in enumerate(vec):
                rows.append(
                    _row(
                        EXAMPLE_COLUMNS, meta,
                        row_kind="trial", session=str(sessions[row_i]),
                        trial_id=int(trials[row_i]), maze=maze, strategy=strategy,
                        strategy_tag=cellmod.STRATEGY_TAG[strategy], cell_name=cell_name,
                        n_trials_in_row=1, row_sum=row_sum, n_nonzero=n_nonzero,
                        dim=d, dim_label=dims[d] if dims is not None else d, value=float(val),
                    )
                )
    return rows


def half_mean_rows(X, mazes, y, *, dims, meta, seed=0):
    """`row_kind="half_mean"` rows: the split-0 (a, b) half-mean pair per cell.

    This is the vector the Pearson actually sees -- illustrative only, not
    part of the formal CV in `eye_pre_flash.corr.matrix` (which reseeds per
    session and draws `n_splits` such pairs).
    """
    rng = np.random.default_rng((seed, 2))  # tag 2: half-mean examples
    rows = []
    for maze, strategy, mask in _cell_masks(mazes, y):
        idx_pool = np.flatnonzero(mask)
        half = idx_pool.size // 2
        if half < 1:
            continue
        order = rng.permutation(idx_pool.size)
        a_idx = idx_pool[order[:half]]
        b_idx = idx_pool[order[half : 2 * half]]
        cell_name = f"{maze}{cellmod.STRATEGY_TAG[strategy]}"
        for half_name, sel in (("a", a_idx), ("b", b_idx)):
            vec = X[sel].mean(axis=0)
            row_sum = float(np.nansum(vec))
            n_nonzero = int(np.sum(vec != 0))
            for d, val in enumerate(vec):
                rows.append(
                    _row(
                        EXAMPLE_COLUMNS, meta,
                        row_kind="half_mean", maze=maze, strategy=strategy,
                        strategy_tag=cellmod.STRATEGY_TAG[strategy], cell_name=cell_name,
                        half=half_name, split=0, n_trials_in_row=half, row_sum=row_sum,
                        n_nonzero=n_nonzero, dim=d,
                        dim_label=dims[d] if dims is not None else d, value=float(val),
                    )
                )
    return rows


def _distribution_stats(block, dims, meta, cell_name, n_trials):
    rows = []
    for dcol in range(block.shape[1]):
        col = block[:, dcol]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        vals, counts = np.unique(col, return_counts=True)
        top_i = int(np.argmax(counts))
        rows.append(
            _row(
                DIM_COLUMNS, meta, cell_name=cell_name, n_trials=n_trials, dim=dcol,
                dim_label=dims[dcol] if dims is not None else dcol,
                frac_zero=float(np.mean(col == 0)), min=float(col.min()),
                p05=float(np.percentile(col, 5)), p25=float(np.percentile(col, 25)),
                median=float(np.median(col)), p75=float(np.percentile(col, 75)),
                p95=float(np.percentile(col, 95)), max=float(col.max()),
                mean=float(col.mean()), std=float(col.std(ddof=1)) if col.size > 1 else 0.0,
                n_distinct=int(vals.size), top_value=float(vals[top_i]),
                top_value_frac=float(counts[top_i] / col.size),
            )
        )
    return rows


def _vector_summary(block, meta, cell_name, n_trials):
    finite = np.isfinite(block).all(axis=1)
    vb = block[finite]
    if vb.shape[0] == 0:
        frac_zero, mean_nonzero = np.nan, np.nan
    else:
        frac_zero = float(np.mean(vb == 0))
        mean_nonzero = float(np.mean(np.sum(vb != 0, axis=1)))
    return _row(
        DIM_COLUMNS, meta, cell_name=cell_name, n_trials=n_trials, dim="__vector__",
        dim_label="__vector__", frac_zero=frac_zero, mean_n_nonzero_per_trial=mean_nonzero,
    )


def dimension_stat_rows(X, mazes, y, *, dims, meta, pc1_info=None):
    """Per-dimension distribution stats, per cell and pooled, plus the PC1 diagnostic.

    ``pc1_info``, when given (the `variants.pc1_removed` fit), is
    ``(loading_vector, explained_variance_fraction)`` -- if PC1 explains
    almost all the variance, that number is the headline result of the whole
    variant diagnostic.
    """
    rows = []
    for maze, strategy, mask in _cell_masks(mazes, y):
        cell_name = f"{maze}{cellmod.STRATEGY_TAG[strategy]}"
        rows += _distribution_stats(X[mask], dims, meta, cell_name, int(mask.sum()))
        rows.append(_vector_summary(X[mask], meta, cell_name, int(mask.sum())))
    rows += _distribution_stats(X, dims, meta, "all", X.shape[0])
    rows.append(_vector_summary(X, meta, "all", X.shape[0]))

    if pc1_info is not None:
        pc1_vec, explained = pc1_info
        rows.append(
            _row(
                DIM_COLUMNS, meta, cell_name="all", n_trials=X.shape[0], dim="__pc1__",
                dim_label="__pc1__", pc1_explained_var=float(explained),
                pc1_loading=";".join(f"{v:.4f}" for v in pc1_vec),
            )
        )
    return rows
