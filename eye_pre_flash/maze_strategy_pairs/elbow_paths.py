"""Where the elbow-selected-K analysis' figures and CSVs land.

Shares `paths.OUT_ROOT` with the pooled analysis and both K sweeps, but takes
its own `elbow/` subtree, so nothing here can collide with any of them.

The tree has two halves, because this analysis has two stages that depend on
different things:

``out/elbow/<source>/<monkey>/ev_curves.png``            held-out EV against K, one line per maze
``out/elbow/<source>/<monkey>/selected_k.csv``           K per maze, all three rules, diagnostics

``out/elbow/<method>/<source>/<monkey>/maze<M>.png``     the 2x2 panel at the selected K
``out/elbow/<method>/<source>/<monkey>/results.csv``     one row per (maze, feature, variant)
``out/elbow/<method>/<source>/summary_<monkey>.md``      the same, readable

`<method>` is absent from the selection half on purpose. Selecting K is
label-blind *and* estimator-blind -- it never calls `estimator.estimate`, so
`trial_by_trial` and `block_means` would receive byte-identical files. Writing
them under a `<method>` level would mean writing the same numbers twice and
inviting a reader to diff two files that cannot differ. The estimate half does
depend on the method, so it keeps that level, exactly as `comp_pooled_paths`
does.

As in `comp_pooled_paths.py`, there is no `<variant>` level -- a variant is a
row in the table, not a directory -- and K is not a level either. Here K is not
even an axis: it is one selected number per maze, reported in `selected_k.csv`.
"""

from __future__ import annotations

# Re-exported, not merely imported: `elbow_build` takes its `--out-root`
# default from here, so this tree names its own root exactly once even though
# that root is shared with `paths.py`, `comp_paths.py` and
# `comp_pooled_paths.py`.
from eye_pre_flash.maze_strategy_pairs.paths import OUT_ROOT  # noqa: F401

ELBOW = "elbow"


def select_dir(source, monkey):
    """The method-independent half: everything about choosing K."""
    return f"{ELBOW}/{source}/{monkey}"


def rel_dir(method, source, monkey):
    """The method-dependent half: the estimate at the selected K."""
    return f"{ELBOW}/{method}/{source}/{monkey}"


def stem(maze, feature, variant):
    """All three in the filename, because none of them is a directory level.

    `comp_pooled_paths` gets away with a bare `maze<M>` because its figure has
    a K axis and can carry all six (feature, variant) curves at once. Here K is
    one selected number, so the figure is a single 2x2 panel and each
    (feature, variant) needs its own file.
    """
    return f"maze{maze}_{feature}_{variant}"


EV_STEM = "ev_curves"


def selected_k_csv(out_root, source, monkey):
    return out_root / select_dir(source, monkey) / "selected_k.csv"


def results_csv(out_root, method, source, monkey):
    return out_root / rel_dir(method, source, monkey) / "results.csv"


def summary_md(out_root, method, source, monkey):
    return out_root / f"{ELBOW}/{method}/{source}" / f"summary_{monkey}.md"
