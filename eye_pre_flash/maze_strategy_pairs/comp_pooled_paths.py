"""Where the per-(monkey, maze) pooled sweep's figures and CSVs land.

Shares `paths.OUT_ROOT` with both the pooled analysis and the per-session
sweep, but takes its own `comp_pooled/` subtree, so nothing here can collide
with either:

``out/comp_pooled/<method>/<source>/<monkey>/maze<M>.png``   p against K, 6 curves
``out/comp_pooled/<method>/<source>/<monkey>/results.csv``   one row per (maze, K, feature, variant)
``out/comp_pooled/<method>/<source>/<monkey>/p_by_maze_k.csv``  the maze x K table
``out/comp_pooled/<method>/<source>/summary_<monkey>.md``   the same table, readable

As in `comp_paths.py`, there is no `<variant>` level: a variant is a curve on
the figure, not a directory. K is not a level either -- it is the x axis.
"""

from __future__ import annotations

# Re-exported, not merely imported: `comp_pooled_build` takes its
# `--out-root` default from here, so this tree names its own root exactly
# once even though that root is shared with both `paths.py` and
# `comp_paths.py`.
from eye_pre_flash.maze_strategy_pairs.paths import OUT_ROOT  # noqa: F401

COMP_POOLED = "comp_pooled"


def rel_dir(method, source, monkey):
    return f"{COMP_POOLED}/{method}/{source}/{monkey}"


def stem(maze):
    return f"maze{maze}"


def results_csv(out_root, method, source, monkey):
    return out_root / rel_dir(method, source, monkey) / "results.csv"


def table_csv(out_root, method, source, monkey):
    return out_root / rel_dir(method, source, monkey) / "p_by_maze_k.csv"


def summary_md(out_root, method, source, monkey):
    return out_root / COMP_POOLED / method / source / f"summary_{monkey}.md"
