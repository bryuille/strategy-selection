"""Where the per-(monkey, maze) pooled sweep's figures and CSVs land.

Shares `paths.OUT_ROOT` with both the pooled analysis and the per-session
sweep, but takes its own `comp_pooled/` subtree, so nothing here can collide
with either:

``out/comp_pooled/<source>/<monkey>/maze<M>.png``   p against K, 6 curves
``out/comp_pooled/<source>/<monkey>/results.csv``   one row per (maze, K, feature, variant)
``out/comp_pooled/<source>/<monkey>/p_by_maze_k.csv``  the maze x K table
``out/comp_pooled/<source>/summary_<monkey>.md``   the same table, readable

As in `comp_paths.py`, there is no `<variant>` level: a variant is a curve on
the figure, not a directory. K is not a level either -- it is the x axis.
Scoring method is not a level either: there is only `block_means`.
"""

from __future__ import annotations

# Re-exported, not merely imported: `comp_pooled_build` takes its
# `--out-root` default from here, so this tree names its own root exactly
# once even though that root is shared with both `paths.py` and
# `comp_paths.py`.
from eye_pre_flash.maze_strategy_pairs.paths import OUT_ROOT  # noqa: F401

COMP_POOLED = "comp_pooled"


def rel_dir(source, monkey):
    return f"{COMP_POOLED}/{source}/{monkey}"


def stem(maze):
    return f"maze{maze}"


def results_csv(out_root, source, monkey):
    return out_root / rel_dir(source, monkey) / "results.csv"


def table_csv(out_root, source, monkey):
    return out_root / rel_dir(source, monkey) / "p_by_maze_k.csv"


def summary_md(out_root, source, monkey):
    return out_root / COMP_POOLED / source / f"summary_{monkey}.md"
