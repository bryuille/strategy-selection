"""Where the per-(session, maze) sweep's figures and CSVs land.

Shares `paths.OUT_ROOT` with the pooled analysis but takes the whole `comp/`
subtree under it, so nothing here can collide with a pooled output:

``out/comp/<method>/<source>/<monkey>/maze<M>.png``   p against K, 6 curves
``out/comp/<method>/<source>/<monkey>/results.csv``   every per-session estimate
``out/comp/<method>/<source>/<monkey>/p_by_maze_k.csv``  the maze x K table
``out/comp/<method>/<source>/summary_<monkey>.md``   the same table, readable

Unlike the pooled tree there is no `<variant>` level: a variant is a curve on
the figure, not a directory, because the point of the figure is to compare
them at a glance. K is not a level either, for the same reason as in
`paths.py` -- it is the x axis.
"""

from __future__ import annotations

# Re-exported, not merely imported: `comp_build` takes its `--out-root`
# default from here, so the comp tree names its own root exactly once even
# though that root is shared with the pooled analysis.
from eye_pre_flash.maze_strategy_pairs.paths import OUT_ROOT  # noqa: F401

COMP = "comp"


def rel_dir(method, source, monkey):
    return f"{COMP}/{method}/{source}/{monkey}"


def stem(maze):
    return f"maze{maze}"


def results_csv(out_root, method, source, monkey):
    return out_root / rel_dir(method, source, monkey) / "results.csv"


def table_csv(out_root, method, source, monkey):
    return out_root / rel_dir(method, source, monkey) / "p_by_maze_k.csv"


def summary_md(out_root, method, source, monkey):
    return out_root / COMP / method / source / f"summary_{monkey}.md"
