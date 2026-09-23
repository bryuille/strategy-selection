"""Where this package's figures and CSV land.

``out/<Monkey>/<variant>/maze<M>.png``
``out/<Monkey>/<variant>/results.csv``
``out/<Monkey>/codebook.png``
``out/<Monkey>/codebook_maze<M>.png``
``out/<Monkey>/strategy_maze<M>.png``

Label source (SVM), feature (binary occupancy), K (5) and the scoring method
are all fixed in this package, so none of them is a directory level. The
variant is, because the three variants would otherwise overwrite each other.
"""

from __future__ import annotations

from pathlib import Path

OUT_ROOT = Path(__file__).resolve().parent / "out"


def rel_dir(monkey, variant):
    return f"{monkey}/{variant}"


def stem(maze):
    return f"maze{maze}"


def figure_png(out_root, monkey, variant, maze):
    return out_root / rel_dir(monkey, variant) / f"{stem(maze)}.png"


def results_csv(out_root, monkey, variant):
    return out_root / rel_dir(monkey, variant) / "results.csv"


def codebook_png(out_root, monkey):
    return out_root / monkey / "codebook.png"


def codebook_maze_stem(maze):
    return f"codebook_maze{maze}"


def codebook_maze_png(out_root, monkey, maze):
    return out_root / monkey / f"{codebook_maze_stem(maze)}.png"


def strategy_stem(maze):
    return f"strategy_maze{maze}"


def strategy_png(out_root, monkey, maze):
    return out_root / monkey / f"{strategy_stem(maze)}.png"
