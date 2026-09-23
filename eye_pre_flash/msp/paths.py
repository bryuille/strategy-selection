"""Where this package's figures and CSV land.

``out/r<radius>/<Monkey>/<variant>/maze<M>.png``
``out/r<radius>/<Monkey>/<variant>/results.csv``
``out/r<radius>/<Monkey>/codebook.png``
``out/r<radius>/<Monkey>/codebook_maze<M>.png``
``out/r<radius>/<Monkey>/strategy_maze<M>.png``

Label source (SVM), feature (binary occupancy), K (5) and the scoring method
are all fixed in this package, so none of them is a directory level. The
variant is, because the three variants would otherwise overwrite each other;
so is the assignment radius, because `--radius` changes every number and two
radii must be able to sit side by side (``r1``, ``r0.5``).
"""

from __future__ import annotations

from pathlib import Path

OUT_ROOT = Path(__file__).resolve().parent / "out"


def radius_dir(radius):
    return f"r{float(radius):g}"


def monkey_dir(radius, monkey):
    return f"{radius_dir(radius)}/{monkey}"


def rel_dir(radius, monkey, variant):
    return f"{monkey_dir(radius, monkey)}/{variant}"


def stem(maze):
    return f"maze{maze}"


def figure_png(out_root, radius, monkey, variant, maze):
    return out_root / rel_dir(radius, monkey, variant) / f"{stem(maze)}.png"


def results_csv(out_root, radius, monkey, variant):
    return out_root / rel_dir(radius, monkey, variant) / "results.csv"


def codebook_stem():
    return "codebook"


def codebook_maze_stem(maze):
    return f"codebook_maze{maze}"


def strategy_stem(maze):
    return f"strategy_maze{maze}"
