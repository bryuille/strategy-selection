"""Where this package's figures and CSV land.

The repo convention (see `plotting/plot_io.save_figure`) is that each figure
package owns one `out/` declared here, so the tree template lives in exactly
one place.

``out/<source>/<monkey>/<variant>/maze<M>_<feature>.png``
``out/<source>/<monkey>/<variant>/results.csv``

There is only one scoring method (`block_means` in `estimator.py`), so it is
not a directory level.

`k` is **not** a directory level: both K = 6 and K = 12 are panels inside one
file, so a `k<k>` directory would be a lie. The variant takes that slot --
without it a three-variant sweep would overwrite two thirds of its own output.
"""

from __future__ import annotations

from pathlib import Path

OUT_ROOT = Path(__file__).resolve().parent / "out"


def rel_dir(source, monkey, variant):
    return f"{source}/{monkey}/{variant}"


def stem(maze, feature):
    return f"maze{maze}_{feature}"


def results_csv(out_root, source, monkey, variant):
    return out_root / rel_dir(source, monkey, variant) / "results.csv"
