"""Where this package's figures and CSV land.

The repo convention (see `plotting/plot_io.save_figure`) is that each figure
package owns one `out/` declared here, so the tree template lives in exactly
one place.

``out/<method>/<source>/<monkey>/<variant>/maze<M>_<feature>.png``
``out/<method>/<source>/<monkey>/<variant>/results.csv``

`<method>` is `trial_by_trial` or `block_means` (`estimator.METHODS`): the two
ways of scoring a group pairing, swept side by side over an otherwise
identical figure set, so the same panel can be compared between them.

`k` is **not** a directory level: both K = 6 and K = 12 are panels inside one
file, so a `k<k>` directory would be a lie. The variant takes that slot --
without it a three-variant sweep would overwrite two thirds of its own output.
"""

from __future__ import annotations

from pathlib import Path

OUT_ROOT = Path(__file__).resolve().parent / "out"


def rel_dir(method, source, monkey, variant):
    return f"{method}/{source}/{monkey}/{variant}"


def stem(maze, feature):
    return f"maze{maze}_{feature}"


def results_csv(out_root, method, source, monkey, variant):
    return out_root / rel_dir(method, source, monkey, variant) / "results.csv"
