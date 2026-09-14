"""Where this package's figures and CSVs land.

The repo convention (see `plotting/plot_io.save_figure` and
`plotting/similarities/paths.py`) is that each figure package owns one `out/`
declared here, so the tree template lives in exactly one place and `build.py`,
`build_pooled.py` and the stale-output scan cannot disagree about it.

``out/<source>/<monkey>/<variant>/maze<M>_<feature>.png``
``out/pooled/<source>/<monkey>/<variant>/maze<M>_<feature>.png``

`k` is deliberately **not** a directory level any more: both K = 6 and K = 12
are panels inside one file, so a `k<k>` directory would be a lie. The variant
takes that slot, which also fixes a latent bug -- the old path omitted the
variant, so a three-variant sweep would have overwritten two thirds of its own
output.

The two trees are kept isomorphic (`pooled/` just prefixes the same template)
so a path in one maps mechanically onto the other. Note `pooled` therefore
shares a namespace with the source names: a future label source called
`pooled` would collide.
"""

from __future__ import annotations

import re
from pathlib import Path

OUT_ROOT = Path(__file__).resolve().parent / "out"
POOLED_ROOT = OUT_ROOT / "pooled"

# Directories the previous single-K layout wrote, which the new tree never
# overwrites. Matched strictly -- anchored, with the source and monkey names
# enumerated -- so `--prune-stale` can never touch a variant directory, a CSV,
# or anything outside this package. The optional `pooled/` prefix catches
# `build_pooled.py`'s old tree, which sits one level deeper.
STALE_DIR_RE = re.compile(r"^(?:pooled/)?(?:dendro|svm)/(?:Faure|Nielsen)/k\d+$")

# Both depths: `<source>/<monkey>/k<k>` and `pooled/<source>/<monkey>/k<k>`.
_STALE_GLOBS = ("*/*/k*", "pooled/*/*/k*")


def rel_dir(source, monkey, variant):
    return f"{source}/{monkey}/{variant}"


def stem(maze, feature):
    return f"maze{maze}_{feature}"


def results_csv(out_root, source, monkey, variant):
    return out_root / rel_dir(source, monkey, variant) / "results_pair.csv"


def results_raw_csv(out_root, source, monkey, variant):
    return out_root / rel_dir(source, monkey, variant) / "results_raw_pair.csv"


def stale_dirs(out_root=OUT_ROOT):
    """Single-K directories from the previous layout, with their PNG counts."""
    if not out_root.exists():
        return []
    found = {}
    for pattern in _STALE_GLOBS:
        for path in out_root.glob(pattern):
            rel = path.relative_to(out_root).as_posix()
            if path.is_dir() and STALE_DIR_RE.match(rel):
                found[path] = len(list(path.glob("*.png")))
    return sorted(found.items())
