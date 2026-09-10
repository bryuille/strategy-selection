"""Output paths and CSV writing for the maze x strategy pair package.

A verbatim copy of `eye_pre_flash.corr.corr_io`, and it has to be a copy
rather than an import: ``OUT_ROOT`` is resolved from ``__file__``, so
importing corr's module would drop this package's figures and CSVs into
``eye_pre_flash/corr/out/``. Everything here lands under
``eye_pre_flash/mazestratpair/out/`` instead; figures save at dpi=300 (the
classifier convention, which corr also follows) and long-format rows go out
through one shared `csv.DictWriter` helper.

`write_rows` opens ``"w"`` and truncates, which is why every stem in this
package carries monkey, maze and K -- a stem shared between two mazes or two
K would silently overwrite rather than append.
"""

from __future__ import annotations

import csv
from pathlib import Path

OUT_ROOT = Path(__file__).resolve().parent / "out"


def out_dir(rel_dir=None):
    path = OUT_ROOT / rel_dir if rel_dir else OUT_ROOT
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, stem, *, rel_dir=None, dpi=300):
    path = out_dir(rel_dir) / f"{stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    print(f"Saved {path}")
    return path


def write_rows(rows, stem, *, rel_dir=None):
    """Write `rows` (a list of dicts, same keys) as `<stem>.csv`.

    The header comes from the first row's keys. This truncates, so every stem
    it is given must be unique per leaf -- this package writes three files per
    leaf (`results_raw_<Monkey>_maze<M>_k<K>.csv`, `nulls_...`,
    `null_draws_...`) and each carries monkey, maze and K for that reason.
    An empty `rows` still creates the file, with no header, so a leaf that
    produced nothing is visibly empty rather than silently missing.
    """
    path = out_dir(rel_dir) / f"{stem}.csv"
    if not rows:
        path.write_text("")
        print(f"Saved {path} (empty)")
        return path
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path} ({len(rows)} rows)")
    return path
