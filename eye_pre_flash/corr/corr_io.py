"""Output paths and CSV writing for the maze x strategy correlation package.

Mirrors ``eye_pre_flash.classifier.classifier_io``: everything lands under
``eye_pre_flash/corr/out/``, figures save at dpi=300 (the classifier
convention, since this package mirrors that tree), and long-format rows go
out through one shared `csv.DictWriter` helper instead of each caller
inlining its own.
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

    The header comes from the first row's keys, as `classifier.decoding` and
    `classifier.pairwise` each do inline for their own `results_raw.csv` --
    factored out here because this package writes several such files per leaf
    (`results_raw.csv`, `nulls.csv`, `examples.csv`, `examples_dims.csv`).
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
