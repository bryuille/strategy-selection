"""Codebooks, blocks, variants, tags, and the `out/` layout.

Three codebooks, all with msp's origin ball at the tag's radius:

``balls``   msp's four exit balls at the same radius; labels merge LU+LD ->
            ``left``, RU+RD -> ``right`` (3 states). Tags ``r1``, ``r0.5``.
``quads``   every other point inside the maze, split by quadrant into LU /
            LD / RU / RD (5 states). Tag ``r0.5quads``.
``halves``  the same region split by side into ``left`` / ``right``
            (3 states). Tag ``r0.5halves``.

"Inside the maze" is the outer edge of the four r1 exit balls joined by
straight tangents: every point within `PERIMETER_RADIUS` of the square the
exits span (a rounded square reaching +-2). Both monkeys share every codebook.

Three states, not two: with a 2-vector the Pearson correlation of two group
means is ``sign((a1 - a2) * (b1 - b2))`` -- always +-1 -- so the 2x2 would be
blind to magnitude. That is also why msp's ``no_origin`` variant is absent
here (it would leave two dimensions).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from msp import config as msp_cfg

assert tuple(msp_cfg.STATE_NAMES) == ("origin", "LU", "LD", "RU", "RD")

# Region codebooks: exits at (+-1, +-1), perimeter = the r1 exit balls' hull.
EXIT_HALF_WIDTH = 1.0
PERIMETER_RADIUS = 1.0

N_STATES = 5  # extracted state indices: msp's five


@dataclass(frozen=True)
class Codebook:
    """``names`` in display order; ``state_map[i]`` is the state that
    extracted index ``i`` (origin, LU, LD, RU, RD) maps to. ``region`` means
    the exit states are maze regions rather than balls."""

    key: str
    names: tuple
    state_map: tuple
    region: bool
    label: str

    @property
    def k(self):
        return len(self.names)

    @property
    def map_array(self):
        return np.asarray(self.state_map, dtype=np.int16)


CODEBOOKS = {
    "balls": Codebook("balls", ("left", "origin", "right"), (1, 0, 0, 2, 2), False,
                      "merged balls (left, origin, right)"),
    "quads": Codebook("quads", ("LU", "LD", "origin", "RU", "RD"), (2, 0, 1, 3, 4), True,
                      "origin ball + maze quadrants (LU, LD, RU, RD)"),
    "halves": Codebook("halves", ("left", "origin", "right"), (1, 0, 0, 2, 2), True,
                       "origin ball + maze halves (left, right)"),
}
for _cb in CODEBOOKS.values():
    assert len(_cb.state_map) == N_STATES, _cb.key
    assert sorted(set(_cb.state_map)) == list(range(_cb.k)), _cb.key
DEFAULT_CODEBOOK = "balls"

# Which merged state carries each monkey's hypothesised H / S preference.
TREND = {
    "Faure": {"H": "left", "S": "origin"},
    "Nielsen": {"H": "left", "S": "right"},
}

# Directory name -> msp cache block.
BLOCKS = {"binary": "occ_bin", "dwell": "occ_ms"}
BLOCK_NAMES = tuple(BLOCKS)
BLOCK_LABEL = {
    "binary": "binary occupancy",
    "dwell": "dwell share",
}
BLOCK_YLABEL = {
    "binary": "fraction of trials visiting",
    "dwell": "mean share of assigned dwell",
}

# ``mean_removed`` centres on the label-free maze grand mean (msp's), which
# weights strategies by n and so leaves the majority almost no shared residual.
VARIANTS = ("full", "mean_removed")

# Every maze is tested when the split-half is possible (>= 2 trials per
# strategy, so each half holds one); msp's floor of 10 is not used. A score
# whose smaller strategy has <= LOW_N trials is marked with LOW_N_MARK.
MIN_TRIALS = 2
LOW_N = 5
LOW_N_MARK = "*"
LOW_N_NOTE = f"{LOW_N_MARK} ≤ {LOW_N} trials of one strategy"

MONKEYS = ("Faure", "Nielsen")
MAZES = (2, 3, 4, 5)
POOLED = "pooled"

# Label scopes. ``svm``: msp's svm/top_ten (SVM labels, ten sessions per
# monkey). ``dendro``: the Ward-dendrogram labels (``*_strategy_choices.npz``)
# on the publication sessions only -- as specified for the paper, Nov_6_g0
# rather than msp.labels.PUBLICATION_SESSIONS' Oct_22_g0. All four are inside
# both monkeys' top ten, which is all the geofix extraction covers.
LABEL_SCOPES = {
    "svm": dict(stem="strategy_svm", sessions=None, label="SVM labels, top-ten sessions"),
    "dendro": dict(
        stem="strategy_choices",
        sessions={"Faure": ("june_8_g0", "june_24_g0"), "Nielsen": ("Nov_3_g0", "Nov_6_g0")},
        label="dendrogram labels, publication sessions",
    ),
}
DEFAULT_LABELS = "svm"

# Analysis window, fixed for the whole package: msp's ``geofix``,
# ``[geo_present, fix_start]`` (maze onset to fixation onset). Fixation onset is
# ~1534 ms after maze onset in nearly every trial.
WINDOW = "geofix"
WINDOW_LABEL = "maze onset → fixation"

# The tags `--all-tags` runs. A tag is ``r<radius>[<codebook>][_<scope>]``:
# the codebook suffix is omitted for ``balls``, the scope for ``svm``.
TAGS = ("r1", "r0.5", "r0.5_dendro", "r0.5quads", "r0.5halves")

_TAG_RE = re.compile(
    r"^r(?P<radius>[0-9]+(?:\.[0-9]+)?)(?P<codebook>"
    # longest first, so no codebook key shadows a longer one
    + "|".join(sorted((k for k in CODEBOOKS if k != DEFAULT_CODEBOOK), key=len, reverse=True))
    + r")?(?:_(?P<scope>" + "|".join(LABEL_SCOPES) + r"))?$"
)


@dataclass(frozen=True)
class Tag:
    radius: float
    codebook: Codebook
    scope: str


def parse_tag(tag):
    m = _TAG_RE.match(str(tag))
    if not m:
        raise KeyError(f"unknown tag {tag!r}; e.g. {', '.join(TAGS)}")
    return Tag(
        radius=float(m["radius"]),
        codebook=CODEBOOKS[m["codebook"] or DEFAULT_CODEBOOK],
        scope=m["scope"] or DEFAULT_LABELS,
    )


def make_tag(radius, codebook=DEFAULT_CODEBOOK, scope=DEFAULT_LABELS):
    tag = f"r{float(radius):g}" + ("" if codebook == DEFAULT_CODEBOOK else codebook)
    return tag if scope == DEFAULT_LABELS else f"{tag}_{scope}"


for _t in TAGS:
    _p = parse_tag(_t)
    assert make_tag(_p.radius, _p.codebook.key, _p.scope) == _t, _t

# ---- output paths ------------------------------------------------------------

OUT_ROOT = Path(__file__).resolve().parent / "out"


def monkey_dir(tag, monkey):
    return f"{tag}/{monkey}"


def block_dir(tag, monkey, block):
    return f"{monkey_dir(tag, monkey)}/{block}"


def variant_dir(tag, monkey, block, variant):
    return f"{block_dir(tag, monkey, block)}/{variant}"


def paper_dir(tag, block, variant):
    return f"{tag}/paper/{block}/{variant}"


def stem(maze):
    return f"maze{maze}"


def figure_png(out_root, tag, monkey, block, variant, maze):
    return out_root / variant_dir(tag, monkey, block, variant) / f"{stem(maze)}.png"


def results_csv(out_root, tag, monkey, block, variant):
    return out_root / variant_dir(tag, monkey, block, variant) / "results.csv"


# ---- figure style ------------------------------------------------------------

# Both sets validated CVD-safe in display order (dataviz validator).
STATE_COLOUR = {
    "left": "#2a9d8f", "origin": "#7b4fb5", "right": "#c77700",
    "LU": "#1f8fbf", "LD": "#2a9d5a", "RU": "#c77700", "RD": "#d0457a",
}
STRATEGY_COLOUR = msp_cfg.STRATEGY_COLOUR
CMAP = msp_cfg.CMAP
