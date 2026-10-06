"""Codebooks, blocks, variants, tags, and the `out/` layout.

Five codebooks; all but ``capsules`` keep msp's origin ball at the tag's radius:

``balls``   msp's origin ball plus its four exit balls at the same radius,
            unmerged: LU / LD / origin / RU / RD (5 states). Tags ``r1``,
            ``r0.5``.
``balls3``  the retired merge of those balls: LU+LD -> ``left``, RU+RD ->
            ``right`` (3 states). Tags ``r1balls3``, ``r0.5balls3``.
``quads``   every other point inside the maze, split by quadrant into LU /
            LD / RU / RD (5 states). Tag ``r0.5quads``.
``halves``  the same region split by side into ``left`` / ``right``
            (3 states). Tag ``r0.5halves``.
``capsules`` one capsule per exit: every point within the tag's radius of the
            segment from the origin to that exit (the origin and exit balls
            joined by two straight tangents). No origin state: where the
            capsules overlap, around the origin, the nearest segment wins,
            which is the quadrant (ties as for ``quads``). LU / LD / RU / RD
            (4 states). Tag ``r0.5capsules``.

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
    extracted index ``i`` (origin, LU, LD, RU, RD) maps to (-1: the rule never
    emits it). ``rule`` is the extraction rule: msp's ``balls``, maze
    ``region``s, or exit ``capsule``s."""

    key: str
    names: tuple
    state_map: tuple
    rule: str
    label: str

    @property
    def k(self):
        return len(self.names)

    @property
    def map_array(self):
        return np.asarray(self.state_map, dtype=np.int16)


CODEBOOKS = {
    "balls": Codebook("balls", ("LU", "LD", "origin", "RU", "RD"), (2, 0, 1, 3, 4), "balls",
                      "origin ball + four exit balls (LU, LD, RU, RD)"),
    "balls3": Codebook("balls3", ("left", "origin", "right"), (1, 0, 0, 2, 2), "balls",
                       "merged balls (left, origin, right)"),
    "quads": Codebook("quads", ("LU", "LD", "origin", "RU", "RD"), (2, 0, 1, 3, 4), "region",
                      "origin ball + maze quadrants (LU, LD, RU, RD)"),
    "halves": Codebook("halves", ("left", "origin", "right"), (1, 0, 0, 2, 2), "region",
                       "origin ball + maze halves (left, right)"),
    "capsules": Codebook("capsules", ("LU", "LD", "RU", "RD"), (-1, 0, 1, 2, 3), "capsule",
                         "origin-to-exit capsules (LU, LD, RU, RD)"),
}
for _cb in CODEBOOKS.values():
    assert len(_cb.state_map) == N_STATES, _cb.key
    assert sorted(set(_cb.state_map) - {-1}) == list(range(_cb.k)), _cb.key
DEFAULT_CODEBOOK = "balls"

# Which merged state carries each monkey's hypothesised H / S preference.
TREND = {
    "Faure": {"H": "left", "S": "origin"},
    "Nielsen": {"H": "left", "S": "right"},
}

# Directory name -> msp cache block.
# ``both`` is not a cache block: binary then dwell, concatenated (2k features).
BLOCKS = {"binary": "occ_bin", "dwell": "occ_ms"}
BOTH = "both"
BLOCK_NAMES = (*BLOCKS, BOTH)
BLOCK_LABEL = {
    "binary": "binary occupancy",
    "dwell": "dwell share",
    BOTH: "binary occupancy + dwell share",
}
BLOCK_YLABEL = {
    "binary": "fraction of trials visiting",
    "dwell": "mean share of assigned dwell",
    BOTH: "binary (fraction) | dwell (share)",
}


def feature_names(codebook, block):
    """Column names of `block`'s feature matrix under `codebook`."""
    if block != BOTH:
        return tuple(codebook.names)
    return tuple(f"{n} bin" for n in codebook.names) + tuple(f"{n} dwell" for n in codebook.names)

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
ALL_MAZES = (1, 2, 3, 4, 5, 6)
POOLED = "pooled"

# Label scopes. ``svm``: msp's svm/top_ten (SVM labels, ten sessions per
# monkey). ``dendro``: the Ward-dendrogram labels (``*_strategy_choices.npz``)
# on the publication sessions only -- as specified for the paper, Nov_6_g0
# rather than msp.labels.PUBLICATION_SESSIONS' Oct_22_g0. All four are inside
# both monkeys' top ten, which is all msp's extraction covers.
LABEL_SCOPES = {
    "svm": dict(stem="strategy_svm", sessions=None, label="SVM labels, top-ten sessions"),
    "dendro": dict(
        stem="strategy_choices",
        sessions={"Faure": ("june_8_g0", "june_24_g0"), "Nielsen": ("Nov_3_g0", "Nov_6_g0")},
        label="dendrogram labels, publication sessions",
    ),
}
DEFAULT_LABELS = "svm"

# msp's window, ``[geo_present, fix_start]``; fixation onset is ~1534 ms after
# maze onset in nearly every trial.
WINDOW_LABEL = "maze onset → fixation"

# The tags `--all-tags` runs. A tag is ``r<radius>[<codebook>][_<scope>][_across]``:
# the codebook suffix is omitted for ``balls``, the scope for ``svm``. ``_across``
# pools mazes before the split-half (one 2x2 per feature and variant) instead of
# scoring each maze and pooling the scores; `--all-tags` does not run it.
TAGS = ("r1", "r0.5", "r0.5_dendro", "r0.5quads", "r0.5halves")

_TAG_RE = re.compile(
    r"^r(?P<radius>[0-9]+(?:\.[0-9]+)?)(?P<codebook>"
    # longest first, so no codebook key shadows a longer one
    + "|".join(sorted((k for k in CODEBOOKS if k != DEFAULT_CODEBOOK), key=len, reverse=True))
    + r")?(?:_(?P<scope>" + "|".join(LABEL_SCOPES) + r"))?(?P<across>_across)?$"
)


@dataclass(frozen=True)
class Tag:
    radius: float
    codebook: Codebook
    scope: str
    across: bool = False


def parse_tag(tag):
    m = _TAG_RE.match(str(tag))
    if not m:
        raise KeyError(f"unknown tag {tag!r}; e.g. {', '.join(TAGS)}")
    return Tag(
        radius=float(m["radius"]),
        codebook=CODEBOOKS[m["codebook"] or DEFAULT_CODEBOOK],
        scope=m["scope"] or DEFAULT_LABELS,
        across=bool(m["across"]),
    )


def make_tag(radius, codebook=DEFAULT_CODEBOOK, scope=DEFAULT_LABELS, across=False):
    tag = f"r{float(radius):g}" + ("" if codebook == DEFAULT_CODEBOOK else codebook)
    tag = tag if scope == DEFAULT_LABELS else f"{tag}_{scope}"
    return tag + "_across" if across else tag


for _t in TAGS:
    _p = parse_tag(_t)
    assert make_tag(_p.radius, _p.codebook.key, _p.scope) == _t, _t
assert parse_tag("r0.5quads_dendro_across").across

def default_mazes(tag):
    """Mazes run when `--maze` is not given: all six for dendro labels, else 2-5."""
    return ALL_MAZES if parse_tag(tag).scope == "dendro" else MAZES


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
