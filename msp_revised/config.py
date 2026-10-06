"""msp_revised: scope, window, variants, tags, output layout, figure style.

Self-contained: imports only `data/` and the standard scientific stack. Logic
borrowed from `msp` / `msp_trends` is copied in, never imported.

Scope is fixed and is not a CLI axis:

* the four publication sessions (two per monkey), Ward-dendrogram labels
  (``<session>_strategy_choices.npz``: 0 = hierarchical H, 1 = sequential S);
* all six mazes, each tested on its own, then pooled two ways (see `build`);
* the msp window: maze onset (``geo_present``) to fixation onset (``fix_start``).

Codebooks live in `codebooks.py`; the per-trial feature kinds in
`features.trial_features`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ---- scope -------------------------------------------------------------------

SESSIONS = {
    "Faure": ("june_8_g0", "june_24_g0"),
    "Nielsen": ("Nov_3_g0", "Nov_6_g0"),
}
MONKEYS = tuple(SESSIONS)
LABEL_STEM = "strategy_choices"  # Ward dendrogram labels (data.labels.build_strategy_choices)
LABEL_LABEL = "dendrogram labels, publication sessions"
MAZES = (1, 2, 3, 4, 5, 6)
N_MAZES = 6

# ---- window ------------------------------------------------------------------


def window_bounds(fix_ms):
    """``(lo, hi)`` in ms from geo_present: maze onset to fixation onset."""
    return 0.0, float(fix_ms)


WINDOW_LABEL = "maze onset → fixation"

# ---- geometry ----------------------------------------------------------------

# The maze is an H whose crossbar runs through the fixation point: in degrees
# the crossbar is the segment (-h1, 0) -> (h4, 0), the left bar is x = -h1 for
# y in [-h3, h2], the right bar x = h4 for y in [-h6, h5]. Landmarks:
#   origin (0, 0), LU (-h1, +h2), LD (-h1, -h3), RU (+h4, +h5), RD (+h4, -h6).
# `data.builder.to_maze` maps the exits to (±1, ±1); its vertical scale at
# x = 0 is STEM_DEG, so (0, STEM_DEG) deg is (0, 1) unit H. Gaze clusters
# there too ("top" in the balls6 codebooks); it is not a maze segment.
# Across the six mazes only h4 (10, 7.5, 5) and the h2/h5 swap (7/9 vs 9/7)
# change, so unit-H stretches x on the right side by up to 2x between mazes
# 1-2 and 5-6, where the strategy mix also flips. Degree-frame codebooks are
# the control for that.
STEM_DEG = 7.0
UNIT_LIM = 3.0  # unit-H plot box; warped samples outside are off-maze
DEG_LIM = 20.0

# ---- variants ----------------------------------------------------------------

# full          the feature matrix as extracted
# mean_removed  minus the maze's label-free grand mean (msp's)
VARIANTS = ("full", "mean_removed")

# ---- estimator thresholds ----------------------------------------------------

MIN_TRIALS = 2  # per strategy; below this the split-half is impossible
LOW_N = 5  # scores with <= LOW_N of one strategy are marked
LOW_N_MARK = "*"
LOW_N_NOTE = f"{LOW_N_MARK} ≤ {LOW_N} trials of one strategy"
SOLID_N = 10  # per-maze replication is judged on mazes with >= SOLID_N of each

# ---- pupil -------------------------------------------------------------------

# Per-session pupil tables, written by `pupil_extract` (cluster, from the eye
# mats) and pulled down with the figures: <session>_pupil.npz under
# OUT_ROOT/pupil/. `features` reads them before trying the mats.
PUPIL_EARLY_MS = 300.0  # first 300 ms of the window (maze onset)
PUPIL_LATE_MS = 300.0  # last 300 ms before fixation onset
PUPIL_MIN_VALID = 0.5  # fraction of window samples needed for a trial value
PUPIL_FEATURES = ("pupil_mean", "pupil_change")

# ---- output paths ------------------------------------------------------------

OUT_ROOT = Path(__file__).resolve().parent / "out"
PUPIL_DIR = OUT_ROOT / "pupil"
SEARCH_DIR = OUT_ROOT / "search"


def tag(codebook_key, kind, pupil):
    """Out-dir token: ``<codebook>__<kind>[+pupil]``."""
    return f"{codebook_key}__{kind}" + ("+pupil" if pupil else "")


def paper_dir(tag_, variant):
    return f"{tag_}/paper/{variant}"


def results_csv(out_root, tag_):
    return out_root / tag_ / "results.csv"


# ---- figure style ------------------------------------------------------------

STRATEGY_COLOUR = {"H": "#1f77b4", "S": "#d62728"}
FIG_W, FIG_H = 7.2, 5.2
CBAR_LABEL = "r between group means"

_PARULA_RGB = [
    [0.2422, 0.1504, 0.6603], [0.2810, 0.3228, 0.9579], [0.1786, 0.5289, 0.9682],
    [0.0689, 0.6948, 0.8394], [0.2161, 0.7843, 0.5923], [0.6720, 0.7793, 0.2227],
    [0.9970, 0.7659, 0.2199], [0.9769, 0.9839, 0.0805],
]
BLUE_YELLOW = LinearSegmentedColormap.from_list("parula8", _PARULA_RGB, N=256)
CMAP = BLUE_YELLOW.with_extremes(bad="0.92")  # `.with_extremes` copies

# State colours (cycled when a codebook has more states than this list).
STATE_COLOURS = (
    "#7b4fb5", "#1f8fbf", "#2a9d5a", "#c77700", "#d0457a", "#555555",
    "#8c564b", "#17becf", "#bcbd22", "#e377c2", "#7f7f7f", "#9467bd",
)


def state_colour(i):
    return STATE_COLOURS[int(i) % len(STATE_COLOURS)]


assert len(VARIANTS) == 2
assert all(len(v) == 2 for v in SESSIONS.values())
_ = np  # numpy kept for callers that expect it from here
