"""Package configuration: sessions, windows, codebook, model settings, paths.

Nothing here is fitted. The codebook is the fixed five-state one from
`msp/config.py` (origin + the four unit-H exits), copied so this package
depends on nothing but the shared `data/` layer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ---- scope -------------------------------------------------------------------

SESSIONS = {
    "Faure": ("june_24_g0", "june_8_g0"),
    "Nielsen": ("Nov_6_g0", "Nov_3_g0"),
}
MONKEYS = tuple(SESSIONS)
N_MAZES = 6

# ---- windows -----------------------------------------------------------------
# Each maps a trial's fix_start (ms from geo_present) to (lo, hi), also ms from
# geo_present, and goes through the detect -> warp -> assign path.


def _geofix(fix_ms):
    return 0.0, float(fix_ms)


WINDOWS = {
    "geofix": _geofix,  # [geo_present, fix_start], variable length
}
WINDOW_NAMES = tuple(WINDOWS)

# ---- codebook / assignment (copied from msp/config.py) -----------------------

CODEBOOK_XY = np.array(
    [
        [0.0, 0.0],  # origin: fixation point, foot of the stem
        [-1.0, 1.0],  # LU: left-up exit
        [-1.0, -1.0],  # LD: left-down exit
        [1.0, 1.0],  # RU: right-up exit
        [1.0, -1.0],  # RD: right-down exit
    ],
    dtype=np.float32,
)
STATE_NAMES = ("origin", "LU", "LD", "RU", "RD")
K = 5
ASSIGN_RADIUS = 1.0
MAZE_SCREEN_LIM = 3.0  # unit-H box; warped samples outside are off-maze
MIN_WINDOW_SAMPLES = 6  # clean_eye_data's per-trial floor before detection

assert CODEBOOK_XY.shape == (K, 2)
assert len(STATE_NAMES) == K

# ---- model -------------------------------------------------------------------

DEFAULT_WINDOW = "geofix"  # geo_present -> fix_start; variable length, so a window-length term is fit
MIN_PER_LABEL = 5  # a maze needs this many of each label to keep its offset
VISIT_RATE_RANGE = (0.05, 0.95)  # states outside are dropped from the model
ZERO_SD = 1e-9  # columns with smaller SD are dropped
# A visits column needs this many visited trials off the modal count (almost
# always 1 at the exits in a ~1.5 s window); fewer and a handful of revisit
# trials can perfectly predict the label (quasi-separation, beta -> infinity).
MIN_OFF_MODE = 10
# A column whose variance is more than this fraction explained by the earlier
# predictors of the same state (order: PREDICTORS), or whose leftover variation
# comes from fewer than MIN_OFF_MODE trials, is dropped and the reason recorded:
# it would not add a separately estimable effect.
MAX_R2 = 0.9
ALPHA = 0.05
SEED = 0

# The three predictors, in the order they are tried (and drawn): (key, label).
PREDICTORS = (
    ("occupied", "bin occupancy"),
    ("visits", "visit count"),
    ("duration", "mean fixation duration"),
)

# ---- paths -------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
OUT_DIR = PACKAGE_DIR / "out"


def radius_tag(radius=ASSIGN_RADIUS):
    return f"r{float(radius):g}"


def cache_stem(monkey, window, radius=ASSIGN_RADIUS):
    """Stem under ``data/processed/`` for the fixation table (msp-style)."""
    return f"{monkey}_reg_fixed{K}_{radius_tag(radius)}_{window}"


def out_dir(window):
    return OUT_DIR / window
