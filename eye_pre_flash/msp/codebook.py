"""The fixed five-state gaze codebook: screen origin plus the four maze exits.

Nothing in this package fits a codebook. `data.attractor.to_maze` warps every
trial's gaze into the same unit H -- fixation point at (0, 0), exits at
(+-1, +-1), stem top at (0, 1) -- so the five landmarks a strategy could
plausibly care about have known coordinates, and K = 5 is a statement about
the maze rather than a number chosen from the data.

Geometry worth knowing before reading any output:

* The stem top (0, 1) is at distance exactly 1.0 from the origin AND from
  both upper exits. With `ASSIGN_RADIUS = 1.0` that is a three-way tie on the
  ball boundary. The origin is listed first so `argmin` resolves such ties
  to it.
* Origin-to-exit distance is sqrt(2) < 2 * radius, so neighbouring balls
  overlap; nearest-prototype assignment handles that exactly as it does for
  fitted codebooks.
* Gaze more than `ASSIGN_RADIUS` from every prototype -- the middle of an arm
  at (+-1, 0), anything beyond an exit -- is unassigned and contributes to no
  feature. A fitted K = 5 put prototypes where gaze actually dwelt, so the
  unassigned fraction is higher here. `build --dry-run` measures it.

`ASSIGN_RADIUS` and `MAZE_SCREEN_LIM` are defined here, not imported from
`data.attractor`, so a change there cannot silently move this analysis.
"""

from __future__ import annotations

import numpy as np

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
ORIGIN_STATE = 0

# Assign a fixation only when its centroid lies within this radius of a
# prototype. Same value as `data.attractor.ASSIGN_RADIUS`, pinned locally.
ASSIGN_RADIUS = 1.0
# Unit-H plot box; warped samples outside it are off-maze.
MAZE_SCREEN_LIM = 3.0

assert CODEBOOK_XY.shape == (K, 2)
assert tuple(CODEBOOK_XY[ORIGIN_STATE]) == (0.0, 0.0), "origin must be state 0"
assert len(STATE_NAMES) == K
