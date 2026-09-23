"""Mean gaze per trial over `fix_start` to `flash_one`, aggregated by maze.

Every other pre-flash analysis in this repo integrates a *window* ending at
`fix_start` (see `eye_pre_flash/plotting/plot_io.PRE_FIX_START_MS`). This one
instead integrates the window *starting* at `fix_start` and running to
`flash_one` -- the animal is already fixating and the flash has not yet
appeared, so this is where the eyes sit once the pre-flash fixation is
established but before anything happens.

`collect_fix_start` reduces each trial to one (x, y): the mean gaze over that
window. `maze_stats` then reduces each maze's trials to one mean and SD (per
axis), across trials, of that per-trial mean, plus the SD of each trial's own
distance to that mean (`sd_r`). `maze_pvalues` compares those means pairwise,
scaling the distance between two mazes' means by the reference maze's `sd_r`.
"""

from __future__ import annotations

import math

import numpy as np

from data.builder import trial_qc_ok
from data.loader import load_eye_behavioral_data, load_eye_data

MAZES = tuple(range(1, 7))


def _behavioral_lookup(behavioral):
    return {
        (str(session), int(trial_id)): i
        for i, (session, trial_id) in enumerate(
            zip(behavioral["session"], behavioral["trial_indices_all"])
        )
    }


def collect_fix_start(monkey):
    """One (x, y, maze) per usable trial: mean gaze over `fix_start` to `flash_one`."""
    eye = load_eye_data(monkey)
    behavioral = load_eye_behavioral_data(monkey)
    beh_index = _behavioral_lookup(behavioral)

    x, y, maze = [], [], []
    for eye_i in range(len(eye["session"])):
        session = str(eye["session"][eye_i])
        trial_id = int(eye["trial_indices_all"][eye_i])
        beh_i = beh_index.get((session, trial_id))
        if not trial_qc_ok(behavioral, beh_i):
            continue
        this_maze = int(behavioral["geo_type"][beh_i])
        if this_maze not in MAZES:
            continue

        t_ms = np.asarray(eye["time"][eye_i], dtype=float) * 1000
        fix_ms = (behavioral["fix_start"][beh_i] - behavioral["geo_present"][beh_i]) * 1000
        flash_ms = (behavioral["flash_one"][beh_i] - behavioral["geo_present"][beh_i]) * 1000
        window = (t_ms >= fix_ms) & (t_ms <= flash_ms)

        x.append(np.asarray(eye["eye_x"][eye_i], dtype=float)[window].mean())
        y.append(np.asarray(eye["eye_y"][eye_i], dtype=float)[window].mean())
        maze.append(this_maze)

    return {"x": np.asarray(x), "y": np.asarray(y), "maze": np.asarray(maze, dtype=int)}


def maze_stats(trials):
    """Mean and SD, across trials, of each maze's per-trial mean gaze.

    Also `sd_r`: the SD, across that maze's trials, of each trial's own
    Euclidean distance to the maze's mean -- the spread `maze_pvalues` scales
    against.
    """
    stats = {}
    for m in MAZES:
        mask = trials["maze"] == m
        x, y = trials["x"][mask], trials["y"][mask]
        mean_x, mean_y = float(np.nanmean(x)), float(np.nanmean(y))
        r = np.hypot(x - mean_x, y - mean_y)
        stats[m] = dict(
            mean_x=mean_x,
            sd_x=float(np.nanstd(x, ddof=1)),
            mean_y=mean_y,
            sd_y=float(np.nanstd(y, ddof=1)),
            sd_r=float(np.nanstd(r, ddof=1)),
            n=int(mask.sum()),
        )
    return stats


def maze_pvalues(stats):
    """p-value of the distance between maze i's and maze j's mean gaze.

    Scales the Euclidean distance between the two means by maze j's own
    `sd_r` (how far its trials typically sit from its own mean) and reads off
    the one-sided normal tail: p = P(Z > distance / sd_r). Directional:
    p[i][j] uses maze j's sd_r, so p[i][j] != p[j][i] in general.
    """
    p = {}
    for i in MAZES:
        p[i] = {}
        for j in MAZES:
            if i == j:
                continue
            si, sj = stats[i], stats[j]
            d = math.hypot(si["mean_x"] - sj["mean_x"], si["mean_y"] - sj["mean_y"])
            z = d / sj["sd_r"]
            p[i][j] = 0.5 * math.erfc(z / math.sqrt(2))
    return p
