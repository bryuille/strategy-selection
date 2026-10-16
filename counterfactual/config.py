"""counterfactual: post-choice exit sequences per trial.

After the monkey commits to an exit, where does it look next? Each trial is
reduced to a list of exit codes: item 0 is the chosen exit (decoded from the
behavioral record), every later item is an exit the monkey fixated afterwards,
in time order, repeats kept. Likelihoods against the strategy label are not
defined yet; this package only annotates the data.

Scope: the four publication sessions (two per monkey), the dendrogram-label
sessions. Codes follow DATA_DICTIONARY.md: 1 = LU, 2 = LD, 3 = RU, 4 = RD;
0 is the origin (fixation point) and -1 everything else.
"""

from __future__ import annotations

from pathlib import Path

from data.config import processed_npz

SESSIONS = {
    "Faure": ("june_8_g0", "june_24_g0"),
    "Nielsen": ("Nov_3_g0", "Nov_6_g0"),
}
MONKEYS = tuple(SESSIONS)

# ---- states -----------------------------------------------------------------
ELSEWHERE = -1
ORIGIN = 0
LU, LD, RU, RD = 1, 2, 3, 4
EXIT_CODES = (LU, LD, RU, RD)
EXIT_NAMES = ("LU", "LD", "RU", "RD")
STATE_NAMES = {ELSEWHERE: "elsewhere", ORIGIN: "origin", LU: "LU", LD: "LD", RU: "RU", RD: "RD"}

# Assignment in degrees, fixation point at the origin: a fixation centroid
# within EXIT_RADIUS_DEG of an exit is that exit (nearest wins), else within
# ORIGIN_RADIUS_DEG of the origin is the origin, else elsewhere. 3 deg is the
# exit-ball radius the classifier package uses.
EXIT_RADIUS_DEG = 3.0
ORIGIN_RADIUS_DEG = 3.0

# ---- window -----------------------------------------------------------------
# From the start of the choice saccade (first finite field in order) to the
# end of the trial, both in ms from geo_present, clipped to the eye trace.
WINDOW_START_FIELDS = ("saccade_init", "answer_time", "fixation_off")
WINDOW_END_FIELD = "trial_end"
MIN_WINDOW_SAMPLES = 6

# Pupil traces for blink masking: the per-session tables the classifier
# package extracts on the cluster, else the eye mats, else none (warned).
PUPIL_DIR = Path(__file__).resolve().parents[1] / "classifier" / "out" / "pupil"

# ---- cache / output ---------------------------------------------------------
SCHEMA = 2
OUT_ROOT = Path(__file__).resolve().parent / "out"


def cache_path(monkey):
    return processed_npz(f"{monkey}_cf_exits")
