"""counterfactual: counterfactual looks scored against choice-conditioned strategy beliefs.

See README.md. Codes follow DATA_DICTIONARY.md: 1 = LU, 2 = LD, 3 = RU,
4 = RD; 0 is the origin and -1 everything else.
"""

from __future__ import annotations

from pathlib import Path

from data.config import processed_npz

SESSIONS = {
    "Faure": ("june_8_g0", "june_24_g0"),
    "Nielsen": ("Nov_3_g0", "Nov_6_g0"),
}
MONKEYS = tuple(SESSIONS)
MODELS = ("H", "S")  # the two strategies compared against the labels (compare, logp)
# Every behavioral model ranked by counterfactual.best_model (counterfactual.noise defines each); the
# revision and lapse variants of v1's best-model table (commit 09e6fea) are left out.
ALL_MODELS = ("H", "S", "O", "P", "T", "C", "F")
MODEL_NAMES = {"H": "Hierarchical", "S": "Second-only (Sequential)", "O": "Optimal", "P": "Postdictive",
               "T": "Total Time", "C": "Total Time (convolution)", "F": "First-only"}
H_LABEL, S_LABEL = 0, 1  # dendrogram label codes

# ---- looks ------------------------------------------------------------------
ELSEWHERE, ORIGIN = -1, 0
EXIT_NAMES = ("LU", "LD", "RU", "RD")
EXIT_RADIUS_DEG = 3.0  # nearest exit within this radius (the classifier's exit-ball radius)
ORIGIN_RADIUS_DEG = 3.0
# Looks overlapping [flash_three, trial_end], ms from geo_present.
WINDOW_START_FIELD = "flash_three"
CHOICE_FIELD = "saccade_init"
LOOKS_EPOCH = "trial"

# ---- beliefs (counterfactual.beliefs) ----------------------------------------------------------
# Published mean Weber fraction per monkey (Figure One panel G, stored in
# Data/Pre_Physiology/Model_Results/Behavioral_Psychometrics/timing_scalars.mat).
WEBER = {"Faure": 0.165, "Nielsen": 0.153}
N_DRAWS = 100_000  # per (maze, chosen exit) cell
SEED = 45
EPS = 1e-3  # added to each unchosen-exit probability before renormalising

# ---- statistics -------------------------------------------------------------------
N_BOOT = 2_000

SCHEMA = 3
OUT_ROOT = Path(__file__).resolve().parent / "out"


def trials_cache(monkey):
    return processed_npz(f"{monkey}_cfv2_trials")


def q_cache(monkey):
    return processed_npz(f"{monkey}_cfv2_q")


def out_dir(monkey):
    d = OUT_ROOT / monkey
    d.mkdir(parents=True, exist_ok=True)
    return d
