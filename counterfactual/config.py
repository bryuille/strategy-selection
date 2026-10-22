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
# end of the `trial` looks epoch (trial_end), both in ms from geo_present.
WINDOW_START_FIELDS = ("saccade_init", "answer_time", "fixation_off")
WINDOW_END_FIELD = "trial_end"

# ---- choice models (counterfactual.models) ------------------------------------
# Published mean Weber fraction per monkey: `MR_AG_MJ/load_timing_scalar.m`
# reading `Data/Pre_Physiology/Model_Results/Behavioral_Psychometrics/
# timing_scalars.mat` (values read from that file on 2026-10-22). It is the
# `scalar` of `D_likelihood_Physiology.m`: normpdf(measured, arm, arm * scalar).
WEBER_SCALAR = {"Faure": 0.165, "Nielsen": 0.153}
N_SAMPLES = 10_000  # noise draws per trial, as Figure Five
MODEL_SEED = 45  # Figure Five's rng(45)
# Smoothing added to each exit probability before renormalising over the
# unchosen exits, so a model that never picks an exit still gives it a finite
# log likelihood. Sensitivity-checked in the analysis.
MODEL_EPS = 1e-3

# ---- cache / output ---------------------------------------------------------
SCHEMA = 3  # looks-based (data.looks, timestamp_preprocessing)
OUT_ROOT = Path(__file__).resolve().parent / "out"


def cache_path(monkey):
    return processed_npz(f"{monkey}_cf_exits")
