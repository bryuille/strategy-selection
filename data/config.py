"""Data trees, session identity, and path helpers."""

from pathlib import Path

MAT_ROOT = Path("./data/mat")
NPZ_ROOT = Path("./data/npz")
PROCESSED_ROOT = Path("./data/processed")

MONKEYS = ("Faure", "Nielsen")
KINDS = {
    "behavioral": "Behavioral_Data",
    "eye": "Eye_Data",
    "neural": "Neural_Data",
    "single_trial": "Single_Trial_List",
}

MONKEY = "Faure"
SESSION = "june_24_g0"


def mat_dir(kind, monkey):
    return MAT_ROOT / KINDS[kind] / monkey


def npz_dir(kind, monkey):
    return NPZ_ROOT / KINDS[kind] / monkey


def behavioral_mat_path(monkey, session):
    return mat_dir("behavioral", monkey) / f"{session}_good_trials_concat.mat"


def behavioral_npz_path(monkey, session):
    return npz_dir("behavioral", monkey) / f"{session}_good_trials_concat.npz"


def eye_mat_path(monkey, session):
    return mat_dir("eye", monkey) / f"Eye_Data_{session}.mat"


def eye_npz_path(monkey, session):
    return npz_dir("eye", monkey) / f"Eye_Data_{session}.npz"


def neural_mat_path(monkey, session):
    return mat_dir("neural", monkey) / f"{session}_Whole_Trial_FR_Causal.mat"


def neural_npz_path(monkey, session):
    return npz_dir("neural", monkey) / f"{session}_Whole_Trial_FR_Causal.npz"


def processed_npz(stem):
    return PROCESSED_ROOT / f"{stem}.npz"


BEHAVIORAL_MAT = behavioral_mat_path(MONKEY, SESSION)
BEHAVIORAL_NPZ = behavioral_npz_path(MONKEY, SESSION)
EYE_MAT = eye_mat_path(MONKEY, SESSION)
EYE_NPZ = eye_npz_path(MONKEY, SESSION)
NEURAL_MAT = neural_mat_path(MONKEY, SESSION)
NEURAL_NPZ = neural_npz_path(MONKEY, SESSION)
