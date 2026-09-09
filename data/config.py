import os
from pathlib import Path

# Raw/derived data root. Defaults to the repo-relative `./data/` the laptop
# checkout still uses (it holds only a handful of sessions' mat and is not
# space-constrained in a way that matters). On the cluster, `data/mat/` used
# to compete with the code and every cache for the same 200 GB `/home` quota
# -- see `cloud.md`'s "Space: the number to watch" -- which is why
# `data.labeler.sweep_strategy_labels` deleted neural intermediates as it
# went. That quota problem is gone now that the cluster's mat/npz/processed
# live on a separate, much larger scratch volume
# (`/home/byuille/orcd/scratch/strategy_selection_data/`); every `slurm/*.sbatch`
# job sets `STRATEGY_DATA_ROOT` to point here instead.
_DATA_ROOT = Path(os.environ["STRATEGY_DATA_ROOT"]) if os.environ.get("STRATEGY_DATA_ROOT") else Path("./data")

MAT_ROOT = _DATA_ROOT / "mat"
NPZ_ROOT = _DATA_ROOT / "npz"
PROCESSED_ROOT = _DATA_ROOT / "processed"

MONKEYS = ("Faure", "Nielsen")
KINDS = {
    "behavioral": "Behavioral_Data",
    "eye": "Eye_Data",
    "neural": "Neural_Data",
    "single_trial": "Single_Trial_List",
}

MONKEY = "Faure"
SESSION = "june_24_g0"

SESSION_MONTH_MONKEY = {
    "june": "Faure",
    "April": "Faure",
    "Dec": "Nielsen",
    "Nov": "Nielsen",
    "Oct": "Nielsen",
}


def monkey_for_session(session):
    return SESSION_MONTH_MONKEY[session.split("_", 1)[0]]


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


def single_trial_mat_path(monkey, session):
    return mat_dir("single_trial", monkey) / f"single_trial_list_{session}.mat"


def single_trial_npz_path(monkey, session):
    return npz_dir("single_trial", monkey) / f"single_trial_list_{session}.npz"


def processed_npz(stem):
    return PROCESSED_ROOT / f"{stem}.npz"


BEHAVIORAL_MAT = behavioral_mat_path(MONKEY, SESSION)
BEHAVIORAL_NPZ = behavioral_npz_path(MONKEY, SESSION)
EYE_MAT = eye_mat_path(MONKEY, SESSION)
EYE_NPZ = eye_npz_path(MONKEY, SESSION)
NEURAL_MAT = neural_mat_path(MONKEY, SESSION)
NEURAL_NPZ = neural_npz_path(MONKEY, SESSION)
