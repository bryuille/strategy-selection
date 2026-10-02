from pathlib import Path

# ---------------------------------------------------------------------------
# SET THESE TWO PATHS. Nothing else needs configuring.
#
# MAT_ROOT       Folder holding the raw `.mat` files (required, no default):
#                    <MAT_ROOT>/{Behavioral_Data,Eye_Data,Neural_Data,
#                                Single_Trial_List}/{Faure,Nielsen}/*.mat
#                The code only ever reads from it. It is never written to.
# PROCESSED_ROOT Folder for everything the code computes (labels, eye
#                tables, feature caches). Created if missing. Any folder
#                outside MAT_ROOT will do.
# ---------------------------------------------------------------------------
MAT_ROOT = None  # e.g. Path("/Volumes/archive/mat")
PROCESSED_ROOT = Path("./data/processed")

if MAT_ROOT is not None:
    MAT_ROOT = Path(MAT_ROOT).expanduser().resolve()
PROCESSED_ROOT = Path(PROCESSED_ROOT).expanduser().resolve()

if MAT_ROOT is not None and (
    PROCESSED_ROOT == MAT_ROOT or MAT_ROOT in PROCESSED_ROOT.parents
):
    raise RuntimeError(
        f"PROCESSED_ROOT ({PROCESSED_ROOT}) is inside MAT_ROOT ({MAT_ROOT}). "
        "Outputs must not be written into the raw data; pick another folder "
        "in data/config.py."
    )

MONKEYS = ("Faure", "Nielsen")
KINDS = {
    "behavioral": "Behavioral_Data",
    "eye": "Eye_Data",
    "neural": "Neural_Data",
    "single_trial": "Single_Trial_List",
}

# The analysis window: step 1 of the pipeline clips every trial to
# [fix_start - PRE_FIX_START_MS, fix_start - PRE_FIX_END_MS] before anything
# else happens, so these define what "in-window" means for detection, for the
# k-means pool and for every feature block. Modules must import these rather
# than define their own.
#
# 1466 is the measured minimum `geo_present -> fix_start` gap (Nielsen). A wider
# window would reach back past `geo_present` on some trials and plot gaze from
# before the maze was on screen.
PRE_FIX_START_MS = 1466
PRE_FIX_END_MS = 0
PRE_FIX_WINDOW_MS = float(PRE_FIX_START_MS)

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
    if MAT_ROOT is None:
        raise RuntimeError(
            "MAT_ROOT is not set. Open data/config.py and set MAT_ROOT to the "
            "folder that holds your Behavioral_Data/, Eye_Data/, Neural_Data/ "
            "and Single_Trial_List/ folders."
        )
    return MAT_ROOT / KINDS[kind] / monkey


def behavioral_mat_path(monkey, session):
    return mat_dir("behavioral", monkey) / f"{session}_good_trials_concat.mat"


def eye_mat_path(monkey, session):
    return mat_dir("eye", monkey) / f"Eye_Data_{session}.mat"


def neural_mat_path(monkey, session):
    return mat_dir("neural", monkey) / f"{session}_Whole_Trial_FR_Causal.mat"


def single_trial_mat_path(monkey, session):
    return mat_dir("single_trial", monkey) / f"single_trial_list_{session}.mat"


def processed_npz(stem):
    return PROCESSED_ROOT / f"{stem}.npz"
