"""classifier: predict the neural strategy label from pre-fixation gaze.

Window: maze onset -> fixation onset. Two label scopes (``--labels``):

``dendro`` (default)  the four publication sessions, Ward dendrogram labels
                      (0 = H, 1 = S), mazes 1-6.
``svm``               the ten best sessions per monkey by `snr_auc`, with the
                      maze-1-vs-6 SVM labels projected onto every trial,
                      mazes 2-5. The msp scope; mazes 1 and 6 are the SVM's
                      own anchors and are left out, as in msp.

Imports only `data/` and the scientific stack. The fixation table behind the
engineered / regions branches is built by `classifier.fixations` and the
pupil tables by `classifier.pupil_extract`.

Two feature branches, one evaluation:

``raw``         the eye trace itself: gaze x, y (degrees) and pupil, averaged
                in `RAW_BIN_MS` bins over the last `RAW_WINDOW_MS` before
                fixation onset, each bin session-z-scored. Nothing hand-picked.
``engineered``  the handful of quantities an earlier codebook search pointed at:
                origin-disc quadrant dwell, exit-ball occupancy and dwell in
                degrees, origin share, fixation count, eccentricity rings,
                first / last fixation state, pupil mean and change.

Maze is almost a function of strategy on these sessions (maze 1 all H, maze
6 all S), and the maze is readable from gaze, so every model gets the maze
one-hot and is judged on the **within-maze pairwise AUC**: the fraction of
(H, S) trial pairs *from the same maze* ranked correctly by out-of-fold
probability. Maze identity cannot earn credit there. Its chance level comes
from refitting with labels shuffled within (session, maze). A
leave-one-session-out fit is reported alongside.
"""

from __future__ import annotations

from pathlib import Path

from data.config import monkey_for_session
from data.labels import CLUSTERING_SESSIONS, SNR_AUC

SESSIONS = {
    "Faure": ("june_8_g0", "june_24_g0"),
    "Nielsen": ("Nov_3_g0", "Nov_6_g0"),
}
MONKEYS = tuple(SESSIONS)
MAZES = (1, 2, 3, 4, 5, 6)
STEM_DEG = 7.0  # vertical scale of the unit-H warp at x = 0 (data.builder)

# ---- label scopes -------------------------------------------------------------
TOP_N = 10


def top_sessions(monkey, top_n=TOP_N):
    """The monkey's `CLUSTERING_SESSIONS` ranked by `snr_auc`, truncated: the
    same ten sessions `msp.labels.top_ten_sessions` selects."""
    pool = [s for s in CLUSTERING_SESSIONS if monkey_for_session(s) == monkey]
    return tuple(sorted(sorted(pool), key=lambda s: SNR_AUC.get(s, -1.0), reverse=True)[:top_n])


SVM_SESSIONS = {m: top_sessions(m) for m in MONKEYS}

LABEL_SCOPES = {
    "dendro": dict(stem="strategy_choices", sessions=SESSIONS, mazes=MAZES,
                   label="dendrogram labels, publication sessions", tag=""),
    "svm": dict(stem="strategy_svm", sessions=SVM_SESSIONS, mazes=(2, 3, 4, 5),
                label="SVM labels, top-ten sessions", tag="svm"),
}
DEFAULT_LABELS = "dendro"
LABEL_STEM = LABEL_SCOPES[DEFAULT_LABELS]["stem"]

# ---- raw branch ---------------------------------------------------------------
RAW_WINDOW_MS = 1500.0  # last 1.5 s before fixation onset (window is ~1534 ms)
RAW_BIN_MS = 50.0
RAW_N_BINS = int(RAW_WINDOW_MS // RAW_BIN_MS)
RAW_GRID = 6  # dwell map, RAW_GRID x RAW_GRID over [-GRID_LIM, GRID_LIM] deg
GRID_LIM_DEG = 15.0
POSITION_LIMIT_DEG = 30.0

# ---- engineered branch -------------------------------------------------------
ORIGIN_DISC_DEG = 5.0
EXIT_BALL_DEG = 3.0
RING_EDGES_DEG = (2.0, 6.0, 10.0)

# ---- pupil tables (classifier.pupil_extract) ---------------------------------
PUPIL_DIR = Path(__file__).resolve().parent / "out" / "pupil"
PUPIL_EARLY_MS = 300.0
PUPIL_LATE_MS = 300.0
PUPIL_MIN_VALID = 0.5

# ---- models and evaluation ---------------------------------------------------
MODELS = ("logreg", "gbm")
N_FOLDS = 5
N_REPEATS = 4
N_PERM = 200
MIN_PAIRS = 1  # a maze contributes to the within-maze AUC when it has an (H, S) pair
SOLID_N = 10
SEED = 0

OUT_ROOT = Path(__file__).resolve().parent / "out"

assert all(len(v) == 2 for v in SESSIONS.values())
assert all(set(SESSIONS[m]) <= set(SVM_SESSIONS[m]) for m in MONKEYS)
