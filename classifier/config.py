"""classifier: predict the neural strategy label from pre-fixation gaze.

Window: maze onset -> fixation onset. Two label scopes (``--labels``):

``dendro`` (default)  the four publication sessions, Ward dendrogram labels
                      (0 = H, 1 = S), mazes 1-6.
``svm``               the ten best sessions per monkey by `snr_auc`, with the
                      maze-1-vs-6 SVM labels projected onto every trial,
                      mazes 2-5. The msp scope; mazes 1 and 6 are the SVM's
                      own anchors and are left out, as in msp.

Imports only `data/` and the scientific stack. The fixation table behind the
regions branches is built by `classifier.fixations` and the
pupil tables by `classifier.pupil_extract`.

Two feature branches, one evaluation:

``raw``         the eye trace itself: gaze x, y (degrees) and pupil, averaged
                in `RAW_BIN_MS` bins over the last `RAW_WINDOW_MS` before
                fixation onset, each bin session-z-scored. Nothing hand-picked.
``regions``     dwell share and / or visited over nine regions in degrees:
                the four origin-disc quadrants, the four exit balls and
                everything else (`regions_dwell` / `regions_binary` for one set).
``regions_plus`` the same with "elsewhere" split at `FAR_DEG` and pupil change
                added (`regions_plus_dwell` / `regions_plus_binary`).
``exits``       the four exit balls alone (`exits_dwell` / `exits_binary`).
``lor``         origin ball + left / right half of the maze, in degrees
                (`lor_dwell` / `lor_binary`); msp's `halves` codebook with a
                perimeter that hugs the exit balls.
``balls``       msp's r0.5 codebook in degrees: origin ball + four exit balls
                (`balls_dwell` / `balls_binary`), nothing else.
``capsules``   origin ball + four capsules (parallel tangents between the
                origin and exit balls) + elsewhere, in degrees
                (`capsules_tight` shrinks only the origin ball).

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
STEM_DEG = 7.0  # physical height of the centre stem in degrees (data.builder), for drawings

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

# ---- regions branch ------------------------------------------------------------
ORIGIN_DISC_DEG = 5.0
EXIT_BALL_DEG = 3.0
FAR_DEG = 10.0  # regions_plus: "elsewhere" split into near (disc edge to FAR_DEG) and far
# Everything in classifier is in degrees; nothing here is warped to unit H.
# Where a codebook mirrors one of msp's (whose balls are r0.5 unit H), a 3 deg
# ball stands in: 0.5 unit H is 2.5-4.5 deg depending on maze and direction.
# balls: msp's r0.5 codebook in degrees, an origin ball + the four exit balls
# (EXIT_BALL_DEG) at each maze's exits, everything else unassigned.
BALLS_ORIGIN_DEG = 3.0
# capsules: msp's r0.5 capsules in degrees. Each capsule is every point within
# CAPSULE_DEG of the segment from the origin to one exit, i.e. the two balls
# joined by parallel tangents. The origin ball is carved out as its own region
# and everything outside the capsules is "elsewhere". `tight` keeps the same
# capsules and shrinks the origin ball to the r0.25 stand-in.
CAPSULE_DEG = 3.0
CAPSULE_ORIGIN_DEG = 3.0
CAPSULE_ORIGIN_TIGHT_DEG = 1.5
# lor: origin ball + maze halves. "Inside the maze" is every point within
# LOR_PERIMETER_DEG of the quadrilateral the four exits span, i.e. the outer
# edge of exit balls of that radius joined by straight tangents; x < 0 is left.
LOR_ORIGIN_DEG = 3.0
LOR_PERIMETER_DEG = 3.0

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
