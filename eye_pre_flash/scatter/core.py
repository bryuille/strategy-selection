"""Where the eyes are at the instant of `fix_start`, per trial.

Every other pre-flash analysis in this repo integrates a *window* ending at
`fix_start` (see `eye_pre_flash/plotting/plot_io.PRE_FIX_START_MS`). This one
takes the single sample at `fix_start` itself -- one (x, y) per trial -- so a
trial is a point rather than a cloud, and maze/strategy structure shows up as a
shift of the point cloud instead of a shift of a heatmap.

Trial filters match the rest of the pre-flash family: gaze-tracked, QC-passed,
unfaded, fixed-geometry trials of mazes 1-6 (`path_type != -99`,
`photodiode_qc_bad` false, `trial_fade == 0`). On top of that the `fix_start`
sample itself must exist: eye time is sampled at 1 ms, so the nearest sample is
required to sit within `TOL_MS` of `fix_start` and to be finite and inside the
tracker range (`POSITION_LIMIT_DEG`).

Two coordinate frames, because they answer different questions:

    deg     raw degrees from the fixation target. A maze difference here can
            be geometric -- the arms of maze 1 and maze 6 are not the same
            lengths, so "look further left" and "the left arm is longer" are
            not separable.
    unith   each trial warped onto the unit H by its own arm lengths
            (`data.attractor.to_maze`), which divides that geometry out. A
            difference that survives here is a difference in where the animal
            put its eyes relative to the maze it was about to solve.
"""

from __future__ import annotations

import csv

import numpy as np

from data.attractor import to_maze
from data.builder import POSITION_LIMIT_DEG
from data.loader import load_eye_behavioral_data, load_eye_data
from eye_pre_flash.classifier.labels import LABEL_NAMES

from pathlib import Path

OUT_ROOT = Path(__file__).resolve().parent / "out"

MAZES = tuple(range(1, 7))
LABELS = (0, 1)
# Eye data is sampled at 1 ms, so the sample "at fix_start" is always within
# half a sample of it. Anything further out means the trial's gaze trace does
# not cover fix_start and the trial is dropped rather than extrapolated.
TOL_MS = 2.0

SPACES = {
    "deg": dict(axis="deg", label="Gaze at fix_start (deg from fixation target)"),
    "unith": dict(axis="unit H", label="Gaze at fix_start (unit H)"),
}

# --- Palette ---------------------------------------------------------------
# Maze 1-6 is *ordered* (1-3 hierarchical, 5-6 sequential, 4 the boundary), so
# it takes a single-hue ordinal ramp rather than six categorical hues: six
# evenly lightness-spaced steps of the blue ramp, validated for monotone
# lightness, adjacent ΔL >= 0.06 and light-end contrast on a light surface.
MAZE_COLORS = {
    1: "#86b6ef",
    2: "#599be8",
    3: "#307edc",
    4: "#2265b7",
    5: "#164c91",
    6: "#0d366b",
}
# Strategy is two identities, not a magnitude: categorical slots 2 and 3, which
# clear the all-pairs CVD and normal-vision floors. Aqua sits below 3:1 on a
# light surface, so every strategy mark is directly labelled as well.
STRATEGY_COLORS = {0: "#eb6834", 1: "#1baf7a"}
STRATEGY_NAMES = dict(enumerate(LABEL_NAMES))

INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8880"
SURFACE = "#fcfcfb"
CLOUD = "#d8d6cf"


def output_rel_dir(space, monkey, scope, *, centered=False):
    """`<monkey>/<scope>[_centered]` under `out/`, dropping default-value folders.

    `deg` and scope `"all"` are the canonical defaults, so a run with both
    doesn't get its own folder level for them -- only a non-default space or
    scope (or centring) adds one, so the common case stays flat.
    """
    parts = [] if space == "deg" else [space]
    parts.append(monkey)
    tail = "" if scope == "all" else scope
    if centered:
        tail = f"{tail}_centered" if tail else "centered"
    if tail:
        parts.append(tail)
    return "/".join(parts)


def save_figure(fig, stem, *, rel_dir=None, dpi=200):
    """Save `fig` under eye_pre_flash/scatter/out/[rel_dir/]<stem>.png."""
    out_dir = OUT_ROOT / rel_dir if rel_dir else OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    print(f"Saved {path}")
    return path


def style_axes(ax, *, grid=True):
    """Recessive frame and grid; data is the only thing with weight."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SOFT, labelsize=8, width=0.8, length=3)
    if grid:
        ax.grid(True, color="#e8e6df", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
    return ax


def _behavioral_lookup(behavioral):
    return {
        (str(session), int(trial_id)): i
        for i, (session, trial_id) in enumerate(
            zip(behavioral["session"], behavioral["trial_indices_all"])
        )
    }


def _fix_start_ms(behavioral, idx):
    return (behavioral["fix_start"][idx] - behavioral["geo_present"][idx]) * 1000


def _choice(behavioral, idx):
    """Chosen exit as 1 left-up, 2 left-down, 3 right-up, 4 right-down, else -1.

    `trial_answer1..4` are correctness indicators relative to the trial's
    ground truth `LR` (1 left / 2 right) and `LR2` (1 up / 2 down), so the
    chosen exit is the ground truth flipped on whichever axis was answered
    wrong (see DATA_DICTIONARY.md § Events and responses).
    """
    lr = behavioral["LR"][idx]
    lr2 = behavioral["LR2"][idx]
    if not (np.isfinite(lr) and np.isfinite(lr2)):
        return -1
    answers = [bool(behavioral[f"trial_answer{k}"][idx]) for k in range(1, 5)]
    if sum(answers) != 1:
        return -1
    which = answers.index(True)
    flip_h = which in (2, 3)
    flip_v = which in (1, 3)
    left = (int(lr) == 1) != flip_h
    up = (int(lr2) == 1) != flip_v
    return (1 if left else 3) + (0 if up else 1)


def collect_fix_start(
    monkey,
    *,
    sessions=None,
    label_lookup=None,
    space="deg",
    tol_ms=TOL_MS,
):
    """One (x, y) per trial: the gaze sample at `fix_start`.

    `sessions` restricts to a session tuple (a classifier scope); `label_lookup`
    is ``(session, trial_id) -> 0/1`` from
    `eye_pre_flash.classifier.labels.strategy_label_lookup`. Trials outside
    `sessions` are dropped; trials without a label keep a NaN label rather than
    being dropped, so a maze-only figure still uses every trial.

    Returns a dict of parallel arrays plus a `dropped` tally of why trials left.
    """
    if space not in SPACES:
        raise ValueError(f"unknown space {space!r}; choose from {sorted(SPACES)}")

    eye = load_eye_data(monkey)
    behavioral = load_eye_behavioral_data(monkey)
    beh_index = _behavioral_lookup(behavioral)
    wanted = None if sessions is None else set(map(str, sessions))

    cols = {k: [] for k in ("x", "y", "maze", "label", "session", "trial_id", "choice")}
    dropped = dict(session=0, no_behavioral=0, qc=0, maze=0, no_sample=0, bad_sample=0)

    for eye_i in range(len(eye["session"])):
        session = str(eye["session"][eye_i])
        if wanted is not None and session not in wanted:
            dropped["session"] += 1
            continue
        trial_id = int(eye["trial_indices_all"][eye_i])
        beh_i = beh_index.get((session, trial_id))
        if beh_i is None:
            dropped["no_behavioral"] += 1
            continue
        if (
            behavioral["path_type"][beh_i] == -99
            or behavioral["photodiode_qc_bad"][beh_i]
            or behavioral["trial_fade"][beh_i] != 0
        ):
            dropped["qc"] += 1
            continue
        maze = int(behavioral["geo_type"][beh_i])
        if maze not in MAZES:
            dropped["maze"] += 1
            continue

        t_ms = np.asarray(eye["time"][eye_i], dtype=float) * 1000
        if t_ms.size == 0:
            dropped["no_sample"] += 1
            continue
        fix_ms = _fix_start_ms(behavioral, beh_i)
        k = int(np.argmin(np.abs(t_ms - fix_ms)))
        if abs(t_ms[k] - fix_ms) > tol_ms:
            dropped["no_sample"] += 1
            continue

        x = float(np.asarray(eye["eye_x"][eye_i], dtype=float)[k])
        y = float(np.asarray(eye["eye_y"][eye_i], dtype=float)[k])
        if not (np.isfinite(x) and np.isfinite(y)):
            dropped["bad_sample"] += 1
            continue
        if abs(x) > POSITION_LIMIT_DEG or abs(y) > POSITION_LIMIT_DEG:
            dropped["bad_sample"] += 1
            continue

        if space == "unith":
            h = tuple(float(behavioral[f"h{i}"][beh_i]) for i in range(1, 7))
            if not np.all(np.isfinite(h)):
                dropped["bad_sample"] += 1
                continue
            xn, yn = to_maze(np.array([x]), np.array([y]), h)
            x, y = float(xn[0]), float(yn[0])

        label = np.nan
        if label_lookup is not None:
            got = label_lookup.get((session, trial_id))
            if got is not None and np.isfinite(got):
                label = float(got)

        cols["x"].append(x)
        cols["y"].append(y)
        cols["maze"].append(maze)
        cols["label"].append(label)
        cols["session"].append(session)
        cols["trial_id"].append(trial_id)
        cols["choice"].append(_choice(behavioral, beh_i))

    out = {
        "x": np.asarray(cols["x"], dtype=float),
        "y": np.asarray(cols["y"], dtype=float),
        "maze": np.asarray(cols["maze"], dtype=int),
        "label": np.asarray(cols["label"], dtype=float),
        "session": np.asarray(cols["session"], dtype=object),
        "trial_id": np.asarray(cols["trial_id"], dtype=int),
        "choice": np.asarray(cols["choice"], dtype=int),
        "space": space,
        "monkey": monkey,
        "dropped": dropped,
    }
    return out


TRIAL_KEYS = ("x", "y", "maze", "label", "session", "trial_id", "choice")


def save_trials_csv(trials, *, rel_dir=None, stem="trials"):
    """Write the per-trial table a figure was built from next to it.

    Every view in `plots.py` is some aggregation of this same table, so
    saving it once per run means a figure can be redrawn, or re-aggregated a
    different way, from `<stem>.csv` alone -- without re-running
    `collect_fix_start` against the raw eye/behavioral data.
    """
    out_dir = OUT_ROOT / rel_dir if rel_dir else OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.csv"
    n = trials["x"].size
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(TRIAL_KEYS)
        cols = [trials[k] for k in TRIAL_KEYS]
        for i in range(n):
            writer.writerow([col[i] for col in cols])
    print(f"Saved {path}")
    return path


def subset(trials, mask):
    """The same trial dict restricted to `mask`, so footers report its own n.

    Views that only plot labelled trials would otherwise caption themselves
    with the whole pool's trial count.
    """
    out = dict(trials)
    for key in TRIAL_KEYS:
        out[key] = trials[key][mask]
    return out


def center_by_session(trials):
    """Subtract each session's mean gaze, in place on a copy.

    Eye-tracker calibration drifts between sessions, so a chunk of the raw
    spread across the pooled cloud is between-session offset rather than
    anything about mazes or strategies. Centring each session on its own mean
    holds session fixed and leaves only within-session structure.
    """
    out = dict(trials)
    x = trials["x"].copy()
    y = trials["y"].copy()
    for session in np.unique(trials["session"]):
        m = trials["session"] == session
        x[m] -= x[m].mean()
        y[m] -= y[m].mean()
    out["x"], out["y"] = x, y
    out["centered"] = True
    return out


def limits(x, y, *, pct=99.0, pad=0.08):
    """Robust shared axis limits: the central `pct`% of the cloud, padded."""
    lo_p, hi_p = (100 - pct) / 2, 100 - (100 - pct) / 2
    if x.size == 0:
        return (-1.0, 1.0), (-1.0, 1.0)
    xlo, xhi = np.percentile(x, [lo_p, hi_p])
    ylo, yhi = np.percentile(y, [lo_p, hi_p])
    span = max(xhi - xlo, yhi - ylo, 1e-6)
    cx, cy = (xlo + xhi) / 2, (ylo + yhi) / 2
    half = span / 2 * (1 + pad)
    return (cx - half, cx + half), (cy - half, cy + half)


def clipped_count(x, y, xlim, ylim):
    """Trials outside the plotted box -- reported rather than silently cut."""
    out = (x < xlim[0]) | (x > xlim[1]) | (y < ylim[0]) | (y > ylim[1])
    return int(out.sum())


def centroid_ci(v, *, z=1.96):
    """Mean and 95% CI half-width of `v` (NaN-safe, needs 2+ points)."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan
    if v.size < 2:
        return float(v.mean()), np.nan
    return float(v.mean()), float(z * v.std(ddof=1) / np.sqrt(v.size))
