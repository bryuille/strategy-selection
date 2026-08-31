"""Generate eye-tracking QC movies for the june_24 session.

Port of temp.m for processed june_24 data. Supports consecutive trials on one
continuous session timeline by anchoring each trial's geo_present-normalized
eye times to absolute session time.

Usage:
    uv run python -m eye_pre_flash.plotting.movie
    uv run python -m eye_pre_flash.plotting.movie --start-trial 41 --n-trials 30
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from data.config import MONKEY, behavioral_mat_path, eye_mat_path
from data.convert import (
    convert_mat_field,
    load_data,
    load_eye_data,
)
from eye_pre_flash.plotting.plot_io import OUT_ROOT

SESSION = "june_24_g0"
DATA_PATH = behavioral_mat_path(MONKEY, SESSION)
EYE_PATH = eye_mat_path(MONKEY, SESSION)
OUT_DIR = OUT_ROOT / "movie"

# Movie settings
N_TRIALS = 30
START_TRIAL = 41

FRAME_STEP_MS = 10
PLAYBACK_SPEED = 0.5  # 1.0 = real-time; 0.5 = half speed
VIDEO_FRAME_RATE = int(1000 / FRAME_STEP_MS * PLAYBACK_SPEED)
FLASH_DURATION_MS = 100

GAZE_TRAIL_MS = 150
MAX_EYE_SAMPLE_ERROR_MS = 20
BAD_PUPIL_VALUE = -32768

AX_LIMS = (-15, 15, -15, 15)
FIXATION_STEM_DEG = 7.0  # vertical stem height (fixation target → junction)

# trial_answer1-4 select maze-orientation branch for LR/LR2 decode (not the correct exit).
MONKEY_EXIT_MAP = {
    1: {(-1, 1): 1, (-1, -1): 2, (1, 1): 3, (1, -1): 4},
    2: {(-1, 1): 2, (-1, -1): 1, (1, 1): 4, (1, -1): 3},
    3: {(-1, 1): 3, (-1, -1): 4, (1, 1): 1, (1, -1): 2},
    4: {(-1, 1): 4, (-1, -1): 3, (1, 1): 2, (1, -1): 1},
}


@dataclass
class TrialSegment:
    trial_id: int
    path_type_disp: int
    h_vals: tuple[float, float, float, float, float, float]
    eye_x: np.ndarray
    eye_y: np.ndarray
    eye_t_ms: np.ndarray
    flash1_ms: float
    flash2_ms: float
    flash3_ms: float
    fix_start_ms: float
    fixation_off_ms: float
    answer_ms: float
    session_offset_ms: float
    vel_deg_s: float
    correct_exit: int | None
    monkey_exit: int

    @property
    def global_eye_start_ms(self) -> float:
        return self.session_offset_ms + self.eye_t_ms.min()

    @property
    def global_eye_end_ms(self) -> float:
        return self.session_offset_ms + self.eye_t_ms.max()

    def local_time(self, global_ms: float) -> float:
        return global_ms - self.session_offset_ms

    def contains_global(self, global_ms: float) -> bool:
        return self.global_eye_start_ms <= global_ms <= self.global_eye_end_ms


def load_optional_timing(data):
    extra = {}
    with h5py.File(DATA_PATH, "r") as f:
        all_data = f["save_all_data"]
        for field in ("fixation_off", "answer_time", "trial_end"):
            if field in all_data:
                extra[field] = convert_mat_field(f, all_data[field]).squeeze()
    return extra


def load_pupil_traces():
    with h5py.File(EYE_PATH, "r") as f:
        eye = f["Eye_Data"]
        if "pupil_size" not in eye:
            return None
        return convert_mat_field(f, eye["pupil_size"])


def load_lr2():
    with h5py.File(DATA_PATH, "r") as f:
        return convert_mat_field(f, f["save_all_data"]["LR2"]).squeeze()


def load_two_hier_exits():
    """Optional random-geometry exit labels; None arrays if fields are absent."""
    with h5py.File(DATA_PATH, "r") as f:
        all_data = f["save_all_data"]
        if "two_hier_left" not in all_data or "two_hier_right" not in all_data:
            return None, None
        left = convert_mat_field(f, all_data["two_hier_left"]).squeeze()
        right = convert_mat_field(f, all_data["two_hier_right"]).squeeze()
    return left, right


def answer_branch(row, data):
    """Which trial_answer* flag is set (maze orientation / LR decode branch)."""
    for i in range(1, 5):
        if bool(data[f"trial_answer{i}"][row]):
            return i
    return None


def monkey_exit_from_lr(lr, lr2, branch):
    return MONKEY_EXIT_MAP[branch][(int(lr), int(lr2))]


def correct_exit_from_path(row, data, two_hier_left, two_hier_right):
    """Rewarded exit in display coordinates: 1=LU, 2=LD, 3=RU, 4=RD."""
    path_type = data["path_type"][row]
    if path_type != -99 and not np.isnan(path_type):
        return int((int(path_type) - 1) % 4 + 1)
    # path_type -99: use hierarchical exit labels when available
    if two_hier_left is None or two_hier_right is None:
        return None
    for val in (two_hier_right[row], two_hier_left[row]):
        exit_id = int(val)
        if 1 <= exit_id <= 4:
            return exit_id
    return None


def exit_positions(h1, h2, h3, h4, h5, h6):
    return {
        1: (-h1, h2),
        2: (-h1, -h3),
        3: (h4, h5),
        4: (h4, -h6),
    }


def choice_reveal_ms(segment):
    if not np.isnan(segment.answer_ms):
        return segment.answer_ms
    if not np.isnan(segment.fixation_off_ms):
        return segment.fixation_off_ms
    return segment.flash3_ms


def load_vel():
    with h5py.File(DATA_PATH, "r") as f:
        return convert_mat_field(f, f["save_all_data"]["vel"]).squeeze()


def drop_duration_ms(vel_deg_s):
    """Time for ball to fall the fixation stem at constant maze travel speed."""
    return FIXATION_STEM_DEG / vel_deg_s * 1000


def trial_row(data, trial_id):
    rows = np.where(data["trial_indices_all"].astype(int) == trial_id)[0]
    if rows.size == 0:
        return None
    return int(rows[0])


def path_type_display(path_type, all_path_types):
    valid = all_path_types[~np.isnan(all_path_types)]
    if valid.size and valid.min() == 0 and valid.max() <= 23:
        return int(path_type) + 1
    return int(path_type)


def flash_times_ms(data, row):
    geo_s = data["geo_present"][row]
    return (
        (data["flash_one"][row] - geo_s) * 1000,
        (data["flash_two"][row] - geo_s) * 1000,
        (data["flash_three"][row] - geo_s) * 1000,
    )


def relative_ms(data, row, field, extra):
    if field in extra:
        value = extra[field][row]
    elif field in data:
        value = data[field][row]
    else:
        return np.nan
    geo_s = data["geo_present"][row]
    return (value - geo_s) * 1000


def is_valid_trial(data, gaze, trial_id):
    row = trial_row(data, trial_id)
    if row is None or trial_id < 1 or trial_id > len(gaze):
        return False

    g = gaze[trial_id - 1]
    if g.size == 0:
        return False

    flash1_ms, flash2_ms, flash3_ms = flash_times_ms(data, row)
    if (
        np.isnan([flash1_ms, flash2_ms, flash3_ms]).any()
        or flash1_ms <= 0
        or flash2_ms <= flash1_ms
        or flash3_ms <= flash2_ms
    ):
        return False

    eye_t_ms = g[:, 2] * 1000
    eye_t_ms = eye_t_ms[~np.isnan(eye_t_ms)]
    if eye_t_ms.size == 0:
        return False

    return eye_t_ms.max() >= flash3_ms


def prepare_eye_trace(gaze, pupil_traces, trial_id):
    g = gaze[trial_id - 1]
    eye_x = g[:, 0].astype(float)
    eye_y = g[:, 1].astype(float)
    eye_t_ms = g[:, 2].astype(float) * 1000

    keep = ~np.isnan(eye_x) & ~np.isnan(eye_t_ms)
    eye_x = eye_x[keep]
    eye_y = eye_y[keep]
    eye_t_ms = eye_t_ms[keep]

    bad = (
        (eye_x < AX_LIMS[0])
        | (eye_x > AX_LIMS[1])
        | (eye_y < AX_LIMS[2])
        | (eye_y > AX_LIMS[3])
    )

    if pupil_traces is not None and trial_id - 1 < len(pupil_traces):
        p = np.asarray(pupil_traces[trial_id - 1], dtype=float)
        if p.ndim == 2 and p.shape[1] == 2:
            pupil_size = p[:, 0]
            pupil_t_ms = p[:, 1] * 1000
            keep_p = ~np.isnan(pupil_size) & ~np.isnan(pupil_t_ms)
            pupil_size = pupil_size[keep_p]
            pupil_t_ms = pupil_t_ms[keep_p]
            if pupil_size.size >= 2:
                bad_pupil = (pupil_size == BAD_PUPIL_VALUE).astype(float)
                bad_pupil_on_eye = np.interp(
                    eye_t_ms,
                    pupil_t_ms,
                    bad_pupil,
                    left=0.0,
                    right=0.0,
                )
                bad |= bad_pupil_on_eye > 0

    eye_x[bad] = np.nan
    eye_y[bad] = np.nan
    return eye_x, eye_y, eye_t_ms


def build_trial_segment(
    data,
    gaze,
    pupil_traces,
    extra,
    trial_id,
    session_origin_s,
    vel_all,
    lr2_all,
    two_hier_left,
    two_hier_right,
) -> TrialSegment | None:
    if not is_valid_trial(data, gaze, trial_id):
        return None

    row = trial_row(data, trial_id)
    eye_x, eye_y, eye_t_ms = prepare_eye_trace(gaze, pupil_traces, trial_id)
    flash1_ms, flash2_ms, flash3_ms = flash_times_ms(data, row)

    h_vals = tuple(float(data[f"h{i}"][row]) for i in range(1, 7))
    session_offset_ms = (data["geo_present"][row] - session_origin_s) * 1000
    vel_deg_s = float(vel_all[row])
    branch = answer_branch(row, data)
    if branch is None:
        return None
    monkey_exit = monkey_exit_from_lr(data["LR"][row], lr2_all[row], branch)
    correct_exit = correct_exit_from_path(row, data, two_hier_left, two_hier_right)

    return TrialSegment(
        trial_id=trial_id,
        path_type_disp=path_type_display(data["path_type"][row], data["path_type"]),
        h_vals=h_vals,
        eye_x=eye_x,
        eye_y=eye_y,
        eye_t_ms=eye_t_ms,
        flash1_ms=flash1_ms,
        flash2_ms=flash2_ms,
        flash3_ms=flash3_ms,
        fix_start_ms=relative_ms(data, row, "fix_start", extra),
        fixation_off_ms=relative_ms(data, row, "fixation_off", extra),
        answer_ms=relative_ms(data, row, "answer_time", extra),
        session_offset_ms=session_offset_ms,
        vel_deg_s=vel_deg_s,
        correct_exit=correct_exit,
        monkey_exit=monkey_exit,
    )


def build_trial_sequence(
    data,
    gaze,
    pupil_traces,
    extra,
    vel_all,
    lr2_all,
    two_hier_left,
    two_hier_right,
    start_trial,
    n_trials,
) -> list[TrialSegment]:
    trial_ids = list(range(start_trial, start_trial + n_trials))
    missing = [tid for tid in trial_ids if not is_valid_trial(data, gaze, tid)]
    if missing:
        raise SystemExit(f"Invalid trials in sequence: {missing}")

    origin_row = trial_row(data, start_trial)
    session_origin_s = data["geo_present"][origin_row]

    segments = []
    for trial_id in trial_ids:
        segment = build_trial_segment(
            data,
            gaze,
            pupil_traces,
            extra,
            trial_id,
            session_origin_s,
            vel_all,
            lr2_all,
            two_hier_left,
            two_hier_right,
        )
        if segment is None:
            raise SystemExit(f"Failed to build trial segment for trial {trial_id}")
        segments.append(segment)
    return segments


def active_segment(segments, global_ms):
    for segment in segments:
        if segment.contains_global(global_ms):
            return segment
    return None


def build_frame_times(segments):
    """Session timeline sampled only where eye data exist (no inter-trial gaps)."""
    chunks = []
    for segment in segments:
        local = np.arange(
            segment.eye_t_ms.min(),
            segment.eye_t_ms.max() + FRAME_STEP_MS,
            FRAME_STEP_MS,
        )
        chunks.append(segment.session_offset_ms + local)
    return np.unique(np.concatenate(chunks))


def draw_h_maze(ax, h1, h2, h3, h4, h5, h6):
    """Draw H-maze with the T junction at (0, 0), matching eye-coordinate origin."""
    left_x = -h1
    right_x = h4

    ax.plot([left_x, 0], [0, 0], "k", linewidth=2)
    ax.plot([0, right_x], [0, 0], "k", linewidth=2)
    ax.plot([left_x, left_x], [-h3, h2], "k", linewidth=2)
    ax.plot([right_x, right_x], [-h6, h5], "k", linewidth=2)
    ax.plot([0, 0], [0, FIXATION_STEM_DEG], "k", linewidth=2)

    for x, y in (
        (left_x, h2),
        (left_x, -h3),
        (right_x, h5),
        (right_x, -h6),
    ):
        ax.plot(x, y, "ko", markersize=7, markerfacecolor="w")

    ax.plot(0, 0, "ko", markersize=5, markerfacecolor="k", zorder=4)


def draw_ball(ax, t_ms, flash1_ms, vel_deg_s):
    """Drop ball from fixation stem top at vel deg/s, arriving at junction at flash 1."""
    drop_ms = drop_duration_ms(vel_deg_s)
    drop_start_ms = flash1_ms - drop_ms
    eps = np.finfo(float).eps

    if t_ms < drop_start_ms:
        ball_y = FIXATION_STEM_DEG
    elif t_ms < flash1_ms:
        frac = (t_ms - drop_start_ms) / max(drop_ms, eps)
        ball_y = FIXATION_STEM_DEG * (1.0 - frac)
    else:
        ball_y = 0.0

    if t_ms < flash1_ms:
        ax.plot(0, ball_y, "o", markersize=13, markerfacecolor="k", markeredgecolor="k")
    else:
        ax.plot(
            0,
            ball_y,
            "o",
            markersize=8,
            markerfacecolor=(0.3, 0.3, 0.3),
            markeredgecolor="k",
        )


def draw_flash(ax, t_ms, flash1_ms, flash2_ms, flash3_ms):
    is_flash1 = flash1_ms <= t_ms < flash1_ms + FLASH_DURATION_MS
    is_flash2 = flash2_ms <= t_ms < flash2_ms + FLASH_DURATION_MS
    is_flash3 = flash3_ms <= t_ms < flash3_ms + FLASH_DURATION_MS
    if not (is_flash1 or is_flash2 or is_flash3):
        return

    ax.plot(
        0,
        0,
        "o",
        markersize=28,
        markerfacecolor=(1, 1, 0.2),
        markeredgecolor="k",
        markeredgewidth=2,
    )
    flash_label = "FLASH 1" if is_flash1 else "FLASH 2" if is_flash2 else "FLASH 3"
    ax.text(0, 1.3, flash_label, ha="center", fontsize=16, fontweight="bold", color="k")


def draw_exit_highlights(ax, h_vals, correct_exit, monkey_exit):
    """Highlight chosen exit (green=correct, red=wrong) and correct exit if wrong."""
    if correct_exit is None:
        return
    positions = exit_positions(*h_vals)
    is_correct = correct_exit == monkey_exit

    for exit_id, (x, y) in positions.items():
        if exit_id == monkey_exit:
            color = (0.0, 0.75, 0.0) if is_correct else (0.9, 0.0, 0.0)
            ax.plot(
                x,
                y,
                "o",
                markersize=18,
                markerfacecolor=color,
                markeredgecolor="k",
                markeredgewidth=2,
                zorder=6,
            )
        elif not is_correct and exit_id == correct_exit:
            ax.plot(
                x,
                y,
                "o",
                markersize=18,
                markerfacecolor=(0.0, 0.75, 0.0),
                markeredgecolor="k",
                markeredgewidth=2,
                zorder=6,
            )


def draw_gaze(ax, eye_x, eye_y, eye_t_ms, t_ms):
    trail_idx = (eye_t_ms >= t_ms - GAZE_TRAIL_MS) & (eye_t_ms <= t_ms)
    if np.any(trail_idx):
        ax.plot(
            eye_x[trail_idx],
            eye_y[trail_idx],
            color=(0.3, 0.75, 0.85),
            linewidth=1.2,
        )

    idx = int(np.argmin(np.abs(eye_t_ms - t_ms)))
    if abs(eye_t_ms[idx] - t_ms) > MAX_EYE_SAMPLE_ERROR_MS:
        return

    gaze_x = eye_x[idx]
    gaze_y = eye_y[idx]
    if np.isnan(gaze_x) or np.isnan(gaze_y):
        return

    ax.plot(gaze_x, gaze_y, ".", markersize=18, color="c")


def render_trial_frame(ax, segment, local_ms, global_ms):
    ax.clear()
    draw_h_maze(ax, *segment.h_vals)
    draw_ball(ax, local_ms, segment.flash1_ms, segment.vel_deg_s)
    draw_flash(ax, local_ms, segment.flash1_ms, segment.flash2_ms, segment.flash3_ms)

    if local_ms >= choice_reveal_ms(segment):
        draw_exit_highlights(
            ax, segment.h_vals, segment.correct_exit, segment.monkey_exit
        )

    draw_gaze(ax, segment.eye_x, segment.eye_y, segment.eye_t_ms, local_ms)

    ax.set_aspect("equal")
    ax.set_xlim(AX_LIMS[0], AX_LIMS[1])
    ax.set_ylim(AX_LIMS[2], AX_LIMS[3])
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_title(
        f"{SESSION} | trial {segment.trial_id} | path {segment.path_type_disp} | "
        f"local t = {local_ms:.0f} ms | session t = {global_ms:.0f} ms"
    )
    ax.set_facecolor("white")


def make_movie(
    start_trial=START_TRIAL,
    n_trials=N_TRIALS,
    output=None,
    show=False,
):
    data = load_data()
    gaze = load_eye_data()["gaze"]
    extra = load_optional_timing(data)
    vel_all = load_vel()
    lr2_all = load_lr2()
    two_hier_left, two_hier_right = load_two_hier_exits()
    pupil_traces = load_pupil_traces()

    segments = build_trial_sequence(
        data,
        gaze,
        pupil_traces,
        extra,
        vel_all,
        lr2_all,
        two_hier_left,
        two_hier_right,
        start_trial,
        n_trials,
    )
    print(
        f"Trials {start_trial}–{start_trial + n_trials - 1} "
        f"({len(segments)} segments on continuous session timeline)"
    )

    frame_times_ms = build_frame_times(segments)
    movie_start_ms = frame_times_ms.min()
    movie_end_ms = frame_times_ms.max()

    out_path = (
        Path(output)
        if output
        else OUT_DIR
        / f"{SESSION}_trials_{start_trial}-{start_trial + n_trials - 1}_eye_QC.mp4"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor="white")
    if not show:
        plt.ioff()

    writer = FFMpegWriter(fps=VIDEO_FRAME_RATE)
    with writer.saving(fig, str(out_path), dpi=100):
        for global_ms in frame_times_ms:
            segment = active_segment(segments, global_ms)
            if segment is None:
                continue
            render_trial_frame(
                ax,
                segment,
                segment.local_time(global_ms),
                global_ms,
            )
            writer.grab_frame()

    plt.close(fig)
    duration_s = (movie_end_ms - movie_start_ms) / 1000
    print(f"Saved movie: {out_path}")
    print(f"  Session span: {movie_start_ms:.0f}–{movie_end_ms:.0f} ms ({duration_s:.1f} s)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate eye QC movie for june_24.")
    parser.add_argument(
        "--start-trial",
        type=int,
        default=START_TRIAL,
        help="First trial ID in consecutive sequence",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=N_TRIALS,
        help="Number of consecutive trials to include",
    )
    parser.add_argument("--output", type=str, default=None, help="Output mp4 path")
    parser.add_argument("--show", action="store_true", help="Show figure while rendering")
    args = parser.parse_args()
    make_movie(
        start_trial=args.start_trial,
        n_trials=args.n_trials,
        output=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
