"""Read-only access to the raw `.mat` files under `data.config.MAT_ROOT`.

Every reader opens the file with h5py mode "r" and returns plain numpy data in
memory. Nothing in the repo writes to, renames or deletes a `.mat`, and no raw
data is cached on disk: derived results go to `PROCESSED_ROOT` only.
"""

from __future__ import annotations

import h5py
import numpy as np

from data.config import (
    behavioral_mat_path,
    eye_mat_path,
    monkey_for_session,
    neural_mat_path,
    single_trial_mat_path,
)

BAD_PUPIL_VALUE = -32768

BEHAVIORAL_FIELDS = {
    "LR",
    "LR2",
    "answer_time",
    "feedback_time",
    "fix_start",
    "fixation_cue_present",
    "fixation_off",
    "flash_one",
    "flash_three",
    "flash_two",
    "geo_present",
    "geo_type",
    "h",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "nrns",
    "path_type",
    "photodiode_qc_bad",
    "saccade_init",
    "trial_end",
    "trial_answer1",
    "trial_answer2",
    "trial_answer3",
    "trial_answer4",
    "trial_fade",
    "trial_indices_all",
    "vel",
}


def load_npz(path):
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def xy_t(arr):
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return np.empty((0, 2))
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.shape[1] == 2:
        return a
    if a.shape[0] == 2:
        return a.T
    return a[:, :2]


def convert_mat_field(f, dataset):
    if dataset.dtype == h5py.ref_dtype:
        refs = dataset[()]
        return [np.array(f[r]).squeeze() for r in refs.flat]
    return np.array(dataset).squeeze()


def read_behavioral_mat(path):
    data = {}
    with h5py.File(path, "r") as f:
        all_data = f["save_all_data"]
        for field in BEHAVIORAL_FIELDS:
            data[field] = convert_mat_field(f, all_data[field])
    return data


def read_neural_mat(path):
    with h5py.File(path, "r") as f:
        return {"FR_WH": convert_mat_field(f, f["smooth_session"])}


def read_eye_mat(path):
    """Gaze traces plus pupil.

    Each gaze trial is (N, 3): [x, y, t] (deg, deg, s from geo_present).
    Each pupil trial is (M, 2): [size, t]; tracker-loss codes are NaN.
    """
    with h5py.File(path, "r") as f:
        eye = f["Eye_Data"]
        xs = convert_mat_field(f, eye["eye_x"])
        ys = convert_mat_field(f, eye["eye_y"])
        pupils = (
            convert_mat_field(f, eye["pupil_size"]) if "pupil_size" in eye else None
        )

    gaze = []
    pupil_size = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = min(len(x), len(y))
        t = x[:n, 1]
        gaze.append(np.column_stack([x[:n, 0], y[:n, 0], t]))
        if pupils is None or i >= len(pupils):
            pupil_size.append(np.empty((0, 2)))
        else:
            p = xy_t(pupils[i])
            if p.size:
                p = p.copy()
                p[p[:, 0] == BAD_PUPIL_VALUE, 0] = np.nan
            pupil_size.append(p)
    return {"gaze": gaze, "pupil_size": pupil_size}


def read_single_trial_mat(path):
    data = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            if key == "#refs#":
                continue
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                data[key] = convert_mat_field(f, obj)
            elif isinstance(obj, h5py.Group):
                for sub in obj.keys():
                    data[f"{key}/{sub}"] = convert_mat_field(f, obj[sub])
    return data


def _as_object_arrays(data):
    return {
        k: np.asarray(v, dtype=object) if isinstance(v, list) else v
        for k, v in data.items()
    }


def load_behavioral(monkey, session):
    return _as_object_arrays(read_behavioral_mat(behavioral_mat_path(monkey, session)))


def load_eye(monkey, session):
    """`{"gaze": [...], "pupil_size": [...]}`, one entry per trial."""
    return _as_object_arrays(read_eye_mat(eye_mat_path(monkey, session)))


def load_neural(monkey, session):
    return _as_object_arrays(read_neural_mat(neural_mat_path(monkey, session)))


def load_single_trial(monkey, session):
    return _as_object_arrays(
        read_single_trial_mat(single_trial_mat_path(monkey, session))
    )


def load_session_data(session):
    monkey = monkey_for_session(session)
    data = load_behavioral(monkey, session)
    data.update(load_neural(monkey, session))
    return data
