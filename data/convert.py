"""mat → npz conversion for all data kinds.

Examples:
    uv run -m data.convert all
    uv run -m data.convert eye --monkey Faure
    uv run -m data.convert behavioral neural --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from data.config import (
    BEHAVIORAL_NPZ,
    EYE_NPZ,
    KINDS,
    MAT_ROOT,
    MONKEYS,
    NEURAL_NPZ,
    NPZ_ROOT,
)

BAD_PUPIL_VALUE = -32768

BEHAVIORAL_FIELDS = {
    "LR",
    "LR2",
    "fix_start",
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


def _xy_t(arr):
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


def mirror_npz_path(mat_file):
    rel = Path(mat_file).relative_to(MAT_ROOT)
    return NPZ_ROOT / rel.with_suffix(".npz")


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
            p = _xy_t(pupils[i])
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


READERS = {
    "Behavioral_Data": read_behavioral_mat,
    "Eye_Data": read_eye_mat,
    "Neural_Data": read_neural_mat,
    "Single_Trial_List": read_single_trial_mat,
}


def kind_dir_of(mat_file):
    for kind_dir in KINDS.values():
        if kind_dir in Path(mat_file).parts:
            return kind_dir


def iter_mats(kind_dir, monkey=None):
    monkeys = (monkey,) if monkey else MONKEYS
    for name in monkeys:
        yield from sorted((MAT_ROOT / kind_dir / name).glob("*.mat"))


def convert_mat(mat_file, *, overwrite=False):
    mat_file = Path(mat_file)
    out_path = mirror_npz_path(mat_file)
    if out_path.exists() and not overwrite:
        return out_path, "skipped"

    data = READERS[kind_dir_of(mat_file)](mat_file)
    out = {
        k: np.asarray(v, dtype=object) if isinstance(v, list) else v
        for k, v in data.items()
    }
    np.savez(out_path, **out)
    return out_path, "wrote"


def convert_kind(kind_dir, *, monkey=None, overwrite=False):
    results = []
    for mat_file in iter_mats(kind_dir, monkey):
        out_path, status = convert_mat(mat_file, overwrite=overwrite)
        results.append((mat_file, out_path, status))
        print(f"[{status}] {mat_file} -> {out_path}")
    return results


def convert_kinds(kinds, *, monkey=None, overwrite=False):
    if kinds == ["all"]:
        kind_dirs = list(KINDS.values())
    else:
        kind_dirs = [KINDS[alias] for alias in kinds]

    results = []
    for kind_dir in kind_dirs:
        print(f"=== {kind_dir} ===")
        results.extend(convert_kind(kind_dir, monkey=monkey, overwrite=overwrite))
    return results


def load_behavioral():
    return load_npz(BEHAVIORAL_NPZ)


def load_neural():
    return load_npz(NEURAL_NPZ)


def load_eye_data():
    raw = load_npz(EYE_NPZ)
    return {k: list(raw[k]) for k in raw}


def load_data():
    """Behavioral fields + FR_WH from converted npzs."""
    data = load_behavioral()
    data.update(load_neural())
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert data/mat/*.mat → data/npz/*.npz"
    )
    parser.add_argument(
        "kinds",
        nargs="+",
        choices=("all", *KINDS.keys()),
    )
    parser.add_argument("--monkey", choices=MONKEYS, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    results = convert_kinds(args.kinds, monkey=args.monkey, overwrite=args.overwrite)
    n_wrote = sum(1 for *_, s in results if s == "wrote")
    n_skip = sum(1 for *_, s in results if s == "skipped")
    print(f"done: {n_wrote} wrote, {n_skip} skipped, {len(results)} total")


if __name__ == "__main__":
    main()
