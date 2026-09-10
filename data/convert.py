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
    behavioral_npz_path,
    monkey_for_session,
    neural_npz_path,
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
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a hidden sibling and rename. `np.savez` straight onto `out_path`
    # leaves a truncated-but-existing npz if the process dies mid-write -- a
    # Slurm TIMEOUT, a scancel, a full disk -- and the next run sees the file,
    # reports "skipped", and carries the corruption downstream. `replace` is
    # atomic within a filesystem, so a file at `out_path` is always complete.
    # The temp name is passed as an open handle because `np.savez` appends
    # ".npz" to a *path* that lacks it, which would defeat the rename.
    tmp = out_path.with_name(f".{out_path.name}.tmp")
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, **out)
        tmp.replace(out_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return out_path, "wrote"


def convert_kind(kind_dir, *, monkey=None, overwrite=False, sessions=None):
    results = []
    for mat_file in iter_mats(kind_dir, monkey):
        if sessions is not None and not any(s in mat_file.name for s in sessions):
            continue
        out_path, status = convert_mat(mat_file, overwrite=overwrite)
        results.append((mat_file, out_path, status))
        print(f"[{status}] {mat_file} -> {out_path}")
    return results


def convert_kinds(kinds, *, monkey=None, overwrite=False, sessions=None):
    """Convert the requested kinds, optionally restricted to named sessions.

    `sessions` exists because npz is roughly three times the size of the mat it
    came from -- `np.savez` is uncompressed, `.mat` v7.3 is compressed HDF5 -- and
    neural files are the big ones, a couple of gigabytes each before conversion.
    Converting every neural session on a quota-limited filesystem is the fastest
    way to run out of space, and only the sessions in
    `data.labeler.CLUSTERING_SESSIONS` carry the strategy labels anything
    downstream needs.
    """
    if kinds == ["all"]:
        kind_dirs = list(KINDS.values())
    else:
        kind_dirs = [KINDS[alias] for alias in kinds]

    results = []
    for kind_dir in kind_dirs:
        print(f"=== {kind_dir} ===")
        results.extend(
            convert_kind(
                kind_dir, monkey=monkey, overwrite=overwrite, sessions=sessions
            )
        )
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


def load_session_data(session):
    monkey = monkey_for_session(session)
    data = load_npz(behavioral_npz_path(monkey, session))
    data.update(load_npz(neural_npz_path(monkey, session)))
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
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=None,
        help="only convert files whose name contains one of these session ids",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    results = convert_kinds(
        args.kinds,
        monkey=args.monkey,
        overwrite=args.overwrite,
        sessions=args.sessions,
    )
    n_wrote = sum(1 for *_, s in results if s == "wrote")
    n_skip = sum(1 for *_, s in results if s == "skipped")
    print(f"done: {n_wrote} wrote, {n_skip} skipped, {len(results)} total")


if __name__ == "__main__":
    main()
