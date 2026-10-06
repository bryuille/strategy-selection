"""Write per-session pupil traces to ``out/pupil/``: the four publication
sessions by default, or any ``--session`` list (the classifier's svm scope
wants the top-ten sessions, `classifier.config.SVM_SESSIONS`).

Runs where the eye mats live (the cluster):

    sbatch -J pupil slurm/run_module.sbatch msp_revised.pupil_extract
    sbatch -J pupil slurm/run_module.sbatch msp_revised.pupil_extract --session Oct_22_g0 Nov_18_g0

and the tables come down with the figures:

    rsync -av engaging:strategy-selection/msp_revised/out/pupil/ ./msp_revised/out/pupil/

One ``<session>_pupil.npz`` per session: ``pupil`` is an object array, one
``(M, 2) float32`` ``[size, t_s]`` per trial (t in seconds from
geo_present, tracker-loss codes already NaN -- `data.mat.read_eye_mat`),
indexed by ``trial_indices_all - 1``. `features` feeds these to the blink
detector and the pupil features, so a laptop run matches a cluster run.
"""

from __future__ import annotations

import argparse

import numpy as np

from data.config import eye_mat_path, monkey_for_session
from data.mat import load_eye
from msp_revised.config import PUPIL_DIR, SESSIONS


def extract(monkey, session):
    path = eye_mat_path(monkey, session)
    raw = load_eye(monkey, session)
    pupils = [np.asarray(p, dtype=np.float32) for p in raw["pupil_size"]]
    arr = np.empty(len(pupils), dtype=object)
    for i, p in enumerate(pupils):
        arr[i] = p
    PUPIL_DIR.mkdir(parents=True, exist_ok=True)
    out = PUPIL_DIR / f"{session}_pupil.npz"
    tmp = out.with_name(f".{out.name}.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, pupil=arr, trial_indices_all=np.arange(1, len(pupils) + 1),
                 session=np.asarray(session), source=np.asarray(str(path)))
    tmp.replace(out)
    n_samples = sum(len(p) for p in pupils)
    print(f"{session}: {len(pupils)} trials, {n_samples} samples -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", nargs="*", default=None,
                    help="sessions to extract (default: the publication sessions)")
    args = ap.parse_args()
    pairs = ([(monkey_for_session(s), s) for s in args.session] if args.session
             else [(m, s) for m, ss in SESSIONS.items() for s in ss])
    for monkey, session in pairs:
        extract(monkey, session)


if __name__ == "__main__":
    main()
