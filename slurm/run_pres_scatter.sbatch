#!/bin/bash
# Presentation fixation-scatter figures (eye_pre_flash/pres/scatter), on Engaging.
#
#   mkdir -p logs          # Slurm will not create the -o/-e directory for you
#   sbatch slurm/run_pres_scatter.sbatch
#
# Anything after the script name is forwarded to the module, so a single
# monkey, source or space can be re-run without editing this file:
#
#   sbatch slurm/run_pres_scatter.sbatch --monkey Faure
#   sbatch slurm/run_pres_scatter.sbatch --source svm --space unith
#
# Light and read-only: it loads the already-built raw `{monkey}_eye_data.npz`
# / `{monkey}_eye_behavioral.npz` caches (3-5 GB each, see cloud.md) plus the
# ~5 KB strategy labels, takes the one `fix_start` gaze sample per trial, and
# writes two PNGs per monkey. Still a job rather than a login-node command,
# because the login node's RLIMIT_NPROC kills numpy.
#
# Results leave ~/strategy-selection/eye_pre_flash/pres/scatter/ (run on the
# laptop):
#   rsync -av engaging:strategy-selection/eye_pre_flash/pres/scatter/ \
#       ./eye_pre_flash/pres/scatter/

#SBATCH -J eyepresscatter
#SBATCH -p ou_bcs_low
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/eyepresscatter-%j.out
#SBATCH -e logs/eyepresscatter-%j.err

# One core: `collect_fix_start` is a single serial pass over each monkey's
# trial table (~40k trials pooled) plus one scatter draw per figure -- no
# joblib, no `n_jobs`. The 32G ask covers holding both the raw eye and
# behavioral caches resident at once (`run_saccades.sbatch` sizes the same
# raw-cache load at 32G); 1h is headroom over a load-and-plot job that should
# finish in minutes once the caches are warm.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export STRATEGY_DATA_ROOT="${STRATEGY_DATA_ROOT:-/home/byuille/orcd/scratch/strategy_selection_data}"
export PATH="$HOME/.local/bin:$PATH"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "host        $(hostname)"
echo "args        $*"
echo "started     $(date)"
echo

uv run python -u -m eye_pre_flash.pres.scatter.build "$@"

echo
echo "finished    $(date)"
