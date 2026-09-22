#!/bin/bash
# Presentation fixation-scatter figures (eye_pre_flash/scatter), on Engaging.
#
#   mkdir -p logs          # Slurm will not create the -o/-e directory for you
#   sbatch slurm/run_scatter.sbatch
#
# Anything after the script name is forwarded to the module, so a single
# monkey can be re-run without editing this file:
#
#   sbatch slurm/run_scatter.sbatch --monkey Faure
#
# Light and read-only: it loads the already-built raw `{monkey}_eye_data.npz`
# / `{monkey}_eye_behavioral.npz` caches (3-5 GB each, see cloud.md), averages
# the gaze samples between `fix_start` and `flash_one` per trial, and per
# monkey writes `by_maze.png` (six per-maze mean-gaze dots with +-1 SD bars)
# and `maze_pvalues.png` (a 6x6 table: p-value of maze i's mean position under
# maze j's own per-axis SD, chi-square(2 df),
# `eye_pre_flash.scatter.core.maze_pvalues` -- directional and uncorrected for
# the 30 comparisons per monkey). No strategy labels needed. Still a job
# rather than a login-node command, because the login node's RLIMIT_NPROC
# kills numpy.
#
# Results leave ~/strategy-selection/eye_pre_flash/scatter/ (run on the
# laptop):
#   rsync -av engaging:strategy-selection/eye_pre_flash/scatter/out/ \
#       ./eye_pre_flash/scatter/out/

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
# behavioral caches resident at once (`run_rebuild.sbatch` sizes the same
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

uv run python -u -m eye_pre_flash.scatter.build "$@"

echo
echo "finished    $(date)"
