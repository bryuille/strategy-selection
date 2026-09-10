#!/bin/bash
# Presentation copies of the H/S census heatmaps (eye_pre_flash/pres/counts), on Engaging.
#
#   mkdir -p logs          # Slurm will not create the -o/-e directory for you
#   sbatch slurm/run_pres_counts.sbatch                       # both sources, both monkeys, allplus
#   sbatch slurm/run_pres_counts.sbatch --monkey Faure
#   sbatch slurm/run_pres_counts.sbatch --scope publication
#
# Anything after the script name is forwarded to the module. Same cheap,
# read-only load as `run_counts.sbatch` (the already-built windowed feature
# cache, ~12-30 MB, plus ~5 KB of labels) -- still a job rather than a
# login-node command, because the login node's RLIMIT_NPROC kills numpy.
#
# It needs the feature cache to already *exist*: see `run_counts.sbatch`'s
# note on `load_features` racing a cold attractor conversion.
#
# Results leave ~/strategy-selection/eye_pre_flash/pres/counts/ (run on the
# laptop):
#   rsync -av --exclude='__pycache__' \
#       engaging:strategy-selection/eye_pre_flash/pres/counts/ ./eye_pre_flash/pres/counts/

#SBATCH -J eyeprescounts
#SBATCH -p ou_bcs_low
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o logs/eyeprescounts-%j.out
#SBATCH -e logs/eyeprescounts-%j.err

# One core: loading one npz and drawing one heatmap, both serial.

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

uv run python -u -m eye_pre_flash.pres.counts.build "$@"

echo
echo "finished    $(date)"
