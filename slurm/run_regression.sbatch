#!/bin/bash
# Per-maze-offset logistic regression (regression/), on Engaging.
#
#   mkdir -p logs
#   sbatch slurm/run_regression.sbatch --dry-run
#   sbatch slurm/run_regression.sbatch
#
# Anything after the script name is forwarded to `python -m regression.build`.
# Fixation caches (<Monkey>_reg_fixed5_r1.npz) are built on first use.
# DO NOT run two instances concurrently on a cold cache.

#SBATCH -J regression
#SBATCH -p ou_bcs_low
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o logs/regression-%j.out
#SBATCH -e logs/regression-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PATH="$HOME/.local/bin:$PATH"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "host     $(hostname)"
echo "args     $*"
echo "started  $(date)"
echo

uv run python -u -m regression.checks
uv run python -u -m regression.build "$@"

echo
echo "finished $(date)"
