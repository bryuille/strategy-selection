#!/bin/bash
# Maze-strategy pairs with a FIXED five-state codebook (msp/), on Engaging.
#
#   mkdir -p logs
#   sbatch slurm/run_msp.sbatch --dry-run
#   sbatch slurm/run_msp.sbatch
#   sbatch slurm/run_msp.sbatch --radius 0.5
#   sbatch slurm/run_msp.sbatch --monkey Faure --maze 4 --n-perm 200
#
# Anything after the script name is forwarded to `python -m msp.build`.
#
# PRECONDITIONS. Feature caches build on first run. Each trial is
# re-detected from the pooled ``<Monkey>_eye_data.npz`` (pupil from
# the per-session eye mats under ``<MAT_ROOT>/Eye_Data/<Monkey>/``
# when present) plus behavioral + svm labels.
#
# DO NOT run two instances concurrently on a cold cache.

#SBATCH -J msp
#SBATCH -p ou_bcs_low
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o logs/msp-%j.out
#SBATCH -e logs/msp-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PATH="$HOME/.local/bin:$PATH"
# MAT_ROOT / PROCESSED_ROOT come from data/config.py; read the processed one
# back for the checks below.
processed="$(uv run python -c 'import data.config as c; print(c.PROCESSED_ROOT)')"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "host        $(hostname)"
echo "args        $*"
echo "started     $(date)"
echo

shopt -s nullglob
eye=("$processed"/*_eye_data.npz)
behav=("$processed"/*_eye_behavioral.npz)
svm=("$processed"/*_strategy_svm.npz)
shopt -u nullglob

if [ "${#eye[@]}" -eq 0 ] || [ "${#behav[@]}" -eq 0 ]; then
  echo "FATAL: missing eye_data or eye_behavioral caches in $processed"
  exit 1
fi
if [ "${#svm[@]}" -eq 0 ]; then
  echo "FATAL: no *_strategy_svm.npz in $processed"
  exit 1
fi

echo "eye_data caches    ${#eye[@]} (needs 2)"
echo "behavioral caches  ${#behav[@]} of 2"
echo "svm label caches   ${#svm[@]}"
echo

uv run python -u -m msp.build "$@"

echo
echo "finished    $(date)"
