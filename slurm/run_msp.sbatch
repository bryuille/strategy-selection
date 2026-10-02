#!/bin/bash
# Maze-strategy pairs with a FIXED five-state codebook (msp/), on Engaging.
#
#   mkdir -p logs
#   sbatch slurm/run_msp.sbatch --dry-run
#   sbatch slurm/run_msp.sbatch
#   sbatch slurm/run_msp.sbatch --radius 0.5 --window pre1466   # legacy window
#   sbatch slurm/run_msp.sbatch --monkey Faure --maze 4 --n-perm 200
#
# Anything after the script name is forwarded to `python -m msp.build`.
#
# PRECONDITIONS. Feature caches build on first run. The default ``geofix``
# window re-detects from the pooled ``<Monkey>_eye_data.npz`` (pupil from
# per-session eye npzs under ``$STRATEGY_DATA_ROOT/npz/Eye_Data/<Monkey>/``
# when present) plus behavioral + svm labels. ``--window pre1466`` needs the
# attractor + clean-eye + behavioral caches instead.
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
export STRATEGY_DATA_ROOT="${STRATEGY_DATA_ROOT:-/home/byuille/orcd/scratch/strategy_selection_data}"
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
attractor=("$STRATEGY_DATA_ROOT"/processed/*_attractor_eye_data.npz)
eye=("$STRATEGY_DATA_ROOT"/processed/*_eye_data.npz)
behav=("$STRATEGY_DATA_ROOT"/processed/*_eye_behavioral.npz)
svm=("$STRATEGY_DATA_ROOT"/processed/*_strategy_svm.npz)
shopt -u nullglob

# The window decides which gaze cache the features need.
case " $* " in
  *" pre1466 "*) source_name="attractor"; n_source=${#attractor[@]} ;;
  *)             source_name="eye_data";  n_source=${#eye[@]} ;;
esac
if [ "$n_source" -eq 0 ] || [ "${#behav[@]}" -eq 0 ]; then
  echo "FATAL: missing ${source_name} or eye_behavioral caches in \$STRATEGY_DATA_ROOT/processed"
  exit 1
fi
if [ "${#svm[@]}" -eq 0 ]; then
  echo "FATAL: no *_strategy_svm.npz in \$STRATEGY_DATA_ROOT/processed"
  exit 1
fi

echo "${source_name} caches   ${n_source} (needs 2)"
echo "behavioral caches  ${#behav[@]} of 2"
echo "svm label caches   ${#svm[@]}"
echo

uv run python -u -m msp.build "$@"

echo
echo "finished    $(date)"
