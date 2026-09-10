#!/bin/bash
# Presentation-ready single-panel mazestratpair figures (eye_pre_flash/pres/mazestratpair),
# on Engaging.
#
#   mkdir -p logs          # Slurm will not create the -o/-e directory for you
#   sbatch slurm/run_pres_mazestratpair.sbatch                  # default sweep: Nielsen, mazes 2+4
#   sbatch slurm/run_pres_mazestratpair.sbatch --monkey Faure --maze 2
#
# Anything after the script name is forwarded to the module. Same
# read-only feature-cache load as `run_mazestratpair.sbatch` -- still a job
# rather than a login-node command, because the login node's RLIMIT_NPROC
# kills numpy.
#
# PRECONDITIONS, and like `run_mazestratpair.sbatch` this script does NOT
# build them: the per-session strategy labels and the windowed unit-H feature
# caches must already exist. This job is read-only on those caches and fails
# fast below rather than triggering a cold build that races a concurrent job
# on the attractor conversion (see run_mazestratpair.sbatch, job 22404715).
#
# Results leave ~/strategy-selection/eye_pre_flash/pres/mazestratpair/ (run on
# the laptop):
#   rsync -av --exclude='__pycache__' \
#       engaging:strategy-selection/eye_pre_flash/pres/mazestratpair/ \
#       ./eye_pre_flash/pres/mazestratpair/

#SBATCH -J eyepresmazepair
#SBATCH -p ou_bcs_low
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH -o logs/eyepresmazepair-%j.out
#SBATCH -e logs/eyepresmazepair-%j.err

# One core: `pair.build_pools` / `pair.permutation_null` are plain serial
# Python loops, same as `run_mazestratpair.sbatch`. The default sweep here is
# at most 2 mazes x 2 features x 2 (source, scope) pairs = 8 of those calls,
# a fraction of `run_mazestratpair.sbatch`'s full label sweep, so 45 minutes
# is comfortable headroom.

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
labels=("$STRATEGY_DATA_ROOT"/processed/*_strategy_choices.npz)
caches=("$STRATEGY_DATA_ROOT"/processed/*_clf2_k*_unith_s1466_e0.npz)
shopt -u nullglob

if [ "${#labels[@]}" -eq 0 ] || [ "${#caches[@]}" -eq 0 ]; then
  echo "FATAL: missing prerequisites in \$STRATEGY_DATA_ROOT/processed"
  echo "       dendro labels: ${#labels[@]}, unit-H feature caches: ${#caches[@]}"
  echo "       run 'sbatch slurm/run_corr.sbatch' first -- it builds both."
  echo "       Do NOT let this job build them: it would race the attractor"
  echo "       conversion with any concurrent job (see job 22404715)."
  exit 1
fi

echo "dendro labels  ${#labels[@]} of 23"
echo "unith caches   ${#caches[@]} of 4"
echo

uv run python -u -m eye_pre_flash.pres.mazestratpair.build "$@"

echo
echo "finished    $(date)"
