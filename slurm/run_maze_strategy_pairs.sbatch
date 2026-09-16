#!/bin/bash
# Merged-K single-maze H-vs-S similarity figures (eye_pre_flash/maze_strategy_pairs),
# on Engaging. Each figure carries both K=6 and K=12 as side-by-side panels.
#
# NOTE: this job is optional now. The estimator is a few seconds per panel and
# reads only the label and feature caches, so the whole sweep runs on a laptop
# in about 15 minutes -- see cloud.md for the rsync that brings those caches
# down. This script stays for running it beside run_rebuild.sbatch, which
# still chains it.
#
#   mkdir -p logs          # Slurm will not create the -o/-e directory for you
#   sbatch slurm/run_maze_strategy_pairs.sbatch                  # full sweep (see below)
#   sbatch slurm/run_maze_strategy_pairs.sbatch --monkey Faure --maze 2
#
# Anything after the script name is forwarded to the module. Same
# read-only feature-cache load as `run_rebuild.sbatch` -- still a job
# rather than a login-node command, because the login node's RLIMIT_NPROC
# kills numpy.
#
# PRECONDITIONS, and like `run_rebuild.sbatch` this script does NOT
# build them: the per-session strategy labels and the windowed unit-H feature
# caches must already exist. This job is read-only on those caches and fails
# fast below rather than triggering a cold build that races a concurrent job
# on the attractor conversion (see job 22404715).
#
# Results leave ~/strategy-selection/eye_pre_flash/maze_strategy_pairs/ (run on
# the laptop):
#   rsync -av --exclude='__pycache__' \
#       engaging:strategy-selection/eye_pre_flash/maze_strategy_pairs/ \
#       ./eye_pre_flash/maze_strategy_pairs/
#
# Run `python -m eye_pre_flash.maze_strategy_pairs.checks` first if the
# estimator has been touched: it is synthetic-data only and needs no cache.

#SBATCH -J eyepresmazepair
#SBATCH -p ou_bcs_low
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 6:00:00
#SBATCH -o logs/eyepresmazepair-%j.out
#SBATCH -e logs/eyepresmazepair-%j.err

# One core: the permutation loop is serial Python, but each round's 100
# groupings go through one BLAS call, so extra cores would not help without
# restructuring.
#
# The default sweep is 2 estimator methods x 2 monkeys x 6 mazes x 2 features
# x 2 (source, scope) x 3 variants = 240 figures, each needing one estimator
# run per K = 480 runs at n_perm=1000 and n_rounds=100. Measured at ~15 min
# on a laptop; the 6 h wall is headroom for a cold label build (see
# PRECONDITIONS). `--dry-run` lists targets and pooled trial counts without
# computing anything, and `--maze 4 --n-perm 200` is the smoke test.

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

# The feature caches are still a hard precondition: building those means the
# whole eye pipeline, which is `run_rebuild.sbatch`'s job, not this one.
if [ "${#caches[@]}" -eq 0 ]; then
  echo "FATAL: no unit-H feature caches in \$STRATEGY_DATA_ROOT/processed"
  echo "       run 'sbatch slurm/run_rebuild.sbatch' first."
  exit 1
fi

# Labels are NOT a precondition: `classifier.labels.scope_sessions` builds them
# on miss like every other cache, and a session that cannot be labelled raises
# rather than dropping out of the pool. That build costs a neural conversion
# per session, which is why the time budget here is hours rather than minutes.
# Do not run this concurrently with another label-building job (see job
# 22404715, where a cold conversion raced one).
echo "dendro labels  ${#labels[@]} of 23 already built"
echo "unith caches   ${#caches[@]} of 4"
echo

uv run python -u -m eye_pre_flash.maze_strategy_pairs.build "$@"

echo
echo "finished    $(date)"
