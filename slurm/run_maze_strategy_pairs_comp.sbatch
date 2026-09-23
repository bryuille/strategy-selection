#!/bin/bash
# Per-(session, maze) maze-strategy pairs, swept over codebook size K
# (eye_pre_flash/maze_strategy_pairs/comp_build), on Engaging.
#
# The comparison arm to run_maze_strategy_pairs.sbatch. That job fits one
# codebook per (monkey, K) and pools every in-scope session's trials of a maze;
# this one fits a codebook per (session, maze) and estimates one session at a
# time, then averages the p-values across sessions. It answers: for each
# (monkey, maze), which K gives the lowest mean p?
#
#   mkdir -p logs          # Slurm will not create the -o/-e directory for you
#   sbatch slurm/run_maze_strategy_pairs_comp.sbatch              # full sweep
#   sbatch slurm/run_maze_strategy_pairs_comp.sbatch --dry-run    # counts only
#   sbatch slurm/run_maze_strategy_pairs_comp.sbatch --monkey Faure --k 6 12
#
# Anything after the script name is forwarded to the module.
#
# RUN --dry-run FIRST. `estimator.MIN_TRIALS` is 10 per strategy and was
# written for pools spanning ten sessions; one session's trials of one maze
# will often not clear it. The dry run reports the per-(session, maze) H/S
# counts and exactly which cells qualify, and costs no compute beyond building
# one feature cache. If most cells are skipped, lower it with --min-trials
# rather than reading an average that rests on two sessions.
#
# PRECONDITIONS. Unlike run_maze_strategy_pairs.sbatch this job builds its own
# caches on first run -- one `<Monkey>_mspcomp_k<K>_unith_s1466_e0.npz` per
# (monkey, K), 26 of them for the default K grid. They are small (occupancy
# blocks only, no bigram), but everything they are built FROM must exist:
#
#   $STRATEGY_DATA_ROOT/processed/<Monkey>_attractor_eye_data.npz
#   $STRATEGY_DATA_ROOT/processed/<Monkey>_clean_eye_data_s1466_e0.npz
#   $STRATEGY_DATA_ROOT/processed/<session>_strategy_svm.npz
#
# Build them with `python -m eye_pre_flash.classifier.pipeline` or
# run_rebuild.sbatch first. The checks below fail fast rather than triggering a
# cold eye-data conversion, which needs the raw per-session behavioral npz and
# 64 GB, and which would race any concurrent job on the same conversion.
#
# DO NOT run two instances concurrently on a cold cache: both would build the
# same npz and `savez_atomic` only makes the *rename* atomic, not the work in
# front of it. Let the first finish, then fan out with --k.
#
# Results leave ~/strategy-selection/eye_pre_flash/maze_strategy_pairs/out/comp/
# (run on the laptop):
#   rsync -av --exclude='__pycache__' \
#       engaging:strategy-selection/eye_pre_flash/maze_strategy_pairs/out/comp/ \
#       ./eye_pre_flash/maze_strategy_pairs/out/comp/

#SBATCH -J eyepresmazecomp
#SBATCH -p ou_bcs_low
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/eyepresmazecomp-%j.out
#SBATCH -e logs/eyepresmazecomp-%j.err

# One core, for the same reason as the pooled job: the permutation loop is
# serial Python and each round's groupings already go through one BLAS call.
#
# Memory is 32G rather than 16G because the sweep loads the attractor eye data
# once per (monkey, K) to build a cache -- 26 times for the default grid. Each
# load is transient and the caches themselves are small, but the peak during a
# cold build is well above what the pooled job ever touches.
#
# Sizing. Extraction: 2 monkeys x 13 K = 26 passes, each fitting one k-means
# per (session, maze) over all six mazes. Estimation: 2 monkeys x 13 K x
# 4 mazes (2-5) x ~10 sessions x 2 features x 3 variants ~= 6,250
# cells at n_perm=1000, minus everything that fails coverage -- which on this
# data is most of them, so the real count is far lower and the wall below is
# mostly headroom for the cold cache build. `--k 6 12 --n-perm 200` is the
# smoke test.

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
clean=("$STRATEGY_DATA_ROOT"/processed/*_clean_eye_data_s1466_e0.npz)
svm=("$STRATEGY_DATA_ROOT"/processed/*_strategy_svm.npz)
comp=("$STRATEGY_DATA_ROOT"/processed/*_mspcomp_k*_unith_s1466_e0.npz)
shopt -u nullglob

# The eye pipeline is a hard precondition: building it is run_rebuild.sbatch's
# job, not this one. Fail before the first k-means rather than three hours in.
if [ "${#attractor[@]}" -eq 0 ] || [ "${#clean[@]}" -eq 0 ]; then
  echo "FATAL: missing attractor or clean eye data in \$STRATEGY_DATA_ROOT/processed"
  echo "       run 'sbatch slurm/run_rebuild.sbatch' first."
  exit 1
fi

# SVM labels are this sweep's only label source (comp_build pins
# SOURCE_SCOPES_WANTED to svm/top_ten), so their absence is fatal here even
# though the pooled job tolerates it.
if [ "${#svm[@]}" -eq 0 ]; then
  echo "FATAL: no *_strategy_svm.npz in \$STRATEGY_DATA_ROOT/processed"
  echo "       this sweep uses SVM labels only; build them first."
  exit 1
fi

echo "attractor caches  ${#attractor[@]} of 2"
echo "clean eye caches  ${#clean[@]} of 2"
echo "svm label caches  ${#svm[@]}"
echo "mspcomp caches    ${#comp[@]} already built (26 for the default K grid)"
echo

uv run python -u -m eye_pre_flash.maze_strategy_pairs.comp_build "$@"

echo
echo "finished    $(date)"
