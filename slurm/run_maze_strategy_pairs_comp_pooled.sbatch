#!/bin/bash
# Per-(monkey, maze) pooled maze-strategy pairs, swept over codebook size K
# (eye_pre_flash/maze_strategy_pairs/comp_pooled_build), on Engaging.
#
# The fix to run_maze_strategy_pairs_comp.sbatch's averaging problem: that job
# fits a codebook per (session, maze) and estimates one session at a time,
# then averages the p-values across sessions -- which is not a valid combined
# significance test and measurably washes out real effects. This job fits one
# codebook per (monkey, maze, K), pooling that maze's fixation samples across
# every in-scope session, then runs ONE pooled estimate per (maze, K), the
# same pooling `run_maze_strategy_pairs.sbatch` already uses -- just swept
# over the full K grid instead of only K=6/K=12. It answers: for each
# (monkey, maze), which K gives the lowest genuine (not averaged) p?
#
#   mkdir -p logs          # Slurm will not create the -o/-e directory for you
#   sbatch slurm/run_maze_strategy_pairs_comp_pooled.sbatch              # full sweep
#   sbatch slurm/run_maze_strategy_pairs_comp_pooled.sbatch --dry-run    # counts only
#   sbatch slurm/run_maze_strategy_pairs_comp_pooled.sbatch --monkey Faure --k 6 12
#
# Anything after the script name is forwarded to the module.
#
# RUN --dry-run FIRST. It reports the per-maze pooled H/S counts (which must
# match run_maze_strategy_pairs.sbatch's own dry run exactly -- both mask on
# the same label logic) and each maze's pooled fixation-sample count against
# the top of the K sweep, and costs no compute beyond building one feature
# cache at the smallest K.
#
# PRECONDITIONS. This job builds its own caches on first run -- one
# `<Monkey>_mspcomp_pooled_k<K>_unith_s1466_e0.npz` per (monkey, K), 26 of
# them for the default K grid. They are small (occupancy blocks only, no
# bigram, at most 6 codebooks each rather than comp's ~60), but everything
# they are built FROM must exist:
#
#   $STRATEGY_DATA_ROOT/processed/<Monkey>_attractor_eye_data.npz
#   $STRATEGY_DATA_ROOT/processed/<Monkey>_clean_eye_data_s1466_e0.npz
#   $STRATEGY_DATA_ROOT/processed/<session>_strategy_svm.npz
#
# Build them with `python -m eye_pre_flash.classifier.pipeline` or
# run_rebuild.sbatch first. The checks below fail fast rather than triggering
# a cold eye-data conversion, which needs the raw per-session behavioral npz
# and 64 GB, and which would race any concurrent job on the same conversion.
#
# DO NOT run two instances concurrently on a cold cache: both would build the
# same npz and `savez_atomic` only makes the *rename* atomic, not the work in
# front of it. Let the first finish, then fan out with --k.
#
# Results leave ~/strategy-selection/eye_pre_flash/maze_strategy_pairs/out/comp_pooled/
# (run on the laptop):
#   rsync -av --exclude='__pycache__' \
#       engaging:strategy-selection/eye_pre_flash/maze_strategy_pairs/out/comp_pooled/ \
#       ./eye_pre_flash/maze_strategy_pairs/out/comp_pooled/

#SBATCH -J eyepresmazecomppooled
#SBATCH -p ou_bcs_low
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/eyepresmazecomppooled-%j.out
#SBATCH -e logs/eyepresmazecomppooled-%j.err

# One core, same reason as both comparison jobs: the permutation loop is
# serial Python and each round's groupings already go through one BLAS call.
#
# Memory and wall time are sized like run_maze_strategy_pairs_comp.sbatch, not
# the lighter run_maze_strategy_pairs.sbatch, even though the *estimation*
# side of this job is far cheaper (no per-session multiplication -- roughly
# 2 monkeys x 13 K x 4 mazes x 2 features x 3 variants ~= 625
# cells, an order of magnitude below comp's ~6,250, and comparable to
# run_maze_strategy_pairs.sbatch's own 288). The expensive part this job
# shares with comp, not with the pooled job, is *feature extraction*: 26
# passes (2 monkeys x 13 K) over the attractor/behavioral/clean-event caches
# to fit codebooks, one cold pass per K exactly as comp_features.py pays --
# fitting at most 6 per-maze codebooks per pass here instead of ~60
# per-(session,maze) ones there is cheap, but the pass itself (scanning every
# usable trial to build the sample pools) is the same size. `--k 6 12
# --n-perm 200` is the smoke test.

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
comp_pooled=("$STRATEGY_DATA_ROOT"/processed/*_mspcomp_pooled_k*_unith_s1466_e0.npz)
shopt -u nullglob

# The eye pipeline is a hard precondition: building it is run_rebuild.sbatch's
# job, not this one. Fail before the first k-means rather than hours in.
if [ "${#attractor[@]}" -eq 0 ] || [ "${#clean[@]}" -eq 0 ]; then
  echo "FATAL: missing attractor or clean eye data in \$STRATEGY_DATA_ROOT/processed"
  echo "       run 'sbatch slurm/run_rebuild.sbatch' first."
  exit 1
fi

# SVM labels are this sweep's only label source (comp_pooled_build pins
# SOURCE_SCOPES_WANTED to svm/top_ten), so their absence is fatal here too.
if [ "${#svm[@]}" -eq 0 ]; then
  echo "FATAL: no *_strategy_svm.npz in \$STRATEGY_DATA_ROOT/processed"
  echo "       this sweep uses SVM labels only; build them first."
  exit 1
fi

echo "attractor caches       ${#attractor[@]} of 2"
echo "clean eye caches       ${#clean[@]} of 2"
echo "svm label caches       ${#svm[@]}"
echo "mspcomp_pooled caches  ${#comp_pooled[@]} already built (26 for the default K grid)"
echo

uv run python -u -m eye_pre_flash.maze_strategy_pairs.comp_pooled_build "$@"

echo
echo "finished    $(date)"
