#!/bin/bash
# Maze-strategy pairs at a K chosen per maze from held-out data
# (eye_pre_flash/maze_strategy_pairs/elbow_build), on Engaging.
#
# The fix to run_maze_strategy_pairs_comp_pooled.sbatch's selection problem:
# that job sweeps K over 13 values and reports the K minimising p, which picks
# the codebook size ON the outcome statistic -- so its `best_p` is a minimum
# over 13 correlated tests with no correction, and means much less than it
# reads as. This job chooses K with the H/S labels never consulted: split each
# maze's trials 80/20 at the trial level, fit the k-means on train, score the
# held-out fixation centroids, and take the kneedle knee of the averaged curve.
# K is fixed before the estimator runs even once, so the reported p needs no
# correction over K.
#
#   mkdir -p logs          # Slurm will not create the -o/-e directory for you
#   sbatch slurm/run_maze_strategy_pairs_elbow.sbatch                    # full run
#   sbatch slurm/run_maze_strategy_pairs_elbow.sbatch --dry-run          # counts only
#   sbatch slurm/run_maze_strategy_pairs_elbow.sbatch --monkey Faure --maze 5
#
# Anything after the script name is forwarded to the module.
#
# RUN --dry-run FIRST. It reports, per (monkey, maze), the in-scope trial
# count, the pooled fixation-sample and fixation-centroid counts the selection
# has to work with, the H/S label counts, and whether the maze clears
# MIN_TRIALS_FOR_SPLIT. The H/S counts must match
# run_maze_strategy_pairs_comp_pooled.sbatch --dry-run for the same
# (monkey, source, scope, maze) -- both mask on the same label logic. The
# sample counts will NOT match: the pool here is restricted to the in-scope
# sessions, where comp_pooled's is label-blind over every QC-passing session.
#
# READ eye_pre_flash/maze_strategy_pairs/elbow.md BEFORE THE OUTPUT. Two
# things in it are load-bearing. Fixing K removes one degree of freedom, not
# the other four: 4 mazes x 2 features x 3 variants x 2 methods is 48 cells,
# and the pre-committed primary cell plus the Holm family over mazes are named
# there. And selected_k.csv's identifiability columns decide whether the chosen
# K is a measurement at all -- a one-session smoke test had the EV curve still
# climbing at K=10 with no plateau, so check ev_curves.png flattens inside the
# 2..20 grid before trusting any K.
#
# PRECONDITIONS. This job builds its own cache on first run -- ONE
# `<Monkey>_mspelbow_unith_s1466_e0.npz` per monkey, two in total, rather than
# comp_pooled's 26, because K is not in the stem (it differs per maze and lives
# inside the file). Everything they are built FROM must exist:
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
# The attractor cache must be COMPLETE, not merely present. A partial one
# (a laptop checkout can hold a single session while the scope names ten)
# produces numbers that look valid; elbow_features prints a WARNING naming
# every in-scope session that contributed no row. Grep the log for it.
#
# DO NOT run two instances concurrently on a cold cache: both would build the
# same npz and `savez_atomic` only makes the *rename* atomic, not the work in
# front of it. Let the first finish, then fan out with --monkey.
#
# Results leave ~/strategy-selection/eye_pre_flash/maze_strategy_pairs/out/elbow/
# (run on the laptop):
#   rsync -av --exclude='__pycache__' \
#       engaging:strategy-selection/eye_pre_flash/maze_strategy_pairs/out/elbow/ \
#       ./eye_pre_flash/maze_strategy_pairs/out/elbow/

#SBATCH -J eyepresmazeelbow
#SBATCH -p ou_bcs_low
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/eyepresmazeelbow-%j.out
#SBATCH -e logs/eyepresmazeelbow-%j.err

# One core, same reason as every job in this family: the permutation loop is
# serial Python and each round's groupings already go through one BLAS call.
# The k-means sweep is the one part that would thread, but it is a small
# fraction of the wall time and sklearn is pinned to one thread below so it
# cannot contend with anything.
#
# Measured sizing, both parts far below the 12 h ceiling, which is kept only to
# match the sibling jobs and to absorb a cold attractor build:
#
#   Selection: 2 monkeys x 4 mazes x 19 K x 20 splits = 3,040 k-means fits.
#   Benchmarked at 0.14-0.46 s per fit on 64k x 2 float32 (K=6..20,
#   n_init=10), and the real per-maze pools are ~30-55k samples, so ~15-25
#   min. The partition-stability diagnostic reuses those same codebooks rather
#   than refitting -- it adds only the pairwise ARI, 19 K x 190 pairs x 4
#   mazes x 2 monkeys, a few minutes. `--no-stability` skips the ARI and saves
#   no k-means at all.
#
#   Estimation: 48 cells (4 mazes x 2 features x 3 variants x 2 methods),
#   against comp_pooled's ~1,250 and run_maze_strategy_pairs' 480 -- because
#   there is no K axis left to sweep. Well under an hour.
#
#   Plus two attractor scans per monkey: one to build the selection pools, one
#   to assign features at the selected K. `--k-grid 2 4 6 8 10 --n-splits 3
#   --n-perm 200` is the smoke test.
#
# Memory is sized like the sibling comparison jobs. The pools this job holds in
# memory (per-trial samples plus per-fixation centroids for four mazes) are
# a few hundred MB at most; the attractor cache dominates, as everywhere else.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PATH="$HOME/.local/bin:$PATH"
export STRATEGY_DATA_ROOT="${STRATEGY_DATA_ROOT:-/home/byuille/orcd/scratch/strategy_selection_data}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# sklearn's KMeans reaches joblib/loky even when it is not parallelising, and
# loky probes the physical core count on import. Pinning it keeps that probe
# from disagreeing with the single core Slurm actually granted.
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-1}"

echo "host        $(hostname)"
echo "args        $*"
echo "started     $(date)"
echo

shopt -s nullglob
attractor=("$STRATEGY_DATA_ROOT"/processed/*_attractor_eye_data.npz)
clean=("$STRATEGY_DATA_ROOT"/processed/*_clean_eye_data_s1466_e0.npz)
svm=("$STRATEGY_DATA_ROOT"/processed/*_strategy_svm.npz)
elbow=("$STRATEGY_DATA_ROOT"/processed/*_mspelbow_unith_s1466_e0.npz)
shopt -u nullglob

# The eye pipeline is a hard precondition: building it is run_rebuild.sbatch's
# job, not this one. Fail before the first k-means rather than hours in.
if [ "${#attractor[@]}" -eq 0 ] || [ "${#clean[@]}" -eq 0 ]; then
  echo "FATAL: missing attractor or clean eye data in \$STRATEGY_DATA_ROOT/processed"
  echo "       run 'sbatch slurm/run_rebuild.sbatch' first."
  exit 1
fi

# SVM labels are this analysis' only label source (elbow_build pins
# SOURCE_SCOPES_WANTED to svm/top_ten), so their absence is fatal here too.
if [ "${#svm[@]}" -eq 0 ]; then
  echo "FATAL: no *_strategy_svm.npz in \$STRATEGY_DATA_ROOT/processed"
  echo "       this analysis uses SVM labels only; build them first."
  exit 1
fi

echo "attractor caches    ${#attractor[@]} of 2"
echo "clean eye caches    ${#clean[@]} of 2"
echo "svm label caches    ${#svm[@]}"
echo "mspelbow caches     ${#elbow[@]} already built (2 when complete)"
echo

uv run python -u -m eye_pre_flash.maze_strategy_pairs.elbow_build "$@"

echo
echo "finished    $(date)"
