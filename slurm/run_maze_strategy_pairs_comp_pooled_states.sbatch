#!/bin/bash
# Per-state breakdown of a comp_pooled cell
# (eye_pre_flash/maze_strategy_pairs/comp_pooled_states), on Engaging.
#
# run_maze_strategy_pairs_comp_pooled.sbatch reports one p per (maze, K) but
# not which codebook state produced it. This job opens one (monkey, maze) up
# across two or more K: the codebook coordinates at each K, the nearest-centre
# geometry between them, and the per-state visit rate and dwell split H vs S.
#
# The case it was written for: Nielsen maze 4 under svm/top_ten, where p sits
# at the permutation floor at every K from 5 up but not at K=4, across nearly
# every feature and variant. Those are the defaults, so a bare submit answers
# it.
#
#   mkdir -p logs          # Slurm will not create the -o/-e directory for you
#   sbatch slurm/run_maze_strategy_pairs_comp_pooled_states.sbatch
#   sbatch slurm/run_maze_strategy_pairs_comp_pooled_states.sbatch --per-session
#   sbatch slurm/run_maze_strategy_pairs_comp_pooled_states.sbatch \
#       --monkey Faure --maze 2 --k 4 5 6
#
# Anything after the script name is forwarded to the module.
#
# THIS JOB WRITES NO FILES. It is a read-only diagnostic over caches the
# comp_pooled sweep has already built, and its entire result is stdout. Read it
# on the cluster with `cat logs/eyepresmazestates-<jobid>.out`, or pull it down
# (run on the laptop):
#   rsync -av engaging:strategy-selection/logs/eyepresmazestates-*.out ./logs/
#
# PRECONDITIONS. It wants `<Monkey>_mspcomp_pooled_k<K>_unith_s1466_e0.npz` for
# each requested K to exist already -- the comp_pooled sweep builds these, so
# run that job first if this is a cold tree. `load_features` is build-on-miss,
# so a missing cache is not fatal; it just means this short job silently turns
# into a feature-extraction pass, which is why the memory and wall time below
# are sized for one. The check further down reports which K are already warm so
# that is visible in the log rather than a surprise in the runtime.

#SBATCH -J eyepresmazestates
#SBATCH -p ou_bcs_low
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/eyepresmazestates-%j.out
#SBATCH -e logs/eyepresmazestates-%j.err

# One core: this loads two caches and prints tables. There is no permutation
# loop here at all -- it computes no statistics the sweep has not already
# reported -- so the only real cost is the cache read, or a cold feature
# extraction if a requested K was never built. Memory and wall time are sized
# for that cold case, matching run_maze_strategy_pairs_comp_pooled.sbatch;
# warm, this finishes in seconds.

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
svm=("$STRATEGY_DATA_ROOT"/processed/*_strategy_svm.npz)
pooled=("$STRATEGY_DATA_ROOT"/processed/*_mspcomp_pooled_k*_unith_s1466_e0.npz)
shopt -u nullglob

# SVM labels are the only label source here (comp_pooled_states reads
# comp_pooled_build's SOURCE_SCOPES_WANTED, which is pinned to svm/top_ten), so
# their absence is fatal -- unlike a missing feature cache, nothing can rebuild
# them as a side effect.
if [ "${#svm[@]}" -eq 0 ]; then
  echo "FATAL: no *_strategy_svm.npz in \$STRATEGY_DATA_ROOT/processed"
  echo "       this diagnostic uses SVM labels only; build them first."
  exit 1
fi

echo "svm label caches       ${#svm[@]}"
echo "mspcomp_pooled caches  ${#pooled[@]} present"
for f in "${pooled[@]}"; do echo "                       $(basename "$f")"; done
echo

uv run python -u -m eye_pre_flash.maze_strategy_pairs.comp_pooled_states "$@"

echo
echo "finished    $(date)"
