#!/bin/bash
# Maze-strategy pairs with a FIXED five-state codebook (eye_pre_flash/msp),
# on Engaging.
#
# No k-means anywhere: the five prototypes sit at the unit-H origin and the
# four exits (eye_pre_flash/msp/codebook.py), for both monkeys. SVM labels at
# the top-ten scope, mazes 2-5, binary occupancy, three variants. One 2x2 per
# (monkey, variant, maze) plus a codebook/coverage figure per monkey.
#
#   mkdir -p logs                              # Slurm will not create -o/-e dirs
#   sbatch slurm/run_msp.sbatch --dry-run      # counts + codebook coverage, no stats
#   sbatch slurm/run_msp.sbatch                # full: 2 monkeys x 3 variants x 4 mazes
#   sbatch slurm/run_msp.sbatch --monkey Faure --maze 4 --n-perm 200   # smoke test
#   sbatch slurm/run_msp.sbatch --radius 1.2   # different ball radius (new cache)
#
# Anything after the script name is forwarded to the module.
#
# RUN --dry-run FIRST. It reports per-maze pooled H/S counts (which must equal
# the svm/top_ten counts of every other maze-strategy-pairs pipeline) and the
# share of in-window fixations the fixed codebook actually assigns, per maze
# and per state. That coverage number is the main caveat of this analysis
# (see eye_pre_flash/msp/msp.md) and should be read before any panel.
#
# PRECONDITIONS. The job builds its own feature cache on first run, one per
# (monkey, radius): `<Monkey>_msp_fixed5_r<radius>_unith_s1466_e0.npz`. What
# it is built FROM must already exist:
#
#   $STRATEGY_DATA_ROOT/processed/<Monkey>_attractor_eye_data.npz
#   $STRATEGY_DATA_ROOT/processed/<Monkey>_clean_eye_data_s1466_e0.npz
#   $STRATEGY_DATA_ROOT/processed/<Monkey>_eye_behavioral.npz
#   $STRATEGY_DATA_ROOT/processed/<session>_strategy_svm.npz   (20 in scope)
#
# Build them with run_rebuild.sbatch first. The checks below fail fast rather
# than triggering a cold eye-data conversion. Label files are read directly
# and a missing one raises inside the module, naming the session -- it never
# starts a neural conversion.
#
# DO NOT run two instances concurrently on a cold cache: both would build the
# same npz, and `savez_atomic` only makes the rename atomic, not the work in
# front of it. Let the first finish, then fan out with --monkey if you like.
#
# Results leave ~/strategy-selection/eye_pre_flash/msp/out/ (run on the laptop):
#   rsync -av --exclude='__pycache__' \
#       engaging:strategy-selection/eye_pre_flash/msp/out/ ./eye_pre_flash/msp/out/

#SBATCH -J eyepresmsp
#SBATCH -p ou_bcs_low
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o logs/eyepresmsp-%j.out
#SBATCH -e logs/eyepresmsp-%j.err

# One core: the permutation loop is serial Python and each round's groupings
# go through one BLAS call. Memory is set by the extraction pass, which
# materialises a monkey's attractor npz (~2.6 GB on disk for Nielsen) plus
# float64 working copies; 32G is the figure the per-maze pooled job ran at.
# Wall: two extraction passes plus 24 estimator runs at n_perm=1000 is about
# ten minutes on a laptop; 2 h is headroom for a slow node and `uv sync`.

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
behav=("$STRATEGY_DATA_ROOT"/processed/*_eye_behavioral.npz)
svm=("$STRATEGY_DATA_ROOT"/processed/*_strategy_svm.npz)
msp=("$STRATEGY_DATA_ROOT"/processed/*_msp_fixed5_r*_unith_s1466_e0.npz)
shopt -u nullglob

if [ "${#attractor[@]}" -eq 0 ] || [ "${#clean[@]}" -eq 0 ] || [ "${#behav[@]}" -eq 0 ]; then
  echo "FATAL: missing attractor, clean-eye or eye_behavioral caches in \$STRATEGY_DATA_ROOT/processed"
  echo "       run 'sbatch slurm/run_rebuild.sbatch' first."
  exit 1
fi

if [ "${#svm[@]}" -eq 0 ]; then
  echo "FATAL: no *_strategy_svm.npz in \$STRATEGY_DATA_ROOT/processed"
  echo "       this analysis uses SVM labels only; build them first."
  exit 1
fi

echo "attractor caches   ${#attractor[@]} of 2"
echo "clean eye caches   ${#clean[@]} of 2"
echo "behavioral caches  ${#behav[@]} of 2"
echo "svm label caches   ${#svm[@]} (20 in scope; a missing in-scope one raises)"
echo "msp feature caches ${#msp[@]} already built (2 for the default radius)"
echo

uv run python -u -m eye_pre_flash.msp.build "$@"

echo
echo "finished    $(date)"
