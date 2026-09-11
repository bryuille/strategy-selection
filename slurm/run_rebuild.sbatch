#!/bin/bash
# Cold rebuild of every cache, then the attractor and maze_strategy_pairs
# figures, on Engaging.
#
#   mkdir -p logs          # Slurm will not create the -o/-e directory for you
#   sbatch slurm/run_rebuild.sbatch
#   sbatch slurm/run_rebuild.sbatch june_24_g0        # one session's figures
#
# Anything after the script name overrides the session list for step 5.
#
# This script assumes `$STRATEGY_DATA_ROOT/processed/` is EMPTY or absent, and
# is the intended way to come back from deleting it. Nothing here moves caches
# aside or forces a refresh: every loader builds on miss, so an empty
# `processed/` is the invalidation. `npz/` must survive -- it is the source
# these caches are derived from, and rebuilding it means re-running
# `data.convert` against `mat/`, which this script does not do.
#
# Order matters; each step consumes what the previous one wrote:
#
#   0. `{session}_strategy_choices.npz` / `{session}_strategy_svm.npz` -- the
#      per-session strategy labels, both kinds off one conversion pass. A label
#      is ~5 KB but comes from a neural recording of up to 14 GB, so this is by
#      far the longest step and the reason for the 12 h budget. `--with-svm`
#      builds the dendrogram and SVM labels off the same conversion instead of
#      paying for it twice.
#   1. `{monkey}_eye_behavioral.npz`, `{monkey}_eye_data.npz` -- concatenated
#      from `npz/` (~3 GB and ~5 GB).
#   2. `{monkey}_clean_eye_data_s1466_e0.npz` -- I-DT fixations and saccades on
#      the clipped, unwarped window. Detection only; nothing is merged.
#   3. `{monkey}_attractor_eye_data.npz` -- unit-H warped positions, validity
#      mask and timebase. K-independent, so one cache serves every K.
#   4. `{monkey}_clf2_k{6,12}_unith_s1466_e0.npz` -- the k-means codebook and
#      the three feature blocks (occ_ms, occ_bin, bigram).
#   5. Figures.
#
# Steps 1-4 all happen inside `classifier.features`, which pulls each loader in
# turn; they are listed separately because that is the order the log shows and
# the order to read a failure in.
#
# Step 0 is separate and must not be skipped after an empty `processed/`: the
# label caches live there too, and everything label-dependent
# (`maze_strategy_pairs`, `counts`, `scatter`, `classifier.decoding`) silently
# skips with "no labelled trial" without them rather than failing loudly.
#
# Results leave ~/strategy-selection/eye_pre_flash/ (run on the laptop):
#   rsync -av engaging:strategy-selection/eye_pre_flash/plotting/saccades/out/ \
#       ./eye_pre_flash/plotting/saccades/out/
#   rsync -av --exclude='__pycache__' \
#       engaging:strategy-selection/eye_pre_flash/maze_strategy_pairs/ \
#       ./eye_pre_flash/maze_strategy_pairs/

#SBATCH -J eyerebuild
#SBATCH -p ou_bcs_low
#SBATCH -c 1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/eyerebuild-%j.out
#SBATCH -e logs/eyerebuild-%j.err

# One core: every step is a serial numpy/matplotlib loop, and an idle core is
# allocation this job holds without using. 64G because step 1 concatenates
# whole-session gaze into one object array before writing it; the figure steps
# need a fraction of that. Check `sacct -o MaxRSS` after the first run and size
# this down if it is comfortable.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export STRATEGY_DATA_ROOT="${STRATEGY_DATA_ROOT:-/home/byuille/orcd/scratch/strategy_selection_data}"
export PATH="$HOME/.local/bin:$PATH"
# No display on a compute node.
export MPLBACKEND=Agg
# Serial BLAS: -c 1 means extra threads would oversubscribe the one core.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

SESSIONS=("$@")
if [ ${#SESSIONS[@]} -eq 0 ]; then
  SESSIONS=(june_24_g0 june_8_g0 Nov_3_g0 Oct_22_g0)
fi

K_VALUES=(6 12)

monkey_of() {
  case "$1" in
    june_*|April_*) echo Faure ;;
    Nov_*|Oct_*|Dec_*) echo Nielsen ;;
    *) echo "unknown monkey for session $1" >&2; return 1 ;;
  esac
}

echo "host        $(hostname)"
echo "partition   ${SLURM_JOB_PARTITION:-?}"
echo "mem         ${SLURM_MEM_PER_NODE:-?} MB"
echo "data root   $STRATEGY_DATA_ROOT"
echo "sessions    ${SESSIONS[*]}"
echo "started     $(date)"
echo

echo "=== 0/5  strategy labels (dendrogram + SVM, one conversion pass) ==="
# Skips any session whose label already exists, so an interrupted run resumes
# by resubmitting, and one session that fails to convert is reported at the end
# rather than stopping the other 22.
uv run python -u -m data.labeler --type strategy --sweep --with-svm
echo "dendro labels on disk  $(ls "$STRATEGY_DATA_ROOT"/processed/*_strategy_choices.npz 2>/dev/null | wc -l) of 23"
echo "svm labels on disk     $(ls "$STRATEGY_DATA_ROOT"/processed/*_strategy_svm.npz 2>/dev/null | wc -l) of 23"

echo
echo "=== 1-4/5  caches: eye data -> I-DT events -> unit-H warp -> codebook + features ==="
# Builds every cache in dependency order for both monkeys, at both K. The
# attractor cache is built once per monkey and reused across K.
uv run python -u -m eye_pre_flash.classifier.features

echo
echo "=== 5/5  figures ==="
for session in "${SESSIONS[@]}"; do
  monkey="$(monkey_of "$session")"
  for maze in 1 2 3 4 5 6; do
    for k in "${K_VALUES[@]}"; do
      for view in average trials; do
        echo "--- $session maze $maze attractor k$k ($view)"
        uv run python -u -m eye_pre_flash.plotting.saccades.attractor \
          --session "$session" --monkey "$monkey" --maze "$maze" \
          --view "$view" --k "$k" \
          || echo "    FAILED: attractor k$k $session maze $maze $view"
      done
    done
  done
done

echo
echo "--- maze_strategy_pairs"
uv run python -u -m eye_pre_flash.maze_strategy_pairs.build \
  || echo "    FAILED: maze_strategy_pairs"

echo
echo "figures     $(find eye_pre_flash/plotting/saccades/out -name '*.png' 2>/dev/null | wc -l) png under saccades/out"
echo "finished    $(date)"
