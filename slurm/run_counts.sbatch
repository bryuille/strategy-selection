#!/bin/bash
# H/S per-(session, maze) census heatmaps, on Engaging.
#
#   mkdir -p logs
#   sbatch slurm/run_counts.sbatch                       # both sources, Faure
#   sbatch slurm/run_counts.sbatch --monkey Nielsen      # forwarded to the module
#   sbatch slurm/run_counts.sbatch --source svm --scope all
#
# Light and read-only: it loads the already-built windowed feature cache
# (`{monkey}_clf2_k6_unith_s1466_e0.npz`, ~12-30 MB) plus the ~5 KB strategy
# labels, and writes two PNGs. Minutes, not hours -- but it is still a job
# rather than a login-node command, because the login node's RLIMIT_NPROC
# kills numpy.
#
# It needs the feature cache to *exist*: `load_features` would otherwise build
# it, which means a cold attractor conversion and a race with any job doing
# the same (see `run_saccades.sbatch`'s note on detector constants, and job
# 22404715, which died exactly that way). Run this after the job that builds
# the caches has finished, or -- as on 2026-09-10 -- for the monkey whose
# caches are already written.
#
# Both sources come off the same feature table on purpose: the `dendro` and
# `svm` figures then differ only in the label source, which is the comparison
# the pair exists to make. Do not regenerate one of them against a feature
# cache built under different detector constants and compare it with the other.
#
# Results leave ~/strategy-selection/eye_pre_flash/counts/ (run on the laptop):
#   rsync -av engaging:strategy-selection/eye_pre_flash/counts/ \
#       ./eye_pre_flash/counts/

#SBATCH -J eyecounts
#SBATCH -p ou_bcs_low
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o logs/eyecounts-%j.out
#SBATCH -e logs/eyecounts-%j.err

# One core: loading one npz and drawing one heatmap, both serial.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export STRATEGY_DATA_ROOT="${STRATEGY_DATA_ROOT:-/home/byuille/orcd/scratch/strategy_selection_data}"
export PATH="$HOME/.local/bin:$PATH"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "host        $(hostname)"
echo "args        $*"
echo "started     $(date)"
echo

if [ "$#" -gt 0 ]; then
  uv run python -u -m eye_pre_flash.counts.census "$@"
else
  for source in svm dendro; do
    echo "=== source $source ==="
    uv run python -u -m eye_pre_flash.counts.census --source "$source" --monkey Faure
    echo
  done
fi

echo "finished    $(date)"
