#!/bin/bash
# Single-maze 2x2 (H vs S) split-half occupancy similarity, on Engaging.
#
#   mkdir -p logs          # Slurm will not create the -o/-e directory for you
#   sbatch slurm/run_mazestratpair.sbatch
#
# Anything after the script name is forwarded to `mazestratpair.run`, so a
# single maze, monkey, feature, source or scope can be re-run without editing
# this file:
#
#   sbatch slurm/run_mazestratpair.sbatch --n-perm 50            # smoke run
#   sbatch slurm/run_mazestratpair.sbatch --monkey Faure --maze 2
#   sbatch slurm/run_mazestratpair.sbatch --all-monkeys --all-mazes
#
# The single-maze sibling of `eye_pre_flash.corr`. corr's 12x12 argues that its
# decisive read is the same-maze H-vs-S comparison -- geometry is identical on
# both sides, so a cross-strategy r below that maze's own split-half
# reliability is strategy-dependent gaze sampling and not a visual confound --
# and this package is that read on its own terms, defaulting to Nielsen maze 4
# (the one cell where both label sources agree the split is genuinely mixed,
# and the maze `classifier.labels.ANCHOR_MAJORITY` leaves unscored).
#
# Restricting to one maze is not just a crop. corr's shared half size is the
# minimum over every cell clearing `min_stable`, so maze 4's comfortable counts
# get halved at the size set by maze 1's near-empty sequential cell: Nielsen
# `Nov_1_g0` has 45 H and 36 S maze-4 trials and corr splits it into halves of
# 5, where these two cells alone give 18. Same estimator, 3.6x the trials per
# half, and therefore different -- better -- numbers for the same cells. The
# report prints both half sizes side by side so the difference is never read as
# a finding, and `--n-half` forces corr's size back for a like-for-like check.
#
# Each figure carries all three feature variants as three 2x2 panels, so the
# variant is a panel rather than a directory. Significance is the
# within-(session, maze) label-shuffle permutation null on
# delta = 0.5*(r_HH + r_SS) - r_HS. See eye_pre_flash/mazestratpair/mazestratpair.md.
#
# PRECONDITIONS, and unlike `run_corr.sbatch` this script does NOT build them:
# the per-session strategy labels and the windowed unit-H feature caches must
# already exist. Building them concurrently with another job races on the
# attractor conversion and kills both (see `run_saccades.sbatch`'s note on
# detector constants, and job 22404715, which died exactly that way). This job
# is read-only on those caches and fails fast below rather than triggering a
# cold build; run `sbatch slurm/run_corr.sbatch` first if either is missing.
#
# Results leave ~/strategy-selection/eye_pre_flash/mazestratpair/out/
# (run on the laptop):
#   rsync -av engaging:strategy-selection/eye_pre_flash/mazestratpair/out/ \
#       ./eye_pre_flash/mazestratpair/out/

#SBATCH -J eyepair
#SBATCH -p ou_bcs_low
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o logs/eyepair-%j.out
#SBATCH -e logs/eyepair-%j.err

# One core and 16G. This reads one already-built feature cache (~10-27 MB) plus
# ~5 KB of labels per session and does no neural conversion, so
# `run_corr.sbatch`'s 32G/12h -- a budget set by its label sweep, not by its
# analysis -- does not apply; `run_counts.sbatch`'s sizing does. Everything is
# serial: the CV and permutation loops are plain Python `for`s with no joblib
# and no `n_jobs`, and BLAS is pinned to one thread below, so extra cores would
# sit idle. The 2x2 permutation null is far cheaper per resample than corr's
# 12x12, and the pool is built once and only the labels reshuffled, so the
# default sweep for one (monkey, maze) is minutes. The 4h ask is headroom for
# a full `--all-monkeys --all-mazes` sweep.

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

# Fail fast rather than let `load_features` start a cold build that races any
# concurrent job on the same attractor conversion.
shopt -s nullglob
labels=("$STRATEGY_DATA_ROOT"/processed/*_strategy_choices.npz)
svm_labels=("$STRATEGY_DATA_ROOT"/processed/*_strategy_svm.npz)
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
echo "svm labels     ${#svm_labels[@]} of 23"
echo "unith caches   ${#caches[@]} of 4"
echo

uv run python -u -m eye_pre_flash.mazestratpair.run "$@"

echo
echo "finished    $(date)"
