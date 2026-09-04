#!/bin/bash
# Strategy-labelled gaze similarity matrices on Engaging.
#
#   mkdir -p logs          # Slurm will not create the -o/-e directory for you
#   sbatch slurm/run_similarity.sbatch
#
# Two steps, in order, because the second needs what the first writes:
#
#   1. `data.labeler --sweep` builds any missing strategy label. It is the
#      expensive half: a label is ~5 KB but comes from a neural recording of up
#      to 14 GB, converted and freed one session at a time. Sessions whose label
#      already exists are skipped, so this is cheap once warm.
#
#   2. `similarities.heatmap_labels` builds the 12x12 (maze x decoded strategy)
#      matrices for both monkeys, every scope, both bin widths.
#
# Why this run exists: `data.labeler.CLUSTER_NAMING` now defaults to
# "relative", which names the clusters by which one maze 6 favours over maze 1
# rather than by absolute majority. It always names, so all 23 eligible sessions
# label, against 13 under the reference rule. Existing labels are untouched --
# the relative rule returns the same answer wherever the reference one
# succeeded -- so this only adds sessions, and `allplus` becomes the full 23.
#
# Sessions whose two clusters separate the anchor mazes poorly print a Fisher p
# and a WEAKLY SEPARATED flag. They are labelled anyway; anchor-maze vetting
# decides downstream whether they enter `all`.
#
# Results leave ~/strategy-selection/eye_pre_flash/plotting/out/similarities/
# (run on the laptop):
#   rsync -av engaging:strategy-selection/eye_pre_flash/plotting/out/similarities/ \
#       ./eye_pre_flash/plotting/out/similarities/

#SBATCH -J eyesim
#SBATCH -p mit_normal
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH -o logs/eyesim-%j.out
#SBATCH -e logs/eyesim-%j.err

# Sizing:
#
#   mem    set by the labeller, not the plots: `read_neural_mat` materialises
#          the whole FR_WH (~14 GB for june_24_g0) and `build_trial_timebins`
#          allocates a second ~1.5 GB array while it is still live. ~20 GB peak.
#          The plotting half loads one monkey's eye npz (~6 GB) plus per-trial
#          count maps, comfortably under the same ceiling. Do not go below 24G.
#
#   cores  one. Both halves are serial and BLAS is pinned to one thread below,
#          so nothing here fans out -- no joblib, no n_jobs. Requested cores are
#          held for the job whether or not it uses them, so asking for more
#          would take allocation from other users and queue worse for nothing.
#          (`run_classifier.sbatch` asks for 4 because RandomForest there uses
#          n_jobs=-1; that is the exception, not the pattern to copy.)
#
#   time   dominated by the 10 neural conversions the sweep still has to do
#          (7 of which then fail the naming gate, after conversion). Both halves
#          are resumable: labels that exist are skipped, and re-running the
#          plots is minutes.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# uv installs to ~/.local/bin and is not on PATH in a non-login shell.
export PATH="$HOME/.local/bin:$PATH"

# Pin BLAS to one thread; nothing here wants nested parallelism.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# No display on a compute node.
export MPLBACKEND=Agg

echo "host        $(hostname)"
echo "cores       ${SLURM_CPUS_PER_TASK:-?}"
echo "mem         ${SLURM_MEM_PER_NODE:-?} MB"
echo "started     $(date)"
echo "disk        $(du -sh data 2>/dev/null | cut -f1) in data/"
echo

echo "=== 1/2  strategy labels (naming rule: relative) ==="
uv run python -m data.labeler --type strategy --sweep

echo
echo "labels on disk  $(ls data/processed/*_strategy_choices.npz 2>/dev/null | wc -l) of 23"
echo

echo "=== 2/2  strategy-labelled similarity matrices ==="
for monkey in Faure Nielsen; do
    uv run python -m eye_pre_flash.plotting.similarities.heatmap_labels --monkey "$monkey"
done

echo
echo "neural npz  $(du -sh data/npz/Neural_Data 2>/dev/null | cut -f1)"
echo "disk        $(du -sh data 2>/dev/null | cut -f1) in data/"
echo "finished    $(date)"
