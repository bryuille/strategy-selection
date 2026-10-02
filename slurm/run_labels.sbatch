#!/bin/bash
# Strategy labels (dendrogram + maze-1-vs-6 SVM), one array task per session.
#
#   mkdir -p logs
#   sbatch slurm/run_labels.sbatch                 # every CLUSTERING_SESSIONS entry
#   sbatch --array=0 slurm/run_labels.sbatch       # one session, as a test
#
# Task i labels `data.labels.CLUSTERING_SESSIONS[i]` with
# `python -m data.labels --type strategy --sweep --with-svm --session <s>`,
# reading the session's neural/behavioral/single-trial mats from MAT_ROOT and
# writing `<s>_trial_timebins`, `<s>_strategy_choices` and `<s>_strategy_svm`
# to PROCESSED_ROOT (both set in data/config.py). Sessions share no files, so
# tasks are independent; the sweep skips a session whose labels exist.
#
# Memory: the largest neural mat is ~3 GB compressed; its firing-rate matrix
# plus the dense (trial, neuron, time) tensor run to tens of GB. Single core:
# nothing here fans out.

#SBATCH -J labels
#SBATCH -p ou_bcs_low
#SBATCH -c 1
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH --array=0-22%8
#SBATCH -o logs/labels-%A_%a.out
#SBATCH -e logs/labels-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PATH="$HOME/.local/bin:$PATH"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

session="$(uv run python -c "from data.labels import CLUSTERING_SESSIONS as s; print(s[${SLURM_ARRAY_TASK_ID}])")"
echo "host        $(hostname)"
echo "session     ${session} (task ${SLURM_ARRAY_TASK_ID})"
echo "started     $(date)"
echo

uv run python -u -m data.labels --type strategy --sweep --with-svm --session "$session"

echo
echo "finished    $(date)"
