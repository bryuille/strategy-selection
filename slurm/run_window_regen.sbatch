#!/bin/bash
# Regenerate every figure whose pre-fixation window just changed.
#
#   mkdir -p logs          # Slurm will not create the -o/-e directory for you
#   sbatch slurm/run_window_regen.sbatch
#
# Why this run exists: `plot_io.PRE_FIX_END_MS` was 300 and is now 0, so the
# window runs `fix_start − 1466 ms → fix_start` everywhere instead of stopping
# 300 ms short. Before the change the repo was split -- the heatmap and
# strategy-similarity plots inherited the 300 while the occupancy, transition
# and decoding analyses shadowed it with 0 -- so only the figures that
# inherited it are stale. Those are exactly the four families below.
#
# NOT rerun, because their window was already 0 and their output is unchanged:
# similarities/{occupancy,occupancy_bin,transition,heatmap_log,heatmap,
# heatmap_fixations,heatmap_saccades}, saccades/*, movie/*, and the whole
# eye_pre_flash/classifier pipeline. `data/occupancy.py` takes the constant as
# a default argument, but every caller passes `end_ms` explicitly, so that
# default change moves nothing on its own.
#
# Strategy labels are NOT rebuilt here: the label sweep is the expensive half
# of `run_similarity.sbatch` and the window has no bearing on labelling. The
# job prints the label count and the scope-based plots skip what is missing.
#
# Results leave ~/strategy-selection/eye_pre_flash/plotting/out/
# (run on the laptop):
#   rsync -av engaging:strategy-selection/eye_pre_flash/plotting/out/ \
#       ./eye_pre_flash/plotting/out/

#SBATCH -J eyewin
#SBATCH -p mit_normal
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -o logs/eyewin-%j.out
#SBATCH -e logs/eyewin-%j.err

# Sizing:
#
#   mem    one monkey's eye npz is ~6 GB and the per-maze renderer holds
#          320x320 count maps alongside it. 32G matches run_similarity.sbatch
#          and leaves room for the 12x12 label matrices, which are the
#          heaviest step here.
#
#   cores  one. Every module here is serial -- no joblib, no n_jobs, and BLAS
#          is pinned to a single thread below. Requested cores are held whether
#          used or not, so asking for more would take allocation from other
#          users and queue worse for nothing.
#
#   time   plotting only, no conversions and no labelling: the similarity job
#          notes the plotting half runs in minutes. 6 h is slack for the
#          per-maze pass at the fine bin width across both monkeys.

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
echo "labels      $(ls data/processed/*_strategy_choices.npz 2>/dev/null | wc -l) of 23 on disk"
echo "window      $(uv run python -c 'from eye_pre_flash.plotting.plot_io import window_label; print(window_label())')"
echo

# The six renderings of the maze heatmaps, plus the per-maze pass. Each takes
# --monkey and defaults to Faure, so both are named explicitly.
echo "=== 1/4  heatmap_maze ==="
for mod in core sum sum_log sum_log_norm sum_log_norm_diff sum_norm sum_norm_diff; do
    for monkey in Faure Nielsen; do
        echo "--- heatmap_maze.$mod --monkey $monkey"
        uv run python -m "eye_pre_flash.plotting.heatmap_maze.$mod" --monkey "$monkey"
    done
done

# These default to every monkey and every scope, so a bare call is the full set.
echo
echo "=== 2/4  heatmap_label ==="
for mod in sum sum_log sum_log_norm sum_log_norm_diff sum_norm sum_norm_diff; do
    echo "--- heatmap_label.$mod"
    uv run python -m "eye_pre_flash.plotting.heatmap_label.$mod"
done

echo
echo "=== 3/4  similarities.heatmap_eq ==="
for monkey in Faure Nielsen; do
    echo "--- heatmap_eq --monkey $monkey"
    uv run python -m eye_pre_flash.plotting.similarities.heatmap_eq --monkey "$monkey"
done

echo
echo "=== 4/4  similarities.heatmap_labels ==="
for monkey in Faure Nielsen; do
    echo "--- heatmap_labels --monkey $monkey"
    uv run python -m eye_pre_flash.plotting.similarities.heatmap_labels --monkey "$monkey"
done

echo
echo "disk        $(du -sh data 2>/dev/null | cut -f1) in data/"
echo "finished    $(date)"
