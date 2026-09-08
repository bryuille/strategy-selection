#!/bin/bash
# Maze x decoded-strategy occupancy similarity matrices, on Engaging.
#
#   mkdir -p logs          # Slurm will not create the -o/-e directory for you
#   sbatch slurm/run_corr.sbatch
#
# Anything after the script name is forwarded to `corr.run`, so a single
# source, feature or scope can be re-run without editing this file:
#
#   sbatch slurm/run_corr.sbatch --scope allplus
#   sbatch slurm/run_corr.sbatch --source svm --feature occupancy --monkey Faure
#
# Three steps, in order, because each needs what came before:
#
#   1. `data.labeler --sweep --with-svm` builds any missing strategy label,
#      of BOTH kinds, off one conversion per session -- a label is ~5 KB but
#      comes from a neural recording of up to 14 GB, converted and freed one
#      session at a time. This is why `--with-svm` matters here: converting
#      once and building both labels before freeing the neural npz halves the
#      cluster cost against running the dendrogram sweep and a later SVM
#      sweep separately. Sessions whose label(s) already exist are skipped,
#      so this is cheap once warm.
#
#   2. `classifier.features` builds any missing occupancy/bigram feature
#      cache (both K, both spaces, both monkeys) -- cheap once warm, since it
#      is keyed off the eye data, not the neural data.
#
#   3. `corr.run` sweeps 2 label sources x 3 features x 3 variants x 3 scopes
#      x 2 monkeys x 2 K x 2 spaces and writes figures plus `results_raw.csv`
#      / `nulls.csv` / `examples*.csv` per leaf.
#
# `corr` revives the retired `similarities.heatmap_labels` (a 12x12 over
# (maze x decoded strategy) built the same split-half way, against
# 102,400-dim spatial gaze maps) against low-dimensional codebook-occupancy
# vectors instead, computing every cell regardless of count rather than
# needing both the H and S side of a maze to clear a coverage floor, and adds
# a second, supervised label source (a maze-1-vs-6 linear SVM) alongside the
# unsupervised dendrogram clustering. See `eye_pre_flash/corr/corr.md`.
#
# Results leave ~/strategy-selection/eye_pre_flash/corr/out/
# (run on the laptop):
#   rsync -av engaging:strategy-selection/eye_pre_flash/corr/out/ \
#       ./eye_pre_flash/corr/out/

#SBATCH -J eyecorr
#SBATCH -p mit_normal
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/eyecorr-%j.out
#SBATCH -e logs/eyecorr-%j.err

# Sizing:
#
#   mem    set by the label sweep, not the analysis. `read_neural_mat`
#          materialises the whole FR_WH (~14 GB for june_24_g0) while
#          `build_trial_timebins` allocates another ~1.5 GB while it is still
#          live -- ~20 GB peak. The analysis half loads one `clf2_*` feature
#          cache (megabytes) plus an (n_trials, d) block per leaf, nowhere
#          near that ceiling. Do not go below 24G.
#
#   cores  one. Both halves are serial and BLAS is pinned to one thread below
#          -- the linear SVM is single-threaded and the CV/permutation loops
#          are plain Python `for`s, no joblib, no n_jobs anywhere in this
#          path. Requested cores are held for the job whether or not it uses
#          them, so asking for more takes allocation from other users and
#          queues worse for nothing. (`run_classifier.sbatch` asks for 4
#          because RandomForest there uses n_jobs=-1; that is the exception,
#          not the pattern to copy.)
#
#   time   two costs stack here. The label sweep is a cold ~23-session neural
#          conversion the first time either label kind is built (only 4
#          sessions' neural npz stay on disk locally; the sweep frees the
#          rest as it goes) -- this is why the job runs on the cluster rather
#          than the laptop. The analysis half is vectorised (the 12x12
#          per-split correlation is two matrix multiplies, not 144 individual
#          `pearson()` calls) -- measured at ~40s for one (source, feature,
#          variant, monkey, k, space) leaf's `allplus` scope (23 sessions,
#          --n-perm 1000, the expensive scope; `publication`/`all` are much
#          cheaper). The full sweep is 144 such leaves, so well under 2h once
#          labels exist; the 12h budget is headroom for a cold label sweep,
#          which is resumable if it is cut short. Both halves are resumable:
#          labels that exist are skipped, and a partial `corr.run` can be
#          re-run scoped to what didn't finish via the `"$@"` forwarding
#          above.

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

echo "=== 1/3  strategy labels (dendrogram + SVM, one conversion pass) ==="
uv run python -u -m data.labeler --type strategy --sweep --with-svm

echo
echo "dendro labels on disk  $(ls data/processed/*_strategy_choices.npz 2>/dev/null | wc -l) of 23"
echo "svm labels on disk     $(ls data/processed/*_strategy_svm.npz 2>/dev/null | wc -l) of 23"
echo

echo "=== 2/3  occupancy/bigram feature caches (k6+k12, deg+unith, both monkeys) ==="
uv run python -u -m eye_pre_flash.classifier.features

echo
echo "=== 3/3  12x12 (maze x strategy) occupancy correlation matrices ==="
uv run python -u -m eye_pre_flash.corr.run "$@"

echo
echo "neural npz  $(du -sh data/npz/Neural_Data 2>/dev/null | cut -f1)"
echo "disk        $(du -sh data 2>/dev/null | cut -f1) in data/"
echo "finished    $(date)"
