#!/bin/bash
# Maze x decoded-strategy occupancy similarity matrices, on Engaging.
#
#   mkdir -p logs          # Slurm will not create the -o/-e directory for you
#   sbatch slurm/run_corr.sbatch
#
# Anything after the script name is forwarded to `corr.run`, so a single
# source, feature or scope can be re-run without editing this file:
#
#   sbatch slurm/run_corr.sbatch --scope top_ten
#   sbatch slurm/run_corr.sbatch --source svm --feature occupancy --monkey Faure
#
# Three steps, in order, because each needs what came before:
#
#   1. `data.labeler --sweep --with-svm` builds any missing strategy label,
#      of BOTH kinds, off one conversion per session -- a label is ~5 KB but
#      comes from a neural recording of up to 14 GB. This is why `--with-svm`
#      matters here: converting once and building both labels off it avoids a
#      second cold conversion pass against running the dendrogram sweep and a
#      later SVM sweep separately. Sessions whose label(s) already exist are
#      skipped, so this is cheap once warm. The converted neural npz and
#      `trial_timebins` cache are left in place afterward, not deleted --
#      `STRATEGY_DATA_ROOT` (below) points them at scratch, not the 200G
#      /home quota that used to make deleting them worth it -- so a later
#      rerun or a different analysis over the same sessions does not pay the
#      conversion cost again. Pass `--reclaim-neural` to `data.labeler` to
#      restore the old delete-as-you-go behaviour.
#
#   2. `classifier.features` builds any missing occupancy/bigram feature
#      cache (both K, both monkeys) -- cheap once warm, since it is keyed off
#      the eye data, not the neural data. `corr` is unit-H only (`corr.md`
#      section 5), so the `deg` caches are not built for it.
#
#   3. `corr.run` sweeps 3 source-scope pairs (dendro/publication,
#      dendro/top_four, svm/top_ten -- scopes are per source, see
#      `corr.labels.SOURCE_SCOPES`) x 3 features x 3 variants x 2 monkeys
#      x 2 K = 108 figures, plus `results_raw_k<K>.csv` / `nulls_k<K>.csv` /
#      `examples*_k<K>.csv` per leaf.
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
#SBATCH -p ou_bcs_low
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 12:00:00
# `-c 1` is one core, but `squeue` will report CPUS=2 on this partition and
# `billing=2`. That is not a stray request: `SelectTypeParameters` is
# `CR_CORE_MEMORY`, so Slurm's smallest allocatable unit is a whole physical
# core, and `ou_bcs_low` nodes expose both hyperthreads (32 cores x 2 sockets
# x 2 threads = CPUTot 128). One core therefore *is* 2 CPUs here. `mit_normal`
# nodes are configured with CPUTot = core count (96 = 48 x 2), which is why the
# same `-c 1` reports 1 there. Nothing to fix -- the job holds one core either
# way and `--hint=nomultithread` cannot go below that -- but it is billed at 2,
# which is the price of `ou_bcs_low` starting immediately where `mit_normal`
# left this job in PENDING (Priority).
# `ou_bcs_low` rather than `mit_normal`: it allows 1-day jobs and had far
# more free nodes, and this job kept sitting in PENDING (Priority) on
# mit_normal. The 12h ask is unchanged and well inside the partition limit.
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
#          sessions' neural npz stay on disk on the laptop; scratch on the
#          cluster has room to keep all 23, so a rerun after this job is warm
#          throughout) -- this is why the job runs on the cluster rather than
#          the laptop. The analysis half is vectorised (the 12x12
#          per-split correlation is two matrix multiplies, not 144 individual
#          `pearson()` calls) -- measured at ~40s for one (source, feature,
#          variant, monkey, k) leaf over 23 sessions at --n-perm 1000. Every
#          scope is now smaller than that -- `svm/top_ten` is 10 sessions per
#          monkey, `dendro/top_four` 4 and `publication` 2 -- so the analysis
#          stage costs less than that estimate. The sweep is 108 monkey-level
#          figures, so well under 2h once
#          labels exist; the 12h budget is headroom for a cold label sweep,
#          which is resumable if it is cut short. Both halves are resumable:
#          labels that exist are skipped, and a partial `corr.run` can be
#          re-run scoped to what didn't finish via the `"$@"` forwarding
#          above.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# uv installs to ~/.local/bin and is not on PATH in a non-login shell.
export PATH="$HOME/.local/bin:$PATH"

# mat/npz/processed live on scratch, not the 200G /home quota -- see
# cloud.md "Storage: scratch, not /home".
export STRATEGY_DATA_ROOT="/home/byuille/orcd/scratch/strategy_selection_data"

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
echo "disk        $(du -sh "$STRATEGY_DATA_ROOT" 2>/dev/null | cut -f1) in \$STRATEGY_DATA_ROOT"
echo

echo "=== 1/3  strategy labels (dendrogram + SVM, one conversion pass) ==="
uv run python -u -m data.labeler --type strategy --sweep --with-svm

echo
echo "dendro labels on disk  $(ls "$STRATEGY_DATA_ROOT"/processed/*_strategy_choices.npz 2>/dev/null | wc -l) of 23"
echo "svm labels on disk     $(ls "$STRATEGY_DATA_ROOT"/processed/*_strategy_svm.npz 2>/dev/null | wc -l) of 23"
echo

echo "=== 2/3  occupancy/bigram feature caches (k6+k12, unith, both monkeys) ==="
uv run python -u -m eye_pre_flash.classifier.features --space unith

echo
echo "=== 3/3  12x12 (maze x strategy) occupancy correlation matrices ==="
uv run python -u -m eye_pre_flash.corr.run "$@"

echo
echo "neural npz  $(du -sh "$STRATEGY_DATA_ROOT/npz/Neural_Data" 2>/dev/null | cut -f1)"
echo "disk        $(du -sh "$STRATEGY_DATA_ROOT" 2>/dev/null | cut -f1) in \$STRATEGY_DATA_ROOT"
echo "finished    $(date)"
