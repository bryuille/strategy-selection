#!/bin/bash
# Per-maze codebook-size (K) sweep against strategy information
# (eye_pre_flash/clusters), on Engaging.
#
# Answers, for each (monkey, maze): which codebook size K carries the most
# information about the trial's strategy label? The codebook is fitted per
# maze, so 2 monkeys x 6 mazes x 11 K = 132 k-means fits, all of them driven
# off one cached fixation table per monkey.
#
#   mkdir -p logs          # Slurm will not create the -o/-e directory for you
#   sbatch slurm/run_clusters.sbatch                        # full sweep (see Sizing)
#   sbatch slurm/run_clusters.sbatch --dry-run              # cells + counts, no compute
#   sbatch slurm/run_clusters.sbatch --monkey Faure --block occ_ms occ_bin
#
# Anything after the script name is forwarded to the module.
#
# PRECONDITIONS. Unlike run_maze_strategy_pairs.sbatch, this job *does* build a
# cache on its first run: the per-monkey fixation table
# (`<Monkey>_clusters_fixtable_s1466_e0.npz`, ~120 MB Faure / ~190 MB Nielsen).
# Everything it needs to build that must already exist:
#
#   data/processed/<Monkey>_attractor_eye_data.npz
#   data/processed/<Monkey>_clean_eye_data_s1466_e0.npz
#   data/processed/<session>_strategy_{choices,svm}.npz
#
# Build them with `python -m eye_pre_flash.classifier.pipeline` (stages
# `labels` then `features`) or run_rebuild.sbatch first. The checks below fail
# fast rather than triggering a cold eye-data conversion, which needs the raw
# per-session behavioral npz and 64 GB, and which would race any concurrent job
# on the same conversion (see job 22404715).
#
# DO NOT run two instances of this script concurrently on a cold fixation
# table: both would build it and `savez_atomic` only makes the *rename* atomic,
# not the work in front of it. Let the first one finish, then fan out.
#
# Results leave ~/strategy-selection/eye_pre_flash/clusters/ (run on the laptop):
#   rsync -av --exclude='__pycache__' \
#       engaging:strategy-selection/eye_pre_flash/clusters/out/ \
#       ./eye_pre_flash/clusters/out/
#
# Run `python -m eye_pre_flash.clusters.checks` first if fixtable.py,
# codebook.py or information.py has been touched. Its synthetic half needs no
# cache; its equivalence half asserts, against the clf2 caches already on disk,
# that this package's feature blocks equal `classifier.features`' to the bit.

#SBATCH -J eyeclusters
#SBATCH -p ou_bcs_low
#SBATCH -c 1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/eyeclusters-%j.out
#SBATCH -e logs/eyeclusters-%j.err

# Sizing:
#
# One core. The cross-validation loop is serial Python and every fit is tiny
# (n <= ~900, d <= 576), so extra cores would idle; sklearn is pinned to one
# thread below for the same reason. Fan out with one job per monkey if the wall
# clock matters more than the allocation.
#
# 64 GB is for the fixation-table build, not the sweep. The attractor cache is
# ~98 KB/trial, so reading it costs ~1.4 GB (Faure) / ~2.1 GB (Nielsen) and the
# table itself ~120-190 MB. Once the table is cached the sweep needs well under
# 16 GB; drop the request if you know the table exists.
#
# The default sweep is 2 CV schemes x 2 monkeys x 2 (source, scope) x 6 mazes
# x (11 + 11 + 8) K = 1,440 cells. Each cell is one k-means plus roughly 2,000
# cross-validated logistic refits (the inner C grid on the observed run, then
# n_perm=200 shuffles at fixed C), so ~3M fits total: a few hours, dominated by
# the `bigram` block (d = K*K) and the `pooled` arm's 5 repeats.
# `--block occ_ms occ_bin` roughly halves it; `--dry-run` lists every target
# cell with its trial counts and computes nothing; `--monkey Faure --maze 2
# --block occ_ms --k 2 4 6 12 --n-perm 50` is the smoke test.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PATH="$HOME/.local/bin:$PATH"
export STRATEGY_DATA_ROOT="${STRATEGY_DATA_ROOT:-/home/byuille/orcd/scratch/strategy_selection_data}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-1}"

echo "host:    $(hostname)"
echo "args:    $*"
echo "data:    $STRATEGY_DATA_ROOT"
echo "started: $(date)"

# PRECONDITION checks. Missing caches are a setup error, not something to
# discover four hours in -- or worse, to have silently papered over: a partial
# attractor cache builds a partial fixation table that looks entirely healthy
# (measured: one session of Faure's 27 gave a 473-trial table with no warning).
missing=0
for monkey in Faure Nielsen; do
    for stem in "${monkey}_attractor_eye_data" "${monkey}_clean_eye_data_s1466_e0"; do
        if [ ! -f "$STRATEGY_DATA_ROOT/processed/${stem}.npz" ]; then
            echo "MISSING: processed/${stem}.npz" >&2
            missing=1
        fi
    done
done
if ! compgen -G "$STRATEGY_DATA_ROOT/processed/*_strategy_choices.npz" > /dev/null; then
    echo "MISSING: no *_strategy_choices.npz; run the labels stage first" >&2
    missing=1
fi
if [ "$missing" -ne 0 ]; then
    echo "Preconditions unmet -- see the PRECONDITIONS block in this script." >&2
    exit 1
fi

uv run python -u -m eye_pre_flash.clusters.build "$@"

echo "finished $(date)"
