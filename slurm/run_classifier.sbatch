#!/bin/bash
# eye_pre_flash/classifier pipeline on Engaging.
#
#   mkdir -p logs          # Slurm will not create the -o/-e directory for you
#
# The phases want different resources, so they can run as separate jobs with
# the directives below overridden on the command line:
#
#   # 1. labels: neural conversions. The long, memory-hungry pole. Skipped
#   #    cheaply once every label exists.
#   sbatch slurm/run_classifier.sbatch --only convert labels
#
#   # 2. features + decoding: features builds the k=12 attractor caches and the
#   #    degree-space codebooks (eye npz in memory, ~1-2 h); decoding itself is
#   #    minutes.
#   sbatch slurm/run_classifier.sbatch --from features
#
# Everything after the script name is passed straight to the pipeline module.
# The labels sweep is resumable: sessions whose label already exists are skipped
# (and their stale npz reclaimed), so a TIMEOUT is fixed by resubmitting.

#SBATCH -J eyeclf
#SBATCH -p mit_normal
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -o logs/eyeclf-%j.out
#SBATCH -e logs/eyeclf-%j.err

# Sizing:
#
#   cores  the decoding stages fit RandomForest with n_jobs=-1 (capped to
#          SLURM_CPUS_PER_TASK via LOKY_MAX_CPU_COUNT); everything else is
#          serial with BLAS pinned to one thread. 4 cores covers it.
#
#   mem    `read_neural_mat` does `np.array(dataset)` on the whole FR_WH, and
#          the npz is uncompressed, so RAM tracks npz size -- up to ~14 GB
#          (june_24_g0, 144 neurons, the largest of the 23).
#          `build_trial_timebins` then allocates a second ~1.5 GB array while
#          the first is still live. ~20 GB peak; 32G leaves headroom. The
#          features stage loads one monkey's eye npz (~6 GB) plus its attractor
#          cache -- comfortably under the same ceiling. Do not go below 24G.
#
#   time   dominated by whichever of the 23 neural conversions still need to
#          run, plus the k=12 attractor builds in `features`. Resubmit with
#          --from to resume if it runs out.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# uv installs to ~/.local/bin and is not on PATH in a non-login shell.
export PATH="$HOME/.local/bin:$PATH"

# Nested parallelism is the classic way to make an HPC job slower than a laptop.
# RandomForest uses n_jobs=-1, so joblib forks one worker per core; if each
# worker also gets a full BLAS thread pool they fight over the same cores.
# Give the parallelism to joblib and pin BLAS to one thread per worker.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-1}"

# No display on a compute node, and the tables are rendered through pyplot.
export MPLBACKEND=Agg

echo "host        $(hostname)"
echo "cores       ${SLURM_CPUS_PER_TASK:-?}"
echo "mem         ${SLURM_MEM_PER_NODE:-?} MB"
echo "started     $(date)"
echo "disk        $(du -sh data 2>/dev/null | cut -f1) in data/"
echo

uv run python -m eye_pre_flash.classifier.pipeline "$@"

echo
# The sweep frees each session before starting the next, so this should sit near
# one session's size. If it has grown to tens of GB, the reclaim is not firing.
echo "neural npz  $(du -sh data/npz/Neural_Data 2>/dev/null | cut -f1)"
echo "disk        $(du -sh data 2>/dev/null | cut -f1) in data/"
echo "finished    $(date)"
