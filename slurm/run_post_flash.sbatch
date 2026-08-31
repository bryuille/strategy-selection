#!/bin/bash
# Post-flash counterfactual-saccade analysis on Engaging.
#
#   mkdir -p logs          # Slurm will not create the -o/-e directory for you
#   sbatch slurm/run_post_flash.sbatch
#
# Does the one step that cannot run on a laptop checkout -- re-converting the
# behavioral npz so they carry feedback_time / answer_time / saccade_init /
# trial_end (added to data.convert.BEHAVIORAL_FIELDS after the original
# conversion) -- then rebuilds the eye-behavioral caches and runs
# eye_post_flash.counterfactual under both alignments. Behavioral mats are
# small; this is a light job. Pull results back with the same rsync as the
# classifier (out/ directories are not excluded), e.g.:
#
#   rsync -av engaging:strategy-selection/eye_post_flash/out/ ./eye_post_flash/out/

#SBATCH -J postflash
#SBATCH -p mit_normal
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -o logs/postflash-%j.out
#SBATCH -e logs/postflash-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

echo "host        $(hostname)"
echo "started     $(date)"
echo

# 1. Behavioral npz now include the post-flash event fields.
uv run python -m data.convert behavioral --overwrite

# 2. The eye-behavioral caches were built from the old npz; rebuild them.
rm -f data/processed/Faure_eye_behavioral.npz data/processed/Nielsen_eye_behavioral.npz

# 3. The analysis (rebuilds the caches on first load).
uv run python -m eye_post_flash.counterfactual --align feedback
uv run python -m eye_post_flash.counterfactual --align response

echo
echo "finished    $(date)"
