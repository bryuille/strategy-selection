#!/bin/bash
# Run any `python -m <module> [args...]` on a compute node, on Engaging.
#
#   mkdir -p logs
#   sbatch -J eyecaches slurm/run_module.sbatch data.loader
#   sbatch -J counts    slurm/run_module.sbatch counts.build
#   sbatch -J persess   slurm/run_module.sbatch msp.per_session --radius 0.5
#   sbatch -J trends    slurm/run_module.sbatch msp_trends.build --radius 0.5 --labels dendro
#   sbatch -J scatter   slurm/run_module.sbatch scatter.build
#
# For packages with their own script (msp.build, regression.build) prefer
# run_msp.sbatch / run_regression.sbatch, which check their inputs first.
# Override the resources below on the command line (`--mem`, `-t`) when a
# job needs more; keep `-c 1` unless the module itself runs parallel work.
# MAT_ROOT / PROCESSED_ROOT come from data/config.py.

#SBATCH -J module
#SBATCH -p ou_bcs_low
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ "$#" -lt 1 ]; then
  echo "usage: sbatch slurm/run_module.sbatch <module> [args...]"
  exit 2
fi

export PATH="$HOME/.local/bin:$PATH"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "host        $(hostname)"
echo "module      $1"
echo "args        ${*:2}"
echo "started     $(date)"
echo

uv run python -u -m "$@"

echo
echo "finished    $(date)"
