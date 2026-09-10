#!/bin/bash
# Pre-fixation saccade/gaze figures, on Engaging.
#
#   mkdir -p logs          # Slurm will not create the -o/-e directory for you
#   sbatch slurm/run_saccades.sbatch
#
# Anything after the script name overrides the session list, so a single
# session can be redone without editing this file:
#
#   sbatch slurm/run_saccades.sbatch june_24_g0
#   sbatch slurm/run_saccades.sbatch Nov_3_g0 Oct_22_g0
#
# Separate from `run_corr.sbatch` on purpose: this job only needs the eye
# pipeline (behavioral + eye conversions, the attractor cache, and the
# `classifier.features` windowed codebook). It never touches the neural
# recordings or the strategy labels, so it does not have to wait behind, or
# risk clobbering, the label sweep that `run_corr` runs first.
#
# Four steps, in order:
#
#   1. Re-convert the behavioral `.mat` files, but only if needed.
#      `fixation_cue_present` was only just added to
#      `data.convert.BEHAVIORAL_FIELDS`, so npz written before that read the
#      field back as all-NaN and the cue marker would silently never render.
#      The check is cheap and the conversion is skipped once it has been done,
#      which also stops step 2 from deleting a cache another job is reading.
#
#   2. Delete the `*_eye_behavioral.npz` caches so `load_eye_behavioral_data`
#      rebuilds them from the re-converted npz. `load_eye_behavioral_data`
#      only auto-rebuilds when a *wanted field is missing from the cache*, and
#      `fixation_cue_present` is present-but-NaN, so it would otherwise be
#      considered satisfied and never refresh.
#
#   3. `classifier.features --space unith` builds the K=6/K=12 windowed
#      codebooks, and on the way the windowed `clean_eye_data_s1466_e0` event
#      table -- I-DT run on the clipped, unwarped segment rather than the whole
#      trial. That is a different analysis, not a cheaper route to the same
#      one: I-DT is a greedy sequential scan, so clipping before vs after
#      detection agrees on only 7.8% of trials. It is also 4.2x cheaper
#      (measured: 11.0s vs 45.8s per 400 trials).
#
#   4. The three `saccades` modules over every maze, in the `average` and
#      `trials` views, plus the 3D trace view. `attractor` runs once per
#      `K_VALUES` entry (k6 and k12); the other two are codebook-independent.
#
# Changing a detector constant in `data.builder` (I-DT dispersion, I-VT
# velocity, the saccade or fixation duration floors) does NOT invalidate the
# caches on its own: `{monkey}_clean_eye_data_s*_e*`, `{monkey}_attractor_k*`
# and `{monkey}_clf2_*` stems carry the window and k, not the detector
# settings, so a rerun happily reloads events detected under the old value.
# Move them aside first, e.g.
#   cd "$STRATEGY_DATA_ROOT/processed"
#   for f in *_clean_eye_data_s1466_e0.npz *_attractor*_eye_data.npz \
#            *_clf2_*_unith_s1466_e0.npz; do mv "$f" "$f.idt2"; done
# Renaming rather than deleting keeps the previous run's caches recoverable.
#
# What changed in these figures (see `eye_pre_flash/classifier/features.py`):
#
#   * Window is 1466 ms, not 1533. `PRE_FIX_WINDOW_MS` was 1533.0 -- the
#     *median* `geo_present -> fix_start` gap rather than the minimum. The
#     measured minimum is 1466.2 ms (Nielsen), so the old window reached back
#     past `geo_present` in 25.1% of Nielsen trials and 0.2% of Faure's,
#     plotting gaze from before the maze was on screen. At 1466 no trial in
#     either animal is affected (0 of 40,314).
#   * I-DT only. `data.builder` runs I-DT and I-VT and stores both; the state
#     assignment used to take both via a substring match, letting a
#     longest-span tie-break pick between them (I-DT won ~99.5% of the time
#     anyway). Now I-DT alone, matching `saccades.labeled`, which already
#     filtered on `fixation_idt`.
#   * Windowed k-means. Prototypes come from `classifier.features`, fit on
#     in-window samples, not on whole-trial gaze.
#   * `fixation_cue_present` marked on every time axis. The cue sits a median
#     34 ms before fix_start and inside the window on 99.3% (Faure) / 99.9%
#     (Nielsen) of trials; the remainder are long outliers (out to -9.8 s) and
#     are omitted rather than forcing the axis open. Group views mark the
#     median across the trials plotted and say so in the label.
#
# Results leave ~/strategy-selection/eye_pre_flash/plotting/out/saccades/
# (run on the laptop):
#   rsync -av engaging:strategy-selection/eye_pre_flash/plotting/out/saccades/ \
#       ./eye_pre_flash/plotting/out/saccades/

#SBATCH -J eyesacc
#SBATCH -p ou_bcs_low
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/eyesacc-%j.out
#SBATCH -e logs/eyesacc-%j.err

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
# One core: every step here is a serial matplotlib/numpy loop, and idle cores
# are allocation this job holds without using. `ou_bcs_low` allows 1-day jobs;
# the 12h ask is headroom for a cold behavioral reconversion plus the figure
# sweep, which is dominated by per-trial PNG writes rather than computation.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export STRATEGY_DATA_ROOT="${STRATEGY_DATA_ROOT:-/home/byuille/orcd/scratch/strategy_selection_data}"
export PATH="$HOME/.local/bin:$PATH"
# No display on a compute node.
export MPLBACKEND=Agg
# Serial BLAS: -c 1 means extra threads would oversubscribe the one core.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

SESSIONS=("$@")
if [ ${#SESSIONS[@]} -eq 0 ]; then
  SESSIONS=(june_24_g0 june_8_g0 Nov_3_g0 Oct_22_g0)
fi

# Codebook sizes for the `attractor` module, which writes into a k-keyed
# directory (`out/saccades/attractor/<session>/k<k>/maze_<n>/`). Passed
# explicitly because the module otherwise falls back to `data.attractor`'s
# DEFAULT_K (12) and the k6 figures never get refreshed -- which is what
# happened on 2026-09-09, leaving k6 on disk older than k12. `traces` and
# `labeled` are not codebook-dependent and run once per view.
K_VALUES=(6 12)

monkey_of() {
  case "$1" in
    june_*|April_*) echo Faure ;;
    Nov_*|Oct_*|Dec_*) echo Nielsen ;;
    *) echo "unknown monkey for session $1" >&2; return 1 ;;
  esac
}

echo "host        $(hostname)"
echo "partition   ${SLURM_JOB_PARTITION:-?}"
echo "cores       ${SLURM_CPUS_PER_TASK:-?}"
echo "mem         ${SLURM_MEM_PER_NODE:-?} MB"
echo "sessions    ${SESSIONS[*]}"
echo "started     $(date)"
echo

echo "=== 1/4  behavioral mats: convert only if fixation_cue_present is missing ==="
# Idempotent on purpose. The re-conversion is only needed once, when
# `fixation_cue_present` is first added to `data.convert.BEHAVIORAL_FIELDS`;
# repeating it costs minutes and, worse, step 2 would then delete an
# `eye_behavioral` cache another job may be mid-read on.
need_convert=$(uv run python - <<'PYEOF'
import glob, os
import numpy as np
from data.convert import load_npz
R = os.environ["STRATEGY_DATA_ROOT"]
paths = sorted(glob.glob(R + "/npz/Behavioral_Data/*/*_good_trials_concat.npz"))
if not paths:
    print("yes"); raise SystemExit
for p in paths:
    d = load_npz(p)
    if "fixation_cue_present" not in d:
        print("yes"); raise SystemExit
    if not np.isfinite(np.asarray(d["fixation_cue_present"], float)).any():
        print("yes"); raise SystemExit
print("no")
PYEOF
)
if [ "$need_convert" = "yes" ]; then
  echo "  field missing -> re-converting"
  uv run python -u -m data.convert behavioral --overwrite
  echo
  echo "=== 2/4  drop eye_behavioral caches so the new field is picked up ==="
  rm -fv "$STRATEGY_DATA_ROOT"/processed/*_eye_behavioral.npz || true
else
  echo "  fixation_cue_present already present in every behavioral npz -> skipping"
  echo
  echo "=== 2/4  eye_behavioral caches left alone (nothing to pick up) ==="
fi

echo
echo "=== 3/4  windowed occupancy codebooks (k6+k12, unith, both monkeys) ==="
uv run python -u -m eye_pre_flash.classifier.features --space unith

echo
echo "=== 4/4  saccade figures ==="
for session in "${SESSIONS[@]}"; do
  monkey="$(monkey_of "$session")"
  for maze in 1 2 3 4 5 6; do
    for view in average trials; do
      for mod in traces labeled; do
        echo "--- $session maze $maze $mod ($view)"
        uv run python -u -m "eye_pre_flash.plotting.saccades.$mod" \
          --session "$session" --monkey "$monkey" --maze "$maze" --view "$view" \
          || echo "    FAILED: $mod $session maze $maze $view"
      done
      for k in "${K_VALUES[@]}"; do
        echo "--- $session maze $maze attractor k$k ($view)"
        uv run python -u -m eye_pre_flash.plotting.saccades.attractor \
          --session "$session" --monkey "$monkey" --maze "$maze" --view "$view" \
          --k "$k" \
          || echo "    FAILED: attractor k$k $session maze $maze $view"
      done
    done
    echo "--- $session maze $maze traces (3d average)"
    uv run python -u -m eye_pre_flash.plotting.saccades.traces \
      --session "$session" --monkey "$monkey" --maze "$maze" \
      --view average --mode 3d --save \
      || echo "    FAILED: traces 3d $session maze $maze"
  done
done

echo
echo "figures     $(find eye_pre_flash/plotting/out/saccades -name '*.png' 2>/dev/null | wc -l) png under out/saccades"
echo "finished    $(date)"
