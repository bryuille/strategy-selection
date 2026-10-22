# Current eye preprocessing handoff

Use `current.preprocess_trial` or `run_current.py` for the final stable-look/ROI rules. The collaborator's original `data/`, `msp/`, `regression/`, and `scatter/` implementations are untouched. The historical timestamp-native implementation (`core.py`, `run_preprocessing.py`) remains available to reproduce the first audit; its I-DT fixation events are NOT the current fixation definition.

## Run without paper analysis code

Install the repository dependencies (`uv sync`), then:

```sh
uv run python timestamp_preprocessing/run_current.py \
  --eye-root /path/to/Data/Physiology/Eye_Data \
  --trials /path/to/selected_trials.csv \
  --output /path/to/new_eye_output
```

Input CSV columns: `monkey,session,trial_id,epoch_ms,h1,h2,h3,h4,h5,h6`; optional `geometry_kind`. Trial IDs are one-based eye-cell indices. `epoch_ms` is the measured behavioral fixation-acquisition time minus maze presentation in milliseconds, not a sample count. Geometry must be six positive raw-degree arm lengths. `eye-root` contains `Monkey/Eye_Data_session.mat` (MATLAB v7.3): `Eye_Data/eye_x`, `eye_y`, `pupil_size` cell references, each Nx2 `[value, seconds_from_geo_present]`. Selection is explicit: the caller supplies behavior-QC-valid trials and their actual task/photodiode timestamps. No geometry is inferred from maze number; canonical and random geometry use the same code. No neural data or labels required.

Outputs: observed samples, raw-source matching indices, flags, full-trial and epoch segment IDs, local velocities in HDF5; events/stable intervals and provenance in JSON. Output directory must be new, outside the raw eye directory. Invalid inputs stop with session/trial context; no silent imputation. No image or movie exports in this command. Missing timestamps, invalid samples and discontinuities can be located using preserved original X-array row indices, `t`, `flags`, source indices and segment boundaries; this includes values outside the selected epoch, which are not themselves grounds to reject valid within-epoch samples.

For programmatic use:

```python
from timestamp_preprocessing.current import preprocess_trial
result = preprocess_trial(eye_x, eye_y, pupil_size, h, epoch_end_ms)
# result['looks'], ['saccades'], ['quality'], ['epoch_segment']
```

## Final definitions

| Setting | Interpretation |
|---|---|
| Time | Actual timestamps; no nominal 1000-Hz indexing, resampling, interpolation or filling |
| Clock | Assumed task/photodiode clock, relative to geo_present; no independent synchronization or clock repair claimed |
| X/Y/pupil match | Unique nearest observed timestamp within 0.002 ms; unmatched samples stay invalid |
| Missingness | Nonfinite/nonpositive pupil, invalid coordinates, unmatched/duplicate/nonincreasing time are flagged; raw pupil sentinel retained |
| Gaps | Break segments above max(5×median dt, 3×90th-percentile dt), computed on central90% of positive dt; not a universal physiological cutoff |
| Blinks | Existing loss/rapid-pupil heuristic;20-ms minimum,10-ms merge,final50-ms padding |
| Velocity | Time-weighted local observed-position slope within±4ms, at least3 points spanning2ms; separate per segment |
| Saccades | Engbert2015 scale, factor8 elliptical threshold,12-ms minimum,40-ms merge only within usable segment and finite velocity support |
| Stable fixation support | At least75ms elapsed duration; maximum pairwise Euclidean spread≤2 raw degrees |
| Qualified look | One candidate between detected saccades or segment breaks; union of qualifying stable windows within it |
| Counted duration | Sum of elapsed stable-island support, not candidate-envelope duration; nonstable intervals excluded |
| Position | Elapsed-time-weighted raw centroid over stable islands; no gap bridging |
| Center ROI | Raw centroid within2.5° of(0,0), priority over side regions |
| Side ROIs | Within2.5° of that side's finite horizontal/vertical arms, outsidecenter, respective x-sign; rest unassigned |
| Display | Warp raw centroid after assignment; normalized maze does not define current ROI membership |

A qualified look can contain disconnected stable islands or gradual drift, so the entire look need not have a≤2° diameter. This intentionally avoids splitting a continuous look merely when a greedy dispersion window ends. It does not merge across a detected saccade or unusable segment. Differing island ROI labels are flagged (`island_roi_disagreement`) and retained, matching the saved analysis; consumers should decide explicitly how to treat them. If velocity thresholds are degenerate/insufficient, `saccade_status` records that fact; absence of detected saccades in that case is not confirmed absence of movement.

The75ms criterion is for fixation qualification, NOT saccade duration. `Config.fixation_min_ms=100` belongs to historical I-DT; `current.py` discards those historical fixation events and applies the separate75ms detector. Quality/saccade settings are inherited unchanged. Current75ms/2° and raw2.5° ROI rules are fixed to reproduce the reviewed results.

## Reproduction and validation

```sh
uv run python -m unittest discover -s timestamp_preprocessing -p 'test_*.py'
uv run python timestamp_preprocessing/verify_handoff.py \
  --paper-root /path/to/MR_AG_MJ \
  --canonical-samples /path/to/top2_timestamp_20261006_v2
```

The optional second command compares every canonical+random selected trial in the saved four-session paper cohort against its reviewed per-look results: counts, raw/normalized centroids, intervals, durations, labels and ambiguity flags. It needs local recordings/outputs not distributed in Git. The self-contained unit tests require no recordings. Maximum-pairwise-distance computation uses O(n²) candidate memory; very long candidates may require substantial memory. No model equivalence to the original collaborator's pymovements event labels is asserted.

This handoff includes only preprocessing and focused validation. No private recordings, large output bundles, machine-specific AGENTS.md, exploratory neural regressions or paper labels are committed. The existing ranking-driven historical runner can recreate canonical QC inputs from a paper-root path; current runner takes an explicit selection to avoid hard-coding sessions. A manuscript claim that eye/task clocks are synchronized still depends on acquisition provenance.
