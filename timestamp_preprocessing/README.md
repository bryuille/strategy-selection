# Current handoff entry point

**Start with [HANDOFF.md](HANDOFF.md)** for the final75-ms, maximum-Euclidean-spread stable looks and raw center/side ROIs. Use `run_current.py` for portable canonical/random trial selections.

The documentation below describes the historical100-ms I-DT timestamp audit, retained for reproduction. Its fixation events and normalized five-ROI assignment are not the final current definitions.

# Timestamp-native eye preprocessing

New, separate implementation for the four sessions currently selected by Figure Five.
Original collaborator scripts and all paper scripts are unchanged. No neural data or
strategy labels are loaded. This stage prepares and visually checks eye observations.

## Run

From `/Volumes/Portable/strategy-selection`, using the isolated environment:

```bash
MPLCONFIGDIR=/private/tmp/strategy-selection-mpl .venv/bin/python timestamp_preprocessing/test_preprocessing.py
MPLCONFIGDIR=/private/tmp/strategy-selection-mpl .venv/bin/python timestamp_preprocessing/run_preprocessing.py \
  --paper-root /Volumes/Portable/MR_AG_MJ \
  --output timestamp_preprocessing/outputs/NEW_RUN
.venv/bin/python timestamp_preprocessing/verify_outputs.py timestamp_preprocessing/outputs/NEW_RUN
```

Each run requires a new output directory. Environment: upstream Python 3.14 / uv.lock,
plus `imageio-ffmpeg==0.6.0` for MP4 encoding. Installed versions and script hashes are
recorded in the manifest. Installing the extra encoder does not change uv.lock.

## Session selection and trial scope

Read `Figure_Five/01_Session_Ranking/Outputs/all_session_mahalanobis_results.csv`.
Within each monkey, require status beginning with `passed` and finite
`cv_mahalanobis`, sort descending stably, and take the first two sessions. This
matches the selection rule in Figure Five stage 2, including stable input-order ties.
The CSV hash and selected rows are saved. No session names are hard-coded.

Trial keys are monkey/session/trial ID. Repeated behavioral fields must agree across
neuron rows before collapsing. Retain published single-trial bounds, photodiode-QC
passing, nonfaded, canonical-maze trials. Export every selection/exclusion reason.
Eye cell index + 1 follows the existing acquisition convention; Eye_Data itself has
no independent trial ID. Random geometries are not included in this initial pipeline.

## Clock and epoch

Eye second columns are retained as float64 seconds converted to milliseconds,
relative to `geo_present`. The user's convention is that these share the task /
photodiode-aligned clock. We do not independently fit or certify a clock transform.
Behavioral `flash_one`, `flash_two`, and `flash_three` are the measured flash times;
they are retained for alignment reports and movie annotations, not replaced with
nominal schedules. The analysis interval is [maze onset, fixation onset]. The
fixation-onset event is behavioral; it is not relabeled as a photodiode event.
There is no assumed sampling rate and no integer-ms timestamp rounding.

## Missing observations and matching

- X is the anchor observation stream. Original X values and timestamps are unchanged.
- Y and pupil are matched to actual observations with a maximum 0.002 ms residual
  (2 microseconds, explicit tolerance for the observed export precision differences).
- Matching is unique: ambiguous duplicate timestamps / many-to-one claims are
  unavailable. Both source indices and residuals are exported. Unmatched values
  are NaN placeholders with explicit flags, never fabricated observations.
- Original pupil sentinel values are preserved and flagged, rather than overwritten.
- Invalid timestamps, nonfinite values, nonpositive pupil, missing pupil/Y matches,
  gaze outside +/-30 degrees, and blink padding make a sample unusable.
- No resampling, zero padding, forward filling, or temporal interpolation occurs.
  The geometric unit-H coordinate transform only transforms observed positions.

## Continuity rule (explicit initial QC policy)

For each trial, compute positive finite intersample intervals. Take their lower
90% to reduce the influence of long interruptions. The gap threshold is the larger
of 5 times that subset's median and 3 times its 90th percentile. Both factors are
CLI options and are recorded. This is a measured-cadence rule, not an assumed Hz
value or the earlier 1.5 ms screening flag. Fewer than ten positive intervals means
continuity cannot be estimated: samples are isolated and no events are inferred.

Break segments at any invalid sample, non-increasing time, or interval exceeding
that trial's measured gap threshold. No event, merge, velocity estimate, or QC
trajectory crosses a break. The numeric threshold is exported per trial. This
initial policy is explicit and reviewable; it is not a universally validated
physiological threshold. Cadence is estimated from the recorded trial, while gaze
velocity and fixation fitting use only the selected epoch.

## Retained concepts and necessary algorithm changes

| Component | New implementation |
| --- | --- |
| Fixations | Greedy I-DT, same 2-degree dispersion / 100 ms minimum; window determined by actual elapsed time, independently per valid segment. |
| Velocity | Time-weighted local linear slope of observed positions within +/-4 ms, at least 3 observations and 2 ms elapsed span. No slope through a gap and no imputed position. This replaces the fixed-rate five-point derivative. |
| Saccades | Same Engbert-2015 robust scale and factor 8; actual-time 12 ms minimum; merge at most 40 ms, only inside the same valid segment with finite velocity throughout. |
| Blink candidates | Preserve pupil loss/zero/nonfinite flags; detect large pupil changes using 5 times the 95th percentile of within-segment valid differences. Group/merge candidates with actual-time 10 ms gaps; blink event minimum 20 ms. |
| Blink masking | 50 ms padding, matching the original final sample mask, now applied BEFORE detection to avoid contamination and guarantee segmentation. Short pupil-loss samples are still invalid even if no 20 ms blink is declared. |
| Event overlap | Remove saccade samples from fixation candidates, split, and re-test actual duration. |
| Failed saccade threshold | Explicit status. Dispersion candidates are `fixation_unconfirmed`, excluded from confirmed fixation ROI maps/dwell, rather than assuming no saccades. |
| Unit H | Same piecewise geometry, evaluated algebraically on each real observation. No temporal interpolation. |
| Five ROIs | Original origin/LU/LD/RU/RD prototypes, nearest within radius 1.0; ties use first ROI. `origin` is the junction (0,0); the stem target is (0,1). |
| Fixation centroid | Duration-weighted observed positions using half adjacent intervals inside each fixation. |
| Dwell | Sum observed fixation elapsed spans per ROI; no median-period cap and no time assigned to gaps or intervals between fixations. |

Changes in derivatives, actual-time duration, pre-detection masking, centroids,
and event fragmentation mean outputs are not expected to match the original
sample-index pipeline exactly. This is an explicit replacement preprocessing
variant, not a claim of unchanged detection results. Keep the original as baseline.

## Outputs

Per session:

- `sample_quality.csv.gz`: one row per recorded X observation inside the epoch,
  source indices, actual matched values, matching residuals, quality flags, segment
  IDs, derived velocities and unit-H coordinates. No synthetic sampling grid.
- `samples.h5`: compressed equivalent arrays grouped by trial; original pupil
  observations are additionally retained on their native timestamps.
- `quality_intervals.csv`: invalid runs and timestamp discontinuities, with exact
  source file, source indices, time endpoints, and whether they intersect the epoch.
  Unlocatable nonfinite timestamps retain their source indices; time is not guessed.
- `trial_selection.csv`, `trials.csv`, `segments.csv`, `events.csv`: selection,
  coverage, segment boundaries, event times, duration, ROI, and boundary-touch flags.
- `event_alignment.csv`: nearest recorded observation to maze onset, fixation onset,
  and photodiode flash events, signed timing residual, and sample validity. This
  measures observation coverage, not independent clock synchronization accuracy.
- `normalized_H_ROIs.png`: per-maze maps with all confirmed fixation centroids.
- `clean_trial_*.png`, `interrupted_trial_*.png`: traces, velocity and normalized map.
- `QC_trial_*.mp4`: one movie per session, raw degrees beside normalized H/5 ROIs.
  Rank-1 session uses greatest observed coverage; rank-2 uses an interrupted trial
  with greatest usable duration. Choices are diagnostic, not random representatives.

Movies use 20 ms display steps at 50 fps (1x). Gaze markers select an actual nearest
usable sample only while the frame is inside the same continuous segment and the
residual is at most half the adaptive gap threshold. No usable observation means
no marker. Trails are limited to one segment. Flash annotations last 100 ms for
visibility; that is a DISPLAY duration, not a measured flash duration. Movies extend
through flash three +300 ms; only maze-onset-to-fixation gaze is event-preprocessed.

## Verification

Tests cover elapsed-time velocity/duration, gaps, pupil loss, unmatched samples,
duplicate time, dwell integration, geometry, dynamic ranking, and exact agreement
with the installed library's Engbert-2015 threshold definition.
`verify_outputs.py` independently compares every saved selected observation with
its raw source, checks every event stays within one usable continuous segment,
checks duration criteria, and decodes/counts all movie frames.

The interrupted first development run `top2_timestamp_20261006` is superseded;
use `top2_timestamp_20261006_v2` for the completed Engbert-2015 run.
