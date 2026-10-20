# Timestamp preprocessing: QC review

Completed the four sessions selected dynamically from Figure Five’s current cross-validated Mahalanobis ranking. No session names are hard-coded in the pipeline.

Processed **1,530 selected canonical trials** and **2,345,089 recorded epoch samples**. All ten tests passed. Independent verification matched every saved observation to its source, verified all 14,686 detected events remain inside a usable continuous segment, and decoded all four movies.

Original paper and collaborator analyses are unchanged. The pipeline uses recorded float64 timestamps on the user-specified task/photodiode clock; it does not assume 1,000 Hz, refit clock synchronization, interpolate positions, or fill gaps. Actual measured flash events appear in alignment tables and movie annotations.

| Monkey | Session | Trials | Confirmed fixations | Saccades | Mean usable epoch time |
| --- | --- | ---: | ---: | ---: | ---: |
| Faure | june_24_g0 | 367 | 1844 | 1460 | 87.1% |
| Faure | june_8_g0 | 293 | 1688 | 1328 | 95.5% |
| Nielsen | Nov_3_g0 | 498 | 2703 | 2019 | 94.4% |
| Nielsen | Nov_6_g0 | 372 | 1936 | 1708 | 92.2% |

Usable time excludes invalid observations, blink masks, and detected interruptions. These percentages are per-trial averages, not the percentage of trials retained. No minimum coverage exclusion has been chosen for a later strategy analysis.

## Movies and normalized five-ROI maps

Movies show recorded gaze in degrees alongside the normalized H-maze. Playback is real time. Markers disappear inside unusable intervals; trails never cross segment breaks. Rank-1 sessions show a high-coverage trial, and rank-2 sessions show an interrupted trial. These are deliberately selected QC examples, not random samples. Movies continue past fixation onset for context; the event-preprocessing epoch remains maze onset through fixation onset.

### Faure: june_24_g0

[Play QC movie — trial 119](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Faure_june_24_g0/QC_trial_119.mp4) · [Normalized H-maze / five ROIs](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Faure_june_24_g0/normalized_H_ROIs.png)

[Sample-quality table (CSV.gz)](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Faure_june_24_g0/sample_quality.csv.gz) · [Trial summaries](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Faure_june_24_g0/trials.csv) · [Events](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Faure_june_24_g0/events.csv) · [Segments](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Faure_june_24_g0/segments.csv) · [Timing/invalidity intervals](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Faure_june_24_g0/quality_intervals.csv) · [Photodiode-event alignment](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Faure_june_24_g0/event_alignment.csv)

[Clean trial diagnostic](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Faure_june_24_g0/clean_trial_119.png)
[Interrupted trial diagnostic](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Faure_june_24_g0/interrupted_trial_493.png)

### Faure: june_8_g0

[Play QC movie — trial 366](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Faure_june_8_g0/QC_trial_366.mp4) · [Normalized H-maze / five ROIs](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Faure_june_8_g0/normalized_H_ROIs.png)

[Sample-quality table (CSV.gz)](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Faure_june_8_g0/sample_quality.csv.gz) · [Trial summaries](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Faure_june_8_g0/trials.csv) · [Events](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Faure_june_8_g0/events.csv) · [Segments](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Faure_june_8_g0/segments.csv) · [Timing/invalidity intervals](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Faure_june_8_g0/quality_intervals.csv) · [Photodiode-event alignment](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Faure_june_8_g0/event_alignment.csv)

[Clean trial diagnostic](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Faure_june_8_g0/clean_trial_295.png)
[Interrupted trial diagnostic](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Faure_june_8_g0/interrupted_trial_366.png)

### Nielsen: Nov_3_g0

[Play QC movie — trial 522](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Nielsen_Nov_3_g0/QC_trial_522.mp4) · [Normalized H-maze / five ROIs](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Nielsen_Nov_3_g0/normalized_H_ROIs.png)

[Sample-quality table (CSV.gz)](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Nielsen_Nov_3_g0/sample_quality.csv.gz) · [Trial summaries](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Nielsen_Nov_3_g0/trials.csv) · [Events](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Nielsen_Nov_3_g0/events.csv) · [Segments](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Nielsen_Nov_3_g0/segments.csv) · [Timing/invalidity intervals](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Nielsen_Nov_3_g0/quality_intervals.csv) · [Photodiode-event alignment](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Nielsen_Nov_3_g0/event_alignment.csv)

[Clean trial diagnostic](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Nielsen_Nov_3_g0/clean_trial_522.png)
[Interrupted trial diagnostic](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Nielsen_Nov_3_g0/interrupted_trial_520.png)

### Nielsen: Nov_6_g0

[Play QC movie — trial 199](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Nielsen_Nov_6_g0/QC_trial_199.mp4) · [Normalized H-maze / five ROIs](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Nielsen_Nov_6_g0/normalized_H_ROIs.png)

[Sample-quality table (CSV.gz)](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Nielsen_Nov_6_g0/sample_quality.csv.gz) · [Trial summaries](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Nielsen_Nov_6_g0/trials.csv) · [Events](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Nielsen_Nov_6_g0/events.csv) · [Segments](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Nielsen_Nov_6_g0/segments.csv) · [Timing/invalidity intervals](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Nielsen_Nov_6_g0/quality_intervals.csv) · [Photodiode-event alignment](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Nielsen_Nov_6_g0/event_alignment.csv)

[Clean trial diagnostic](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Nielsen_Nov_6_g0/clean_trial_236.png)
[Interrupted trial diagnostic](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/Nielsen_Nov_6_g0/interrupted_trial_199.png)

## What changed from the original preprocessing

- Preserved the original 2-degree / 100-ms fixation criteria, Engbert-2015 factor 8, 12-ms saccade minimum, 40-ms within-segment merge, final 50-ms blink padding, H-maze transform, and radius-1 five-ROI assignment.
- Replaced sample-count durations and fixed-rate derivatives with elapsed-time criteria and local time-weighted slopes of observed positions. This is a documented algorithm change; identical event labels are not expected.
- Apply invalidity/blink masking before detection, split at unusable samples and timestamp discontinuities, and never merge across a segment boundary.
- Match actual X/Y/pupil observations within 2 microseconds; preserve unmatched/ambiguous observations as unavailable. Save original source indices and matching residuals.
- Use a documented cadence-based gap rule, rather than the earlier 1.5-ms screening flag. Per-trial thresholds and configurable factors are saved.
- Compute duration-weighted fixation centroids and elapsed-span ROI dwell without the original median-interval cap. No duration is assigned across gaps.

The continuity threshold and local velocity window are explicit initial preprocessing choices. These outputs are ready for visual QC and sensitivity review, not evidence that strategy effects are already validated. The intended shared clock is retained as supplied; nearest-event residuals are not an independent synchronization test.

## Reproduce and inspect

[Implementation and complete parameter documentation](/Volumes/Portable/strategy-selection/timestamp_preprocessing/README.md) · [Run script](/Volumes/Portable/strategy-selection/timestamp_preprocessing/run_preprocessing.py) · [Core algorithms](/Volumes/Portable/strategy-selection/timestamp_preprocessing/core.py) · [Tests](/Volumes/Portable/strategy-selection/timestamp_preprocessing/test_preprocessing.py)

[Run manifest / hashes](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/manifest.json) · [Independent verification](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/verification.json) · [Selected ranking rows](/Volumes/Portable/strategy-selection/timestamp_preprocessing/outputs/top2_timestamp_20261006_v2/selected_sessions.csv)

The earlier partial `top2_timestamp_20261006` development output is superseded. Use `top2_timestamp_20261006_v2`.
