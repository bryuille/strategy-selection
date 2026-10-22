# Handoff validation (2026-10-20)

-18 focused unit tests passed: timestamp-dependent velocity/duration, matching, gaps, invalid pupil, duplicate times, saccade boundaries, Euclidean spread versus summed ranges, gradual stable-look drift, raw ROI priority, brute-force window comparison, and ranking selection.
-Current look pipeline reproduced all7,563 reviewed looks across1,645 trials in the four ranked sessions (canonical and random). Exact agreement for count, stable intervals, raw/normalized coordinates, counted duration, onset/offset, ROI and island disagreement. Maximum absolute numerical difference:0.
-New raw-data CLI tested on eight trials (one canonical and one random per session). All resulting saccade onset/offset/durations and current looks matched saved references within1e-7.
-No data or saved trial results are included. verify_handoff.py can repeat full cohort checks with local data. Test results establish implementation reproduction, not physiological ground truth or equivalence to the original collaborator event labels.
