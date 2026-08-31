# Data dictionary

This dictionary documents the variables on which the public analyses rely. A
row in a physiology behavioral file is a neuron–trial observation, so trial
values repeat across neurons. `trial_indices_all` is the stable trial key and
must be used for cross-file joins; row position must not be used as a trial ID.

Paths below are given as they appear in the published distribution
(`Data/Physiology/...`). In this repo the same files live under `data/mat/`, and
all path construction goes through `data/config.py`:

| Published path | This repo | Read by |
| -------------- | --------- | ------- |
| `Data/Physiology/Behavioral_Data/<Monkey>/` | `data/mat/Behavioral_Data/<Monkey>/` | `data.convert behavioral` |
| `Data/Physiology/Neural_Data/<Monkey>/` | `data/mat/Neural_Data/<Monkey>/` | `data.convert neural` |
| `Data/Physiology/Eye_Data/<Monkey>/` | `data/mat/Eye_Data/<Monkey>/` | `data.convert eye` |
| `Data/Physiology/Single_Trial_List/<Monkey>/` | `data/mat/Single_Trial_List/<Monkey>/` | `data.convert single_trial` |
| `Data/Physiology/Decoding_Data/<Monkey>/` | `data/mat/Decoding_Data/<Monkey>/` | nothing — reference only |
| `Data/Pre_Physiology/` | not present | nothing — reference only |

The last two sections of this file describe trees the Python code never opens.
They are kept because they define the published record this analysis has to
stay consistent with.

## Physiology behavior

Files: `Data/Physiology/Behavioral_Data/<Monkey>/<session>_good_trials_concat.mat`
(`data.config.behavioral_mat_path`)

The top-level MATLAB variable is `save_all_data`.

### Identity, condition, and selection fields

| Field | Meaning and coding |
| --- | --- |
| `trial_indices_all` | Stable session-specific trial ID, repeated across neuron rows |
| `cids` | Cluster/neuron identifier associated with each row |
| `nrns` | Neuron index used by the analysis code |
| `geo_type` | Published maze identity: integers 1–6; random geometries are `-99` |
| `path_type` | Published fixed-geometry condition: integers 1–24; random geometries are `-99`; fixed trials satisfy `geo_type = ceil(path_type/4)` |
| `trial_fade` | Binary indicator of whether the ball was visible or invisible/faded; publication physiology analyses retain `trial_fade == 0` where applicable |
| `LR` | Ground-truth first decision: `-1` left, `+1` right |
| `LR2` | Ground-truth second decision: `-1` down, `+1` up |
| `rand_geo` | Random-geometry trial metadata |
| `h` | Length of the initial hallway through which the visible ball travels before entering the H-maze |
| `h1` | Left horizontal arm length |
| `h2` | Left-up vertical arm length |
| `h3` | Left-down vertical arm length |
| `h4` | Right horizontal arm length |
| `h5` | Right-up vertical arm length |
| `h6` | Right-down vertical arm length |
| `vel` | Ball velocity used with arm length to derive expected random-geometry interval duration |

The same definitions of `h` and `h1`–`h6` apply to both fixed and random
geometries.

### Photodiode quality control

| Field | Meaning and coding |
| --- | --- |
| `photodiode_qc_bad` | Boolean exclusion flag; true when either flash interval is at least one 120-Hz frame from its expected duration |
| `photodiode_qc_interval_code` | Bit code: `0` neither interval, `1` flash 1–2 only, `2` flash 2–3 only, `3` both |
| `photodiode_qc_error_12_ms` | Observed minus expected flash 1–2 duration, milliseconds |
| `photodiode_qc_error_23_ms` | Observed minus expected flash 2–3 duration, milliseconds |
| `photodiode_qc_screen_refresh_hz` | Screen refresh rate, `120` Hz |
| `photodiode_qc_threshold_frames` | Absolute-error threshold, `1` frame |
| `photodiode_qc_version` | QC procedure version (`1.0` in the current files) |

The threshold in milliseconds is stored exactly by its defining metadata:
`photodiode_qc_threshold_frames / photodiode_qc_screen_refresh_hz * 1000`.
Analyses exclude flagged trials immediately after loading, then retain the
original published simultaneous-neuron trial range.

### Events and responses

`flash_one`, `flash_two`, and `flash_three` are the measured flash event times
used to calculate the QC errors. Other retained event/response fields include
`fix_start`, `fixation_off`, `fixation_cue_present`, `geo_present`,
`saccade_init`, `answer_time`, `feedback_time`, `trial_end`, and `reward`.

The four `trial_answer` fields are mutually exclusive indicators defined
relative to the trial's ground-truth `LR` and `LR2` values:

| Field | Meaning |
| --- | --- |
| `trial_answer1` | Correct horizontal (left/right) and vertical (up/down) decisions |
| `trial_answer2` | Correct horizontal decision and incorrect vertical decision |
| `trial_answer3` | Incorrect horizontal decision and correct vertical decision |
| `trial_answer4` | Incorrect horizontal and vertical decisions |

They are not fixed spatial-choice labels. The chosen exit is recovered by
combining the active answer indicator with `LR` and `LR2`. The analysis code
uses that combination to derive choices coded as `1` left-up, `2` left-down,
`3` right-up, and `4` right-down. For analyses comparing error strategies,
wrong-horizontal trials may additionally distinguish the chosen vertical arm
using its length similarity to the correct vertical arm.

### Acquisition/provenance payload

`Area`, `Grid_hole`, `probe_type`, `probe_sync`, `which_sync`, `all_synctimes`,
`sync_params`, `date_of_recording`, `labels`, `old_nrns`, `spikes`, and
`spikes_all` are retained acquisition/provenance fields. `spikes_all` is a
large row-by-time matrix and is not read by the active publication analyses.
These fields can be distributed as a separately labelled raw/provenance
component if they are not included in the analysis-ready files.

## Neural data

Files: `Data/Physiology/Neural_Data/<Monkey>/<session>_Whole_Trial_FR_Causal.mat`
(`data.config.neural_mat_path`)

| Variable | Meaning |
| --- | --- |
| `smooth_session` | Causally smoothed whole-trial firing-rate matrix. Rows align exactly with rows in the matching physiology behavioral file; columns are time samples. |

Files `nrn_av_Faure.mat` and `nrn_av_Nielsen.mat` contain the QC-filtered,
condition-averaged neural arrays generated by the Figure Two PCA script and
consumed by the Isomap notebook.

## Eye data

Files: `Data/Physiology/Eye_Data/<Monkey>/Eye_Data_<session>.mat`
(`data.config.eye_mat_path`)

The top-level `Eye_Data` structure contains per-trial cell arrays `eye_x`,
`eye_y`, `eye_RX`, `eye_RY`, and `pupil_size`.

## Published single-trial ranges

Files: `Data/Physiology/Single_Trial_List/<Monkey>/single_trial_list_<session>.mat`
(`data.config.single_trial_mat_path`)

| Variable | Meaning |
| --- | --- |
| `min_value` | First trial ID in the original published optimized range |
| `max_value` | Last trial ID in the original published optimized range |

QC is applied after this range is loaded; the optimization is not rerun.

## Decoder outputs (reference only)

Files: `Data/Physiology/Decoding_Data/<Monkey>/`. Present in `data/mat/` but not
read by any Python module here; the DV decoders in `neural_traces/decoders/` refit from
firing rates rather than loading these.

| Filename family | Principal variable/content |
| --- | --- |
| `_decoding_data_performance_<session>.mat` | `model_pred_accuracy`, one cell per fixed condition |
| `_model_pred_L_R_trials_<session>.mat` | `model_pred_L_R_trials`, trial-level decoder predictions by condition |
| `_IC_answer_train_<session>.mat` | `IC_answer_train`, decoder training/condition outputs |
| `_random_geos_<session>.mat` | `bootstrap_mean`, random-geometry decoder summary |
| Files containing `_shuffle_` | Corresponding shuffled-control result |
| `_decoding_trial_ids_<session>.mat` | Explicit identity/provenance variables described below |

Trial-ID files contain `trial_ids_per_condition`, `choose_trials_original`,
`choose_trials_after_qc`, `bad_photodiode_trial_ids`, `random_geo_trial_ids`,
`min_value`, and `max_value`. These files are mandatory: Figures Three and
Five use them to prevent positional misalignment after QC exclusions.

## Pre-physiology behavior and model results (reference only)

Not present in this repo.

Behavior files contain the top-level structure `S`. Core behavioral variables
used across Figure One include `LR`, `LR2`, `targetCorrect`, `h`, `h1`–`h6`,
`vel`, fading/task timing fields, and session/trial metadata. Many entries are
task structures with a `.data` member. Preserve the structure without
flattening because the Figure One scripts access that layout directly.

The pre-physiology response events use the same correctness categories as the
physiology `trial_answer` indicators, but have different stored names:

| Pre-physiology event | Equivalent physiology field | Meaning |
| --- | --- | --- |
| `t_0_o` / T1 | `trial_answer1` | Correct horizontal and vertical decisions |
| `t_1_o` / T2 | `trial_answer2` | Correct horizontal and incorrect vertical decision |
| `t_2_o` / T3 | `trial_answer3` | Incorrect horizontal and correct vertical decision |
| `t_3_o` / T4 | `trial_answer4` | Incorrect horizontal and vertical decisions |

These categories are relative to ground truth. Figure One combines the event
category with `LR` and `LR2` to derive the selected spatial exit.

`Data/Pre_Physiology/Model_Fitting/Published_Trial_Splits` stores the five
published `testIdx` vectors for each monkey. Seeds 42–46 reproduce runs 1–5,
and the Figure One validator checks the saved vectors against the legacy
`randperm` procedure before model evaluation.

`Data/Pre_Physiology/Model_Results` contains behavioral psychometric summaries,
five fitted-parameter runs for the Optimal Lapse and Revision X_LR_Max models,
and five likelihood runs for each retained fixed and revision model.
