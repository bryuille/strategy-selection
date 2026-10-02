# `data/processed/` — derived caches

All paths resolve through `data.config.processed_npz(stem)` →
`<PROCESSED_ROOT>/<stem>.npz`. `PROCESSED_ROOT` is set in `data/config.py`
(default `./data/processed/`).

Files are NumPy `.npz` archives written atomically (`data.loader.savez_atomic`).
Load with `data.mat.load_npz(path)` or the typed helpers in `data.loader`.

**Join key.** Eye and behavioral tables join on `(session, trial_indices_all)`.
Label arrays are indexed by `trial_id - 1` (MATLAB 1-based trial IDs).

**Not here.** Raw data stays in the `.mat` files under `MAT_ROOT`, read directly
and never copied (see [DATA_DICTIONARY.md](../DATA_DICTIONARY.md)). Single-trial
published ranges are read from `Single_Trial_List/`.

---

## Catalog

| Stem | Built by | Loader |
| --- | --- | --- |
| `<session>_trial_timebins` | `data.builder.build_trial_timebins` | `load_trial_timebins` |
| `<session>_strategy_choices` | `data.labels.build_strategy_choices` | `load_strategy_choices` |
| `<session>_strategy_svm` | `data.labels.build_svm_choices` | `load_svm_choices` |
| `<Monkey>_eye_data` | `data.builder.build_eye_data` | `load_eye_data` |
| `<Monkey>_eye_behavioral` | `data.builder.build_eye_behavioral_data` | `load_eye_behavioral_data` |
| `<Monkey>_clean_eye_data` | `data.builder.clean_eye_data` (whole trial) | `load_clean_eye_data` |
| `<Monkey>_clean_eye_data_s<S>_e<E>` | same, window `[fix−S, fix−E]` ms | `load_clean_eye_data(start_ms=…)` |
| `<Monkey>_attractor_eye_data` | `data.builder.build_attractor_eye_data` | `load_attractor_eye_data` |
| `<Monkey>_msp_fixed5_r<radius>` | `msp.features` (geofix window) | `msp.features.load_features` |
| `<Monkey>_msp_trends_r<radius>[_region]` | `msp_trends.features` (geofix) | `msp_trends.features.load` |
| `<Monkey>_reg_fixed5_<tag>_<window>` | `regression.features` | `regression.features.load_fixations` |

`<Monkey>` is `Faure` or `Nielsen`. `<session>` is e.g. `june_24_g0`.
`<tag>` is a radius tag such as `r1` or `r0.5`. The MSP window is `geofix`,
`[geo_present, fix_start]`, re-detected per trial over the svm/top_ten
sessions (since 2026-10-02). The fixed `s1466_e0` window (`PRE_FIX_START_MS` /
`PRE_FIX_END_MS` in `data.config`) is read only by `msp.legacy.saccades`,
through the attractor and `clean_eye_data_s1466_e0` caches.

After changing detector or QC constants, delete the affected stems and rebuild
— loaders check key presence, not that settings match. Typical wipe:

```bash
cd /path/to/your/PROCESSED_ROOT
rm -f *_clean_eye_data*.npz *_attractor_eye_data.npz \
      *_msp_fixed5_*.npz *_msp_trends_*.npz *_reg_fixed5_*.npz
```

---

## Per-session neural / labels

### `<session>_trial_timebins.npz`

| Key | Shape / dtype | Meaning |
| --- | --- | --- |
| `trial_timebins` | `(n_trials, n_neurons, T)` float | Firing-rate segments from flash 1 to flash 3 + 300 ms, aligned to `geo_present` (1 ms bins). Pads shorter trials with NaN. |

Source: neural + behavioral `.mat` via `session_data`. Used as input to both label builders.

### `<session>_strategy_choices.npz`

| Key | Shape / dtype | Meaning |
| --- | --- | --- |
| `strategy_choices` | `(n_trials,)` float | Ward dendrogram labels on 3 PCs of early activity. `0` = hierarchical, `1` = sequential, `NaN` = unlabelled. Index `trial_id - 1`. |

Built only for trials in the published single-trial range for that session.

### `<session>_strategy_svm.npz`

Same `strategy_choices` convention, plus SVM diagnostics from
`build_svm_choices`:

| Key | Meaning |
| --- | --- |
| `strategy_choices` | As above (anchor mazes: out-of-fold; mazes 2–5: mode across folds) |
| `fold_labels` | `(K, n_labelled)` fold predictions |
| `fold_agreement` | Per labelled trial, fraction of folds agreeing |
| `oof_pred`, `oof_score` | Out-of-fold class / decision value on anchors |
| `svm_auc`, `oof_balanced_acc` | Anchor CV metrics |
| `n_folds`, `n_maze_1`, `n_maze_6`, `n_neurons`, `seed`, `status` | Run metadata |

`load_svm_choices` returns the full dict; `load_strategy_choices` returns only
the 1-d `strategy_choices` array.

---

## Per-monkey eye concatenations

### `<Monkey>_eye_data.npz`

One row per trial across every `Eye_Data_*.mat` for the monkey.

| Key | Shape / dtype | Meaning |
| --- | --- | --- |
| `time` | `(n,)` object → 1-d float | Seconds from `geo_present` |
| `eye_x`, `eye_y` | `(n,)` object → 1-d float | Gaze degrees |
| `session` | `(n,)` | Session name |
| `trial_indices_all` | `(n,)` int | Trial ID within session (1-based) |

### `<Monkey>_eye_behavioral.npz`

One row per trial across every `*_good_trials_concat.mat` for the monkey.
Neuron-level behavioral fields are collapsed to trial level.

| Key | Meaning |
| --- | --- |
| `session`, `trial_indices_all` | Join key |
| Required (`EYE_BEHAVIORAL_FIELDS`) | `h1`–`h6`, `trial_fade`, `path_type`, `geo_present`, `fix_start`, `flash_one`/`_two`/`_three`, `photodiode_qc_bad`, `vel`, `trial_answer1`–`4`, `LR`, `LR2`, `geo_type`, `fixation_off` |
| Optional (`EYE_BEHAVIORAL_OPTIONAL_FIELDS`) | `answer_time`, `fixation_cue_present`, `saccade_init`, `trial_end` — NaN if the cache predates them |

Times are absolute session seconds (same as the behavioral mat). Maze identity
is `geo_type` (1–6). QC used everywhere for pooled analyses:
`path_type != -99`, `trial_fade == 0`, `photodiode_qc_bad == 0`
(`data.builder.qc_mask`).

---

## Events and warped gaze

### `<Monkey>_clean_eye_data[_s<S>_e<E>].npz`

Long table of saccade / fixation events (one row per event). Detection runs
**after** optional clipping to
`[fix_start − S, fix_start − E]` ms from `geo_present`. Whole-trial and
windowed stems must not overwrite each other: I-DT is order-dependent, so the
two are different analyses.

| Key | Meaning |
| --- | --- |
| `name` | Event type (`fixation_idt`, `saccade`, …) |
| `onset`, `offset`, `duration` | ms from `geo_present` (within the detection segment) |
| `peak_velocity`, `amplitude`, `dispersion` | pymovements event properties |
| `location_x`, `location_y` | Event location (degrees) |
| `session`, `trial_indices_all` | Trial join key |

Trials that fail QC contribute no events. `msp.legacy.saccades` reads
`s1466_e0`; msp and regression re-detect inside `geofix` instead.

### `<Monkey>_attractor_eye_data.npz`

Unit-H warped gaze for the same trial set as `eye_data`, validity aligned to
the **whole-trial** (or default) events used at build time.

| Key | Shape / dtype | Meaning |
| --- | --- | --- |
| `time` | `(n,)` object → 1-d float32 | Seconds from `geo_present` |
| `eye_x`, `eye_y` | `(n,)` object → 1-d float32 | Warped unit-H coordinates |
| `valid` | `(n,)` object → 1-d bool | On-screen, non-blink fixation samples usable for assignment |
| `session`, `trial_indices_all` | Trial join key |

State assignment is **not** stored here — the consumer
(`msp.legacy.saccades`) assigns against its own codebook. Regression **cannot** reuse this mask for
`geofix` / alternate windows because validity is tied to the events window
used when the attractor was built.

---

## Analysis feature caches

### `<Monkey>_msp_fixed5_r<radius>.npz`

Fixed five-state occupancy for MSP (`msp.features`). Example:
`Faure_msp_fixed5_r1.npz`. Validity is checked against the
stored `assign_radius` / `window` / codebook, not the tag string.

`<Monkey>_msp_trends_r<radius>.npz` is the same extraction (geofix) cached by
`msp_trends`, which adds `assignment_rule`; `…_region` swaps the exit balls
for maze regions.

**Per-trial** (length `N` = usable trials):

| Key | Shape | Meaning |
| --- | --- | --- |
| `session`, `trial_indices_all`, `maze_id` | `(N,)` | Identity |
| `occ_ms` | `(N, 5)` float32 | Assigned fixation dwell (ms) per state |
| `occ_bin` | `(N, 5)` float32 | `1` if `occ_ms > 0`, else `0` |
| `n_fix`, `n_fix_assigned` | `(N,)` int | Fixation counts |

**Per-fixation** (length `F`):

| Key | Shape | Meaning |
| --- | --- | --- |
| `fix_xy` | `(F, 2)` float32 | Centroid in unit H |
| `fix_state` | `(F,)` int16 | Assigned state `0…4`, or `-1` |
| `fix_row` | `(F,)` int64 | Index into the per-trial tables |

**Metadata:** `codebook_xy` `(5, 2)`, `state_names`, `codebook_k`,
`radius_tag`, `assign_radius`, `window_start_ms`, `window_end_ms`.

State order: `origin`, `LU`, `LD`, `RU`, `RD`
(coordinates `(0,0)`, `(±1,±1)` in unit H).

### `<Monkey>_reg_fixed5_<tag>_<window>.npz`

Regression fixation tables (`regression.features`). `<window>` is `geofix`.
Example: `Faure_reg_fixed5_r1_geofix.npz`.

Always built for the publication session set; session subsets are filtered on
load.

**Per-trial** (length `N`):

| Key | Meaning |
| --- | --- |
| `session`, `trial_indices_all`, `maze_id` | Identity |
| `fix_ms` | `fix_start − geo_present` (ms) |
| `window_lo_ms`, `window_hi_ms` | Analysis clip for that trial |

**Per-fixation** (length `F`):

| Key | Meaning |
| --- | --- |
| `fix_row` | Index into per-trial tables |
| `fix_state` | Assigned state, or `-1` |
| `fix_dwell_ms` | Assigned dwell for that fixation (0 if unassigned) |
| `fix_onset_ms` | Onset within the window |
| `fix_xy` | Centroid `(F, 2)` |

**Metadata:** `codebook_xy`, `state_names`, `assign_radius`, `window`,
`sessions`, `drops_json` (per-session drop tallies), `schema_version`.

Measures (`occ`, `visits`, `dur_ms`, …) are **not** cached — derived on load
by `regression.features.trial_measures`.

---

## Legacy stems (zarchive)

Older packages may still write into the same directory. They are not part of
the current `data/` / `msp` / `regression` loaders:

| Pattern | Package |
| --- | --- |
| `<Monkey>_clf2_k<K>_unith_s…` | `zarchive.classifier.features` |
| `<Monkey>_mspcomp_k<K>_unith_s…` | `zarchive.maze_strategy_pairs.comp_features` |
| `<Monkey>_mspcomp_pooled_…` | `zarchive.maze_strategy_pairs.comp_pooled_features` |
| `<Monkey>_mspelbow_unith_s…` | `zarchive.maze_strategy_pairs.elbow_features` |
| `<session>_trial_metadata`, `_pre_flash_*`, `_post_flash_*`, `_lr_choices` | `zarchive.data.loader` |

Safe to delete if nothing in `zarchive/` is being re-run.

---

## Dependency sketch

```
mat/Neural + mat/Behavioral ──► <session>_trial_timebins
                                    └─► <session>_strategy_choices
                                    └─► <session>_strategy_svm

mat/Eye ──► <Monkey>_eye_data ──► <Monkey>_clean_eye_data[_s…_e…]
                 │                    └─► <Monkey>_attractor_eye_data
                 │                              └─► msp.legacy.saccades (pre1466 plots)
mat/Behavioral ──► <Monkey>_eye_behavioral ──┘

mat/Eye + eye_behavioral ──► <Monkey>_msp_fixed5_r<radius> (geofix), <Monkey>_msp_trends_…

mat/Eye + eye_behavioral ──► <Monkey>_reg_fixed5_…_<window>
  (re-detects per window; does not reuse attractor validity)
```
