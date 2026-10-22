# `counterfactual/`: post-choice exit sequences

Question: after the monkey commits to an exit, which exits does it look at,
and does that depend on the strategy it used? This step only **annotates**
the data: one variable-length list of exit codes per trial. How a sequence's
likelihood under a strategy is scored is not defined yet.

Commands: [README_COMMANDS.md](../README_COMMANDS.md#counterfactual).

## Inputs

- **Sessions**: the four publication sessions only, for now
  (`config.SESSIONS`: Faure june_8_g0, june_24_g0; Nielsen Nov_3_g0,
  Nov_6_g0).
- **Trials**: QC-passing (`data.builder.trial_qc_ok`) with exactly one
  `trial_answer*` flag set.
- **Window**: from the start of the choice saccade (`saccade_init`, falling
  back to `answer_time`, then `fixation_off`) to `trial_end`, in ms from
  `geo_present`, clipped to the eye trace. Detection runs on the clipped
  samples (`data.builder.trial_events`: saccades from pymovements'
  microsaccade detector; a fixation ends only at a saccade or when gaze moves
  more than 2° from the fixation's anchored centroid).

## Codes

`1 = LU, 2 = LD, 3 = RU, 4 = RD` (DATA_DICTIONARY.md), `0` = origin
(fixation point), `-1` = elsewhere. Each fixation's centroid in degrees is
assigned to the nearest exit within `EXIT_RADIUS_DEG` (3°), with the exits
placed from the trial's arm lengths at `(-h1, h2)`, `(-h1, -h3)`, `(h4, h5)`,
`(h4, -h6)`; otherwise origin within 3°, otherwise elsewhere.

The **chosen exit** is decoded from the active `trial_answer{1..4}` flag and
`LR`, `LR2` (`exits.chosen_exit`; `checks.py` verifies it against the
archived 16-entry table). The **exit list** is the chosen exit followed by
every post-choice fixation whose state is an exit, in time order, repeats
kept, e.g. `[3, 3, 1, 2, 3]` = `RU RU LU LD RU`. Origin / elsewhere
fixations are left out of the list but kept in `state_seq`. The first
fixation in the window is where the choice saccade landed; when it is on the
chosen exit (`landing_on_chosen`, true on essentially every trial, which is
the sanity check of the decoding) item 0 already stands for it and the
post-choice items start with the next fixation.

## Outputs

- Cache `data/processed/<Monkey>_cf_exits.npz` (`extract.load`): per trial
  `session`, `trial_indices_all`, `maze_id`, `h`, `chosen_exit`,
  `correct_exit` (−1 for random geometries), `strategy` (0 = H, 1 = S from
  `<session>_strategy_choices`, NaN if unlabelled), `window_lo_ms`,
  `window_hi_ms`, `n_fix`, `landing_on_chosen`, `exit_seq`, `state_seq`; per fixation `fix_row`,
  `fix_xy_d`, `fix_onset_ms`, `fix_offset_ms`, `fix_state`; plus `dropped`
  per session and reason, `sessions`, `pupil_source`, `schema`.
- `counterfactual/out/<Monkey>/exit_sequences.csv`: one row per trial with
  the list as codes and as names.
- `counterfactual/out/<Monkey>/summary.txt`: trials kept and dropped,
  list-length distribution, and the sanity check that the first post-choice
  exit fixation equals the decoded chosen exit.

## Movies

`counterfactual.movie` renders one `.mp4` per trial
(`out/movies/<Monkey>/<session>/trial_<id>.mp4`), or with `--batch` one
video of many trials back to back (`batch_<first>-<last>.mp4`, a one-second
hold at each trial start). The clock is ms from `flash_three`, so the choice
saccade is visible and the analysed window's start (`saccade_init`) is marked
on the timeline. The title and the right panel carry the trial's dendrogram
strategy label (H / S / unlabelled); `--labelled` restricts a batch to
labelled trials and `--start N` skips the first N. Each frame shows:
gaze trail and current sample on the maze, the exit and origin circles, and
for the active fixation a centroid marker, a tether to the gaze sample and a
status box ("fixation #k, age / duration, COUNTED as LU → item [1]" or
"elsewhere: not counted") that stay on for the fixation's whole duration. The
first 150 ms of every fixation add an expanding ring and a "NEW FIXATION #k"
banner, so two abutting fixations are visibly two. Saccades turn the trail
orange. A timeline strip shows the numbered fixation blocks coloured by state
with the counted exits marked, and the right panel lists the exit items as
they are counted. Events are re-detected per trial with the same detector
call the cache used (no pupil, so the count can differ from a cache built
with blink masking; the movie prints when it does).

Blink masking needs a pupil trace: the classifier's per-session pupil tables
(`classifier/out/pupil/`), else the eye mats. Without either (this laptop)
masking is off and the extraction warns.
