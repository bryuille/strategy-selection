# Post-flash eye data: counterfactual saccades by strategy regime

The reviewer bracket this analysis fills (Response 1):

> we quantified the probability that animals made voluntary post-feedback
> saccades toward the most likely unchosen alternative and compared this
> measure across the neurally inferred strategy regimes. [Insert result here.]

```bash
uv run python -m eye_post_flash.counterfactual                      # both monkeys
uv run python -m eye_post_flash.counterfactual --monkey Faure
uv run python -m eye_post_flash.counterfactual --align response     # runs today
```

Output: `eye_post_flash/out/counterfactual/<align>/<monkey>/` — `summary.md`,
`trials.csv`, `regime_rates.png`.

## The measure

Per trial: did the animal, in a window after feedback, fixate within
`--radius` (3°) of the **most likely unchosen alternative** exit? A look is a
clean fixation event (`data/processed/<Monkey>_clean_eye_data.npz`, either
pymovements detector) whose centroid falls in the exit region and whose onset
falls in the window (default 1500 ms).

**Voluntary** means the look is at an exit where nothing appears: the chosen
exit is excluded throughout (the answer saccade lands there and the outcome is
revealed there), and the primary claim uses **correct trials only** — on error
trials the correct exit is where the ball actually emerged, so a look there is
stimulus-driven. Error trials are tabulated separately.

**Most likely unchosen alternative** (`eye_post_flash.targets`): the unchosen
exit whose expected flash intervals are most confusable with the evidence the
animal received, under scalar timing. Exit timing is fully determined by
geometry (interval 1 = horizontal arm `h1|h4`, interval 2 = vertical arm, each
at `h/vel`); the distance is `Σ_j |t_obs_j − t_alt_j| / t_alt_j`, i.e. timing
noise proportional to the timed duration. Normalizing by the *alternative's*
interval is what a scalar-timing likelihood prescribes, and it breaks
raw-millisecond ties (the longer alternative is more confusable). This is an
assumption of the analysis, stated in the deliverable; swap
`targets.timing_distances` to change the model (e.g. to a fitted behavioral
model's posterior). On correct trials this alternative is the
second-most-likely target overall, which is the quantity the reviewer asked
about.

**Controls reported alongside the rate.** (i) `P(look per other alt)` — the
per-exit rate at the two *less* likely unchosen exits; matched for everything
except likelihood, so it prices general post-feedback looking-around.
(ii) first-look identity among trials with any unchosen-exit look, against the
uniform 1/3 chance.

## The regime comparison

Primary split: **maze regime** (`data.labeler.MAZE_GROUPS` — Faure 1–3
hierarchical / 4–6 sequential, Nielsen 1–4 / 5–6), available for every session
with eye data. Fisher exact on the trial-level 2×2, plus a per-session table
so the pooled number can be checked against session scatter (trial-level
pooling admits session-level leakage — same caveat as classifier.md).

Secondary split: **per-trial neural strategy labels** where built
(`{session}_strategy_choices.npz`; unvetted pool — see the vetting discussion
in classifier.md before leaning on it).

## Alignment modes and the data prerequisite

`--align feedback` (default, the reviewer's definition) starts the window at
`feedback_time`. That field was added to `data.convert.BEHAVIORAL_FIELDS`
after the local behavioral npz were converted, so it needs one cluster step
(where `data/mat/` lives):

```bash
sbatch slurm/run_post_flash.sbatch     # re-converts behavioral npz, reruns analysis
```

or by hand: `uv run python -m data.convert behavioral --overwrite`, delete
`data/processed/<Monkey>_eye_behavioral.npz`, rerun. The analysis detects
stale caches and prints this.

`--align response` needs no new data: the window starts at the onset of the
answer fixation (first post-flash-three fixation at the chosen exit), detected
from the eye data alone. It starts slightly before feedback, so anticipatory
looks between answer and feedback are included — treat these numbers as
provisional and quote the feedback-aligned ones.

## Caveats

1. **The alternative is defined by an assumed evidence model.** Scalar-timing
   confusability of the correct path, not a fitted decision model. If the
   revision needs the model-derived alternative, replace the ranking, not the
   pipeline.
2. **Regime is confounded with maze geometry across mazes** — the same issue
   the pre-flash classifier work turned into a defense (alt_defense.md). A
   hierarchical/sequential difference here can reflect the mazes rather than
   the regime; the per-trial neural-label split within the same maze pool is
   the stronger (but lower-powered, label-noisier) comparison.
3. **Eye traces end ~7 s after geo_present.** Coverage of the full window
   after feedback is not guaranteed on every trial; `n_fixations_in_window`
   and the skip counts in `summary.md` show how much data the window saw.
4. **Fixation-based looks.** A saccade that lands on the alternative but is
   not followed by a ≥50 ms stable fixation is not counted; the measure is
   conservative in that direction.
