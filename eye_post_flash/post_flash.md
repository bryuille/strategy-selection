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

**Most likely unchosen alternative** (`eye_post_flash.targets`): which exit
counts as the most likely alternative depends on the decision model, so it is
computed under three models mirroring the manuscript's model set
(`--alt-model` picks the one used for the headline rate tables; default
`optimal`):

**The distribution.** One family, used by all the likelihood-based models:
scalar timing, `p(tm | t) = N(tm; t, wm·t)` — Gaussian, standard deviation
proportional to the timed duration. `wm` is `targets.DEFAULT_WM = 0.15`,
exposed as `--wm`. It is **not fitted here**: the manuscript estimates it from
the separate T-maze experiment, whose data are not in this repo, so 0.15 is a
stand-in and the `--wm` sensitivity has to be reported alongside any result.

| model | scored quantity | implied alternative | manuscript model |
| ----- | --------------- | ------------------- | ---------------- |
| `optimal` | `log p(tm1,tm2 \| t_e1,t_e2) = Σ_j log N(tm_j; t_ej, wm·t_ej)` (conditionally independent marginals) | highest-likelihood unchosen exit | 4AFC optimal |
| `total_time` | `log p(tm1+tm2 \| t_e1+t_e2)`, the convolution — so `sd = wm·√(t_e1²+t_e2²)`, **not** `wm·(t_e1+t_e2)` | highest-likelihood unchosen exit | 4AFC total-time |
| `hierarchical` | none — structural | the sibling exit: the other vertical arm on the *chosen* branch, i.e. what the committed second stage actually evaluated. The only `wm`-free prediction of the three | 2×2AFC hierarchical |

The `−log(sd)` normalizer matters and is kept: it is what makes a *longer*
expected interval the more confusable one at equal absolute error. An earlier
version of this module ranked by absolute-error proxies (`Σ|t_obs−t_e|/t_e`
and `|T_obs−T_e|/T_e`) instead of likelihoods. Those are not
monotone-equivalent to the model they claimed to implement — they reorder the
top alternative on 4/24 path types for `optimal` and 6/24 for `total_time`,
and 2 of the `total_time` disagreements (path types 6 and 8, maze 2) hold at
every `wm` because of the wrong convolution normalizer.

**The evidence is noiseless.** `ranked_alternatives` uses the *correct path's
expected* intervals as `tm`, not the measured flash times. So `tm` equals
`t_e` exactly for the correct exit, no trial-to-trial timing variability
enters, and the ranking is a deterministic function of (path type, chosen
exit) — every trial of a path type gets the identical ranking. The rankings
are therefore best read as a per-path-type lookup table, which is what makes
the geometry confound below so sharp.

Exit timing is fully determined by geometry (interval 1 = horizontal arm
`h1|h4`, interval 2 = vertical arm, each at `h/vel`). The lapse, postdictive
and revision variants are not separately operationalized: lapses do not
change the ranking, the postdictive model ranks like the optimal one, and
revision predictions depend on fitted confidence thresholds (the natural next
step if the discrimination result warrants it).

## Strategy discrimination: does the looked-at alternative track the strategy?

The models dissociate on real geometries — maze 5, truth LU: hierarchical
implies LD (same branch) while total-time implies RU; maze 3, truth LU:
hierarchical implies LD while total-time implies RD. The *strategy
discrimination* tables in `summary.md` restrict to correct trials on which
two models imply different exits and ask which implied exit was fixated
first. If counterfactual evaluation follows the inferred strategy,
hierarchical-regime (or hierarchical-labelled) trials should be directed to
the sibling exit and sequential trials to the total-time exit; the Fisher
exact on regime × first-looked-alternative is the correlation being hunted.
Reported for hierarchical-vs-total-time (primary), hierarchical-vs-optimal
(supplementary), by maze, and by per-trial neural label. `trials.csv` carries
`alt_optimal` / `alt_total_time` / `alt_hierarchical` and the full ordered
`unchosen_look_seq`, so any other model pair can be scored offline.

### The geometry confound, and the controlled tests

The hierarchical model's implied alternative is the *sibling* exit, which
shares the horizontal arm with the chosen exit and is therefore **usually the
nearer of the two candidates** — mean 15–16° away versus 17–21° for the
total-time exit, nearer on 69–100 % of discriminating trials in mazes 1–5.
Maze 6 inverts it (sibling 12.0° vs 10.2°). Since regime is a deterministic
function of the maze, the raw regime × first-look Fisher exact is confounded
with geometry: a regime difference can be pure saccade amplitude. `trials.csv`
carries `alt_dist_optimal` / `alt_dist_total_time` / `alt_dist_hierarchical`
(degrees from the chosen exit) so this is auditable.

Three controls follow the raw table:

1. **Proximity split** — the same contrast within trials where the sibling
   *is* nearer and where it is not. A pure amplitude account predicts
   `frac hierarchical-directed` ≈ 1 in the first stratum and ≈ 0 in the
   second, with no regime effect inside either.
2. **Stratified regime effect** (`_stratified_discrimination`) — a
   Mantel–Haenszel common odds ratio over strata, which holds a nuisance
   variable fixed and asks whether regime still moves the first look inside a
   stratum. Because maze regime is a function of the maze, maze-stratification
   is only defined for the **per-trial neural labels**, where the label varies
   within a maze; that is precisely the "identical path type, different
   strategy" comparison. Run stratified by `maze`, by `session` (the
   session-leakage caveat from classifier.md), and by `maze × session` jointly
   — the decisive test.
3. `OR > 1` means hierarchical-labelled trials are *more* sibling-directed,
   i.e. the direction the hierarchical account predicts. `OR < 1` is the
   opposite.

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

1. **The alternative is defined by an assumed evidence model, not a fitted
   one.** The three rankings are parameter-free monotone proxies built from
   the path geometry — no `wm`, lapse rate, or revision threshold is fitted,
   so the models are distinguished only by *which* exit they rank first, not
   by calibrated likelihoods. The lapse and postdictive variants collapse onto
   rankings already covered; the revision variants do not, and fitting them is
   the natural next step.
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

## Current result (`--align response`, `--wm 0.15`, both monkeys)

The raw regime contrast is significant (Fisher p ~ 1e-14) but not
interpretable — it tracks maze geometry, not strategy. The controlled tests,
on the per-trial neural labels within identical mazes:

| test | Faure | Nielsen |
| ---- | ----- | ------- |
| maze-stratified MH | OR 0.465, p = 0.19 | OR 0.710, p = 0.031 |
| session-stratified MH | OR 1.111, p = 0.98 | OR 0.780, p = 0.072 |
| maze × session MH (decisive) | OR 0.534, p = 0.38 | OR 0.697, p = 0.039 |

`wm` sensitivity of the decisive test: Faure p = 0.38 at every `wm` in
0.10–0.25; Nielsen p = 0.039 (`wm` 0.10, 0.15), 0.078 (0.20), 0.084 (0.25).

**Read this as a null with a hint, not a result.** Every odds ratio bar one is
below 1, i.e. hierarchically-labelled trials are *less* sibling-directed than
sequentially-labelled trials in the same maze and session — the opposite of
the hierarchical account's prediction. But it clears p < 0.05 only in Nielsen,
only at the lower half of the plausible `wm` range, and Faure's much smaller
labelled pool (~230 informative trials) is null throughout. A consistent sign
across two animals is worth noting; these p-values are not worth a claim.

Three readings, none separable with these tables: post-feedback looks are not
a readout of the decision model at all (the strong overall sibling bias —
0.87–0.96 in Faure — would then be a spatial default, and the proximity split
is the relevant control); the within-maze minority labels are label noise
rather than genuine strategy switches (see the vetting discussion in
classifier.md); or `wm` is genuinely lower than 0.15 and the effect is real.
Fitting `wm` on the T-maze data would remove one of the three. The revision
models, which predict *which trials* get re-evaluated rather than a fixed
target exit, are the discriminating next experiment — and unlike the three
implemented here they do not inherit the sibling-proximity confound.

## Catch-all model comparison (`eye_post_flash.model_comparison`)

The pairwise discrimination tests above ask "model A's exit or model B's?" on
a subset of trials. The model comparison replaces them with a single fit of
every model to every trial, so nothing hinges on which pair is contrasted.

**Observable.** On each correct trial with a post-feedback look at an unchosen
exit, which of the three unchosen exits was looked at first. Geometry is
exactly constant per path type, so trials collapse into 24 cells each holding
a 3-way count — a multinomial likelihood over thousands of trials against at
most five parameters.

**Predictions.** A model maps measured intervals to a posterior over the four
exits; the predicted target is its highest-ranked *unchosen* exit. `tm` is
never observed, so it is integrated out against the model's own noise, `tm ~
N(t_true, wm·t_true)`, on a Gauss-Hermite grid — that marginalization is what
turns a deterministic argmax into a distribution and gives the comparison its
power. Every model also carries `eps`, a uniform-over-unchosen rate absorbing
looks no decision model explains, without which one look at a
zero-probability exit would send the log-likelihood to −inf.

**Two readouts of the hierarchical model, fitted separately.** `hierarchical`
scores exits by `p(side|tm1)·p(vert|tm2,side)` and takes the argmax; that does
*not* reliably pick the sibling, since `p(sibling)` is a large number times a
small one and the far branch can beat it. `sibling` is the committed readout —
once the side is chosen, the other arm on that branch is the only alternative
the second stage ever evaluated. `targets.py` implements the committed one, so
`sibling` is the model its `hierarchical` ranking actually corresponds to.
Reporting only one of the two makes the hierarchical account look far better
or far worse than it is.

**Reference models.** `uniform` (0 parameters) is the floor; `saturated`
(per-cell empirical) is the ceiling; `proximity` (`p ~ exp(-d/scale)`) prices
the spatial account that the sibling-is-nearer confound implies.

**Read the diagnostics before the ranking.** The summary flags parameters
resting on a bound and Weber fractions above 0.40. A large `wm` is not a
timing estimate — it flattens the posterior toward uniform, which is how a
model that cannot match the data's shape buys likelihood. A ranking whose
winner has pinned parameters is not evidence for that model's mechanism.

### Result (`--align response`)

The headline metric is **how often a model names the exit the saccade actually
went to**. Three unchosen exits, so guessing scores 33%. The ceiling is set by
always naming each cell's most-looked-at exit — 89% for Faure, 67% for
Nielsen. `of achievable` places a model between those two; `whole
distribution` is the same scale applied to the log-likelihood, which also
scores whether the model got the three-way *split* right, not just its top
guess. Figures: `out/model_comparison/<align>/model_accuracy.png` and
`cell_agreement.png`.

| model | Faure correct | Nielsen correct |
| ----- | ------------- | --------------- |
| *ceiling* | *88.9%* | *66.5%* |
| **sibling** (1 param) | **88.7%** | 38.9% |
| only_first | 77.1% † | 41.7% |
| revision (5 params) | 73.5% | 57.4% † |
| optimal = postdictive = optimal_lapse | 71.0% | 53.4% † |
| only_second | 52.3% | 62.4% † |
| total_time | 43.2% † | 48.7% † |
| hierarchical | 41.8% † | 42.9% † |
| *uniform (chance)* | *33.3%* | *33.3%* |
| proximity | 17.1% | 35.9% |

(† a fitted parameter hit its limit, or the timing noise came out too large to
be real — the number was bought, not earned.)

**Faure: one parameter reaches the ceiling.** `sibling` — the counterfactual
look goes to the other arm of the chosen branch — names the right exit on
88.7% of trials against an 88.9% ceiling, using one parameter, and is right in
23 of 24 cells. No manuscript model comes close, and none of the five-parameter
fits buys its way past it.

**The favoured exit is not the nearest one.** `proximity` scores 17% for
Faure — *below* chance. The sibling is usually nearer than the total-time
alternative, but it is often not the nearest of all three, and the animal
still prefers it. The spatial confound that invalidated the raw regime
contrast does not explain the effect itself.

**Nielsen: nothing fits cleanly.** `sibling` falls to 39%, and every model
above chance carries a flag — `wm` pinned near 2.0 (ten times any real Weber
fraction, which flattens the posterior toward uniform rather than describing
timing), or `theta` and `alpha` at their limits. Accuracy and likelihood also
disagree about the winner. The two animals are qualitatively different here,
matching their first-look rates.

**Discount the revision wins.** `theta` fits at 0.95–0.99 in every variant and
both animals — "always revise", which the manuscript notes makes revision
indistinguishable from the optimal model. Its edge comes from `alpha` and
`beta` acting as a free noise-plus-bias fudge, not from selective revision.

**Not identifiable under this readout:** `optimal`, `postdictive` and
`optimal_lapse` return identical numbers in both animals. A lapse randomizes
the ranking, which is exactly what `eps` does, so `lam` is unidentified; and
the postdictive side-marginalization does not change which unchosen exit ranks
first. These three cannot be separated by counterfactual saccades, whatever
they do to choice accuracy.

Refits are slow (~16 min per alignment; four 5-parameter revision variants
dominate) — submit `slurm/run_post_flash.sbatch` rather than running them on a
laptop. `--plots-only` rebuilds the summaries and figures from the saved
`fits.csv` in seconds, which is exact: the fits are deterministic given the
seed.
