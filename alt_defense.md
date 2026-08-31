# The alternative defense: gaze cannot explain the neural strategy state

The reviewer-response passages in
[eye_pre_flash/classifier/classifier.md](eye_pre_flash/classifier/classifier.md) were
drafted hoping for a *positive* finding — strategy-dependent maze inspection as
behavioural evidence of active evaluation. The decoding tables do not support
that bracket: within a maze, gaze does not detectably differ by the subsequent
strategy state. This document records the framing the results *do* support,
which answers the same reviewer concern from the opposite direction.

## The claim

> Within the same maze — where the stimulus is identical — the animals'
> pre-fixation eye movements are statistically indistinguishable between
> strategy states. There is therefore no evidence that strategy-correlated
> differences in overt sampling account for the within-maze neural strategy
> signal: with the screen fixed and eye position matched in every feature
> tested, foveal and peripheral input are matched to the same extent.

The reviewer's worry is that the early neural state is "a passive response to
visual geometry." A passive (stimulus-driven) account needs the *input* to
differ where the response differs. Within a maze the screen does not differ,
and — because peripheral input is anchored to eye position, not an independent
channel — matched gaze means matched retinal input throughout the visual
field. The eye analysis closes the one gap a passive account had left within
maze: that different eye movements might have created different visual input
from the same screen. What is *lost* relative to the drafts is the
active-evaluation inference — the sentence "such strategy-dependent inspection
would provide behavioural evidence of active evaluation" cannot be written
from these data.

## What this defense can and cannot rule out

Ruled out (within maze): the passive pathway *through overt sampling* —
strategy-correlated eye movements delivering different visual input from an
identical screen.

Not ruled out:

- **Covert attention.** The animal can take in different peripheral
  information from identical retinal input by attending covertly to different
  maze parts. The gaze data are blind to this. Note, though, that covert
  attention deployed by strategy state is an *active, internal* process — it
  contradicts "passive," not the paper. The residual objection it supports is
  subtler: "the early neural state may be a visual-attention state rather than
  an abstract strategy state," and the eye data cannot distinguish those.
- **Undetected gaze differences — a decoding null is not input equivalence.**
  "Statistically matched" means indistinguishable in the features tested
  (occupancy, scan order, spatial density over 1466 ms) — not equivalence of
  the full retinal input stream, which varies trial to trial with the gaze
  path. A passive account can retreat to "the relevant input differences are
  invisible to those features." The null does not exclude that; it prices it:
  the surviving story needs sampling differences potent enough to flip the
  neural regime and predict subsequent choice, yet invisible to position,
  occupancy, order and density statistics that demonstrably read maze
  geometry out of the same signals. Phrase claims evidentially ("we found no
  evidence that…"), never as exclusion ("cannot be attributed to…").
- **The identity of the neural labels.** The eye data cannot address whether
  the two neural regimes are strategies at all, rather than two visual-
  response classes to two families of geometries — strategy ≈ maze at 0.84 in
  the vetted pool, so most of what the clustering separates could in principle
  be geometry readout. That question belongs to the paper's other analyses,
  and two carry it: (i) **within-cell behavioural validation** — if the
  neural label were geometry readout + noise it would predict behaviour no
  better than maze identity, so the decisive test is whether the neural label
  predicts choices *within* (session, maze) cells (the strongest version of
  Figure 5); (ii) **random-geometry bimodality** — a passive readout across
  many random geometries should vary as richly as the geometry space, whereas
  the initial states collapse into two clusters. The "internal selection"
  conclusion rests on the conjunction of early bifurcation, within-cell
  behavioural validation, bimodality, and the gaze null — not on the gaze
  null alone.

## The three supporting results

All numbers from `eye_pre_flash/classifier/out/decoding/all/` (the label-vetted
pool: 4 sessions per monkey; see classifier.md for the vetting rule) at
stratified 5-fold CV, balanced accuracy, chance = 0.500.

**1. Within a maze, gaze is uninformative about strategy.** Per-maze decoding
(degree coordinates, geometry fixed, maze identity unusable by construction)
sits at chance for every feature set and both monkeys: CV 0.46–0.56 across
codebook occupancy (ms and binary, K=6 and K=12), state bigrams, and gaze
heatmaps, while training-fold accuracy runs up to 0.99 — the models had the
capacity and found nothing that generalises. No single maze × feature × model
cell exceeds 0.635, uncorrected over 96 cells per monkey.

**2. The cross-maze gaze–strategy correlation is entirely visual geometry.**
Across mazes, gaze features do predict strategy (CV 0.54–0.64) — but maze
identity alone predicts it at 0.84, and adding gaze to maze identity does not
improve on maze identity:

| | maze only | gaze only (heatmap 10×10) | maze + gaze |
| --- | --- | --- | --- |
| Faure (logistic / forest) | 0.841 | 0.597 / 0.595 | 0.816 / 0.835 |
| Nielsen (logistic / forest) | 0.837 | 0.623 / 0.636 | 0.824 / 0.840 |

Every bit of strategy information in gaze is already contained in maze
identity: gaze is a degraded readout of which maze is on the screen. Combined
with result 1 (conditioning on the maze removes the correlation entirely),
this is the signature of confounding through a common cause — maze → gaze and
maze → strategy — not of a direct strategy → gaze link. This is the pattern
pure passive inspection produces, and it is why unconditioned analyses cannot
speak to the claim.

**3. The null is not a sensitivity failure.** The same features and models
read maze geometry out of gaze at 0.6 across mazes (result 2), so the pipeline
demonstrably detects structure in these signals when it exists. The within-maze
null is a property of the data, not of a weak decoder. It is also conservative
in design: trial-level CV pools sessions, which permits any session-level
leakage to *inflate* decodability, and the result is chance anyway.

## How this slots into the two reviewer responses

**Response 2 (adaptive-selection overclaim) — the null is load-bearing.** The
reviewer argues neural geometry differences follow from task structure. The
response's counter-evidence is the early bifurcation plus within-maze strategy
variation (same maze, occasionally solved in the non-dominant regime, early
neural state predicting it). The reviewer's next move against that evidence is
"the within-maze variation is gaze-driven — different looking, different
retinal input, still a visual response." The eye null closes exactly that
route: the early neural state differs where the stimulus and the sampling do
not, which no passive (stimulus-driven) account can produce. A *positive* gaze
result would have backfired here — strategy-dependent gaze would let the
reviewer re-explain the early bifurcation as differential retinal input.

**Supporting positives.** Gaze demonstrably tracks the maze — maze identity is
decodable from the 10×10 gaze heatmap (degree space, vetted pool, 6-way
logistic) at 0.346 (Faure) and 0.307 (Nielsen) against 0.167 chance — so the
animals inspect the maze; the inspection simply does not differ by upcoming
strategy. Selection is internal.

**What is lost.** The standalone "behavioral evidence of active evaluation"
sentence. The active-evaluation case rests on the early-bifurcation and
simulation analyses; the eye data play defense.

### Draft for Response 1's bracket

> Gaze during the pre-fixation viewing period reliably reflected the maze on
> the screen (maze identity was decodable from gaze density well above
> chance), confirming that animals visually inspected the maze. However, gaze
> did not detectably differ as a function of the subsequently inferred
> strategy state: within a maze, classifiers trained on gaze position, region
> occupancy, and scan order performed at chance in both monkeys (balanced
> accuracy ≈ 0.50), and across mazes gaze carried no strategy information
> beyond maze identity itself (maze alone: 0.84; maze plus gaze: 0.82–0.84).
> We therefore found no evidence that animals gathered different visual
> information depending on the upcoming strategy: the within-maze variation in
> the initial neural state is not accompanied by detectable differences in
> overt sampling, arguing against accounts in which that variation is a
> passive response to trial-to-trial differences in visual input. Together
> with the behavioural validation of the strategy states and their early,
> pre-flash divergence, this supports an internal selection process.

### Draft for Response 2's bracket

The sentence following the bracket ("Such strategy-dependent inspection
would provide…") presumes a positive result and needs replacing.

> We found that they do not: although gaze robustly reflected the maze
> geometry itself, within-maze gaze was statistically indistinguishable
> between strategy states (balanced decoding accuracy ≈ 0.50 in both monkeys),
> and gaze carried no strategy information beyond maze identity. This
> strengthens the central inference of this analysis: the early, within-maze
> divergence of the neural state occurs with the stimulus fixed and with no
> detectable difference in overt sampling, leaving no evidence for a passive
> visual account of that divergence — consistent instead with an internal
> commitment to a strategy made on the basis of comparably sampled visual
> evidence.

Replacement for the follow-on sentence: "The absence of strategy-dependent
inspection indicates that strategy selection is not driven by differential
information gathering during viewing; combined with the early divergence of
neural trajectories, it places the selection process internally, during
evaluation of a comparably inspected stimulus."

## Caveats

1. **Structural limit of the labels.** Strategy is nearly determined by maze
   in exactly the sessions whose labels pass quality checks; the sessions rich
   in within-maze strategy variability either fail to label at all (10 of 23:
   the labeler selects the same neural cluster for maze 1 and maze 6) or sit
   near coin-flip against the anchor-maze map (5 of 13, vetted out). The
   within-maze question is therefore mostly answered where strategy rarely
   varied within a maze. The `allplus` scope re-runs everything without the
   vetting as a robustness check — and it shows why the caveat matters: with
   the unvetted sessions included, Faure's per-maze logistic CV rises to a
   small but consistent 0.54–0.59 across all eight feature sets (Nielsen stays
   at chance). Pure label noise cannot be decoded above chance, so those
   labels carry *some* structure; but with sessions pooled inside a maze, a
   decoder can also reach above chance by recognising the recording day
   (calibration, gaze style) when days differ in strategy mix — and the
   retired day-conditioned analysis saw exactly this Faure pattern evaporate
   when tested across days (cell-level p = 0.002 → day-level p = 0.92, driven
   by 2 of 7 sessions). Adjudicating between "weak real signal" and "session
   leakage" requires session-held-out folds, which is outside the current
   simple design.
2. **Power.** Eight sessions; anchor-maze minorities run 6–67 trials, so
   individual per-maze cells are noisy (their scatter is symmetric about
   0.507 ± 0.057, i.e. chance-like).
3. **Label noise attenuates.** Labels are neural-derived; residual noise can
   only push decoding toward chance. The cleanest-label sessions were kept,
   but "not detectably" is the honest strength of the claim.
4. This is a decoding null, not an equivalence test. If a bound is needed,
   a TOST-style equivalence margin or a Bayes factor on the per-maze AUC
   would formalise "indistinguishable."
