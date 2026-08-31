# The alternative defense: gaze cannot explain the neural strategy state

The reviewer-response passages in
[eye_data_classifier/classifier.md](eye_data_classifier/classifier.md) were
drafted hoping for a *positive* finding — strategy-dependent maze inspection as
behavioural evidence of active evaluation. The decoding tables do not support
that bracket: within a maze, gaze does not detectably differ by the subsequent
strategy state. This document records the framing the results *do* support,
which answers the same reviewer concern from the opposite direction.

## The claim

> Within the same maze — where visual input is identical — the animals'
> pre-fixation eye movements are statistically indistinguishable between
> strategy states. The within-maze neural strategy signal therefore cannot be
> attributed to differences in visual input or overt sampling behaviour.

The reviewer's worry is that the early neural state is "a passive response to
visual geometry." The paper's neural claim rests on within-maze evidence (the
same maze, solved both ways), and for that signal this defense holds: the
neural state differentiates strategies where gaze does not. What is *lost*
relative to the drafts is the active-evaluation inference — the sentence
"such strategy-dependent inspection would provide behavioural evidence of
active evaluation" cannot be written from these data.

**Scope carefully.** The defense is a within-maze statement. Across mazes,
gaze *does* correlate with strategy — see result 2 — so an unscoped "gaze
cannot explain the neural state" would be false as written.

## The three supporting results

All numbers from `eye_data_classifier/out/decoding/all/` (the label-vetted
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

## Suggested bracket text

> Across mazes, gaze features predicted the strategy state (balanced accuracy
> ≈ 0.6) but carried no information beyond maze identity itself (maze alone:
> 0.84; maze + gaze: 0.82–0.84), and within a maze — where geometry is fixed —
> gaze was uninformative about the subsequent strategy (≈ 0.50 in both
> monkeys, across gaze-position, state-occupancy and scan-order features). The
> strategy-predictive structure in gaze is therefore fully accounted for by
> visual geometry, and the within-maze neural strategy state cannot be
> attributed to differences in visual input or overt sampling.

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
