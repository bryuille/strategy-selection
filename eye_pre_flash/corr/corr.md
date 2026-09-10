# Maze x decoded-strategy occupancy similarity

A 12x12 split-half CV correlation matrix over `(maze, strategy)` cells —
mazes 1-6 hierarchical, then mazes 1-6 sequential — built from codebook
occupancy vectors rather than raw gaze maps. Not to be confused with the
`part_corr` name in older docs: see §2.

```bash
uv run python -m eye_pre_flash.corr.run
uv run python -m eye_pre_flash.corr.run --source svm --scope top_ten --monkey Faure
```

## 1. The claim this exists to support

The 6x6 maze-only similarity matrices (`similarity.md`) cannot separate
strategy from geometry: strategy is close to a deterministic function of
`(session, maze)`, so "mazes 4-6 resemble each other" and "sequentially-solved
trials resemble each other" are the same statement about the same trials.
Holding maze type fixed and splitting each maze's trials by a *decoded neural*
strategy label removes that confound by construction. The decisive read is
the same-maze cells — `(maze m, H)` vs `(maze m, S)` — where geometry is identical
on both sides, so anything below that maze's own split-half reliability is
strategy-dependent gaze sampling, not a visual confound.

This revives the retired `eye_pre_flash.plotting.similarities.heatmap_labels`
(`git show HEAD:eye_pre_flash/plotting/similarities/heatmap_labels.py`), which
did the same thing against 102,400-dim spatial gaze maps and was dropped for
a coverage problem (§7). `corr` uses low-dimensional codebook-occupancy
vectors instead, computes every cell regardless of count rather than
requiring both strategies to clear a coverage floor, and adds a second,
supervised label source.

## 2. Not `part_corr`

Earlier drafts of `README.md` and `similarity.md` referenced a planned
`eye_pre_flash/part_corr/` package — single trials scored against a
half-split template of their group, column-centred to remove a
template-quality confound. That package was never built; `corr` absorbed its
doc references (see `similarity.md`'s "Splitting a maze by decoded strategy"
section). The method here is the group-vs-group split-half CV described
throughout this document, not the template-scoring method those older drafts
described.

## 3. Data and features

Three feature blocks, all from
`eye_pre_flash.classifier.features.load_features` (already cached, no new
extraction):

| Feature (`--feature`) | Cache block | d |
| --- | --- | --- |
| `occupancy` | `occ_ms` — seconds per codebook state | K |
| `occupancy_bin` | `occ_bin` — visited/not-visited per state | K |
| `bigram` | ordered state-run pair proportions | K² |

`K in {6, 12}` (`--k`). Window: `fix_start - 1466 ms -> fix_start`
(`eye_pre_flash.plotting.plot_io.window_label`).

### Processing order

`classifier.features.extract_monkey_features` runs these in exactly this
order, and the order is load-bearing:

1. **Clip** each trial to `[fix_start - 1466 ms, fix_start]`. Everything below
   sees in-window samples only.
2. **Warp** to unit-H (`data.attractor.to_maze`; already cached).
3. **K-means** across sessions per monkey, over the pooled **in-window**
   samples — one codebook per (monkey, K).
4. **Fixations** from `data.builder`'s pymovements pass — **I-DT only**
   (`features.FIXATION_EVENT`) — each span clipped to the window.
5. **Assign** each clipped fixation's centroid to the nearest prototype within
   `ASSIGN_RADIUS` = 1.0 unit-H (an L2 ball). Saccades, blinks, and fixations
   outside every ball are unassigned and drop out of the features.

Step 3 is the one that used to be wrong. The codebook was previously taken
from `data.attractor`, which fits it on **whole-trial** gaze — putting
prototypes where gaze went while the maze was being solved, not during the
epoch these features describe. Fitting on windowed gaze moves them: for Faure
at K=6 the largest prototype shift is 0.65 unit-H, and while trial counts are
unchanged (16,060) only 0.01% of `occ_ms` rows, 7.5% of `occ_bin` rows and
12.0% of `bigram` rows survive identical. Aggregates barely move (states
visited per trial 2.58 → 2.53), so this is a re-registration of the state
space rather than a different measurement. `data.attractor`'s cache is still
read for its warped positions, validity mask and timebase; its own
`codebook_xy` and `state_id` are no longer used, so the 6x6 matrices in
`similarity.md` and the other attractor consumers are untouched.

One deliberate deviation from that list: **fixation detection stays in degree
space**, where `data.builder` does it, rather than running after the warp.
I-VT's threshold is 40 deg/s and I-DT's is 2.0 deg of dispersion; unit-H is
dimensionless and warped per trial by that trial's arm lengths, so detecting
on unit-H would make the effective physical threshold vary by trial and by
maze — reintroducing the geometry dependence the warp exists to remove. Step 4
is where the spans are *applied*, not where detection happens.

### I-DT only

`data.builder` runs both I-DT and I-VT and stores both event sets.
`data.attractor.fixation_event_spans` picks up **both**, because it matches the
substring `fixation` — so two detectors with different span boundaries land in
one pool and `assign_fixation_states`' longest-span tie-break silently chooses
between them per fixation. Measured, I-DT wins that ~99.5% of the time (its
p95 duration is roughly twice I-VT's: 944 vs 479 ms for Faure, 951 vs 421 ms
for Nielsen), so the old behaviour was "I-DT nearly always, by accident".

Step 4 now takes I-DT alone, on purpose, which also agrees with the two
modules that already name a detector: `plotting.saccades.labeled` and
`plotting.similarities.heatmap_fixations` both filter on `fixation_idt`.

`data.attractor` itself is unchanged and still pools both, since its
whole-trial `state_id` feeds the 6x6 matrices in `similarity.md`. That
inconsistency is deliberate for now — aligning it would shift those published
numbers.

### `occ_ms` sums to fixation time, not to the window

`_sample_dt_seconds` caps each interval at the trial's sample period. Without
that cap it ran `np.diff` over the *masked* time vector, so the last assigned
sample before a gap absorbed the whole gap and `occ_ms` summed to the full
1466 ms no matter how much of the window was really spent fixating. Capped, the
row sum is the time actually spent in assigned fixations: median **861 ms**
(Faure K=6, 59% of the window) to **922 ms** (Nielsen K=12, 63%), and **no**
row reaches the window length. Cross-checked against the union of clipped I-DT
spans, `occ_ms` recovers 89.8% of it — the shortfall being invalid samples
inside fixations and fixations whose centroid landed outside every ball.

Capping per sample, rather than summing event durations, is also what keeps the
total correct when two detectors overlap: each sample carries one state and
contributes one period, whereas adding I-VT and I-DT span lengths would
double-count the 80-90% of gaze both cover.

### Three invariants

Interpretability is the priority here over squeezing out signal. Every read in
this package is a comparison — a cell against another cell, a feature against
another feature, a variant against another variant — and every one of those
comparisons is only valid if nothing but the thing being compared changed. So:

1. **Unit-H only.** No `deg`, ever, in this package (§5).
2. **Only the processed K=6 and K=12 caches.** `corr` reads the unit-H K=6
   and K=12 blocks straight from `classifier.features.load_features` and
   extracts nothing of its own. No new feature is introduced here, and no
   cache is rebuilt with different settings.
3. **One CV, applied identically to every feature.** All three blocks are just
   (n, d) vectors, so `matrix.session_cv_matrix` treats them as such: split a
   cell's trials in half, take each half's **mean vector**, Pearson the two.
   There is no per-feature branch in the CV and there must not be one.

Invariant 3 is the one with a standing temptation attached. `occ_bin` is
binary per trial, so its half-mean is a *visit fraction* in [0, 1] rather than
a 0/1 vector, and it is periodically tempting to "fix" that by binarising the
half-mean. Don't. Two reasons:

- **It would delete the signal.** For a binary feature the strategy effect
  *is* a rate effect: per trial the only available answer is visited or not,
  and what distinguishes H from S is "H visits this state on 23% of trials, S
  on 71%". That statement exists only in aggregate. 81-88% of half-mean
  entries fall strictly between 0 and 1, and that is precisely the
  information; collapsing it to binary above the trial level throws it away.
- **It would break comparability.** `occupancy`, `occupancy_bin` and `bigram`
  are only readable against each other because one estimator is applied to
  all three.

The measured cost of abandoning invariant 3 was checked directly, by running
the same-maze delta and the same within-session shuffle null two ways on
identical trials — the standard half-mean estimator, and a Pearson averaged
over raw binary trial-vector pairs. The trial-pair version detected nothing
anywhere: max |z| of 1.35 across all 24 (monkey x K x maze) cells, against
3.64 for the half-mean, with every candidate signal collapsing (Faure maze 4
K=6, z 3.64 -> 1.00; Nielsen maze 1 K=6, z 3.28 -> 0.69). Averaging per-pair
correlations measures trial-to-trial similarity, which is dominated by
within-cell trial noise; the half-mean averages that noise down first (half
sizes here are 48-58) and only then correlates. Note also that a Pearson
between two length-6 binary vectors can take just 17 distinct values.

## 4. The dynamic-range problem and the two variants

The existing 6x6 occupancy matrices sit compressed against 1.0
(`similarity.md`, k=12):

| | diag | off-diag | total maze effect |
| --- | --- | --- | --- |
| Faure occupancy | 0.891 | 0.810 | 0.081 |
| Faure occ-bin | 0.937 | 0.894 | 0.043 |
| Nielsen occupancy | 0.888 | 0.716 | 0.172 |
| Nielsen occ-bin | 0.948 | 0.855 | 0.093 |

Every cell shares a dominant common gaze profile, so all r pile up in a
narrow band, and a within-maze strategy effect must be smaller than the
between-maze effect above it. The sampling SD of a Pearson over 12 points is
roughly +-0.06, so for Faure occ-bin the noise on one cell can exceed the
whole effect being differenced. (These particular tables carry a stale-window
banner in `similarity.md` — computed before the window's end moved to
`fix_start` — so treat the numbers as indicative; the compression itself is
structural, not a window artifact.)

Two variants (`--variant`) test whether central fixation is the cause:

- **`full`** — the cached vectors, unmodified. Comparable to the existing 6x6
  figures.
- **`no_origin`** — drop the codebook state nearest the screen origin (0, 0):
  `occ_ms`/`occ_bin` lose one dimension, `bigram` loses every ordered pair
  touching it.
- **`mean_removed`** — subtract the pooled grand-mean profile (the mean
  feature vector over trials), fit once per (monkey, feature, k) over the
  widest scope's labelled trials (not per session, per cell or per scope — a
  profile that differed between sessions would put each session's matrix in a
  different subspace, exactly the cross-session comparability problem
  `heatmap_labels`' comparable-core machinery existed to fix; fitting per
  scope would make `publication` incomparable to `top_four`). Subtracting a
  common profile makes the mean off-diagonal r go negative; that is
  arithmetic, not a finding.

### Why `mean_removed` and not `pc1_removed`

`pc1_removed` — project out PC1 — was the previous third variant and is
retired (`git log`). It was the wrong operation for the intent. Pearson
already centres each vector across its own dimensions, so a per-trial offset
was never the problem; what was missing is subtraction of the mean profile
*across trials*. PC1 is a third thing: `sklearn`'s PCA centres before it
decomposes, so PC1 is a direction of variance *around* the grand mean, not the
grand mean itself, and projecting it out could as easily have deleted the
strategy effect as the common profile.

Two consequences to hold onto:

1. **Absolute r is uninterpretable under `mean_removed`.** The reads this
   package relies on — a maze's `(m, H)` vs `(m, S)` cell against that maze's
   own split-half reliability — are relative, so they survive; any statement
   about the level of r does not.
2. **`no_origin` is the more defensible version of the same intent.** Dropping
   the origin state is a structural claim about which dimension central
   fixation occupies, not a quantity fitted from the data, so it cannot absorb
   strategy variance by accident. Read the two together: if `mean_removed` and
   `no_origin` agree, the finding is robust; if only the fitted one shows an
   effect, distrust it.

### The check that settles what `pc1_removed` was doing

`fit_pc1` is kept for diagnosis only, and the `mean_removed` leaf's
`examples_dims_k<K>.csv` carries a `dim="__pc1__"` row with:

| column | read |
| --- | --- |
| `pc1_explained_var` | how much of the pooled variance PC1 carried |
| `pc1_mean_cos`, `pc1_mean_cos_abs` | cosine of the PC1 loading against the L2-normalised grand-mean profile. Near 1 → `pc1_removed` roughly did what `mean_removed` now does explicitly. Near 0 → it did not, confirming the inversion. PCA fixes no sign convention, so read the absolute value |
| `pc1_label_pointbiserial` | Pearson r of each trial's PC1 score (`X @ pc1`) against its 0/1 strategy label, pooled |
| `pc1_label_pointbiserial_within` | the same after centring score and label inside each (session, maze) group, over the groups that held both labels |
| `pc1_label_within_n_groups`, `pc1_label_within_n_trials` | the denominator the within figure rests on |

If either point-biserial is non-trivial, PC1 was partly a strategy direction,
`pc1_removed` was deleting the effect it was meant to expose, and every result
from that variant needs rereading.

Read the **within** version as the primary one. Strategy is close to a
deterministic function of (session, maze) (§1), so the pooled point-biserial
partly measures which session and maze a trial came from — the same confound
this whole package exists to defeat. A large pooled value beside a near-zero
within value means PC1 tracked recording day and geometry, not strategy.

Read it with its denominator. Only (session, maze) groups holding *both*
labels can speak to a within-group effect; groups where every trial carries the
same label are dropped rather than centred, because their label residuals are
zero while their score residuals are not, so keeping them would drag the
estimate toward 0. That near-determinism is a matter of degree, and in the
limit where no group is mixed the within correlation is **undefined** — NaN
over zero groups, which is not the same claim as zero. A within value resting
on three groups is not the evidence that one resting on forty is.

A `dim="__mean__"` row records the grand-mean profile itself and its L2 norm,
so a result can be re-derived without refitting.

Note the mechanism cuts both ways on the cosine: PCA centres first, so PC1
cannot *be* the mean profile by construction — yet on these features the state
absorbing most dwell time also tends to carry the most variance, so the two can
still run close. Worth measuring rather than predicting. The one spot check on
record for the retired variant (Nielsen occupancy k=12) had PC1 explaining only
~16% of the variance, and removing it widened the off-diagonal spread only
modestly (std 0.053 → 0.071) — central fixation is present but was never the
whole story.

## 5. Unit-H only

Everything here is `unith`. `deg` was swept as a second space and has been
dropped; there is no `--space` flag and no `_<space>` component in the figure
names.

The earlier argument for `deg` was that the same-maze cells hold geometry
fixed, so the unit-H warp (`data.attractor.to_maze`) is an unnecessary
cross-maze device inside one maze. That is true of the *coordinates* and false
of the *features*, which is what the matrix actually correlates. The `deg`
codebook is not a re-expression of the unit-H one:
`classifier.features._fit_degree_codebook` runs its own per-monkey k-means on
screen degrees pooled across every maze, at the same `K`. Arm lengths differ by
maze, so those centroids land where *some* mazes' arms are, and one
degree-space state is on-arm for one geometry and off-arm for another.
`occ_ms`/`occ_bin`/`bigram` built on that book therefore encode which maze was
on screen — the single confound this analysis exists to exclude. Even the
same-maze read is affected, because the states being compared were positioned
by the other five mazes.

`deg` is still correct in `eye_pre_flash.classifier.decoding.REGIMES`' per-maze
regime, where every trial shares one geometry and nothing is pooled across
mazes. It is never right for these cross-maze cells.

## 6. The two label sources

`--source dendro` (default alongside `svm`, i.e. both run unless `--source`
is given) — the existing Ward-clustering labels
(`data.labeler.build_strategy_choices`), vetted against all five anchor
mazes (`eye_pre_flash.classifier.labels.ANCHOR_MAJORITY`,
`MIN_LABEL_AGREEMENT = 0.70`).

`--source svm` — a maze-1-vs-6 linear SVM (`data.labeler.build_svm_choices`)
fit on the same 3-PC neural space the dendrogram clusters, under the
assumption that mazes 1 and 6 anchor the ends of the strategy spectrum. Trials
from those two mazes get 5-fold stratified CV with an out-of-fold prediction;
every other trial (mazes 2-5) gets the **mode** across the 5 fold models'
predictions. This is a supervised alternative that can track a nonlinear
session shape the dendrogram's 2-cluster Ward split cannot.

**Grading the SVM by agreement with the anchor map is circular** (it was
*trained* to say maze 1 = H, maze 6 = S), so it is not anchor-vetted at all.
Instead: the anchor-minority cells `1S` and `6H` are **skipped outright**
(`eye_pre_flash.corr.labels.SOURCE_SKIP_CELLS`), since the classifier's own
training axis makes them near-empty by construction even out-of-fold; and the
only external validity check reported per session
(`dendro_agreement_2345` in `results_raw_k<K>.csv`) is the SVM's agreement with the
dendrogram labels on mazes 2, 3, 5 — the mazes neither fold ever trains or
tests on. State this plainly when reading SVM results: the labels are
validated (via `svm_auc`, held-out ROC-AUC on the anchor split) exactly where
the analysis does not need them, and have only a soft cross-check exactly
where it does.

## 7. Session selection

Scopes are **per source** (`corr.labels.SOURCE_SCOPES`) — each source gets
only the scopes that mean something for the labels it produces. `--scope`
picks one; the default runs every scope each source defines. The retired
`all`/`allplus` scopes are gone.

| Source | Scope | Sessions per monkey | Ranked by |
| --- | --- | --- | --- |
| `dendro` | `publication` | 2 (the 4 original-clustering sessions) | — |
| `dendro` | `top_four` | 4, from the anchor-vetted pool | `data.labeler.SNR_AUC` |
| `svm` | `top_ten` | 10, from the unvetted pool | `snr_auc` |

Both ranked scopes use the same metric, so there is one session ordering in
this package rather than one per source. It comes from
`corr.labels.snr_auc_table()`, read straight from
`zrefs/Dendogram_all/SNR_All_Sessions/all_session_snr_results.csv` — all 61
reference sessions (27 Faure, 34 Nielsen, every one `status=passed`), rather
than from `data.labeler.SNR_AUC`, which is a transcription of the same file
cut down to the 23 that clear its 0.95 gate. The two agree exactly on all 23;
reading the file just means the ranking still works if the pool ever widens.

Ranking is applied **after** vetting, never before — truncating first would
let a session that vetting rejects consume a slot meant for one that passes.

Actual membership, measured:

```
dendro publication Faure    june_8_g0, june_24_g0
dendro publication Nielsen  Nov_3_g0, Oct_22_g0
dendro top_four    Faure    june_24_g0, june_8_g0, june_16_g0, june_22_g0
dendro top_four    Nielsen  Nov_6_g0, Oct_22_g0, Nov_3_g0, Nov_1_g0
svm    top_ten     Faure    june_24_g0, june_8_g0, june_18_g0, june_29_g0, june_12_g0,
                            june_16_g0, june_28_g0, june_10_g0, june_09_09_g0, june_17_g0
svm    top_ten     Nielsen  Oct_22_g0, Nov_3_g0, Nov_6_g0, Oct_21_g0, Nov_7_g0,
                            Nov_18_g0, Nov_4_g0, Oct_25_g0, Nov_9_g0, Nov_1_g0
```

### Two things these scopes are not

**`top_four` is not a selection.** Anchor vetting drops 15 of the 23
`CLUSTERING_SESSIONS` — agreements 0.486-0.653, all under
`MIN_LABEL_AGREEMENT` = 0.70 — leaving exactly 4 per monkey. So the cap binds
on nothing: `top_four` *is* the entire vetted pool, identical to the retired
`all`/`allplus` dendro scopes, and the `SNR_AUC` ranking never gets to choose.
The cap stays only as a guard in case vetting ever passes a fifth. Read the
dendro pair as "publication (2/monkey)" vs "everything whose labels vet
(4/monkey)", not as a quality gradient inside a larger pool.

A related warning falls out of the same numbers: **neural SNR does not predict
dendrogram label quality.** Faure's #3 and #4 by `SNR_AUC` — june_28_g0
(0.9931) and june_29_g0 (0.9834) — both fail vetting, at 0.542 and 0.486,
while june_16_g0 (0.9779) and june_22_g0 (0.9712) pass and take those slots.
Picking sessions by SNR alone would have picked two whose labels do not work.

**`top_ten` only filters one animal — and the pool, not the cap, is why.**
Nielsen has **34** neural recordings in the reference CSV, but only 10 clear
the 0.95 `snr_auc` gate that defines `CLUSTERING_SESSIONS`, so its `top_ten`
is all 10 and nothing is ranked out. For Faure the cap binds, dropping 3 of 13
(june_17_g0, April_8_g0, june_11_g0).

That gate is not an arbitrary cut — it sits just above a cliff in both animals:

```
Nielsen ranks  9-16 by snr_auc:  .965 .958 | .948 | .883 .810 .792 .791 .786
Faure   ranks 11-18 by snr_auc:  .971 .971 | .969 .962 .952 | .942 .892 .870
```

Pool size by threshold: Nielsen 10 at >=0.95, 11 at >=0.92, 12 at >=0.85, 13
at >=0.80; Faure 13, 14, 17, 18. So widening Nielsen past ~11 means admitting
sessions at `snr_auc` <= 0.88, where the neural clustering barely separates
maze 1 from maze 6 and the strategy labels are largely noise. **The asymmetry
is a property of the recordings, not something a bigger pool fixes** — Nielsen
has more sessions, not more usable ones. It still matters when reading one
animal against the other.

### Ranking by `snr_auc`, not by SVM accuracy

The SVM's own `oof_balanced_acc` was considered for `top_ten` and rejected in
favour of one metric shared by both sources. It does have the spread to rank
(0.8785-1.0000 over the 23 sessions, 20 distinct values, 4 tied at the
ceiling) and it would order things differently — june_28_g0 is Faure's #3 by
`snr_auc` and its #7 by out-of-fold accuracy. But `snr_auc` wins on three
counts that matter more here:

1. **One ordering for the whole package.** `top_four` and `top_ten` become the
   same kind of object, differing only in `n`, so a session's rank means one
   thing everywhere.
2. **No dependency on a built label.** `snr_auc` comes from a checked-in CSV
   covering all 61 reference sessions, so a scope can be resolved without
   reading any `_strategy_svm.npz` — which also means scope membership is
   inspectable before the labels stage runs.
3. **Out-of-fold accuracy certifies the wrong mazes anyway.** It is measured
   on mazes 1 and 6, the SVM's own training axis; mazes 2-5 are never in any
   fold, so a high score speaks to the axis, not to the labels on the four
   mazes where most of this analysis reads. `dendro_agreement_2345` in
   `results_raw_k<K>.csv` stays the only handle on those, and ranking by *it*
   would reintroduce exactly the dependence on the dendrogram that a second
   label source exists to avoid.

`svm_auc`, the in-house recomputation of the same maze-1-vs-6 quantity, tracks
`snr_auc` at Spearman **+0.945** across the 23 sessions, so this is close to
the ordering an SVM-side AUC would have given in any case.

## 8. CV design

Split-half CV, `equal_halves` (`eye_pre_flash.plotting.similarities.heatmap_eq`):
each qualifying cell contributes two disjoint equal-size halves per split,
meaned into d-dimensional vectors, correlated
`r_ij = mean(pearson(A_i, B_j), pearson(B_i, A_j))`, averaged over
`--n-splits` (default 20) splits and then over sessions.

**Every cell is computed.** `--min-trials` (default 4) is the arithmetic
floor — a cell must split into two halves of >= 2. Below `--min-stable`
(default 8) a cell is still computed, at its own smaller half size, and
marked thin (`*`) rather than dropped. The shared half size for a session is
set by its **stable** cells only (`min` over cells with n >= min-stable of
`n // 2`), not by the thinnest cell in the session — a single 2-trial cell no
longer drags every other cell in that session down to 1-trial halves.
Unbalanced halves are not merely noisier, they are biased in the direction
being tested: the majority-strategy cell gets cleaner halves and hence a
higher diagonal, so confining that bias to cells marked thin matters. High
variance in the few-session scopes (`publication`, and often `top_four`) is
accepted for now rather than excluded.

The exhaustive-subset "comparable core" restriction the deleted
`heatmap_labels` used (`common_cell_set`, `unreliable_mask`, `matched_matrix`)
is deliberately **not** carried into the plotting path here — every
per-session correlation still lands in `results_raw_k<K>.csv`, so that
restriction remains reconstructable later without re-running anything, at the
cost of not being applied automatically.

## 9. The permutation null

`--n-perm` (default 1000) label-shuffle resamples per maze, shuffling
strategy labels **within each (session, maze)** — which preserves every
cell's trial count exactly — and recomputing the full within-maze delta each
time. A permutation null rather than a bootstrap CI: the question is not "how
much does this vary between recording days" but "is this delta
distinguishable from zero at d as low as 6-12 with these counts", and the
null measures that empirically at the actual dimensionality and coverage
rather than assuming a bootstrap's between-session resampling is the right
error model at this dimensionality. Reported per maze in `nulls_k<K>.csv`: the
observed delta, the null mean and SD, a two-sided p-value (+1 Laplace
smoothing, so p is never exactly 0), and the null's 2.5-97.5th percentiles.

## 10. How to read the output

1. **Census first** (the report's first table) — trials pooled per cell, how
   many sessions each cell appeared in, and which cells are thin.
2. **The within-maze table** — `r(H,H)`, `r(S,S)`, `r(H,S)` on matched
   session sets per maze, `delta = 0.5*(r_HH + r_SS) - r_HS`, and the
   permutation p-value. This is the decisive read.
3. **The full matrix** — every cell, thin ones marked `*`. Never compare a
   marked entry against an unmarked one as if they carried equal weight.
4. Beyond the same-maze cells: same-strategy-different-maze (does a decoded
   strategy look like itself across geometries?), then the two cross-strategy
   quadrants.

## 11. Output files

```
eye_pre_flash/corr/out/
  <source>/                dendro | svm
    <feature>/              occupancy | occupancy_bin | bigram
      <variant>/            full | no_origin | mean_removed
        examples_k<K>.csv
        examples_dims_k<K>.csv
        <scope>/            dendro: publication | top_four;  svm: top_ten
          matrix_<Monkey>_k<K>.png
          results_raw_k<K>.csv
          nulls_k<K>.csv
```

Every CSV stem carries `K`. `corr_io.write_rows` truncates, and the path holds
only (source, feature, variant, scope) — so while `K` lived in the sweep and
not in the name, each `K` silently overwrote the previous one's rows and only
the last `K` written survived on disk.

`results_raw_k<K>.csv` — one row per (leaf config, monkey, session, cell i, cell
j): every per-session correlation plus its trial counts, half sizes, and
stability flags, so the plotted matrix, the within-maze table and (with
`presence`/`stable` reconstructed) a comparable-core restriction are all
rebuildable without recomputation. `nulls_k<K>.csv` — one row per (leaf config,
monkey, maze): the permutation summary from §9.

`examples_k<K>.csv` (long, one row per sampled row x dimension) — `row_kind` in
`{trial, half_mean}`. Both matter: the skew question has a different answer
before and after averaging a half's worth of trials, and it is the half-mean
the Pearson actually sees. `examples_dims_k<K>.csv` (one row per cell x dimension,
plus a `dim="__vector__"` summary row per cell and a `dim="__pc1__"` row for
the `mean_removed` variant, plus a `dim="__mean__"` row beside it) —
`frac_zero`, `n_distinct` and `top_value_frac`
are the direct "heavily skewed to a few values" read; `mean_n_nonzero_per_trial`
says whether a nominally K-dimensional vector is effectively lower-dimensional
in practice.

## 12. Caveats

- **A Pearson over 6-144 points.** The smaller `K`/feature combinations put
  real weight on the permutation null in §9 — do not read a bare delta
  without it.
- **Coverage in the thin scopes is accepted, not solved.** `publication` (2
  sessions/monkey) will show many thin (`*`) cells. `dendro/top_four` (4) and
  `svm/top_ten` (10) are where this analysis has the coverage to be
  informative — and `top_four` is as wide as the dendro source gets, since
  vetting rejects the other 15 sessions outright (§7).
- **Zero-variance `occ_bin` cells** (every trial visited every state) produce
  a degenerate NaN Pearson, counted in `n_degenerate` and noted on the figure
  — an em-dash from this cause means "no variance", not "no coverage".
- **The SVM source's anchor asymmetry** (§6): validated where it matters
  least, soft-checked where it matters most.
- **144 cells across up to 36 leaves, uncorrected.** The within-maze delta on
  one pre-declared (source, feature, variant, k) is the read; the rest is a
  robustness surface, not an independent search.
- **Trials are pooled across sessions**, and strategy is strongly
  session-clustered — this reads as "on these recording days", not as
  generalisation to unseen days.

## 13. Reproduce

```bash
uv run python -m eye_pre_flash.corr.run --source dendro --feature occupancy \
    --variant full --scope top_four --monkey Faure --k 12
```

Full sweep on the cluster: `mkdir -p logs && sbatch slurm/run_corr.sbatch`,
then `rsync -av engaging:strategy-selection/eye_pre_flash/corr/out/
./eye_pre_flash/corr/out/`.
