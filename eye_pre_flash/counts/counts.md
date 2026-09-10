# H/S trial census per (session, maze)

One heatmap per label source. Each cell is a `(session, maze)` pair: the text
is `H/S`, the fill is the sequential share `S / (H + S)`, and a cell whose
minority strategy falls below 10% is printed red and bold — it has trials, but
effectively only on one side, so it cannot carry a within-cell H-vs-S
contrast.

```bash
uv run python -m eye_pre_flash.counts.census --source svm
uv run python -m eye_pre_flash.counts.census --source dendro
uv run python -m eye_pre_flash.counts.census --source svm --monkey Nielsen --scope all
```

Figures land in `eye_pre_flash/counts/<source>/hs_<monkey>_<scope>.png`. On the
cluster, `sbatch slurm/run_counts.sbatch` writes both sources for Faure.

## 1. What it is for

`corr` splits each maze's trials by decoded strategy and reads the same-maze
`(maze m, H)` vs `(maze m, S)` cells, where geometry is identical on both
sides (`corr.md` §1). That read only exists where both sides have trials. This
figure is the precondition, per session and per maze, for either label source —
and it answers a question the pooled `classifier/out/decoding/<scope>/counts_<monkey>.png`
table cannot: pooled counts can look comfortable while every individual
session is one-sided.

## 2. The trial pool is `decoding`'s, exactly

`census_counts` mirrors `classifier.decoding.load_labelled`: rows of the
windowed feature table (`k=6`, `unith` — the row set is identical across
caches, which is why `decoding.run_monkey` reads the same one for its census)
whose trial carries a finite label from the chosen source. So each maze column
here sums to that maze's row in `decoding`'s census table, and the two figures
cannot disagree. Verified on Faure/`allplus`/`dendro`: 449/227, 398/285,
457/279, 307/506, 343/587, 283/617, total 2237 H / 2501 S — identical to
`decoding/allplus/counts_Faure.png`.

Consequence worth knowing: the pool inherits the feature table's dependencies,
so it inherits `data.builder`'s detector constants. Measured on 2026-09-10,
doubling `IDT_DISPERSION_THRESHOLD_DEG` from 2.0 to 4.0 left every cell of
this census unchanged — the fixation threshold moves *within*-trial gaze
structure, not which trials clear the ≥2 valid in-window samples floor. Do not
assume that of a constant that gates trials rather than samples.

## 3. What the two sources look like (`allplus`)

### Faure, 13 sessions

| | dendro | svm |
| --- | --- | --- |
| one-sided cells (minority < 10%) | **15 / 78** | **45 / 78** |
| maze 1 sequential share | 0.336 | 0.055 |
| maze 2 | 0.417 | 0.454 |
| maze 3 | 0.379 | 0.106 |
| maze 4 | 0.622 | 0.908 |
| maze 5 | 0.631 | 0.860 |
| maze 6 | 0.686 | 0.942 |
| total | 2237 H / 2501 S | 1927 H / 2811 S |

The dendrogram labels are a gradient: every maze is genuinely mixed, sequential
share climbing 0.34 → 0.69 across the maze order, and only 15 of 78 cells are
one-sided. The SVM labels are close to a step function — mazes 1 and 3 almost
purely hierarchical, 4-6 almost purely sequential — and **58% of its cells
cannot support the contrast at all**.

That is the label source's definition showing through rather than a coverage
failure. The SVM is trained on maze 1 vs maze 6 (`data.labeler.build_svm_choices`),
so those two mazes are its own training axis; `corr.labels.SOURCE_SKIP_CELLS`
already drops its `1S`/`6H` cells outright for exactly this reason, and
`corr.labels` gives the SVM no anchor-vetted scope because grading it against
a map that calls mazes 1 and 6 hierarchical/sequential would be circular. This
census is that circularity made visible, and extends the observation past the
two skipped cells: maze 3 (0.106) and maze 4 (0.908) are nearly as one-sided
as the training mazes, so the SVM read is thin across most of the maze order,
not just at its ends.

**Maze 2 is the exception, and the interesting one.** It is the only maze where
the SVM finds a near-balanced split (0.454, 373 H / 310 S) — slightly *more*
balanced than the dendrogram's own 0.417 — while its neighbours 1 and 3 are
one-sided. A same-maze H-vs-S contrast under SVM labels therefore rests
mostly on maze 2, where geometry is held fixed and both sides are populated.
Worth knowing before reading an SVM-sourced result as if it were supported
evenly across mazes.

### Nielsen, 10 sessions

| | dendro | svm |
| --- | --- | --- |
| one-sided cells (minority < 10%) | **12 / 60** | **37 / 60** |
| maze 1 sequential share | 0.284 | 0.042 |
| maze 2 | 0.347 | 0.167 |
| maze 3 | 0.310 | 0.097 |
| maze 4 | 0.553 | 0.526 |
| maze 5 | 0.692 | 0.907 |
| maze 6 | 0.706 | 0.960 |
| total | 2268 H / 2279 S | 2313 H / 2234 S |

Same story, same magnitude: the SVM triples the one-sided cell count (12 -> 37
of 60) and pushes every maze but one to a near-pure label. Nielsen's dendro
census likewise reproduces `decoding/allplus/counts_Nielsen.png` exactly
(448/178 ... 255/613, total 2268 H / 2279 S).

### Where each animal's SVM boundary falls

Each animal keeps exactly one maze the SVM does not collapse, and it is not
the same maze:

* **Nielsen: maze 4** (svm 0.526, dendro 0.553). Both sources agree, and it is
  the maze `classifier.labels.ANCHOR_MAJORITY` already leaves unscored as the
  boundary — "genuinely mixed for Nielsen". The SVM sharpens the split on
  either side of it but leaves the boundary itself where the dendrogram has it.
* **Faure: maze 2** (svm 0.454, dendro 0.417) — *inside* the hierarchical
  block, between mazes 1 (0.055) and 3 (0.106), while Faure's own boundary
  maze under the dendrogram (maze 4, 0.622) is the one the SVM collapses
  hardest (0.908).

So the SVM does not merely sharpen the dendrogram's gradient: for Faure it
relocates the transition, putting the step between mazes 3 and 4 and leaving a
balanced maze stranded behind it. Whether maze 2 carries real within-maze
strategy variation for Faure or is where this SVM is simply least decisive is
not something this census can tell you — but it is the cell an SVM-sourced
same-maze contrast will rest on, so it is worth resolving before leaning on
one.

## 5. Per-trial agreement, four sessions

The census above is pooled per maze; it says nothing about whether dendro and
svm are labelling the *same trials* the same way within a session. Checked
directly (both sources' labels compared trial-by-trial, same trial set
`corr.run._dendro_svm_agreement_2345` uses, extended here to all six mazes
rather than just 2/3/5):

| session | monkey | n | agreement (all mazes) | agreement (mazes 2/3/5) |
| --- | --- | --- | --- | --- |
| june_24_g0 | Faure | 367 | 0.866 | 0.870 |
| june_8_g0 | Faure | 293 | 0.986 | 0.979 |
| Oct_22_g0 | Nielsen | 459 | 0.834 | 0.814 |
| Nov_3_g0 | Nielsen | 498 | 0.982 | 0.981 |

Per-maze breakdown:

| session | maze 1 | maze 2 | maze 3 | maze 4 | maze 5 | maze 6 |
| --- | --- | --- | --- | --- | --- | --- |
| june_24_g0 | 0.977 (44) | 0.700 (50) | 0.962 (52) | 0.806 (62) | 0.911 (90) | 0.841 (69) |
| june_8_g0 | 1.000 (44) | 0.976 (41) | 1.000 (43) | 0.980 (49) | 0.968 (62) | 1.000 (54) |
| Oct_22_g0 | 0.710 (62) | 0.652 (69) | 0.766 (77) | 0.841 (82) | 0.978 (91) | 0.987 (78) |
| Nov_3_g0 | 1.000 (65) | 0.965 (86) | 0.976 (82) | 0.953 (85) | 1.000 (96) | 1.000 (84) |

(n in parentheses; n is the same for both label sources here since every trial
in this table already cleared both sources' finite-label filter.)

`june_8_g0` and `Nov_3_g0` track each other closely everywhere — the two
label sources are effectively interchangeable there. `june_24_g0` and
`Oct_22_g0` disagree far more, and specifically at maze 2 (0.700, 0.652) — the
maze `mazestratpair` uses to check cross-maze replication of the maze-4
same-maze result, and the maze §3 above already flagged as Faure's SVM
"exception." `Oct_22_g0` is one of only two sessions in Nielsen's
`dendro/publication` scope, so a maze-2 `dendro` vs `svm` discrepancy there is
not just noise around one ground truth — the two sources are assigning
meaningfully different labels to roughly a third of that session's maze-2
trials.

## 4. Scopes and vetting

Scopes come from `classifier.labels.SCOPES`, not from `corr.labels.SOURCE_SCOPES`
— this figure is a coverage census, so it wants the widest pool that labels at
all rather than the ranked analysis scopes. `allplus` is the default and is
unvetted (`classifier.labels.UNVETTED_SCOPES`), which is what makes it the
right pool here: a session whose labels fail anchor vetting still has a real
trial census, and seeing it is the point.

The `svm` source passes `vet=False` regardless of scope, copying
`corr.labels.source_scope_sessions`'s reasoning rather than re-deriving it.
