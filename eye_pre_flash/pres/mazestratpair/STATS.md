# Reading these figures

Each figure is one (monkey, maze, feature, label source) cell of
`eye_pre_flash.mazestratpair`, restricted to the `mean_removed` variant and
stripped of everything but the numbers that decide whether the result is
real. Full context — every variant, every diagnostic column — lives in
`eye_pre_flash/mazestratpair/mazestratpair.md` and its `out/` CSVs; this is
the slide version.

## The matrix

A 2x2 grid, rows/columns `{maze}H` (hierarchical) and `{maze}S` (sequential):

* **Diagonal** (`HH`, `SS`) — split-half reliability: correlate two random
  halves of that strategy's own trials against each other. This is the
  ceiling — how similar a strategy's gaze pattern is to *itself* across a
  fresh split, i.e. how much of the signal is real structure versus
  split-to-split noise.
* **Off-diagonal** (`HS`) — cross-strategy correlation: one strategy's trials
  against the other's, same maze, same half size. If gaze pattern really
  depends on strategy (not just on the maze, which is held fixed here), this
  number should sit below the diagonal ceiling.
* **`n = ...`** under each diagonal cell is the pooled trial count for that
  strategy across every session used, not the number behind any single
  correlation (each correlation is computed from one session's random half,
  much smaller — see the CSVs for that).
* A tick label with `*` marks a thin cell: the minority strategy fell short
  of a stable split-half size in most sessions it appeared in. Its numbers
  are real but noisier than an unmarked cell's.

**Colour is parula (the same ramp every other similarity figure in this repo
uses) even though `mean_removed` can go negative.** The canonical
`mazestratpair` output gives this variant a zero-centred diverging map
instead, because its sign is arithmetic (mean-subtraction pushes the
off-diagonal negative) rather than informative on its own — see
`eye_pre_flash/mazestratpair/figures.py`. These slides use parula anyway for a
single consistent colour language; read the printed numbers, not the colour,
when comparing `mean_removed` across panels.

## The three statistics

* **Δ (delta)** = `0.5·(r_HH + r_SS) − r_HS`. The raw gap between the
  diagonal ceiling and the cross-strategy correlation, in raw correlation
  units. **Not comparable across features or variants** — different
  transforms live on different scales, so a bigger Δ on one panel does not
  mean a bigger effect than a smaller Δ on another.
* **z** = `(Δ_observed − null_mean) / null_sd`, where the null is built by
  reshuffling the H/S labels within each (session, maze) 200-1000 times and
  recomputing Δ each time — i.e. "how big a gap chance alone produces, given
  this exact data's noise." z **is** comparable across panels, because it is
  already normalised by each panel's own chance variability. This is the
  number to trust when comparing how strong a result is from one panel to
  the next.
* **p** — the two-sided fraction of shuffles that produced a gap at least as
  extreme as the real one. Standard convention: p ≤ 0.05 is called
  significant. A p tagged **(floor)** means *no* shuffle beat the observed
  gap, so p is reported as the smallest value the permutation count allows
  (`1 / (n_perm + 1)`) — that is a statement about the test's resolution, not
  a precise measurement. It means "at least this significant"; z is the
  number that says by how much.

## Caveats that don't fit on a slide but matter

* **One maze, one monkey per figure.** A result here says nothing about
  whether it holds in other mazes or the other animal — check the
  corresponding panel for those directly before generalising.
* **`dendro/publication` and `svm/top_ten` are different session pools and a
  different label source** (Ward clustering vs. a linear SVM, both fit on
  neural population activity — see `data/labeler.py`). They are not two
  independent replications of the same claim; agreement between them is
  informative, disagreement is informative, but neither panel is a rerun of
  the other with more data.
* **Multiple panels, one convention.** Every panel here is its own
  permutation test; p ≤ 0.05 on any single one carries the usual
  multiple-comparisons caveat once you're scanning several of these side by
  side.
