# msp_trends — msp on a merged left / origin / right codebook

`msp/` finds the H-vs-S gaze effect cleanly only in Faure maze 2 and Nielsen
mazes 2 and 4. Its state profiles, though, suggest a consistent direction per
monkey: H trials go to the **left exits**, S trials to the **origin** (Faure)
or the **right exits** (Nielsen). This package reruns the msp 2x2 on a codebook
that states that trend: three states, the same for both monkeys,

| merged state | msp balls |
|---|---|
| `left` | LU (−1, 1), LD (−1, −1) |
| `origin` | origin (0, 0) |
| `right` | RU (1, 1), RD (1, −1) |

and adds a pooled-over-mazes estimate, so a single figure per monkey can cover
mazes 2–5.

Reuses `msp` read-only (features, labels, scope, estimator pieces, figure
helpers); edits nothing outside this folder.

## Why three states and not two

The intended contrast is two states per monkey (left vs origin, left vs right).
That cannot go through the 2x2: Pearson between two **2-vectors** centres each
on its own mean, leaving `±(a₁−a₂)/2 · (1, −1)`, so
`r = sign((a₁−a₂)(b₁−b₂))` — always ±1, blind to magnitude. Every block-mean
pair in Faure maze 2 has left < origin, so all four cells are +1 and Δ = 0
whatever the strategies do. The third state (origin for Nielsen, right for
Faure) is therefore kept. msp's `no_origin` variant is dropped for the same
reason (two dimensions). `checks.py` pins the degeneracy.

## Pipeline

1. **Features**: msp's five-ball extraction at the tag's radius (`msp.features.extract_monkey_features`), cached as `<Monkey>_msp_trends_r<radius>.npz`,
   unchanged — same rows, same fixation assignment, same radius. The merge is a
   relabelling of the five balls:
   * `binary` (primary, as msp): visited(left) = visited(LU) or visited(LD) —
     column max of `occ_bin`.
   * `dwell` (secondary): assigned fixation ms per merged state (column sum of
     `occ_ms`) over the trial's total assigned ms. Trials with no assigned dwell
     stay all-zero and are kept (counted in `n_degenerate`).
2. **Variants**: `full`; `mean_removed` (maze grand mean, label-free, fitted
   once per maze — as msp); `mean_removed_balanced` (√n-weighted centre,
   refit under every shuffle — see below).
3. **Per maze**: the msp estimator exactly (`estimator.run_maze` is
   `msp.estimator.estimate` rebuilt from its pieces; `checks.py` asserts
   identical output for the same seed). Seed for maze `M` is `seed·100 + M`.
4. **Pooled**: each maze keeps its own 2x2 — block means never mix mazes, so the
   very different H/S balance per maze (Faure m3 517 H / 36 S, m4 49 / 577)
   cannot enter. Pooled Δ = trial-count-weighted mean of the per-maze Δs
   (weight ∝ n_H + n_S). Null draw *i* = the same weighted mean of the per-maze
   null draws *i*; mazes shuffle independently, each within its own sessions,
   so the joint draw is a within-(session, maze) shuffle. `z`, `p` as msp.
   The pooled 2x2 shown is the same weighted mean of the cells (display only).

**Window: maze onset to fixation onset, everywhere** (msp's `geofix`,
`[geo_present, fix_start]`, re-detected by msp from the pooled
`*_eye_data.npz`; runs locally). It is not a CLI option. Fixation onset is
~1534 ms after maze onset in nearly every trial (IQR 1533–1650 ms; never
below 1466 ms), so msp's legacy `pre1466` window `[fix_start − 1466, fix_start]`
differs only by the first ~70 ms. When both were run (2026-10-02) z agreed
to within ~1.5 per cell; geofix is kept because it states as "maze onset to
fixation".

Tags: `r1`, `r0.5` (SVM labels, top ten), `r0.5_dendro`, and the region
codebooks `r0.5quads`, `r0.5halves` (below). A tag is
`r<radius>[quads|halves][_dendro]`.

## What the merged means look like

Merged block means per strategy (left / origin / right), r1, from
`build --dry-run`:

| monkey | maze | n_H / n_S | binary H | binary S | dwell H | dwell S |
|---|---|---|---|---|---|---|
| Faure | 2 | 322 / 210 | .95/.94/.08 | .90/.96/.13 | .50/.46/.03 | .41/.54/.05 |
| Faure | 3 | 520 / 36 | .81/.94/.21 | .89/.94/.17 | .38/.54/.07 | .39/.58/.03 |
| Faure | 4 | 50 / 578 | .84/.98/.12 | .88/.98/.15 | .38/.56/.04 | .38/.58/.05 |
| Faure | 5 | 95 / 627 | .81/.92/.20 | .75/.96/.25 | .35/.57/.06 | .31/.60/.08 |
| Nielsen | 2 | 552 / 111 | .46/.95/.49 | .41/.90/.53 | .15/.64/.20 | .14/.58/.24 |
| Nielsen | 3 | 684 / 73 | .57/.93/.45 | .48/.84/.48 | .18/.61/.19 | .15/.56/.21 |
| Nielsen | 4 | 351 / 388 | .41/.91/.39 | .32/.94/.49 | .15/.66/.17 | .10/.66/.22 |
| Nielsen | 5 | 80 / 789 | .49/.97/.40 | .52/.96/.47 | .15/.69/.15 | .16/.65/.17 |

`n_H` / `n_S` are a few trials above msp's pre1466 counts: geofix re-detects
gaze per trial, so a slightly different set of trials clears the
two-valid-samples floor.

Binary `left` saturates for Faure (0.75–0.95 under both strategies): a trial
that visits either left exit counts once. Dwell share keeps the magnitude,
which is why it is carried as a second block. Even so, Faure m3/m4 and
Nielsen m5 show essentially no H − S difference in the means; Nielsen's
`right` is S > H in every maze.

## Results (2026-10-02, n_perm 1000)

`z` per maze 2 / 3 / 4 / 5 → pooled (`*` p < 0.05):

| tag | monkey | variant | binary | dwell |
|---|---|---|---|---|
| r1 | Faure | full | +2.8* / +0.2 / −0.4 / +1.5 → +1.3 | +7.4* / −0.5 / −0.7 / +0.4 → +1.4 |
| r1 | Faure | mean_removed | +1.6 / +0.6 / −1.0 / +1.3 → +1.5 | +2.2* / +0.4 / −1.5 / +0.5 → +1.1 |
| r1 | Nielsen | full | +0.5 / +0.6 / +9.4* / −0.7 → +1.4 | +0.9 / +0.0 / +10.6* / −0.6 → +1.4 |
| r1 | Nielsen | mean_removed | +0.7 / +0.8 / +2.7* / −0.1 → +2.1* | +1.8 / +0.3 / +2.2* / −0.2 → +2.1* |
| r0.5 | Faure | full | +1.6 / −0.7 / −0.5 / +0.4 → −0.3 | +6.5* / −0.6 / −0.6 / +0.1 → +0.8 |
| r0.5 | Faure | mean_removed | +0.9 / −1.6 / −1.1 / +0.4 → −0.4 | +2.2* / −1.1 / −1.5 / +0.1 → +0.2 |
| r0.5 | Nielsen | full | +3.1* / +0.3 / +8.5* / −0.5 → +3.3* | +4.4* / +0.8 / +10.2* / −0.4 → +4.7* |
| r0.5 | Nielsen | mean_removed | +1.8* / +1.3 / +2.0* / −0.3 → +2.4* | +1.8* / +1.3 / +2.1* / +0.4 → +2.9* |

Across all four axes the same cells are significant (Faure m2, Nielsen m4,
Nielsen m2 at r0.5); the choices move magnitudes. `full` gives 3–5× the `z`
of `mean_removed` on the strong cells (near-1 cells, tiny null sd). Radius
moves the monkeys oppositely: r0.5 helps Nielsen (m2, pooled) and hurts
Faure. Dwell matters for Faure only (m2 significant under every dwell
setting, binary saturates). Nielsen pools significant under every
`mean_removed` and every r0.5 setting; Faure never pools significant.

**Weighting caveat.** Under `full`, the per-maze null sd differs up to 10×
between mazes (Faure r1 binary: 0.0008 in m2 vs 0.0075 in m3). Trial-count
weights therefore let a noisy maze dilute a precise one: Nielsen r1's m4
(z +9.4) pools to only +1.4. An inverse-null-variance weighting would give
+8.5 there (Faure r1 binary +2.8). Choosing it after seeing these numbers is a
forking path, so it is not switched; trial weights remain what was
pre-specified.

## `r0.5_dendro`: dendrogram labels, publication sessions

Same features as `r0.5`, but labels are the Ward-dendrogram
`*_strategy_choices.npz` and only four sessions are in scope: Faure
`june_8_g0`, `june_24_g0`; Nielsen `Nov_3_g0`, `Nov_6_g0`. This uses Nov_6_g0
as specified for the paper; `msp/labels.py`'s `PUBLICATION_SESSIONS` lists
Oct_22_g0 instead. Selected with `--labels dendro` (tag suffix `_dendro`,
`config.LABEL_SCOPES`). Every one of these sessions lies inside msp's top-ten
feature scope (geofix re-detects only those), so the `r0.5` cache serves both tags.

Every maze is tested (see "Small strategies" below). With two sessions per
monkey three cells are small and carry the `*`: Faure maze 3 (90 H / 4 S) and
Nielsen maze 5 (3 H / 152 S); Nielsen maze 3 has 9 S. Results (2026-10-02):

| monkey | block / variant | z per maze 2 / 3 / 4 / 5 → pooled |
|---|---|---|
| Faure | binary full | −0.6 / −0.7* / +0.3 / −0.6 → −0.9 |
| Faure | dwell mean_removed | −0.4 / −0.1* / −0.3 / −0.6 → −0.7 |
| Nielsen | binary full | +0.6 / +0.6 / +0.9 / −0.4* → +0.2 |
| Nielsen | binary mean_removed | +1.7 / +1.0 / +1.0 / −0.9* → +1.7 |
| Nielsen | dwell mean_removed | +2.4 / −0.3 / +1.2 / −0.3* → +1.8 (p 0.07) |

Nielsen maze 2 under these labels has S dwelling *less* at the origin and more
on both exits (S origin share .36 vs H .72, n_S = 15), which is not the
left-vs-right pattern the SVM labels show.

## `mean_removed_balanced`

The grand mean weights each strategy by its trial count. Centre on
`c = w_H·μ_H + w_S·μ_S` and each strategy's expected residual is
`w_other·(μ_X − μ_other)`, while its split-half noise falls as `1/√n_X`, so
its diagonal's signal-to-noise goes as `w_other·√n_X`. Under the grand mean
(`w ∝ n`) the minority is favoured by `(n_maj/n_min)^1.5` — about 40× for
Faure m4 (577 : 49) — and the majority, centred almost on its own mean, sits
near the split-half mirror floor. Equal weights overshoot the other way (the
majority favoured by `√(n_maj/n_min)`). `w_X ∝ √n_X` equalises the two
(`features.balanced_centre`).

The centre uses the labels, so it goes through the estimator's `transform`
and is refit under every shuffle; fitting it once on the real labels would
leak them into the null. `checks.py` pins the centre algebra, exact agreement
with `msp.estimator.estimate(transform=…)`, and that at 500 : 50 with a planted
effect `|r_HH − r_SS|` falls from 0.90 (grand mean) to 0.08.

On the data (2026-10-02) it does what it is for: the diagonals become
comparable (Nielsen r1 binary m2 −0.42 / +0.29 → +0.03 / +0.08; Faure r1
binary m3 −0.74 / +0.28 → +0.07 / +0.06). **`z` hardly changes** — per maze
within ~0.4, pooled within 0.5 and mostly lower:

| tag | monkey | block | grand mean: m2 / m3 / m4 / m5 → pooled | √n-balanced |
|---|---|---|---|---|
| r1 | Faure | dwell | +2.2* / +0.4 / −1.5 / +0.5 → +1.1 | +2.1* / +0.3 / −1.5 / +0.4 → +0.6 |
| r1 | Nielsen | binary | +0.7 / +0.8 / +2.7* / −0.1 → +2.1* | +0.6 / +1.0 / +2.7* / −0.1 → +1.9 |
| r1 | Nielsen | dwell | +1.8 / +0.3 / +2.2* / −0.2 → +2.1* | +1.8 / +0.3 / +2.2* / −0.1 → +1.8 |
| r0.5 | Nielsen | binary | +1.8* / +1.3 / +2.0* / −0.3 → +2.4* | +1.8* / +1.3 / +2.0* / −0.4 → +2.2* |
| r0.5 | Nielsen | dwell | +1.8* / +1.3 / +2.1* / +0.4 → +2.9* | +1.7 / +1.2 / +2.1* / +0.5 → +2.7* |

That is the expected outcome: the within-session shuffle already carries
the count skew, so the grand-mean `z` was calibrated; balancing fixes the raw
cells, not the test. Nielsen r1's pooled result drops just below p 0.05
under the balanced centre; at r0.5 it stays significant.

## Region codebooks: `r0.5quads`, `r0.5halves`

The same origin ball (radius 0.5), but instead of four exit balls every other
point **inside the maze** is assigned:

* **Perimeter**: the outer edge of the four r1 exit balls joined by straight
  tangents, i.e. every point within 1.0 of the square the exits span (a
  rounded square reaching ±2; `features.inside_maze`). Outside it, unassigned.
* **`quads`**: by quadrant, LU / LD / RU / RD (5 states, display order
  LU, LD, origin, RU, RD). x ≥ 0 counts as right and y ≥ 0 as up.
* **`halves`**: by side, left / right (3 states, as `balls`).

One extraction serves both (`<Monkey>_msp_trends_r0.5_region.npz`;
`halves` merges `quads`). It is msp's geofix extraction with msp's ball rule
swapped for `features.region_states` during the call (`mock.patch`, scoped
and undone; msp's code is not edited). Same rows as the ball caches (5041 /
6263). Fixations assigned: Faure 84.5% (vs 60.1% under r0.5 balls, of
29 701), Nielsen 70.5% (vs 48.8%, of 35 354); Nielsen's remainder is mostly
the off-maze cloud beyond the upper right.

**Stem caveat.** A dense cluster sits on the upper stem just above the origin
ball, straddling x = 0, so the sign of x splits it between left and right
(LU / RU). 12% of Faure's and 22% of Nielsen's exit-region fixations lie
within 0.25 of x = 0 (10% / 20% on the upper stem); of those, 35% / 46% fall
to the right. That is gaze on the stem, not a choice of side, and it dilutes
the left-vs-right contrast — most for Nielsen, whose trend is exactly
left vs right.

`z`, r0.5, maze 2 / 3 / 4 / 5 → pooled (`*` p < 0.05):

| monkey | block / variant | balls | halves | quads |
|---|---|---|---|---|
| Faure | binary full | +1.6 / −0.7 / −0.5 / +0.4 → −0.3 | +1.3 / +0.1 / −0.1 / +0.7 → +0.5 | +4.6* / +0.5 / +0.4 / +0.2 → +1.3 |
| Faure | binary mean_removed | +0.9 / −1.6 / −1.1 / +0.4 → −0.4 | +1.9* / +1.2 / +0.4 / +0.4 → +1.9 | +2.6* / +1.0 / +0.3 / +0.6 → **+2.3*** |
| Faure | dwell mean_removed | +2.2* / −1.1 / −1.5 / +0.1 → +0.2 | +2.4* / +0.7 / −0.9 / +0.9 → +1.8 | +2.7* / −0.2 / +0.1 / +0.2 → +1.7 |
| Nielsen | binary full | +3.1* / +0.3 / +8.5* / −0.5 → +3.3* | −0.8 / +1.3 / +9.3* / +0.0 → +1.8 | +0.2 / −0.1 / +7.9* / −0.4 → +1.4 |
| Nielsen | binary mean_removed | +1.8* / +1.3 / +2.0* / −0.3 → +2.4* | −1.5 / +0.9 / +2.5* / +0.5 → +1.5 | +1.0 / +0.9 / +2.9* / −0.7 → +2.1* |
| Nielsen | dwell mean_removed | +1.8* / +1.3 / +2.1* / +0.4 → +2.9* | +0.6 / −0.3 / +2.5* / −0.2 → +1.5 | +0.5 / −0.4 / +2.3* / −0.1 → +1.4 |

For Faure the regions help: mazes 3–5 turn from negative to small positive,
and quads gives Faure's first significant pooled result (binary
`mean_removed` +2.3, balanced +2.0). For Nielsen they hurt: maze 2 loses its
effect and the pooled `z` drops, consistent with the stem split blurring the
left / right contrast. Maze 4 holds under every codebook.

## Small strategies

msp skips a maze with fewer than 10 trials of either strategy; this package
does not. A maze is tested whenever the split-half is possible (at least 2
trials per strategy, `config.MIN_TRIALS`), and a score whose smaller strategy
has 5 or fewer trials (`config.LOW_N`) is marked `*` on its figure title, the
paper figure, and the `low_n` csv column. Such mazes enter the pool with their
trial-count weight, which is small.

With so few trials, a shuffled half can have a constant block mean in every
round, so that null draw is undefined (Pearson 0/0). msp would let one such
draw turn `z` into NaN and `p` into a spurious floor. Here those draws are
dropped from the null (`n_null_undefined` in the csv); the result is identical
to msp whenever every draw is defined. If the *observed* score is undefined,
`z` and `p` are reported as missing and the maze is left out of the pool.

## Reading the figures

* **The 2x2 is unsigned.** It says H and S gaze differ, not which way. Every
  2x2 is therefore paired with the H vs S profile (mean ± SE per merged state),
  and the pooled column with the trial-weighted within-maze H − S per state.
  The direction claim comes from those bars; the significance from `z`.
* Read `z`, never the raw `r` (msp.md: cells rise with half size; under
  `mean_removed` a small strategy's diagonal has a floor near −1). With K = 3
  under `full`, every cell sits near 1 — the effect lives in the third decimal
  and in `z`.
* In the paper figure the top row shares one colour scale (`--vmax` fixes it).
* msp.md's estimator caveats (unequal H/S noise, within-session drift,
  non-independent rounds, multiplicity) apply verbatim. The pooled `p` is one
  test per (monkey, tag, block, variant), not a correction of the per-maze ones.

## Running

```bash
uv run python -m msp_trends.checks                   # synthetic, no cache
uv run python -m msp_trends.build --dry-run --all-tags
uv run python -m msp_trends.build                     # r1
uv run python -m msp_trends.build --radius 0.5
uv run python -m msp_trends.build --radius 0.5 --labels dendro
uv run python -m msp_trends.build --radius 0.5 --codebook quads    # r0.5quads (halves likewise)
uv run python -m msp_trends.build --all-tags          # every tag in config.TAGS, incl. _dendro
uv run python -m msp_trends.build --monkey Faure --n-perm 200   # smoke test
```

About 15 s per monkey per tag at the default 1000 shuffles once caches exist.
A radius's first run builds its cache (minutes per monkey).

## Output

```
out/<tag>/<Monkey>/codebook.png, codebook_maze<M>.png    centroids by merged state
out/<tag>/<Monkey>/<block>/strategy_maze<M>.png          H vs S profile, one maze
out/<tag>/<Monkey>/<block>/<variant>/maze<M>.png         the 2x2, one maze
out/<tag>/<Monkey>/<block>/<variant>/results.csv         one row per maze + pooled
out/<tag>/paper/<block>/<variant>/<Monkey>.png          the paper figure
```

Cache: `$STRATEGY_DATA_ROOT/processed/<Monkey>_msp_trends_r<radius>.npz`, one per
(monkey, radius), shared by the label scopes. It holds msp's five-state blocks,
row metadata and fixation centroids unchanged (the merge happens on load) and
is rebuilt when msp's codebook, radius or window check fails.

`<block>` is `binary` or `dwell`. `results.csv` has msp's columns plus `block`,
`weight` (per maze), and on the `pooled` row `mazes_pooled` and the weight list.
Extra columns `low_n` (marked score) and `n_null_undefined` (dropped null draws). A maze with fewer than 2 trials of a strategy cannot be split; it is skipped and left out of the pool.

## Files

| file | purpose |
|---|---|
| `config.py` | codebooks, trend per monkey, blocks, variants, label scopes, tags, `out/` layout, colours |
| `features.py` | caches, region rule, codebook merges, variants; label scope / lookup |
| `estimator.py` | per-maze msp estimate keeping its null draws; pooled estimate |
| `figures.py` | per-maze 2x2 / profile / codebook, paper figure |
| `build.py` | entry point |
| `checks.py` | merge, 2-state degeneracy, equality with msp, pooling, power/calibration |
