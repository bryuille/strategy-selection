# msp_per_session — one recording at a time

Same fixed-codebook MSP estimator as `msp/`, but each
(monkey, session, maze, variant) cell uses only that session's labelled
trials. The within-session label-shuffle null is then a shuffle of the sole
session. Reuses `msp` for features, labels, estimator, and figures; does not
edit that package.

```bash
uv run python -m msp_per_session.build --dry-run
uv run python -m msp_per_session.build --radius 0.5
uv run python -m msp_per_session.build --radius 0.5 --window pre1466   # legacy window
```

The window is msp's: `geofix` (maze onset to fixation onset) by default, tag
`r0.5`; the legacy `pre1466` window is tag `r0.5_pre1466`. The directory
called `out/r0.5/` before 2026-10-02 was pre1466 and is now
`out/r0.5_pre1466/`.

Output under `out/<tag>/<Monkey>/`, grouped by plot type with maze-then-session
names: `<variant>/maze<M>_<session>.png`, `codebook/`, `strategy/`.

## Significant cells (r0.5_pre1466, p < 0.05)

At radius 0.5, among tested `full` and `mean_removed` cells (skipped when
either strategy has fewer than 10 trials), five cells clear p < 0.05 — all
Nielsen, none Faure. Three unique maze×session pairings:

| maze | session | significant under | Δ | z | p | n_H | n_S | balance min/max |
|---|---|---|---|---|---|---|---|---|
| 2 | `Nov_1_g0` | mean_removed only | +0.958 | +2.38 | 0.012 | 41 | 12 | 0.29 |
| 4 | `Nov_18_g0` | full and mean_removed | +0.137 / +1.113 | +4.40 / +2.58 | 0.005 / 0.006 | 43 | 27 | 0.63 |
| 5 | `Oct_21_g0` | full and mean_removed | +0.099 / +0.755 | +4.20 / +2.51 | 0.004 / 0.016 | 15 | 87 | 0.17 |

(`full` / `mean_removed` when both are listed.) Measured from
`out/r0.5_pre1466/<Monkey>/<variant>/results.csv` (then `out/r0.5/`),
2026-10-01, under the legacy pre1466 window.

## Not driven by trial-count balance

These cells are **not** significant merely because H/S counts are better
balanced. Among all tested `full` / `mean_removed` cells:

* Median balance (`min(n_H,n_S) / max`) is **lower** for significant cells
  (0.29) than for non-significant ones (0.33); median minority count is
  similar (15 vs 16).
* The best-balanced sessions of the same mazes are **not** significant. For
  Nielsen maze 4, `Nov_3_g0` (40/45, bal 0.89) and `Nov_9_g0` (34/39, bal
  0.87) sit above `Nov_18_g0` (43/27, bal 0.63) in balance and stay
  non-significant (mean_removed p = 0.66 / 0.11). Faure maze 2 `june_29_g0`
  is perfectly balanced (29/29) with z ≈ 0.7–0.8.
* `Oct_21_g0` maze 5 is among the **worst**-balanced tested cells (15 H /
  87 S, bal 0.17) and is still significant under both variants.
* `Nov_1_g0` maze 2 (41/12, bal 0.29) is also below the median balance of
  non-significant cells.

So z tracks something other than raw H/S count balance: well-balanced cells
can be null, and imbalanced cells can clear the threshold. Power still
requires `min(n_H, n_S) ≥ 10` (the skip rule), but within the tested set
balance does not select the significant ones.
