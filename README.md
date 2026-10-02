# PFC strategy selection

Gaze during the pre-fixation window, scored against neural strategy labels
(hierarchical vs sequential). The main analysis is **msp** (maze-strategy
pairs) under `msp/`.

```bash
uv sync
```

## Data


| Tree              | Contents                                                                                           |
| ----------------- | -------------------------------------------------------------------------------------------------- |
| `data/mat/`       | Raw `.mat` (behavioral, eye, neural, single-trial). **Cluster only** — the one irreplaceable copy. |
| `data/npz/`       | Converted session caches                                                                           |
| `data/processed/` | Derived products (eye concatenations, clean events, unit-H warp, MSP features, strategy labels)    |


On a laptop, paths resolve under `./data/`. On Engaging, every Slurm job sets
`STRATEGY_DATA_ROOT` to the scratch volume
(`/home/byuille/orcd/scratch/strategy_selection_data/`), so `mat/`, `npz/`, and
`processed/` live there instead of under the code checkout.

Raw field reference: [DATA_DICTIONARY.md](DATA_DICTIONARY.md).  
Processed cache formats: [data/PROCESSED.md](data/PROCESSED.md).

```bash
# convert mat → npz (cluster; neural is large)
uv run -m data.convert all
uv run -m data.convert eye --monkey Faure

# strategy labels (cluster; converts neural as needed)
uv run python -m data.labels --type strategy --sweep
```



## MSP — maze-strategy pairs

**Question.** Within one maze, is gaze more similar across two groups that
share a decoded strategy (H or S) than across groups that do not? Geometry is
identical on both sides, so visual confounds cannot explain a difference.

**Design.** Fixed five-state codebook at the unit-H origin and four exits —
no k-means. SVM labels at the top-ten session scope; mazes 2–5; binary
occupancy; variants `full` / `no_origin` / `mean_removed`. Window:
`[fix_start − 1466 ms, fix_start]`.

**Statistic.** Pool labelled trials of one maze across in-scope sessions,
split each strategy into two halves, average each half to a vector, correlate
the four pairings.  
`Δ = 0.5·(r_HH + r_SS) − 0.5·(r_HS + r_SH)`, tested against within-session
label shuffles. Compare `z` across panels, not raw `r`.

Details and caveats: [msp/msp.md](msp/msp.md).

### Run locally

Needs attractor / clean-eye / behavioral / SVM-label caches in
`data/processed/` (eye caches build on first load; SVM labels from
`data.labels`).

```bash
uv run python -m msp.checks                         # estimator invariants (no cache)
uv run python -m msp.features                       # feature caches for both monkeys
uv run python -m msp.build --dry-run                # pooled counts + codebook coverage
uv run python -m msp.build                          # full: 2 monkeys × 3 variants × 4 mazes
uv run python -m msp.build --monkey Faure --maze 4 --n-perm 200   # smoke test
uv run python -m msp.build --radius 0.5             # tighter assignment balls
```

Figures land in `msp/out/r<radius>/<Monkey>/` (window: maze onset to fixation
onset, `geofix`; `--window pre1466` writes `msp/out/r<radius>_pre1466/`).

### Run on the cluster

Login: `orcd-login.mit.edu` (alias `engaging`). Code lives in
`~/strategy-selection`; data on scratch via `STRATEGY_DATA_ROOT`. Authenticate
once with SSH multiplexing, then push / submit / pull without further Duo
prompts.

```bash
# laptop: open the SSH master (Kerberos + Duo once)
ssh engaging

# laptop, from the repo root: open the in-repo control socket (needed for
# Claude Code, which cannot read ~/.ssh; leave it open, rerun if it goes stale
# after sleep or 8 h)
ssh -M -S .ssh-sockets/engaging -o ControlPersist=8h byuille@orcd-login.mit.edu

# laptop: push code (never use --delete-excluded — that can wipe mat/)
rsync -av --delete --exclude-from=.rsync-exclude ./ engaging:strategy-selection/

# cluster (interactive or after first push)
export STRATEGY_DATA_ROOT="/home/byuille/orcd/scratch/strategy_selection_data"
cd ~/strategy-selection && uv sync && mkdir -p logs

# MSP (builds its feature cache on first run if eye/label caches exist)
sbatch slurm/run_msp.sbatch --dry-run
sbatch slurm/run_msp.sbatch

# laptop: pull figures
rsync -av --exclude='__pycache__' \
    engaging:strategy-selection/msp/out/ ./msp/out/
```

Monitor: `squeue --me`, `tail -f logs/eyepresmsp-<JOBID>.out`, `scancel <JOBID>`.

After changing detector / QC constants, delete the derived eye caches and
rebuild — loaders only check that keys exist, not that settings match:

```bash
cd "$STRATEGY_DATA_ROOT/processed"
rm -f *_clean_eye_data_s*.npz *_attractor_eye_data.npz *_msp_fixed5_*.npz *_msp_trends_*.npz *_reg_fixed5_*.npz
```



## Linear regression

