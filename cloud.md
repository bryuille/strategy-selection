# Running the classifier pipeline on Engaging

Login node `orcd-login.mit.edu`, aliased to `engaging` in `~/.ssh/config`.
Scheduler Slurm. One command end to end:
`uv run python -m eye_pre_flash.classifier.pipeline`.

This document is written from the cluster's side: what the cloud tree holds,
what arrives in it, and what it sends back.

## The cloud tree

`~/strategy-selection` on the cluster is the working copy. Every node mounts the
same home directory, so a compute node writes straight into it — nothing is
"sent back" by the job itself.

```
~/strategy-selection/
  *.py, slurm/, ...          code — replaced wholesale on every incoming push
  data/mat/                  THE ONLY COPY. Never arrives, never leaves, never deleted
  data/npz/                  built here from data/mat/, never transferred
  data/processed/            built here, never transferred
  eye_pre_flash/classifier/out/   written here by the job, then pulled down
  logs/                      Slurm stdout/stderr. Must exist before sbatch
  .venv/                     built here by uv sync, never transferred
```

The asymmetry is the whole design: **the cluster owns the data, the laptop owns
the code, and only figures travel back.** `data/mat/` lives here and nowhere
else; the caches derived from it are large enough that rebuilding them here is
cheaper than moving them.

## What crosses the boundary

`.rsync-exclude` is the contract. It excludes all of `data/`, so a push cannot
touch this tree's `data/mat/` — not overwrite it, and (because it is excluded,
not merely absent) not delete it either. **Never add `--delete-excluded`: that
is the one flag that would wipe `data/mat/`.**

rsync is invoked from the laptop in both directions — the login node cannot open
a connection back to it — so the two commands read as "code arrives" and
"figures leave":

```bash
# code arrives in ~/strategy-selection (run on the laptop, from the repo root)
rsync -av --delete --exclude-from=.rsync-exclude ./ engaging:strategy-selection/

# figures leave ~/strategy-selection/eye_pre_flash/classifier/out/
rsync -av engaging:strategy-selection/eye_pre_flash/classifier/out/ ./eye_pre_flash/classifier/out/
```

`out/` directories are deliberately **not** excluded, so one exclude file serves
both directions. The cost falls on this tree: **an incoming push with `--delete`
replaces the cloud's figures with whatever the laptop has, and deletes cloud
figures that do not exist there.** So the cloud's `out/` is only safe until the
next push — pull it down first, or push without `--delete`.

Remote paths are relative to the cluster home, so `engaging:strategy-selection/`
is `~/strategy-selection/`.

## Space: the number to watch

```
nfs001.lb:/home    128G used    195G soft    200G hard
```

Headroom to the soft limit is **67 GB**.

**npz runs 3-6× the mat it came from.** `np.savez` writes uncompressed; `.mat`
v7.3 is compressed HDF5, so the ratio tracks how well the spike data compresses,
which in turn tracks neuron count. Measured on the cluster: `Nov_3_g0` 1.2 GB →
3.9 GB (3.2×), `Oct_22_g0` 1.9 → 6.9 (3.6×), `june_8_g0` 1.3 → 5.0 (3.8×), but
**`june_24_g0` 2.2 → 14 GB (6.4×)** — 144 neurons, the most of the 23. Budget the
peak from 14 GB, not from an average. Two consequences:

- **Converting everything and then deleting `data/mat/` does not help.** It frees
  the mat but spends roughly three times that on npz, and during conversion both
  exist at once. The saving has to come from converting less.
- **So neural data is never held for the whole pool.** Eye and behavioral data
  are pooled across every session to fit the per-monkey codebook, so those
  convert wholesale. Neural is different: `data.labeler.CLUSTERING_SESSIONS` now
  lists **23** label-eligible sessions whose neural mats total ~42 GB; converted
  and kept, that is ~170 GB of npz — past the hard limit on its own.

  What is actually needed from a neural recording is its strategy label, which
  is **~5 KB**. So the `labels` stage sweeps: convert one session, write its
  label, delete the neural npz *and* its `trial_timebins` cache, then move to the
  next. Peak extra usage is one session's worth, not the pool's.

  ```bash
  python -m data.labeler --type strategy --sweep      # convert -> label -> free
  python -m data.labeler --type strategy --sweep --keep-neural   # keep the npz
  ```

  The sweep skips sessions whose label already exists, so an interrupted run
  resumes by re-issuing the same command, and one session that fails to convert
  is reported at the end rather than stopping the other 22. Watch it with
  `du -sh data/npz/Neural_Data` — that directory should stay near the size of a
  single session throughout.

Projected peak, from `du` ratios on a local checkout (which holds neural mat for
four sessions and no eye or behavioral mat, so these are ratios to apply, not
sizes to trust):

| What | Size |
| ---- | ---- |
| `data/npz` neural, one session at a time under `--sweep` | up to 14 GB |
| `data/npz` eye, both monkeys | 12.8 GB |
| `data/npz` behavioral + single-trial | 0.8 GB |
| `data/processed` attractor k6 + k12 | 16.8 GB |
| `data/processed` eye_data | 7.4 GB |
| `data/processed` trial_timebins, freed after each label | 1.5 GB |
| `data/processed` clean / features / labels | 0.4 GB |
| `.venv` + uv cache + Python 3.14 | ~2 GB |
| incoming code push (mostly `eye_pre_flash/plotting/out/`) | 0.6 GB |
| `data/processed` strategy labels, 23 sessions | 0.2 MB |
| **total** | **~48 GB** against 67 GB headroom |

The decoding tables run at **k=6 and k=12**, so the `features` stage builds the
k=12 attractor caches here — another 8.4 GB over the old k=6-only footprint.

Check before committing to a long job:

```bash
du -sh ~/strategy-selection/data/mat/*                 # how much bigger than local?
du -sh ~/strategy-selection/data/{npz,processed} 2>/dev/null
du -sh ~/* ~/.cache 2>/dev/null | sort -rh | head      # where the 128 GB went
```

After the `features` stage, `du -sh data/npz data/processed` gives the real
number; the analysis stages add only megabytes, so that reading is the peak.

**If you get tight**, `data/npz/Neural_Data` is droppable *after* `features`
finishes: it is read only to build `*_trial_timebins.npz` and the strategy
labels, both then cached in `data/processed`. That frees ~18 GB. Any later
`--from labels` rerun will need it back.

`/pool/$USER` and `/nobackup1/$USER` did not exist for this account when checked.
If they appear, symlinking `data/npz` and `data/processed` onto pool is the
cleaner fix — the code uses repo-relative paths and follows symlinks.

## SSH: open one connection, reuse it for everything

Engaging asks for your Kerberos password **and** Duo on every new connection, SSH
key or not — their docs say so explicitly. What changes it is connection
multiplexing. The `engaging` host in `~/.ssh/config` carries:

```
ControlMaster auto
ControlPath ~/.ssh/sockets/%r@%h-%p
ControlPersist 8h
```

So: **authenticate once, then everything else is free.**

```bash
ssh engaging          # Kerberos + Duo, once. Leave this session open.
```

That opens a master socket. For the next 8 hours every `ssh`, `rsync`, and `scp`
rides on it with no prompt — from any terminal, and even after this one closes.
Without it a push-run-pull cycle costs three Duo prompts.

    ssh -O check engaging     # is a master live?
    ssh -O exit engaging      # tear it down (do this if it wedges after a
                              # laptop sleep or network change; the symptom is
                              # a hang, not an error)

## Setup, once

In the `ssh engaging` session:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"

cd ~/strategy-selection
uv sync            # Python 3.14 + deps from uv.lock, into .venv/
mkdir -p logs      # Slurm will not create this, and fails at launch without it
```

**Re-run `uv sync` after any push that changes `pyproject.toml`.** It last
changed to drop `statsmodels`, which only the retired `sampling_effects`
analysis used.

Do **not** `module load miniforge`. `pyproject.toml` requires Python >= 3.14 and
the cluster modules do not carry it; `uv` downloads its own interpreter, which
sidesteps the module system entirely.

Run `uv sync` on the **login node**. Compute nodes frequently have no outbound
network, so a first sync inside a job hangs instead of failing cleanly.

## Submit

```bash
sbatch slurm/run_classifier.sbatch
```

Anything after the script name is forwarded to the pipeline. For a first run,
split it so a typo does not burn a 12-hour slot:

```bash
JOB1=$(sbatch --parsable slurm/run_classifier.sbatch --only convert labels features)
sbatch --dependency=afterok:$JOB1 slurm/run_classifier.sbatch --from decode-pub
```

Or smoke-test the wiring in 15 minutes once the caches exist:

```bash
salloc -p mit_quicktest -c 8 -t 00:15:00
export PATH="$HOME/.local/bin:$PATH" MPLBACKEND=Agg OMP_NUM_THREADS=1
uv run python -m eye_pre_flash.classifier.pipeline --only decode-pub
```

`salloc` swaps your prompt from `[byuille@login009 ...]` to `[byuille@node1806
...]` — you are now typing on a compute node. `sbatch` is the opposite: it queues
a script, prints a job ID, and returns immediately. That return means the request
is filed, not that anything ran.

**Post-flash counterfactual analysis.** One light job; it exists because the
behavioral npz here predate `feedback_time` entering
`data.convert.BEHAVIORAL_FIELDS`, and `data/mat/` never leaves this tree. It
re-converts the behavioral npz (small — minutes), rebuilds the eye-behavioral
caches, and runs `eye_post_flash.counterfactual` under both alignments:

```bash
sbatch slurm/run_post_flash.sbatch
# results leave ~/strategy-selection/eye_post_flash/out/ (run on the laptop):
rsync -av engaging:strategy-selection/eye_post_flash/out/ ./eye_post_flash/out/
```

**Maze x decoded-strategy occupancy similarity (`eye_pre_flash/corr/`).** This
is the job that needs the cluster, not the laptop: its SVM label source
(`--source svm`) needs the same cold ~23-session neural conversion the
dendrogram label sweep already paid once, and only 4 sessions' neural npz
stay on disk locally. `--with-svm` builds both label kinds off one
conversion, so this is one job, not two:

```bash
sbatch slurm/run_corr.sbatch
# a single source, feature or scope re-runs without editing the script:
sbatch slurm/run_corr.sbatch --source svm --scope allplus --monkey Faure
# results leave ~/strategy-selection/eye_pre_flash/corr/out/ (run on the laptop):
rsync -av engaging:strategy-selection/eye_pre_flash/corr/out/ ./eye_pre_flash/corr/out/
```

The analysis half is quick once labels exist — the 12x12 per-split
correlation is vectorised (two matrix multiplies, not 144 individual
`pearson()` calls), measured at ~40s per (source, feature, variant, monkey,
k, space) leaf's `allplus` scope with `--n-perm 1000` — so the 12h budget is
headroom for the label sweep, not the analysis.

## Monitor

```bash
squeue --me                              # R running, PD pending
tail -f logs/eyeclf-<JOBID>.out
scancel <JOBID>
sacct -j <JOBID> -o JobID,State,Elapsed,ReqMem,MaxRSS,AllocCPUS --units=G
```

`MaxRSS` is only accurate once the job is no longer running. Use it to size
`--mem` on the next run.

## Why the job script sets what it sets

`slurm/run_classifier.sbatch` requests 4 cores, 32 GB, 6 h on `mit_normal`.

**Thread pinning is load-bearing, not boilerplate.** `RandomForestClassifier`
uses `n_jobs=-1`, so joblib forks one worker per core. If each worker also opens
a full BLAS thread pool they contend for the same cores and the job runs *slower
than a laptop*. The script gives parallelism to joblib
(`LOKY_MAX_CPU_COUNT=$SLURM_CPUS_PER_TASK`) and pins BLAS to one thread per
worker (`OMP_NUM_THREADS=1` and friends).

`MPLBACKEND=Agg` because the tables render through pyplot and a compute node has
no display.

`PATH` is set explicitly because a batch job runs a non-login shell, so
`~/.bashrc` is not sourced and `~/.local/bin/uv` is not found.

**Stay on `mit_normal`.** `mit_preemptable` offers 48 h but the pipeline has no
checkpointing finer than stage granularity, so a preemption partway through
`features` loses that entire stage.

## When it fails

| Symptom | Fix |
| ------- | --- |
| `uv: command not found` in the job log | `PATH` — the script sets it; check uv really installed to `~/.local/bin` |
| Killed, `State` = `OUT_OF_MEMORY` | raise `--mem` to 128G, rerun just that stage with `--from` |
| `TIMEOUT` | stages are resumable; resubmit with `--from <stage>`, the log names the stage |
| `Disk quota exceeded` | see the space section — drop `data/npz/Neural_Data` after `features` |
| hangs during `uv sync` inside a job | run `uv sync` on the login node instead |
| rsync or ssh to `engaging` hangs | stale control socket — `ssh -O exit engaging`, then reconnect |
| cloud figures vanished | an incoming push with `--delete` — see "What crosses the boundary" |

Every stage is a cache warm, so rerunning is cheap: each loader skips work whose
npz already exists. A failed stage prints its own `--from` resume command.
