# Running the classifier pipeline on Engaging

Login node `orcd-login.mit.edu`, aliased to `engaging` in `~/.ssh/config`.
Scheduler Slurm. One command end to end:
`uv run python -m eye_pre_flash.classifier.pipeline`.

This document is written from the cluster's side: what the cloud tree holds,
what arrives in it, and what it sends back.

## The cloud tree

`~/strategy-selection` on the cluster is the working copy. Every node mounts the
same home directory, so a compute node writes straight into it — nothing is
"sent back" by the job itself. Raw and derived data live on a separate, much
larger scratch volume (`/home/byuille/orcd/scratch/strategy_selection_data/`),
not under `~/strategy-selection/`; every `slurm/*.sbatch` job (and any
interactive session) sets `STRATEGY_DATA_ROOT` to point `data.config`'s
`MAT_ROOT`/`NPZ_ROOT`/`PROCESSED_ROOT` there instead of the repo-relative
`./data/` the laptop still uses.

```
~/strategy-selection/
  *.py, slurm/, ...          code — replaced wholesale on every incoming push
  eye_pre_flash/plotting/*/out/   figures, one out/ per plotting package
  eye_pre_flash/{counts,scatter,maze_strategy_pairs}/   presentation figures
  logs/                      Slurm stdout/stderr. Must exist before sbatch
  .venv/                     built here by uv sync, never transferred

$STRATEGY_DATA_ROOT/            = /home/byuille/orcd/scratch/strategy_selection_data/
  mat/                      THE ONLY COPY. Never arrives, never leaves, never deleted
  npz/                      built here from mat/, never transferred
  processed/                built here, never transferred
```

The asymmetry is the whole design: **the cluster owns the data, the laptop owns
the code, and only figures travel back.** `mat/` lives here and nowhere else;
the caches derived from it are large enough that rebuilding them here is
cheaper than moving them.

This split (code on `/home`, data on scratch) replaced an earlier layout with
everything under `~/strategy-selection/data/`, moved because `/home`'s 200 GB
quota made the neural conversion cache a constant space fight (see the old
`sweep_strategy_labels` `reclaim`-by-default behaviour below). **Caution:**
scratch filesystems at most HPC centers are subject to a purge policy that
`/home` is not — check ORCD's current policy for this volume before treating
it as a long-term archive. `mat/` is the one irreplaceable copy in this whole
pipeline; if scratch purges on an inactivity timer, that copy needs a home
that doesn't.

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

## Storage: scratch, not `/home`

Two volumes, two purposes. `/home` holds the code checkout, `.venv`, and Slurm
logs — small, and quota-capped:

```
nfs001.lb:/home    121G used    195G soft    200G hard      (74G headroom)
```

`$STRATEGY_DATA_ROOT` (`/home/byuille/orcd/scratch/strategy_selection_data/`)
holds `mat/`, `npz/` and `processed/` — on ORCD scratch, which has effectively
no meaningful quota pressure for this project:

```
fstor018.ib:/     294T total     36T used     258T avail
```

This is a recent move (§"The cloud tree"): everything used to live under
`~/strategy-selection/data/`, sharing the 200G `/home` hard limit with the code
and `.venv`, which is what made the neural-conversion dance below necessary in
the first place. It mostly isn't, anymore — kept here because the mechanism
(and its `--reclaim-neural` opt-out) still exists and still matters on a
space-constrained checkout, e.g. the laptop.

**npz runs 3-6× the mat it came from.** `np.savez` writes uncompressed; `.mat`
v7.3 is compressed HDF5, so the ratio tracks how well the spike data compresses,
which in turn tracks neuron count. Measured on the cluster: `Nov_3_g0` 1.2 GB →
3.9 GB (3.2×), `Oct_22_g0` 1.9 → 6.9 (3.6×), `june_8_g0` 1.3 → 5.0 (3.8×), but
**`june_24_g0` 2.2 → 14 GB (6.4×)** — 144 neurons, the most of the 23. Budget the
peak from 14 GB, not from an average.

What is actually needed from a neural recording is its strategy label, which is
**~5 KB**. `data.labeler.CLUSTERING_SESSIONS` lists **23** label-eligible
sessions whose neural mats total ~42 GB; converted and kept, that is ~170 GB of
npz — comfortably inside scratch's headroom now, where it used to be past the
old `/home` hard limit on its own. So the `labels` stage's default changed with
the move: it converts each session and **leaves the neural npz and
`trial_timebins` cache in place**, rather than deleting them as it goes, so a
later rerun or a different analysis over the same sessions does not pay the
conversion cost again.

```bash
python -m data.labeler --type strategy --sweep                  # convert -> label, keep the npz (default)
python -m data.labeler --type strategy --sweep --reclaim-neural # convert -> label -> free (the old default)
```

The sweep skips sessions whose label already exists, so an interrupted run
resumes by re-issuing the same command, and one session that fails to convert
is reported at the end rather than stopping the other 22.

Check before committing to a long job:

```bash
du -sh "$STRATEGY_DATA_ROOT"/mat/*                          # how much bigger than local?
du -sh "$STRATEGY_DATA_ROOT"/{npz,processed} 2>/dev/null
du -sh ~/* ~/.cache 2>/dev/null | sort -rh | head            # /home usage, if that ever gets tight instead
```

**If `/home` gets tight** (the code/venv volume, not the data one): `.venv` and
the uv cache are the only large, safely-droppable things there —
`rm -rf ~/strategy-selection/.venv && uv sync` rebuilds it. If scratch itself
ever gets tight, `--reclaim-neural` is still there.

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
echo 'export STRATEGY_DATA_ROOT="/home/byuille/orcd/scratch/strategy_selection_data"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
export STRATEGY_DATA_ROOT="/home/byuille/orcd/scratch/strategy_selection_data"

cd ~/strategy-selection
uv sync            # Python 3.14 + deps from uv.lock, into .venv/
mkdir -p logs      # Slurm will not create this, and fails at launch without it
```

Every `slurm/*.sbatch` job sets `STRATEGY_DATA_ROOT` itself (a batch job runs a
non-login shell, so `~/.bashrc` is not sourced — same reason `PATH` is set
explicitly there); the `.bashrc` line above is only for interactive sessions
(`salloc`, or `ssh engaging` itself) that run `data.config`-dependent code by
hand.

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
export STRATEGY_DATA_ROOT="/home/byuille/orcd/scratch/strategy_selection_data"
uv run python -m eye_pre_flash.classifier.pipeline --only decode-pub
```

`salloc` swaps your prompt from `[byuille@login009 ...]` to `[byuille@node1806
...]` — you are now typing on a compute node. `sbatch` is the opposite: it queues
a script, prints a job ID, and returns immediately. That return means the request
is filed, not that anything ran.

**Rebuilding every cache from cold (`slurm/run_rebuild.sbatch`).** This is the
job to run after `processed/` has been emptied, and the only one that needs to
be: nothing in this tree moves caches aside or forces a refresh, because an
empty `processed/` *is* the invalidation. Every loader builds on miss, in
dependency order, so one submission walks the whole chain:

```bash
sbatch slurm/run_rebuild.sbatch
sbatch slurm/run_rebuild.sbatch june_24_g0   # figures for one session only
```

What it writes into `$STRATEGY_DATA_ROOT/processed/`, in order:

| step | cache | note |
| ---- | ----- | ---- |
| 0 | `{session}_strategy_choices.npz`, `{session}_strategy_svm.npz` | the per-session strategy labels, 23 sessions, both kinds off one neural conversion. **The long pole** |
| 1 | `{monkey}_eye_behavioral.npz`, `{monkey}_eye_data.npz` | concatenated from `npz/`; ~3 GB and ~5 GB |
| 2 | `{monkey}_clean_eye_data_s1466_e0.npz` | I-DT fixations + saccades on the clipped, unwarped window |
| 3 | `{monkey}_attractor_eye_data.npz` | unit-H warp, validity, timebase — K-independent, one per monkey |
| 4 | `{monkey}_clf2_k{6,12}_unith_s1466_e0.npz` | the k-means codebook and the three feature blocks |

`npz/` is the floor this stands on and is *not* rebuilt here: recreating it
means re-running `data.convert` against `mat/`, which is the one thing in this
tree that cannot be recovered from anywhere else.

**Step 0 is the one that bites.** The strategy labels live in `processed/` like
everything else, so emptying it deletes them too — and every label-dependent
analysis (`maze_strategy_pairs`, `counts`, `scatter`, `classifier.decoding`)
then skips with "no labelled trial" rather than failing loudly. A label is ~5 KB
but comes from a neural recording of up to 14 GB, which is why this step
dominates the runtime and why the job asks for 12 hours. The attractor figures
need none of it and complete regardless, which is exactly what makes the
omission easy to miss.

Then the figures, into the repo rather than into `processed/`:

```bash
# attractor figures leave ~/strategy-selection/eye_pre_flash/plotting/saccades/out/
rsync -av engaging:strategy-selection/eye_pre_flash/plotting/saccades/out/ \
    ./eye_pre_flash/plotting/saccades/out/
```

**The presentation figures (`counts`, `scatter`, `maze_strategy_pairs`).** Each
is read-only on the caches above and fails fast if they are missing, rather
than triggering a cold build that would race a concurrent job. Run them after
`run_rebuild.sbatch`, or with `--dependency=afterok:<jobid>`:

```bash
sbatch slurm/run_counts.sbatch
sbatch slurm/run_scatter.sbatch
sbatch slurm/run_maze_strategy_pairs.sbatch
# each writes into its own package dir; results leave (run on the laptop):
rsync -av --exclude='__pycache__' \
    engaging:strategy-selection/eye_pre_flash/counts/ ./eye_pre_flash/counts/
```

**Caches do not encode the settings they were built under.** A stem carries the
window and K — never `data.builder`'s detector constants
(`IDT_DISPERSION_THRESHOLD_DEG`, `FIXATION_MIN_DURATION_MS`) or its QC
predicate (`qc_mask`). Change one of those and a rerun silently reloads the old
events rather than recomputing them, because every loader checks only that the
expected keys are present. So **after changing a detector constant, delete the
caches it feeds** — there is no refresh flag, and deleting is the invalidation:

```bash
cd "$STRATEGY_DATA_ROOT/processed"
rm -f *_clean_eye_data_s*.npz *_attractor_eye_data.npz *_clf2_*.npz
```

`{monkey}_eye_data.npz` and `{monkey}_eye_behavioral.npz` survive that — raw
gaze concatenation and behavioral fields, neither derived from the detectors nor
the QC screen. They are also the expensive ones (3.1 GB and 4.9 GB), so keeping
them cuts the rebuild from roughly an hour to minutes. Emptying `processed/`
entirely is always safe; it just pays for those two again.

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
| Killed, `State` = `OUT_OF_MEMORY` | raise `--mem` to 128G and resubmit; the caches already written are kept |
| `TIMEOUT` in `run_rebuild` | resubmit — every completed cache is reused, so the rerun starts where it stopped |
| `Disk quota exceeded` on `/home` | see "Storage: scratch, not `/home`" — `.venv` is the droppable thing there, `rm -rf .venv && uv sync` |
| scratch filling up | `data.labeler --sweep --reclaim-neural` frees the neural npz + `trial_timebins` per session |
| hangs during `uv sync` inside a job | run `uv sync` on the login node instead |
| rsync or ssh to `engaging` hangs | stale control socket — `ssh -O exit engaging`, then reconnect |
| cloud figures vanished | an incoming push with `--delete` — see "What crosses the boundary" |
| figures unchanged after editing a detector constant | the caches predate the edit; delete them — see "Caches do not encode the settings they were built under" |

Every stage is a cache warm, so rerunning is cheap: each loader skips work whose
npz already exists. That is also the trap — see the cache-invalidation note
above before concluding a code change had no effect.
