# PFC strategy selection

Python implementation of left/right choice decision-variable (DV) traces from neural firing rates.

## Setup

```bash
uv sync
```

or:

```bash
python -m venv .venv && pip install -e .
```

Session `.mat` files go in `data/raw/` (paths configured in `data/builder.py`). Processed `.npz` caches are written to `data/processed/` on first load.

## Run

To test the lr dv model run:

```bash
uv run -m decoders.lr
```

or:

```bash
python -m decoder.lr
```

To generate decision variable plots run:

```bash
uv run -m plotting.lr_trial_traces
uv run -m plotting.lr_pre_flash_traces
```

### Plotting

Trial traces use neural activity from flash 1 through ~200 ms after flash 3. Pre-flash traces use activity from fixation onset (`fix_start`) through flash 1 (700 ms on the y-axis).
