# PFC strategy selection

Python implementation of left/right choice and strategy decision-variable (DV) traces from neural firing rates.

## Setup

```bash
uv sync
```

Session `.mat` files go in `data/raw/` (paths configured in `data/builder.py`). Processed `.npz` caches are written to `data/processed/` on first load.

## Run

To test the lr dv model:

```bash
uv run -m decoders.lr
```

To generate all decision variable plots:

```bash
uv run -m plotting.lr_trial_traces
uv run -m plotting.lr_pre_flash_traces
uv run -m plotting.strategy_pre_flash_traces
```

### Figures

Figures are saved to `figures/`. The naming scheme is `<decoder>_<window>_traces.png`:

- `lr_trial_traces.png` — left/right DV during the trial (flash_one to ~200 ms after flash_three)
- `lr_pre_flash_traces.png` — left/right DV prior to flash one (geo_present to flash_one)
- `strategy_pre_flash_traces.png` — hierarchical/sequential strategy DV prior to flash one (geo_present to flash_one)
