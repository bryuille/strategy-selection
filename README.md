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
uv run decoders/lr.py
```

or:

```bash
python decoder/lr.py
```

To generate decision variable plots run: 

```bash
uv run dv_graph.py
```

or:

```bash
python dv_graph.py
```

Figures are saved at `figures/lr_traces.png` and `figures/strategy_traces.png`.