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

Session `.mat` files go in `data/` (paths configured in `data/loader.py`).

## Run

To test the model run:

```bash
uv run decoder.py
```

or:

```bash
python decoder.py
```

To generate decision variable plots run: 

```bash
uv run dv_graph.py
```

```bash
python dv_graph.py
```

Figures are saved at `figures/dv_by_maze.png`.
