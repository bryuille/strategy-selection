# PFC strategy selection

Python implementation of left/right choice decision-variable (DV) traces from neural firing rates.

## Setup

```bash
uv sync
# or: python -m venv .venv && pip install -e .
```

Session `.mat` files go in `data/` (paths configured in `data/loader.py`).

## Run

Decoder cross-validation. Picks regularization λ and reports held-out accuracy:

```bash
python decoder.py
```

DV plots. Fits the decoder and saves at `figures/dv_by_maze.png`:

```bash
python dv_graph.py
```
