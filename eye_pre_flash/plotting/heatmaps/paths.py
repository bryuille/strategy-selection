from pathlib import Path

# This package's own figure tree. Each plotting package owns one, so figures
# land beside the code that makes them.
OUT_ROOT = Path(__file__).resolve().parent / "out"
