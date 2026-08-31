from pathlib import Path

OUT_ROOT = Path(__file__).resolve().parent / "out"

# Similarity / heatmap window: fix_start − START → fix_start − END (ms).
PRE_FIX_START_MS = 1466
PRE_FIX_END_MS = 300
# Saccade traces still run through fix_start (ms).
PRE_FIX_WINDOW_MS = 1533.0


def save_figure(
    fig, stem, *, rel_dir=None, dpi=200, bbox_inches=None, pad_inches=0.02
):
    """Save `fig` under eye_pre_flash/plotting/out/[rel_dir/]<stem>.png."""
    out_dir = OUT_ROOT / rel_dir if rel_dir else OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    print(f"Saved {path}")
    return path
