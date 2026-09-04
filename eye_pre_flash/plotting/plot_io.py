from pathlib import Path

OUT_ROOT = Path(__file__).resolve().parent / "out"

# Pre-fixation window: fix_start − START → fix_start − END (ms). END is 0, so
# every analysis runs the free-viewing window right up to fix_start. Modules
# must import this rather than define their own end: when END was 300 the
# heatmap and similarity-matrix plots inherited it while the occupancy,
# transition and decoding analyses shadowed it with 0, and the two families of
# figures silently described different windows.
PRE_FIX_START_MS = 1466
PRE_FIX_END_MS = 0
# Saccade traces run through fix_start (ms).
PRE_FIX_WINDOW_MS = 1533.0


def window_label(start_ms=PRE_FIX_START_MS, end_ms=PRE_FIX_END_MS):
    """The window as a figure caption, naming a zero end as plain `fix_start`.

    Titles that interpolate the end unconditionally would read "→ fix_start −
    0 ms", so every caption goes through here.
    """
    tail = "fix_start" if end_ms == 0 else f"fix_start − {end_ms:g} ms"
    return f"fix_start − {start_ms:g} ms → {tail}"


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
