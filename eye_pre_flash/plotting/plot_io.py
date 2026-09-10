from pathlib import Path

import numpy as np
from matplotlib.transforms import blended_transform_factory

OUT_ROOT = Path(__file__).resolve().parent / "out"

# Pre-fixation window: fix_start − START → fix_start − END (ms). END is 0, so
# every analysis runs the free-viewing window right up to fix_start. Modules
# must import this rather than define their own end: when END was 300 the
# heatmap and similarity-matrix plots inherited it while the occupancy,
# transition and decoding analyses shadowed it with 0, and the two families of
# figures silently described different windows.
PRE_FIX_START_MS = 1466
PRE_FIX_END_MS = 0
# Saccade traces run through fix_start (ms). Identical to `PRE_FIX_START_MS`
# on purpose: this was 1533.0, which is the *median* `geo_present -> fix_start`
# gap, not the minimum. The measured minimum is 1466.2 ms (Nielsen), so a
# 1533 ms trace window reached back past `geo_present` -- plotting gaze from
# before the maze was on screen -- in 25.1% of Nielsen trials and 0.2% of
# Faure's. At 1466 no trial in either animal is affected (0 of 40,314).
PRE_FIX_WINDOW_MS = float(PRE_FIX_START_MS)

# `fixation_cue_present` marker. The cue sits a median 34 ms before fix_start
# and inside the window on 99.3% (Faure) / 99.9% (Nielsen) of trials; the rest
# are long outliers (out to -9.8 s) that would blow the axis open, so they are
# omitted rather than plotted.
CUE_FIELD = "fixation_cue_present"
CUE_COLOR = "#c44e52"


def cue_rel_ms(behavioral, beh_i, *, max_lag_ms=PRE_FIX_START_MS):
    """`fixation_cue_present` relative to fix_start (ms, negative = before).

    `None` when the field is absent from the cache, is not finite, or falls
    outside `[-max_lag_ms, 0]` -- callers then draw no marker.
    """
    if CUE_FIELD not in behavioral:
        return None
    cue = float(np.asarray(behavioral[CUE_FIELD])[beh_i])
    fix = float(np.asarray(behavioral["fix_start"])[beh_i])
    if not (np.isfinite(cue) and np.isfinite(fix)):
        return None
    rel = (cue - fix) * 1000.0
    if rel > 0.0 or rel < -float(max_lag_ms):
        return None
    return rel


def draw_cue_marker(ax, cue_rel, *, median=False, label=CUE_FIELD):
    """Vertical marker for the fixation cue at `cue_rel` ms. No-op if None."""
    if cue_rel is None:
        return False
    ax.axvline(
        cue_rel, color=CUE_COLOR, linewidth=1.2, linestyle="-.", zorder=4, alpha=0.9
    )
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        cue_rel,
        0.02,
        f"{label} (median)" if median else label,
        transform=trans,
        rotation=90,
        ha="right",
        va="bottom",
        fontsize=6,
        color=CUE_COLOR,
        zorder=5,
    )
    return True


def cue_rel_median(cue_rels):
    """Median of the non-None cue offsets, or None if there are none."""
    vals = [c for c in cue_rels if c is not None]
    return float(np.median(vals)) if vals else None


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
