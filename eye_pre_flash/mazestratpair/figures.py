"""The 1x3 panel figure: one 2x2 per feature variant, one maze per figure.

Keeps `eye_pre_flash.corr.figures`' house style -- parula, luminance-switched
annotations, `supxlabel` rather than free-floating figure text, dpi=300,
always emitted however thin the coverage -- and imports corr's label
vocabulary (`FEATURE_LABEL`, `FEATURE_UNIT`, `SOURCE_LABEL`) so the two
packages' titles cannot drift apart.

What is different, and why:

1. **Three panels in one figure, no variant directory.** The variant is a
   panel and a CSV column here, never a path component, because the three
   variants of one maze are the comparison and splitting them across
   directories makes that comparison a file-management exercise.
2. **Per-panel colour scale and per-panel colorbar.** `mean_removed`
   subtracts a common profile, which drives the off-diagonal r negative *by
   arithmetic* (`corr.variants`), while `full` sits at 0.80-0.99. A shared
   scale would compress `full` into a sliver and destroy the one comparison
   that is interpretable, which is within-panel and relative. `mean_removed`
   additionally gets a diverging map centred on zero, since its sign is
   meaningful and its magnitude is not.
3. **`z_perm` leads each panel title, not delta.** Delta is not comparable
   across the three panels -- `full`'s range is compressed and
   `mean_removed`'s is centred -- so a bigger delta on `mean_removed` is a
   scale artefact. The permutation z is on a common scale by construction.
4. **Trial counts on the plot**, as the request asks: pooled per strategy in
   the diagonal cells, per-session half size in the panel title. Those are
   two different n's and the footer says so -- the pooled count is not the n
   behind any single Pearson.
5. **`d` is per panel**, because `no_origin` drops dimensions. One figure with
   three dimensionalities is a real trap, so it is stated in the panel that
   owns it rather than once in the suptitle.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from eye_pre_flash.corr.figures import FEATURE_LABEL, FEATURE_UNIT, SOURCE_LABEL
from eye_pre_flash.mazestratpair import pair as pairmod
from eye_pre_flash.mazestratpair.pair_io import save_figure
from eye_pre_flash.plotting.plot_io import window_label
from eye_pre_flash.plotting.similarities.common import BLUE_YELLOW

def _panel_scale(values):
    """``(vmin, vmax, cmap, norm)`` for one panel, from its own values only.

    Always the panel's own data range, never a fixed [-1, 1] or a range shared
    with the other panels: `full` lives at 0.95-0.99 and `mean_removed`
    somewhere else entirely, so a shared or padded scale renders every panel
    as one flat colour and throws away the only comparison that is
    interpretable, which is within-panel and relative.

    Which map depends on whether the panel straddles zero, and it is decided
    from the values rather than from the variant name. `mean_removed`
    subtracts a common profile, which usually drives the off-diagonal
    negative while the diagonal stays high -- there, sign is information and
    the panel gets a diverging map with white pinned exactly at zero by
    `TwoSlopeNorm`, so the asymmetric range is still used in full. When every
    value shares a sign there is no zero to mark, and parula (the repo's
    similarity map, `plotting.similarities.common.BLUE_YELLOW`) gives more
    contrast across a narrow band than a diverging map anchored off-range.
    """
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 0.0, 1.0, BLUE_YELLOW, None
    vmin, vmax = float(finite.min()), float(finite.max())
    if vmin == vmax:
        vmax = vmin + 1e-12
    if vmin < 0.0 < vmax:
        return (
            vmin, vmax, plt.get_cmap("RdBu_r"),
            mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax),
        )
    return vmin, vmax, BLUE_YELLOW, None


def _fmt(value, digits=3):
    return "—" if not np.isfinite(value) else f"{value:.{digits}f}"


def _half_text(result):
    """Half size per cell, as a median and -- when sessions differ -- a range.

    A bare median reads as though every session split at that size, which is
    exactly the thing this package changed: `corr`'s shared half is one number
    per session set by the thinnest cell anywhere, while this one varies with
    each session's own two cells. The range is the honest summary; the
    per-session values are in `results_raw`.
    """
    parts = []
    for cell in range(pairmod.N_CELLS):
        halves = [h[cell] for h in result.half_sizes if cell in h]
        if not halves:
            parts.append("—")
            continue
        lo, hi = min(halves), max(halves)
        med = int(np.median(halves))
        parts.append(f"{med}" if lo == hi else f"{med} ({lo}-{hi})")
    return parts[0] if parts[0] == parts[1] else " / ".join(parts)


def plot_pair_panels(
    results,
    nulls,
    dims,
    *,
    monkey,
    maze,
    scope,
    source,
    feature,
    k,
    space,
    n_splits,
    min_stable,
    n_perm,
    half_rule,
    variants,
    stem,
    rel_dir,
):
    """`results` / `nulls` / `dims` are ``{variant -> ...}``, keyed by `variants`."""
    n_panels = len(variants)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.5 * n_panels, 5.6), layout="constrained"
    )
    axes = np.atleast_1d(axes)
    names = pairmod.cell_labels(maze)

    census = np.zeros(pairmod.N_CELLS, dtype=int)
    n_sessions = 0
    n_in_scope = 0
    any_degenerate = False
    skipped = ()
    for variant in variants:
        result = results.get(variant)
        if result is None:
            continue
        census = np.maximum(census, result.census)
        n_sessions = max(n_sessions, result.n_sessions)
        n_in_scope = max(n_in_scope, result.n_sessions_in_scope)
        any_degenerate = any_degenerate or bool(result.n_degenerate)
        skipped = skipped or result.skipped_cells

    for ax, variant in zip(axes, variants):
        result = results.get(variant)
        null = nulls.get(variant) or {}
        dim = dims.get(variant, 0)

        if result is None or not result.n_sessions:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            ax.text(
                0.5, 0.5,
                f"{variant}\n\nno usable session\n(both strategies must clear\n"
                f"{result.min_trials if result else '?'} trials in a session)",
                ha="center", va="center", fontsize=9, color="0.35",
                transform=ax.transAxes,
            )
            continue

        corr = result.mean
        thin = pairmod.thin_mask(result)
        star = thin[:, None] | thin[None, :]
        vmin, vmax, cmap, norm = _panel_scale(corr)

        if norm is None:
            im = ax.imshow(
                corr, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal"
            )
        else:
            im = ax.imshow(corr, origin="upper", cmap=cmap, norm=norm, aspect="equal")
        tick_names = [
            f"{name}*" if thin[i] else name for i, name in enumerate(names)
        ]
        ax.set_xticks(range(pairmod.N_CELLS), labels=tick_names, fontsize=11)
        ax.set_yticks(range(pairmod.N_CELLS), labels=tick_names, fontsize=11)
        # One xlabel, on the middle panel: three copies of the same words
        # under three small matrices is noise, and none at all leaves the tick
        # labels to explain themselves.
        if variant == variants[len(variants) // 2]:
            ax.set_xlabel("decoded strategy (H = hierarchical, S = sequential)",
                          fontsize=9)

        for i in range(pairmod.N_CELLS):
            for j in range(pairmod.N_CELLS):
                val = corr[i, j]
                if i in result.skipped_cells or j in result.skipped_cells:
                    ax.text(j, i, "skipped\nby design", ha="center", va="center",
                            color="0.4", fontsize=8)
                    continue
                if not np.isfinite(val):
                    ax.text(j, i, "—", ha="center", va="center", color="0.4",
                            fontsize=11)
                    continue
                r, g, b, _ = im.cmap(im.norm(val))
                colour = "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "white"
                ax.text(
                    j, i, f"{val:.3f}" + ("*" if star[i, j] else ""),
                    ha="center", va="center", color=colour, fontsize=13,
                )
                if i == j:
                    # Trials pooled across sessions for this strategy. NOT the n
                    # behind the correlation above it -- that is 2 x the half
                    # size, in one session. The footer states the distinction.
                    ax.text(
                        j, i + 0.26, f"n = {int(result.census[i])}",
                        ha="center", va="center", color=colour, fontsize=9,
                    )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ticks = [vmin, 0.0, vmax] if norm is not None else [vmin, 0.5 * (vmin + vmax), vmax]
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels([f"{t:.2f}" for t in ticks], fontsize=7)
        # Labelled on every panel: an unlabelled per-panel bar invites the
        # reader to treat the three scales as one.
        cbar.set_label("correlation coefficient", fontsize=7)

        p = null.get("p_two_sided", np.nan)
        p_txt = _fmt(p, 4)
        if np.isfinite(p) and null.get("n_perm") and abs(p - 1.0 / (null["n_perm"] + 1)) < 1e-12:
            p_txt += " (floor)"
        half_txt = _half_text(result)
        ax.set_title(
            f"{variant}\n"
            f"z = {_fmt(null.get('z_perm', np.nan), 2)}  |  "
            f"Δ = {_fmt(null.get('delta_obs', np.nan))}  |  p = {p_txt}\n"
            f"null {_fmt(null.get('null_mean', np.nan))} ± "
            f"{_fmt(null.get('null_sd', np.nan))}  |  d = {dim}  |  "
            f"half = {half_txt}  |  {result.n_sessions} of {result.n_sessions_in_scope} sessions",
            fontsize=9,
        )

    feature_label = FEATURE_LABEL.get(feature, feature)
    unit_label = FEATURE_UNIT.get(feature, feature)
    source_label = SOURCE_LABEL.get(source, source)
    fig.suptitle(
        f"{monkey} maze {maze} — pre-fixation {feature_label} similarity, "
        f"H vs S by {source_label} strategy (K={k}, {space}, CV, balanced n)\n"
        f"split-half Pearson r of cell half-means ({unit_label}) | scope {scope} | "
        f"{maze}H n={int(census[0])} trials, {maze}S n={int(census[1])} trials | "
        f"{n_sessions} of {n_in_scope} sessions | half rule: {half_rule}\n"
        f"{window_label()} | {n_splits} splits | Δ = 0.5·(r_HH + r_SS) − r_HS | "
        f"p: labels shuffled within (session, maze), {n_perm} resamples",
        fontsize=10,
    )

    footer = [
        f"* = fewer than {min_stable} trials in a cell in most sessions, high "
        f"variance, not comparable with unmarked entries",
        "each panel has its own colour scale; absolute r is NOT comparable "
        "across panels (mean_removed is mean-centred, so its off-diagonal is "
        "negative by arithmetic and only Δ reads)",
        "n in a diagonal cell is trials pooled across sessions; the r above it "
        "is over 2 x the half size, within one session (see panel titles)",
    ]
    if any_degenerate:
        footer.append("some split-halves were degenerate (zero variance); see the report")
    if skipped:
        blank = ", ".join(names[s] for s in skipped)
        footer.append(
            f"{blank} skipped as the SVM's anchor-minority cell, not for lack of coverage"
        )
    # `supxlabel`, not `fig.text`: constrained layout reserves vertical space
    # for a supxlabel but not for free-floating text, which lands on top of
    # the tick labels.
    fig.supxlabel(
        "\n".join(footer), x=0.01, ha="left", fontsize=6, color="0.3"
    )

    save_figure(fig, stem, rel_dir=rel_dir)
    plt.close(fig)
    return fig
