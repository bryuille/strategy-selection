"""The printed Markdown report for one (source, feature, variant, scope) leaf.

Ported from the deleted `heatmap_labels._write_report`. Kept to the same
discipline: **printed to the run log, not written beside the png** -- the
`out/` trees carry figures and CSVs only.
"""

from __future__ import annotations

import numpy as np

from eye_pre_flash.corr import cells as cellmod
from eye_pre_flash.corr.figures import cell_thin_mask
from eye_pre_flash.plotting.plot_io import window_label


def write_report(
    result,
    null_rows,
    *,
    monkey,
    source,
    feature,
    variant,
    scope,
    k,
    space,
    dim,
    n_splits,
    min_trials,
    min_stable,
):
    names = cellmod.cell_labels()
    n = cellmod.N_CELLS
    thin = cell_thin_mask(result)

    lines = [
        f"# {monkey} -- {feature} ({variant}) similarity by {source} strategy "
        f"(scope `{scope}`)",
        "",
        f"K={k}, space={space}, d={dim}, {result.n_sessions} sessions, "
        f"{window_label()}, {n_splits} splits, min-trials={min_trials}, "
        f"min-stable={min_stable}.",
        "",
        "## Census (trials pooled across kept sessions; `sessions` = how many "
        "sessions the cell appeared in at all)",
        "",
        "| Cell | " + " | ".join(names) + " |",
        "|---|" + "---|" * n,
        "| trials | " + " | ".join(str(int(c)) for c in result.census) + " |",
        "| sessions | " + " | ".join(str(int(u)) for u in result.used) + " |",
        "| thin (*) | " + " | ".join("*" if t else "" for t in thin) + " |",
        "",
        "## The decisive comparison: within a maze, H vs S",
        "",
        "Geometry is identical on both sides, so `r(H, S)` below the two "
        "same-strategy reliabilities is strategy-dependent sampling that "
        "visual identity cannot explain. `delta = 0.5*(r_HH + r_SS) - r_HS`; "
        "the permutation p-value shuffles strategy labels within each "
        "(session, maze), preserving every cell's trial count.",
        "",
        "| Maze | Sessions | r(H,H) | r(S,S) | r(H,S) | delta | null mean | "
        "null SD | p (two-sided) | null 2.5-97.5% |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    table = cellmod.within_maze_table(result.session_mats, result.presence)
    null_by_maze = {row["maze"]: row for row in null_rows} if null_rows else {}
    for maze in cellmod.MAZES:
        r_hh, r_ss, r_hs, n_both = table[maze]
        if not np.isfinite(r_hs):
            lines.append(f"| {maze} | 0 | — | — | — | — | — | — | — | — |")
            continue
        delta = 0.5 * (r_hh + r_ss) - r_hs
        nrow = null_by_maze.get(maze)
        if nrow is None or not np.isfinite(nrow.get("null_mean", np.nan)):
            null_bits = ("—", "—", "—", "—")
        else:
            null_bits = (
                f"{nrow['null_mean']:.3f}",
                f"{nrow['null_sd']:.3f}" if np.isfinite(nrow["null_sd"]) else "—",
                f"{nrow['p_two_sided']:.4f}",
                f"[{nrow['null_p025']:.3f}, {nrow['null_p975']:.3f}]",
            )
        lines.append(
            f"| {maze} | {n_both} | {r_hh:.3f} | {r_ss:.3f} | {r_hs:.3f} | "
            f"{delta:+.3f} | " + " | ".join(null_bits) + " |"
        )

    lines += ["", "## Matrix as plotted", ""]
    lines.append("| | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * n)
    for i, name in enumerate(names):
        row = " | ".join(
            "—"
            if not np.isfinite(result.mean[i, j])
            else f"{result.mean[i, j]:.3f}" + ("\\*" if thin[i] or thin[j] else "")
            for j in range(n)
        )
        lines.append(f"| **{name}** | {row} |")
    lines.append("")
    if result.n_degenerate:
        lines.append(
            f"{result.n_degenerate} split-half computations were degenerate "
            "(both cells had data but at least one half-mean had zero variance "
            "-- an em-dash on the figure from this cause means \"no variance\", "
            "not \"no coverage\")."
        )
        lines.append("")

    report = "\n".join(lines)
    print(report)
    return report
