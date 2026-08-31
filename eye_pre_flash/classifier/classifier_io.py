"""Output paths and table rendering for the eye-data strategy classifier.

Mirrors ``eye_pre_flash.plotting.plot_io``: everything lands under
``eye_pre_flash/classifier/out/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

OUT_ROOT = Path(__file__).resolve().parent / "out"


def out_dir(rel_dir=None):
    path = OUT_ROOT / rel_dir if rel_dir else OUT_ROOT
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, stem, *, rel_dir=None, dpi=300):
    path = out_dir(rel_dir) / f"{stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    print(f"Saved {path}")
    return path


def save_text(text, stem, suffix, *, rel_dir=None):
    path = out_dir(rel_dir) / f"{stem}{suffix}"
    path.write_text(text)
    print(f"Saved {path}")
    return path


########## Black-and-white publication table


def _fmt(value, decimals=3):
    if value is None:
        return "--"
    if isinstance(value, str):
        return value
    try:
        if value != value:  # NaN
            return "--"
    except TypeError:
        return str(value)
    return f"{value:.{decimals}f}"


def render_table_png(
    columns,
    rows,
    *,
    stem,
    rel_dir=None,
    title=None,
    bold_mask=None,
    col_align=None,
    group_breaks=(),
    row_height=0.30,
    col_pad=0.55,
    fontsize=9.5,
    footnote=None,
):
    """Booktabs-style black-on-white table PNG.

    ``columns``  header strings.
    ``rows``     list of lists of already-formatted cell strings. A cell may
                 also be a pair of ``(text, bold)`` segments, rendered centred
                 as ``left / right`` with each segment's own weight -- used
                 when two values share a cell but are ranked separately.
    ``bold_mask`` optional same-shaped list of bools; True cells are bold.
    ``group_breaks`` row indices to draw a thin rule *above*.
    ``footnote`` optional small note below the bottom rule.
    """

    def cell_text(cell):
        if isinstance(cell, str):
            return cell
        return " / ".join(seg for seg, _bold in cell)

    n_rows = len(rows)
    n_cols = len(columns)
    col_align = col_align or ["left"] + ["center"] * (n_cols - 1)

    widths = []
    for j in range(n_cols):
        cells = [columns[j]] + [cell_text(r[j]) for r in rows]
        widths.append(max(len(c) for c in cells) + col_pad * 2)
    total_w = sum(widths)

    fig_w = total_w * 0.098 + 0.4
    fig_h = (n_rows + 3.6) * row_height + (0.42 if title else 0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, n_rows + 3.6)

    edges = [0.0]
    for w in widths:
        edges.append(edges[-1] + w)

    def x_of(j, align):
        if align == "left":
            return edges[j] + col_pad
        if align == "right":
            return edges[j + 1] - col_pad
        return 0.5 * (edges[j] + edges[j + 1])

    ha = {"left": "left", "right": "right", "center": "center"}
    top = n_rows + 2.2
    header_y = top - 0.75

    ax.plot([0, total_w], [top, top], color="black", lw=1.5, clip_on=False)
    for j, name in enumerate(columns):
        ax.text(
            x_of(j, col_align[j]),
            header_y,
            name,
            ha=ha[col_align[j]],
            va="center",
            fontsize=fontsize,
            color="black",
        )
    mid = top - 1.35
    ax.plot([0, total_w], [mid, mid], color="black", lw=0.9, clip_on=False)

    for i, row in enumerate(rows):
        y = mid - 0.68 - i
        if i in group_breaks and i > 0:
            ax.plot(
                [0, total_w],
                [y + 0.5, y + 0.5],
                color="black",
                lw=0.5,
                clip_on=False,
            )
        for j, cell in enumerate(row):
            if not isinstance(cell, str):
                # Segment pair: left value right-anchored, slash centred,
                # right value left-anchored, each with its own weight.
                (l_text, l_bold), (r_text, r_bold) = cell
                xc = x_of(j, "center")
                common = dict(va="center", fontsize=fontsize, color="black")
                ax.text(xc, y, "/", ha="center", **common)
                ax.text(
                    xc - 0.55, y, l_text, ha="right",
                    fontweight="bold" if l_bold else "normal", **common,
                )
                ax.text(
                    xc + 0.55, y, r_text, ha="left",
                    fontweight="bold" if r_bold else "normal", **common,
                )
                continue
            bold = bool(bold_mask[i][j]) if bold_mask is not None else False
            ax.text(
                x_of(j, col_align[j]),
                y,
                str(cell),
                ha=ha[col_align[j]],
                va="center",
                fontsize=fontsize,
                color="black",
                fontweight="bold" if bold else "normal",
            )
    bottom = mid - 0.68 - (n_rows - 1) - 0.5
    ax.plot([0, total_w], [bottom, bottom], color="black", lw=1.5, clip_on=False)
    if footnote:
        ax.text(
            0,
            bottom - 0.55,
            footnote,
            ha="left",
            va="top",
            fontsize=fontsize - 1.5,
            color="black",
            clip_on=False,
        )

    if title:
        ax.text(
            0,
            top + 0.55,
            title,
            ha="left",
            va="bottom",
            fontsize=fontsize + 1.0,
            color="black",
            fontweight="bold",
        )
    fig.patch.set_facecolor("white")
    return save_figure(fig, stem, rel_dir=rel_dir), fig


def render_table_latex(
    columns, rows, *, caption=None, label=None, col_align=None, footnote=None
):
    n_cols = len(columns)
    spec = col_align or "l" + "r" * (n_cols - 1)
    lines = ["\\begin{table}[t]", "\\centering"]
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines += [
        "\\begin{tabular}{" + spec + "}",
        "\\toprule",
        " & ".join(columns) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(str(c) for c in row) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    if footnote:
        lines += ["\\par\\smallskip", f"{{\\footnotesize {footnote}}}"]
    lines += ["\\end{table}"]
    return "\n".join(lines) + "\n"


def render_table_markdown(columns, rows):
    widths = [
        max(len(str(columns[j])), *(len(str(r[j])) for r in rows)) if rows else len(str(columns[j]))
        for j in range(len(columns))
    ]
    def line(cells):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"
    out = [line(columns), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out += [line(r) for r in rows]
    return "\n".join(out) + "\n"
