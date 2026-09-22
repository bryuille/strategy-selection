"""Output paths and table rendering for the eye-data strategy classifier.

Mirrors ``eye_pre_flash.plotting.plot_io``: everything lands under
``eye_pre_flash/classifier/out/`` by default.

`render_table_png` is the repo's only publication-table renderer, so packages
other than this one legitimately want it. They pass `out_root` to keep their
own `out/` tree, per the one-`out/`-per-package rule in
`plotting/plot_io.save_figure`; omitting it keeps this package's tree, which is
what every caller inside `classifier/` does.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt

# Vertical advance per wrapped caption line, in table-row units.
TITLE_LINE = 0.62
FOOT_LINE = 0.52

OUT_ROOT = Path(__file__).resolve().parent / "out"


def out_dir(rel_dir=None, *, out_root=None):
    root = out_root or OUT_ROOT
    path = root / rel_dir if rel_dir else root
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, stem, *, rel_dir=None, dpi=300, out_root=None):
    path = out_dir(rel_dir, out_root=out_root) / f"{stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    print(f"Saved {path}")
    return path


def save_text(text, stem, suffix, *, rel_dir=None, out_root=None):
    path = out_dir(rel_dir, out_root=out_root) / f"{stem}{suffix}"
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
    out_root=None,
    dpi=300,
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

    # Title and footnote are wrapped to the table's own width. `total_w` is in
    # units of one body character, and these two render at different sizes, so
    # each gets its own budget scaled by its font size -- without this a long
    # caption is drawn as one line and `bbox_inches="tight"` widens the saved
    # png to fit it, leaving the text hanging far past the table's right edge.
    def wrapped(text, size):
        """Wrap to the table width, honouring the caller's own line breaks.

        `textwrap.wrap` treats a newline as ordinary whitespace, so a caption
        written with a deliberate break has to be split on it first or the
        break is silently collapsed into a space.
        """
        if not text:
            return []
        budget = max(24, int(total_w * fontsize / size))
        out = []
        for para in text.split("\n"):
            out.extend(textwrap.wrap(para, width=budget) or [""])
        return out

    title_lines = wrapped(title, fontsize + 1.0)
    foot_lines = wrapped(footnote, fontsize - 1.5)

    # Room for however many lines the wrap produced, above and below the rules.
    head_room = 0.4 + TITLE_LINE * len(title_lines)
    foot_room = 0.55 + FOOT_LINE * len(foot_lines)
    y_hi = n_rows + 2.2 + head_room
    y_lo = -foot_room
    fig_h = (y_hi - y_lo) * row_height
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()
    ax.set_xlim(0, total_w)
    ax.set_ylim(y_lo, y_hi)

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
    for i, line in enumerate(foot_lines):
        ax.text(
            0,
            bottom - 0.55 - FOOT_LINE * i,
            line,
            ha="left",
            va="top",
            fontsize=fontsize - 1.5,
            color="black",
            clip_on=False,
        )

    # Drawn bottom-up so the first line ends up highest and the block sits
    # directly above the top rule however many lines it runs to.
    for i, line in enumerate(reversed(title_lines)):
        ax.text(
            0,
            top + 0.45 + TITLE_LINE * i,
            line,
            ha="left",
            va="bottom",
            fontsize=fontsize + 1.0,
            color="black",
            fontweight="bold",
        )
    fig.patch.set_facecolor("white")
    return save_figure(fig, stem, rel_dir=rel_dir, dpi=dpi, out_root=out_root), fig


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
