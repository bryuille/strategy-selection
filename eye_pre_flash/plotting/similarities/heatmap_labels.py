"""Gaze-map similarity matrices split by decoded neural strategy label.

The 6×6 maze matrices cannot separate strategy from geometry. Strategy is close
to a deterministic function of (session, maze), so "mazes 4-6 resemble each
other" and "sequentially-solved trials resemble each other" are the same
statement about the same trials -- which is exactly the passive-visual-geometry
alternative that `eye_pre_flash/classifier/classifier.md`'s reviewer response
exists to rule out.

This script splits each maze's trials by their **decoded neural label** and
correlates the resulting gaze maps, giving a **12×12 matrix** over the
(maze, strategy) cells: mazes 1-6 labelled hierarchical, then mazes 1-6 labelled
sequential. Cell `(i, j)` is the split-half CV Pearson *r* between cell *i*'s
gaze map and cell *j*'s, exactly as in `similarities.heatmap` -- independent
trial halves throughout, so the diagonal is that cell's own split-half
reliability.

Three readings, in order of how much they can settle:

1. **Within a maze, H vs S** -- the boxed cells, `(maze m, H)` against
   `(maze m, S)`. Geometry is identical on both sides, so anything below that
   maze's own diagonal is strategy-dependent sampling with the visual
   confound removed by construction. This is the decisive comparison.
2. **Same strategy, different maze** -- inside a quadrant. Does a decoded
   strategy look like itself across geometries?
3. **The two off-diagonal quadrants** -- cross-strategy similarity overall.

**Counts are balanced everywhere.** Every half map in a session averages the
same number of trials, ``min over included cells of n // 2``, so no cell enters
a correlation with a cleaner estimate than any other. Without that, the
majority strategy in each maze would carry a higher reliability and correlate
higher with everything -- the same trial-count bias `similarities.heatmap_eq`
removes from the maze matrices, which here would run along the strategy axis
and manufacture the effect being looked for.

**Entries off the comparable core are marked.** Cells qualify in different
numbers of sessions, and sessions differ enormously in overall correlation
level, so averaging each pair over whichever sessions hold its two cells makes
the matrix uncomparable across cells -- a pair that happens to include one
unusually clean recording day is inflated against a pair that does not, and the
inflation reads as structure. `common_cell_set` finds the largest cell set a
single shared set of sessions all supply; `unreliable_mask` then marks every
entry that does not rest on those sessions with ``*``. Nothing is dropped --
hiding thin cells hides how thin the design is -- but a marked entry must not be
read against an unmarked one. The report also gives the core recomputed on its
one session set, where every value is comparable with every other.

A cell needs ``--min-trials`` trials in a session to enter at all, and one thin
cell lowers the half size for every other cell in that session, so raising
`--min-trials` trades cells for cleaner maps. Read the census the run prints
before reading the matrix: labels exist only for the vetted sessions, and most
mazes are heavily one-sided.

Every scope runs separately into its own subfolder.

Usage:
    uv run python -m eye_pre_flash.plotting.similarities.heatmap_labels --monkey Faure
    uv run python -m eye_pre_flash.plotting.similarities.heatmap_labels --monkey Nielsen
    uv run python -m eye_pre_flash.plotting.similarities.heatmap_labels --monkey Nielsen --scope all
    uv run python -m eye_pre_flash.plotting.similarities.heatmap_labels --monkey Faure --bin-w 0.142857 --min-trials 10

Output: `eye_pre_flash/plotting/out/similarities/heatmap_labels/<scope>/<monkey>/`.
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np

from data.config import monkey_for_session
from data.labeler import MAZE_GROUPS
from data.loader import load_eye_behavioral_data, load_eye_data
from eye_pre_flash.classifier.labels import scope_sessions, strategy_label_lookup
from eye_pre_flash.plotting.plot_io import save_figure
from eye_pre_flash.plotting.similarities.common import BLUE_YELLOW
from eye_pre_flash.plotting.similarities.heatmap_eq import equal_halves
from eye_pre_flash.plotting.similarities.heatmap_log import (
    COARSE_BIN,
    DEFAULT_BIN,
    MAZES,
    N_SPLITS,
    PRE_FIX_END_MS,
    PRE_FIX_START_MS,
    _collect_trials,
    _half_profile,
    _pearson,
    _trial_count_maps,
)

VISUALIZER = "similarities/heatmap_labels"
SCOPES = ("publication", "all", "allplus")
# Cells are (maze, strategy): mazes 1-6 hierarchical, then mazes 1-6 sequential.
STRATEGIES = (0, 1)
STRATEGY_TAG = {0: "H", 1: "S"}
N_CELLS = len(MAZES) * len(STRATEGIES)
# A cell needs this many trials in a session to enter, so each half map averages
# at least four. One thin cell drags the whole session's half size down.
MIN_TRIALS = 8


def cell_index(maze, strategy):
    """Row/column of the (maze, strategy) cell in the 12×12 matrix."""
    return strategy * len(MAZES) + (maze - 1)


def cell_labels():
    return [
        f"{maze}{STRATEGY_TAG[strategy]}"
        for strategy in STRATEGIES
        for maze in MAZES
    ]


def _session_cells(maze_trials, bin_w, min_trials):
    """``cell index -> per-trial count maps`` for one session."""
    cells = {}
    for maze in MAZES:
        trials = maze_trials[maze]
        if not trials:
            continue
        labels = np.array([t[2] for t in trials], dtype=int)
        for strategy in STRATEGIES:
            rows = [t for t, lab in zip(trials, labels) if lab == strategy]
            if len(rows) < min_trials:
                continue
            cells[cell_index(maze, strategy)] = _trial_count_maps(
                [(x, y) for x, y, _lab in rows], bin_w
            )
    return cells


def _session_cv_matrix(
    cells, rng, *, n_half, n_splits=N_SPLITS, use_log=False
):
    """12×12 split-half CV Pearson *r* for one session, all halves equal size."""
    acc = np.zeros((N_CELLS, N_CELLS))
    counts = np.zeros((N_CELLS, N_CELLS))

    for _ in range(n_splits):
        mean_a = [None] * N_CELLS
        mean_b = [None] * N_CELLS
        for idx, maps in cells.items():
            if len(maps) < 2 * n_half:
                continue
            idx_a, idx_b = equal_halves(len(maps), n_half, rng)
            mean_a[idx] = _half_profile(maps, idx_a, use_log=use_log)
            mean_b[idx] = _half_profile(maps, idx_b, use_log=use_log)

        present = [m for m in mean_a + mean_b if m is not None]
        if len(present) < 2:
            continue
        active = np.any(np.stack(present, axis=0) > 0, axis=0)

        for i in range(N_CELLS):
            for j in range(N_CELLS):
                if any(
                    m is None
                    for m in (mean_a[i], mean_b[j], mean_b[i], mean_a[j])
                ):
                    continue
                r = np.nanmean(
                    [
                        _pearson(mean_a[i], mean_b[j], active),
                        _pearson(mean_b[i], mean_a[j], active),
                    ]
                )
                if np.isfinite(r):
                    acc[i, j] += r
                    counts[i, j] += 1

    out = np.full((N_CELLS, N_CELLS), np.nan)
    ok = counts > 0
    out[ok] = acc[ok] / counts[ok]
    return out


def label_similarity_matrix(
    eye,
    behavioral,
    *,
    keep_sessions,
    label_lookup,
    bin_w=DEFAULT_BIN,
    n_splits=N_SPLITS,
    min_trials=MIN_TRIALS,
    n_half=None,
    seed=0,
    use_log=False,
):
    """Session-averaged 12×12 CV *r*, the per-session matrices behind it,
    which cells each session contributed, the trial census and the half sizes.
    """
    by_session = _collect_trials(eye, behavioral, label_lookup=label_lookup)
    keep = set(keep_sessions)
    rng = np.random.default_rng(seed)

    session_mats, half_sizes, presence = [], [], []
    census = np.zeros(N_CELLS, dtype=int)
    used = np.zeros(N_CELLS, dtype=int)

    for session, maze_trials in by_session.items():
        if session not in keep:
            continue
        for maze in MAZES:
            for _x, _y, lab in maze_trials[maze]:
                census[cell_index(maze, int(lab))] += 1

        cells = _session_cells(maze_trials, bin_w, min_trials)
        if len(cells) < 2:
            continue
        size = (
            n_half
            if n_half is not None
            else min(len(maps) // 2 for maps in cells.values())
        )
        if size < 1:
            continue
        cells = {i: m for i, m in cells.items() if len(m) >= 2 * size}
        if len(cells) < 2:
            continue
        mat = _session_cv_matrix(
            cells, rng, n_half=size, n_splits=n_splits, use_log=use_log
        )
        if np.isfinite(mat).any():
            session_mats.append(mat)
            half_sizes.append(size)
            here = np.zeros(N_CELLS, dtype=bool)
            for i in cells:
                used[i] += 1
                here[i] = True
            presence.append(here)

    if not session_mats:
        return (
            np.full((N_CELLS, N_CELLS), np.nan),
            [],
            np.zeros((0, N_CELLS), dtype=bool),
            census,
            used,
            0,
            0.0,
        )

    with warnings.catch_warnings():
        # Cells absent from every session are an all-NaN column, not an error.
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(np.stack(session_mats), axis=0)

    return (
        mean,
        session_mats,
        np.stack(presence),
        census,
        used,
        len(session_mats),
        float(np.mean(half_sizes)),
    )


def common_cell_set(presence, min_sessions=2):
    """Largest cell set that every session in some shared subset supplies.

    Averaging each pair over whichever sessions happen to hold both of its
    cells makes the matrix uncomparable across cells: sessions differ a lot in
    overall correlation level, so a pair that happens to include one unusually
    clean recording day is inflated relative to a pair that does not, and the
    difference reads as structure. Restricting the whole matrix to one session
    set removes that, at the cost of the cells which break it.

    Returns ``(cells, sessions)``, both index arrays, maximising the number of
    cells and then the number of sessions. Falls back to a single session only
    when no pair of sessions shares two cells.
    """
    n_sessions = presence.shape[0]
    if n_sessions == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    for floor in (min_sessions, 1):
        best_key, best = (0, 0), None
        for mask in range(1, 1 << n_sessions):
            sess = [k for k in range(n_sessions) if mask >> k & 1]
            if len(sess) < floor:
                continue
            cells = np.flatnonzero(presence[sess].all(axis=0))
            if cells.size < 2:
                continue
            key = (int(cells.size), len(sess))
            if key > best_key:
                best_key, best = key, (cells, np.array(sess, dtype=int))
        if best is not None:
            return best
    return np.array([], dtype=int), np.array([], dtype=int)


def pair_session_sets(presence):
    """``(i, j) -> frozenset of sessions holding both cells``."""
    n = presence.shape[1]
    return {
        (i, j): frozenset(np.flatnonzero(presence[:, i] & presence[:, j]).tolist())
        for i in range(n)
        for j in range(n)
    }


def unreliable_mask(presence, sessions):
    """True where a pair does not rest on the common session set.

    Those entries are still plotted -- dropping data hides how thin the design
    is -- but they are marked, because they average over different recording
    days than the unmarked core and sessions differ widely in overall
    correlation level. Comparing a marked entry against an unmarked one is
    comparing recording days as much as cells.
    """
    common = frozenset(np.asarray(sessions).tolist())
    supports = pair_session_sets(presence)
    mask = np.zeros((N_CELLS, N_CELLS), dtype=bool)
    for (i, j), sup in supports.items():
        mask[i, j] = sup != common
    return mask


def matched_matrix(session_mats, cells, sessions):
    """12×12 mean over `sessions`, with every cell outside `cells` left NaN."""
    out = np.full((N_CELLS, N_CELLS), np.nan)
    if len(cells) == 0 or len(sessions) == 0:
        return out
    stack = np.stack([session_mats[k] for k in sessions])
    sub = np.ix_(cells, cells)
    out[sub] = np.nanmean(stack[:, cells][:, :, cells], axis=0)
    return out


def within_maze_table(session_mats, presence):
    """``maze -> (r_HH, r_SS, r_HS, n sessions)`` on a matched session set.

    A cell can qualify in more sessions than its partner, and sessions differ in
    overall correlation level, so averaging `r(H,H)` over four sessions against
    `r(H,S)` over three would compare them on different recording days. Each
    maze's three numbers come from exactly the sessions where **both** of its
    cells qualified.
    """
    rows = {}
    for maze in MAZES:
        h, s = cell_index(maze, 0), cell_index(maze, 1)
        if presence.size == 0:
            rows[maze] = (np.nan, np.nan, np.nan, 0)
            continue
        both = np.flatnonzero(presence[:, h] & presence[:, s])
        if both.size == 0:
            rows[maze] = (np.nan, np.nan, np.nan, 0)
            continue
        mats = [session_mats[k] for k in both]
        rows[maze] = (
            float(np.mean([m[h, h] for m in mats])),
            float(np.mean([m[s, s] for m in mats])),
            float(np.mean([m[h, s] for m in mats])),
            int(both.size),
        )
    return rows


def _write_report(
    monkey, corr, matched, star, cells, sessions, session_mats, presence, census,
    used, *, scope, scope_line, bin_w, n_sessions, mean_half, min_trials,
):
    hier, seq = MAZE_GROUPS[monkey]
    names = cell_labels()
    end_label = (
        "fix_start" if PRE_FIX_END_MS == 0 else f"fix_start − {PRE_FIX_END_MS} ms"
    )
    lines = [
        f"# {monkey} — gaze similarity by decoded strategy (scope `{scope}`)",
        "",
        scope_line,
        "",
        f"{n_sessions} sessions labelled, {bin_w:g} unit-H bins, window "
        f"fix_start − {PRE_FIX_START_MS} ms → {end_label}, "
        f"{mean_half:.1f} trials per half map (balanced across cells), "
        f"cells need ≥ {min_trials} trials in a session.",
        "",
        f"**Comparable core: {len(sessions)} session(s), {len(cells)} cell(s)** "
        "— the largest cell set every one of those sessions supplies. Sessions "
        "differ widely in overall correlation level, so a pair that averages "
        "over different recording days than another is not comparable with it: "
        "one unusually clean session inflates every pair it happens to touch. "
        "Every cell is still reported; entries marked `*` do not rest on the "
        "core's shared sessions, and must not be read against unmarked ones.",
        "",
        "Core cells: " + (", ".join(names[i] for i in cells) or "none") + ". "
        "Marked `*`: "
        + (", ".join(n for i, n in enumerate(names) if i not in set(cells.tolist())) or "none")
        + " (plus any pair whose two cells never co-occur on exactly those sessions).",
        "",
        f"Maze groups: hierarchical {hier}, sequential {seq}. "
        "Cells are `<maze><H|S>` by decoded label, not by maze type.",
        "",
        "## Census",
        "",
        "| Cell | " + " | ".join(names) + " |",
        "|---|" + "---|" * N_CELLS,
        "| trials | " + " | ".join(str(int(c)) for c in census) + " |",
        "| sessions used | " + " | ".join(str(int(u)) for u in used) + " |",
        "",
        "## The decisive comparison: within a maze, H vs S",
        "",
        "Geometry is identical on both sides, so `r(H, S)` below the two "
        "same-strategy reliabilities is strategy-dependent sampling that visual "
        "identity cannot explain. All three numbers rest on half maps of the "
        "same size, and on the same sessions — only those where both of the "
        "maze's cells qualified.",
        "",
        "| Maze | Sessions | r(H,H) | r(S,S) | r(H,S) | mean same − cross |",
        "|---|---|---|---|---|---|",
    ]
    within = within_maze_table(session_mats, presence)
    for maze in MAZES:
        r_hh, r_ss, r_hs, n_both = within[maze]
        if not np.isfinite(r_hs):
            lines.append(f"| {maze} | 0 | — | — | — | — |")
            continue
        delta = 0.5 * (r_hh + r_ss) - r_hs
        lines.append(
            f"| {maze} | {n_both} | {r_hh:.3f} | {r_ss:.3f} | {r_hs:.3f} | "
            f"{delta:+.3f} |"
        )

    def matrix_block(mat, heading, note, mark=None):
        block = ["", heading, "", note, "",
                 "| | " + " | ".join(names) + " |",
                 "|---|" + "---|" * N_CELLS]
        for i, name in enumerate(names):
            row = " | ".join(
                "—"
                if not np.isfinite(mat[i, j])
                else f"{mat[i, j]:.3f}" + ("\\*" if mark is not None and mark[i, j] else "")
                for j in range(N_CELLS)
            )
            block.append(f"| **{name}** | {row} |")
        return block

    lines += matrix_block(
        corr,
        "## Matrix as plotted (every cell)",
        "`*` marks an entry that does not rest on the comparable core's shared "
        "sessions. Those entries are real measurements, but they average over "
        "different recording days than the unmarked ones, so a marked value and "
        "an unmarked value cannot be compared with each other.",
        mark=star,
    )
    lines += matrix_block(
        matched,
        "## Comparable core (one shared session set)",
        "The same numbers recomputed on a single session set. Everything here "
        "is comparable with everything else here; cells outside the core are "
        "blank.",
    )
    lines.append("")

    # Printed rather than written: the out/ trees carry figures and raw csv
    # only, so this report reaches the run log instead of a .md beside the png.
    report = "\n".join(lines)
    print(report)
    return report


def plot_labels(
    monkey="Faure",
    *,
    scope="all",
    bin_w=DEFAULT_BIN,
    n_splits=N_SPLITS,
    min_trials=MIN_TRIALS,
    n_half=None,
    seed=0,
    use_log=False,
    visualizer=VISUALIZER,
):
    by_monkey, _missing, dropped = scope_sessions(scope)
    keep = by_monkey.get(monkey, ())
    mine = {s: a for s, a in dropped.items() if monkey_for_session(s) == monkey}
    scope_line = f"scope {scope}: {len(keep)} session(s) — {', '.join(keep) or 'none'}"
    if mine:
        scope_line += "; vetted out: " + ", ".join(
            f"{s} ({a:.2f})" for s, a in sorted(mine.items())
        )
    print(f"{monkey}: {scope_line}")
    if not keep:
        print(f"{monkey}: no labelled sessions in scope {scope!r}; skipping")
        return None

    eye = load_eye_data(monkey)
    behavioral = load_eye_behavioral_data(monkey)
    (
        corr,
        session_mats,
        presence,
        census,
        used,
        n_sessions,
        mean_half,
    ) = label_similarity_matrix(
        eye,
        behavioral,
        keep_sessions=keep,
        label_lookup=strategy_label_lookup(keep),
        bin_w=bin_w,
        n_splits=n_splits,
        min_trials=min_trials,
        n_half=n_half,
        seed=seed,
        use_log=use_log,
    )

    cells, sessions = common_cell_set(presence)
    matched = matched_matrix(session_mats, cells, sessions)
    star = unreliable_mask(presence, sessions)
    core = set(np.asarray(cells).tolist())
    names = cell_labels()
    print(
        f"{monkey}: comparable core — {len(sessions)} session(s), "
        f"{len(cells)} cell(s): {', '.join(names[i] for i in cells) or 'none'}"
    )

    finite = corr[np.isfinite(corr)]
    if finite.size:
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmin == vmax:
            vmax = vmin + 1e-12
    else:
        vmin, vmax = 0.0, 1.0
    fig, ax = plt.subplots(figsize=(8.6, 7.4), layout="constrained")
    im = ax.imshow(
        corr, origin="upper", cmap=BLUE_YELLOW, vmin=vmin, vmax=vmax,
        aspect="equal"
    )
    tick_names = [n if i in core else f"{n}*" for i, n in enumerate(names)]
    ax.set_xticks(range(N_CELLS), labels=tick_names, fontsize=8)
    ax.set_yticks(range(N_CELLS), labels=tick_names, fontsize=8)
    ax.set_xlabel("Maze × decoded strategy")
    ax.set_ylabel("Maze × decoded strategy")

    split = len(MAZES) - 0.5
    ax.axhline(split, color="black", lw=1.6)
    ax.axvline(split, color="black", lw=1.6)
    for maze in MAZES:
        h, s = cell_index(maze, 0), cell_index(maze, 1)
        for i, j in ((h, s), (s, h)):
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    lw=1.8,
                )
            )

    rate_label = "log1p(count/trial)" if use_log else "count/trial"
    end_label = (
        "fix_start" if PRE_FIX_END_MS == 0 else f"fix_start − {PRE_FIX_END_MS} ms"
    )
    ax.set_title(
        f"{monkey} pre-fixation gaze similarity by decoded strategy "
        f"(unit H, CV, balanced n) | scope {scope}\n"
        f"split-half Pearson r of {rate_label} | {n_sessions} sessions | "
        f"{mean_half:.1f} trials/half | H = hierarchical, S = sequential\n"
        f"fix_start − {PRE_FIX_START_MS} ms → {end_label} | {bin_w:g} unit H bins | "
        f"{n_splits} splits\n"
        f"boxed = same maze, different strategy | "
        f"* = off the comparable core ({len(cells)} cells on "
        f"{len(sessions)} shared sessions), not comparable with unmarked entries",
        fontsize=9,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("correlation coefficient")

    for i in range(N_CELLS):
        for j in range(N_CELLS):
            val = corr[i, j]
            if not np.isfinite(val):
                ax.text(j, i, "—", ha="center", va="center", color="0.4", fontsize=6)
                continue
            r, g, b, _ = im.cmap(im.norm(val))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(
                j,
                i,
                f"{val:.2f}".lstrip("0") + ("*" if star[i, j] else ""),
                ha="center",
                va="center",
                color="black" if lum > 0.55 else "white",
                fontsize=6,
            )

    stem = f"labels_bin{bin_w:g}"
    rel_dir = f"{visualizer}/{scope}/{monkey}"
    save_figure(fig, stem, rel_dir=rel_dir)
    plt.close(fig)
    _write_report(
        monkey,
        corr,
        matched,
        star,
        cells,
        sessions,
        session_mats,
        presence,
        census,
        used,
        scope=scope,
        scope_line=scope_line,
        bin_w=bin_w,
        n_sessions=n_sessions,
        mean_half=mean_half,
        min_trials=min_trials,
    )
    return matched


def plot_all(monkey="Faure", **kwargs):
    plot_labels(monkey=monkey, bin_w=DEFAULT_BIN, **kwargs)
    plot_labels(monkey=monkey, bin_w=COARSE_BIN, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monkey", default="Faure", choices=("Faure", "Nielsen"))
    parser.add_argument(
        "--scope",
        default=None,
        choices=SCOPES,
        help="Label scope (default: run all of %s separately)" % ", ".join(SCOPES),
    )
    parser.add_argument(
        "--bin-w",
        type=float,
        default=None,
        help=f"Bin width in unit H (default: both {DEFAULT_BIN:g} and {COARSE_BIN:g})",
    )
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument(
        "--min-trials",
        type=int,
        default=MIN_TRIALS,
        help="Trials a (maze, strategy) cell needs in a session to enter",
    )
    parser.add_argument(
        "--n-half",
        type=int,
        default=None,
        help="Fix the half size across sessions too (default: per-session maximum)",
    )
    parser.add_argument("--log", action="store_true", help="log1p(count/trial) maps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    kwargs = dict(
        n_splits=args.n_splits,
        min_trials=args.min_trials,
        n_half=args.n_half,
        use_log=args.log,
        seed=args.seed,
    )
    for scope in ((args.scope,) if args.scope else SCOPES):
        if args.bin_w is not None:
            plot_labels(monkey=args.monkey, scope=scope, bin_w=args.bin_w, **kwargs)
        else:
            plot_all(monkey=args.monkey, scope=scope, **kwargs)


if __name__ == "__main__":
    main()
