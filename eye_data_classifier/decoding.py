"""Strategy decoding tables: balanced CV accuracy for simple feature sets.

One question per table: **can pre-fixation gaze predict the trial's strategy
label?** Each monkey is analysed separately (codebook state ids are
monkey-local), in two session scopes, under two training regimes:

``per-maze``     one classifier per maze, trained and cross-validated on that
                 maze's trials pooled across sessions. Features are in raw
                 **degrees** -- geometry is constant inside a maze, so no
                 unit-H warp is applied. Maze identity cannot help by
                 construction; scores are pooled over mazes.
``across-maze``  one classifier on all mazes' trials pooled across sessions.
                 Features are in **unit-H** maze coordinates, which warps every
                 maze onto the same unit H to mitigate the visual-geometry
                 differences a classifier could otherwise exploit. A
                 ``Maze identity (reference)`` row -- one-hot maze id, no eye
                 data -- shows how much of any accuracy plain maze identity
                 explains anyway.

The metric is **balanced accuracy** under stratified 5-fold CV: the mean of
per-class recall, so chance is 0.5 regardless of how lopsided the strategy
split is. Each model reports the training-fold score next to the held-out
score; a large gap is overfitting.

Trials are pooled across sessions in both regimes, so a fold can hold trials
from a day the model trained on. The tables therefore read as "is the label
decodable from gaze on these recording days", not as generalisation to new
days.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from eye_data_classifier.classifier_io import (
    out_dir,
    render_table_latex,
    render_table_markdown,
    render_table_png,
    save_text,
)
from eye_data_classifier.features import GRID_HI, GRID_LO, N_MAZES, load_features
from eye_data_classifier.labels import (
    DEFAULT_SCOPE,
    MIN_LABEL_AGREEMENT,
    SCOPES,
    labels_for_rows,
    scope_sessions,
    strategy_label_lookup,
)
from eye_data_classifier.models import make_models

# Both strategies must appear this often in a training pool before it is CV'd.
MIN_PER_CLASS = 5
N_SPLITS = 5
SEED = 0

# The table rows: block, codebook size, display label. Heatmaps are
# codebook-free, so their two rows vary the grid instead of K.
FEATURE_ROWS = (
    ("Codebook occupancy (ms), K=6", "occ_ms", 6),
    ("Codebook occupancy (ms), K=12", "occ_ms", 12),
    ("Codebook occupancy (binary), K=6", "occ_bin", 6),
    ("Codebook occupancy (binary), K=12", "occ_bin", 12),
    ("State bigrams, K=6", "bigram", 6),
    ("State bigrams, K=12", "bigram", 12),
    (f"Gaze heatmap, {GRID_LO}x{GRID_LO}", "heatmap_lo", 6),
    (f"Gaze heatmap, {GRID_HI}x{GRID_HI}", "heatmap_hi", 6),
)

REGIMES = {"per-maze": "deg", "across-maze": "unith"}


def load_labelled(monkey, sessions, *, k, space):
    """Feature blocks and strategy labels, pooled over `sessions`."""
    data = load_features(monkey, k=k, space=space)
    rows = np.asarray(data["session"]).astype(str)
    y = labels_for_rows(
        rows, data["trial_indices_all"], strategy_label_lookup(sessions)
    )
    keep = np.isin(rows, list(sessions)) & np.isfinite(y)
    out = {}
    for key, value in data.items():
        arr = np.asarray(value)
        out[key] = arr[keep] if arr.ndim >= 1 and arr.shape[0] == keep.size else value
    return out, y[keep].astype(int)


def cv_pool(X, y, factory, *, n_splits=N_SPLITS, seed=SEED):
    """Pooled (train_true, train_pred, cv_true, cv_pred) over stratified folds.

    Returns ``None`` when either class has fewer than `MIN_PER_CLASS` trials --
    below that a stratified split cannot put both classes on both sides.
    """
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2)
    if counts.min() < MIN_PER_CLASS:
        return None
    skf = StratifiedKFold(
        n_splits=min(n_splits, int(counts.min())), shuffle=True, random_state=seed
    )
    pools = ([], [], [], [])
    for tr, te in skf.split(X, y):
        model = factory().fit(X[tr], y[tr])
        for pool, arr in zip(pools, (y[tr], model.predict(X[tr]), y[te], model.predict(X[te]))):
            pool.append(arr)
    return tuple(np.concatenate(p) for p in pools)


def score_groups(groups, factory, *, n_splits=N_SPLITS, seed=SEED):
    """Balanced train/CV accuracy aggregated over training groups.

    `groups` is a list of ``(id, X, y)`` -- one entry per maze in the per-maze
    regime, a single entry in the across-maze regime. Balanced accuracy is
    computed **within each group** and averaged across groups weighted by trial
    count. Pooling raw predictions across mazes instead would smuggle maze
    identity back in: every per-maze model absorbs its own maze's class prior,
    the priors differ by maze, and the pooled score inflates far above any
    single maze's -- exactly the geometry confound this regime exists to
    remove. ``per_group`` carries each group's own scores for the by-maze
    table; with a single group the aggregate equals it.
    """
    per_group = {}
    for gid, X, y in groups:
        p = cv_pool(X, y, factory, n_splits=n_splits, seed=seed)
        if p is None:
            continue
        per_group[gid] = dict(
            train=balanced_accuracy_score(p[0], p[1]),
            cv=balanced_accuracy_score(p[2], p[3]),
            n=int(p[2].size),
        )
    if not per_group:
        return None
    weights = [g["n"] for g in per_group.values()]
    return dict(
        train=float(
            np.average([g["train"] for g in per_group.values()], weights=weights)
        ),
        cv=float(np.average([g["cv"] for g in per_group.values()], weights=weights)),
        n_groups=len(per_group),
        n_trials=int(sum(weights)),
        per_group=per_group,
    )


def maze_groups(X, y, mazes):
    """Per-maze ``(maze, X, y)`` training pools, mazes with too few of a class
    dropped inside `cv_pool`."""
    return [
        (maze, X[mazes == maze], y[mazes == maze])
        for maze in range(1, N_MAZES + 1)
        if (mazes == maze).any()
    ]


def counts_rows(y, mazes):
    """Per-maze census: trials of each strategy and whether per-maze CV runs."""
    rows = []
    for maze in range(1, N_MAZES + 1):
        idx = mazes == maze
        n0 = int((y[idx] == 0).sum())
        n1 = int((y[idx] == 1).sum())
        rows.append(
            [
                str(maze),
                str(n0 + n1),
                str(n0),
                str(n1),
                "yes" if min(n0, n1) >= MIN_PER_CLASS else "no",
            ]
        )
    rows.append(
        [
            "total",
            str(int(y.size)),
            str(int((y == 0).sum())),
            str(int((y == 1).sum())),
            "",
        ]
    )
    return rows


def fmt(value):
    return "--" if value is None else f"{value:.3f}"


def col_max_mask(rows, cols, skip_rows=()):
    """Bold mask marking each listed column's best value; ties all bold."""
    mask = [[False] * len(row) for row in rows]
    for j in cols:
        vals = []
        for i, row in enumerate(rows):
            if i in skip_rows:
                continue
            try:
                vals.append((float(row[j]), i))
            except ValueError:
                continue
        for v, i in vals:
            if vals and v == max(v_ for v_, _ in vals):
                mask[i][j] = True
    return mask


def bymaze_max_mask(rows):
    """Bold mask marking, per maze column, the cell with the best CV value."""
    mask = [[False] * len(row) for row in rows]
    for j in range(1, len(rows[0])):
        vals = []
        for i, row in enumerate(rows):
            nums = [float(t) for t in row[j].split(" / ") if t != "--"]
            if nums:
                vals.append((max(nums), i))
        for v, i in vals:
            if vals and v == max(v_ for v_, _ in vals):
                mask[i][j] = True
    return mask


def embolden(rows, mask, wrap):
    """Rows with masked cells wrapped -- for the markdown and LaTeX renders."""
    return [
        [wrap(cell) if flag else cell for cell, flag in zip(row, mrow)]
        for row, mrow in zip(rows, mask)
    ]


def md_bold(cell):
    return f"**{cell}**"


def tex_bold(cell):
    return f"\\textbf{{{cell}}}"


def run_monkey(monkey, sessions, *, scope, n_splits, seed):
    """Both regime tables for one monkey; returns raw long-format rows."""
    models = make_models()
    rel_dir = f"decoding/{scope}"
    raw = []

    # Census from the k=6 unit-H cache; row sets are identical across caches.
    data, y = load_labelled(monkey, sessions, k=6, space="unith")
    mazes = np.asarray(data["maze_id"], dtype=int)
    census = counts_rows(y, mazes)
    census_cols = ["Maze", "Trials", "Hierarchical", "Sequential", "Per-maze CV"]
    save_text(
        f"{monkey}, scope `{scope}`: {len(sessions)} session(s) pooled: "
        f"{', '.join(sessions)}\n\n"
        + render_table_markdown(census_cols, census),
        f"counts_{monkey}",
        ".md",
        rel_dir=rel_dir,
    )

    for regime, space in REGIMES.items():
        columns = ["Feature set", "d", "Trials"]
        if regime == "per-maze":
            columns.append("Mazes")
        for model_name in models:
            columns += [f"{model_name} train", f"{model_name} CV"]

        table = []
        by_maze = {name: [] for name in models}
        for label, block, k in FEATURE_ROWS:
            data, y = load_labelled(monkey, sessions, k=k, space=space)
            X = np.asarray(data[block], dtype=np.float64)
            mazes = np.asarray(data["maze_id"], dtype=int)
            groups = (
                maze_groups(X, y, mazes) if regime == "per-maze" else [("all", X, y)]
            )
            row = [label, str(X.shape[1]), ""]
            if regime == "per-maze":
                row.append("")
            per_model_groups = {}
            for model_name, factory in models.items():
                scores = score_groups(groups, factory, n_splits=n_splits, seed=seed)
                per_model_groups[model_name] = scores["per_group"] if scores else {}
                if scores:
                    row[2] = str(scores["n_trials"])
                    if regime == "per-maze":
                        row[3] = str(scores["n_groups"])
                row += [
                    fmt(scores["train"] if scores else None),
                    fmt(scores["cv"] if scores else None),
                ]
                raw.append(
                    dict(
                        scope=scope,
                        monkey=monkey,
                        regime=regime,
                        feature_set=label,
                        model=model_name,
                        maze="pooled",
                        d=X.shape[1],
                        n_trials=scores["n_trials"] if scores else 0,
                        n_groups=scores["n_groups"] if scores else 0,
                        train_bacc=scores["train"] if scores else np.nan,
                        cv_bacc=scores["cv"] if scores else np.nan,
                    )
                )
                for gid, g in (scores["per_group"] if scores else {}).items():
                    if gid == "all":
                        continue
                    raw.append(
                        dict(
                            scope=scope,
                            monkey=monkey,
                            regime=regime,
                            feature_set=label,
                            model=model_name,
                            maze=str(gid),
                            d=X.shape[1],
                            n_trials=g["n"],
                            n_groups=1,
                            train_bacc=g["train"],
                            cv_bacc=g["cv"],
                        )
                    )
            if regime == "per-maze":
                for model_name in models:
                    groups_scored = per_model_groups[model_name]
                    by_maze[model_name].append(
                        [label]
                        + [
                            fmt(g["cv"]) if (g := groups_scored.get(maze)) else "--"
                            for maze in range(1, N_MAZES + 1)
                        ]
                    )
            table.append(row)
            print(f"  {monkey} {regime:11s} {label:32s} " + "  ".join(row[-4:]))

        if regime == "across-maze":
            # No eye data at all: how far does knowing the maze get you? Any
            # eye-feature row must clear this to claim gaze adds information.
            one_hot = np.zeros((mazes.size, N_MAZES))
            one_hot[np.arange(mazes.size), mazes - 1] = 1.0
            row = ["Maze identity (reference)", str(N_MAZES), str(int(y.size))]
            for model_name, factory in models.items():
                scores = score_groups(
                    [("all", one_hot, y)], factory, n_splits=n_splits, seed=seed
                )
                row += [
                    fmt(scores["train"] if scores else None),
                    fmt(scores["cv"] if scores else None),
                ]
                raw.append(
                    dict(
                        scope=scope,
                        monkey=monkey,
                        regime=regime,
                        feature_set="Maze identity (reference)",
                        model=model_name,
                        maze="pooled",
                        d=N_MAZES,
                        n_trials=scores["n_trials"] if scores else 0,
                        n_groups=scores["n_groups"] if scores else 0,
                        train_bacc=scores["train"] if scores else np.nan,
                        cv_bacc=scores["cv"] if scores else np.nan,
                    )
                )
            table.append(row)
            print(f"  {monkey} {regime:11s} {'Maze identity (reference)':32s} "
                  + "  ".join(row[-4:]))

        if regime == "per-maze":
            maze_cols = ["Feature set"] + [f"Maze {m}" for m in range(1, N_MAZES + 1)]
            maze_title = (
                f"{monkey} -- per-maze CV balanced accuracy by maze "
                f"(scope: {scope}). Cells: {' / '.join(models)}"
            )
            maze_note = (
                "Bold: each model's best CV accuracy in each maze. '--': fewer "
                f"than {MIN_PER_CLASS} trials of one strategy in that maze."
            )
            maze_stem = f"permaze_by_maze_{monkey}"
            model_names = list(models)
            # Each model is ranked separately: its mask marks its own best
            # value per maze, and the two verdicts render side by side in one
            # cell with independent bolding.
            masks = {m: bymaze_max_mask(by_maze[m]) for m in model_names}
            png_rows, md_rows, tex_rows = [], [], []
            for i in range(len(by_maze[model_names[0]])):
                label_cell = by_maze[model_names[0]][i][0]
                png_row, md_row, tex_row = [label_cell], [label_cell], [label_cell]
                for j in range(1, N_MAZES + 1):
                    segs = [(by_maze[m][i][j], masks[m][i][j]) for m in model_names]
                    png_row.append(tuple(segs))
                    md_row.append(" / ".join(md_bold(t) if b else t for t, b in segs))
                    tex_row.append(
                        " / ".join(tex_bold(t) if b else t for t, b in segs)
                    )
                png_rows.append(png_row)
                md_rows.append(md_row)
                tex_rows.append(tex_row)
            maze_breaks = {i for i in range(2, len(png_rows), 2)}
            render_table_png(
                maze_cols, png_rows, stem=maze_stem, rel_dir=rel_dir,
                title=maze_title, group_breaks=maze_breaks, footnote=maze_note,
            )
            save_text(
                maze_title
                + "\n\n"
                + render_table_markdown(maze_cols, md_rows)
                + f"\n{maze_note}\n",
                maze_stem,
                ".md",
                rel_dir=rel_dir,
            )
            save_text(
                render_table_latex(
                    maze_cols,
                    tex_rows,
                    caption=maze_title,
                    label=f"tab:{maze_stem}",
                    footnote=maze_note,
                ),
                maze_stem,
                ".tex",
                rel_dir=rel_dir,
            )

        stem = f"{'permaze' if regime == 'per-maze' else 'crossmaze'}_{monkey}"
        space_note = "degrees" if space == "deg" else "unit-H"
        title = (
            f"{monkey} -- {regime} decoding ({space_note} features, "
            f"scope: {scope}). Balanced accuracy, chance = 0.500"
        )
        base = 4 if regime == "per-maze" else 3
        skip = {len(table) - 1} if regime == "across-maze" else set()
        mask = col_max_mask(table, (base + 1, base + 3), skip_rows=skip)
        note = "Bold: best CV accuracy per model" + (
            ", maze-identity reference excluded." if skip else "."
        )
        breaks = {i for i in range(2, len(table), 2)}
        render_table_png(
            columns, table, stem=stem, rel_dir=rel_dir, title=title,
            group_breaks=breaks, bold_mask=mask, footnote=note,
        )
        save_text(
            title
            + "\n\n"
            + render_table_markdown(columns, embolden(table, mask, md_bold))
            + f"\n{note}\n",
            stem,
            ".md",
            rel_dir=rel_dir,
        )
        save_text(
            render_table_latex(
                columns,
                embolden(table, mask, tex_bold),
                caption=title,
                label=f"tab:{stem}",
                footnote=note,
            ),
            stem,
            ".tex",
            rel_dir=rel_dir,
        )
    return raw


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scope", choices=sorted(SCOPES), default=DEFAULT_SCOPE)
    parser.add_argument("--splits", type=int, default=N_SPLITS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    by_monkey, missing, dropped = scope_sessions(args.scope)
    if missing:
        print(
            f"scope {args.scope!r}: {len(missing)} session(s) not labelled yet, "
            f"skipped: {', '.join(missing)}"
        )
    if dropped:
        print(
            f"scope {args.scope!r}: {len(dropped)} session(s) dropped for "
            f"anchor-maze label agreement below {MIN_LABEL_AGREEMENT}: "
            + ", ".join(f"{s} ({a:.2f})" for s, a in sorted(dropped.items()))
        )
    if not by_monkey:
        raise SystemExit("no labelled sessions in scope; run the labels stage first")

    raw = []
    for monkey, sessions in sorted(by_monkey.items()):
        print(f"{monkey}: {len(sessions)} labelled session(s) in scope {args.scope!r}")
        raw += run_monkey(
            monkey, sessions, scope=args.scope, n_splits=args.splits, seed=args.seed
        )

    fields = list(raw[0].keys())
    path = out_dir(f"decoding/{args.scope}") / "results_raw.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(raw)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
