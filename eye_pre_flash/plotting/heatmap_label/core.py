"""Pooled pre-fixation gaze heatmaps grouped by the decoded neural label.

The ``heatmap_maze`` family pools trials by *assumed* strategy -- the maze's
regime in ``data.labeler.MAZE_GROUPS``, so every trial of maze 1 counts as
hierarchical. Here the two pools are instead the trials the neural clustering
actually labelled hierarchical and sequential
(``data.labeler.build_strategy_choices``, 0 = hierarchical, 1 = sequential),
with all six mazes mixed into both pools.

A label exists only for a session whose neural recording was clustered, so the
contributing sessions are a classifier *scope* rather than "all sessions":

    publication  the four sessions the published clustering was defined on
    all          every SNR-eligible session clearing anchor-maze label vetting
    allplus      the same pool, vetting skipped

Scopes come straight from :mod:`eye_pre_flash.classifier.labels`, so these
heatmaps show the same trials the decoding tables are computed on.

This module holds the shared plumbing; the six sibling modules are the
renderings, matching ``heatmap_maze`` suffix for suffix.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from data.attractor import to_maze
from data.loader import load_eye_behavioral_data, load_eye_data
from eye_pre_flash.classifier.labels import (
    LABEL_NAMES,
    MIN_LABEL_AGREEMENT,
    SCOPES,
    scope_sessions,
    strategy_label_lookup,
)
from eye_pre_flash.plotting.heatmap_maze.core import (
    DEFAULT_BIN_DEG,
    HALF_DEG,
    PRE_FIX_END_MS,
    PRE_FIX_START_MS,
    _accumulate_xy as _accumulate_deg,
    _behavioral_lookup,
    _fix_start_ms,
    _map_size as _map_size_deg,
)
from eye_pre_flash.plotting.heatmap_maze.sum import COARSE_BIN_DEG
from eye_pre_flash.plotting.heatmap_maze.sum_norm import (
    COARSE_BIN,
    DEFAULT_BIN,
    DEG_TO_UNIT_H,
    FIXATION_STEM_DEG,
    HALF_UNIT,
    _accumulate_xy as _accumulate_unit,
    _map_size as _map_size_unit,
)
from eye_pre_flash.plotting.plot_io import save_figure, window_label

# The mazes the maze-grouped family covers, so trial totals are comparable.
MAZES = tuple(range(1, 7))
LABELS = (0, 1)

# Everything a rendering needs to know about its coordinate frame: the two
# families differ only in half-extent, binning helpers and axis wording.
SPACES = {
    "deg": dict(
        half=HALF_DEG,
        accumulate=_accumulate_deg,
        map_size=_map_size_deg,
        bin_key="bin_deg",
        axis="deg",
        bins=(DEFAULT_BIN_DEG, COARSE_BIN_DEG),
        extent=lambda bin_size: f"±{HALF_DEG:g}°, {bin_size:g}° bins",
    ),
    "unith": dict(
        half=HALF_UNIT,
        accumulate=_accumulate_unit,
        map_size=_map_size_unit,
        bin_key="bin_w",
        axis="unit H",
        bins=(DEFAULT_BIN, COARSE_BIN),
        extent=lambda bin_size: f"±{HALF_UNIT:.2f} unit H, {bin_size:g} bin_w",
    ),
}


def resolve_scope(scope, *, verbose=True):
    """``monkey -> sessions`` for `scope`, reporting what it could not use.

    Mirrors the reporting in :mod:`eye_pre_flash.classifier.decoding` so a run
    on a partial checkout says which sessions it is missing rather than
    silently plotting a smaller pool.
    """
    by_monkey, missing, dropped = scope_sessions(scope)
    if verbose and missing:
        print(
            f"scope {scope!r}: {len(missing)} session(s) not labelled yet, "
            f"skipped: {', '.join(missing)}"
        )
    if verbose and dropped:
        print(
            f"scope {scope!r}: {len(dropped)} session(s) dropped for anchor-maze "
            f"label agreement below {MIN_LABEL_AGREEMENT}: "
            + ", ".join(f"{s} ({a:.2f})" for s, a in sorted(dropped.items()))
        )
    if not by_monkey:
        raise SystemExit(
            f"scope {scope!r}: no labelled sessions; run the labels stage first"
        )
    return by_monkey


def collect_label_samples(eye, behavioral, sessions, *, space="deg"):
    """``{label: [(x, y), ...]}`` pre-fixation samples for `sessions`.

    Same trial filters and window as ``heatmap_maze.core._collect_maze_samples``
    -- QC-passed, unfaded, fixed-geometry trials over
    fix_start − `PRE_FIX_START_MS` → fix_start − `PRE_FIX_END_MS` -- but keyed
    on the decoded label instead of the maze, and restricted to the scope's
    sessions. With ``space="unith"`` each trial is warped onto the unit H by
    its own geometry before binning.
    """
    lookup = strategy_label_lookup(sessions)
    beh_index = _behavioral_lookup(behavioral)
    wanted = set(sessions)
    samples = {label: [] for label in LABELS}

    for eye_i in range(len(eye["session"])):
        session = str(eye["session"][eye_i])
        if session not in wanted:
            continue
        trial_id = int(eye["trial_indices_all"][eye_i])
        label = lookup.get((session, trial_id))
        if label is None or not np.isfinite(label):
            continue
        beh_i = beh_index.get((session, trial_id))
        if beh_i is None:
            continue
        if int(behavioral["geo_type"][beh_i]) not in MAZES:
            continue
        if behavioral["path_type"][beh_i] == -99:
            continue
        if behavioral["photodiode_qc_bad"][beh_i]:
            continue
        if behavioral["trial_fade"][beh_i] != 0:
            continue

        t_ms = np.asarray(eye["time"][eye_i], dtype=float) * 1000
        x = np.asarray(eye["eye_x"][eye_i], dtype=float)
        y = np.asarray(eye["eye_y"][eye_i], dtype=float)
        fix_ms = _fix_start_ms(behavioral, beh_i)
        keep = (
            np.isfinite(t_ms)
            & np.isfinite(x)
            & np.isfinite(y)
            & (t_ms >= fix_ms - PRE_FIX_START_MS)
            & (t_ms <= fix_ms - PRE_FIX_END_MS)
        )
        if keep.sum() < 2:
            continue
        x, y = x[keep], y[keep]
        if space == "unith":
            h = tuple(float(behavioral[f"h{i}"][beh_i]) for i in range(1, 7))
            x, y = to_maze(x, y, h)
        samples[int(label)].append((x, y))

    return samples


def label_maps(samples, *, bin_size, space="deg"):
    """``({label: eyemap}, {label: n_trials})`` binned at `bin_size`.

    `samples` is collected once per (monkey, space) and re-binned for every bin
    size, since the collection pass dominates the cost.
    """
    spec = SPACES[space]
    size = spec["map_size"](bin_size)
    maps, counts = {}, {}
    for label, trials in samples.items():
        eyemap = np.zeros((size, size), dtype=np.int64)
        for x, y in trials:
            spec["accumulate"](eyemap, x, y, **{spec["bin_key"]: bin_size})
        maps[label] = eyemap
        counts[label] = len(trials)
    return maps, counts


def _scale_note(space, use_log):
    parts = ["sum"] + (["log"] if use_log else [])
    return ", ".join(parts + (["unit H"] if space == "unith" else []))


def _render(display, *, space, cmap, vmin, vmax, title, cbar, stem, rel_dir):
    spec = SPACES[space]
    half = spec["half"]
    fig, ax = plt.subplots(figsize=(6.5, 6.0), layout="constrained")
    im = ax.imshow(
        display,
        origin="lower",
        extent=(-half, half, -half, half),
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_xlabel(f"eye_x ({spec['axis']})")
    ax.set_ylabel(f"eye_y ({spec['axis']})")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.85, label=cbar)
    save_figure(fig, stem, rel_dir=rel_dir)
    plt.close(fig)
    return fig


def out_rel_dir(visualizer, scope, monkey):
    return f"heatmap_label/{scope}/{visualizer}/{monkey}"


def _window_note():
    return f"{window_label()}"


def plot_label_heatmap(
    maps,
    counts,
    label,
    *,
    monkey,
    scope,
    sessions,
    space,
    use_log,
    visualizer,
    bin_size,
):
    """One decoded label's pooled heatmap."""
    n_trials = counts[label]
    if n_trials == 0:
        raise ValueError(
            f"No {LABEL_NAMES[label]} trials for monkey={monkey!r}, scope={scope!r}"
        )
    display = maps[label].astype(float)
    if use_log:
        display = np.log1p(display)
    vmax = float(display.max()) if display.max() > 0 else 1.0
    name = LABEL_NAMES[label]
    return _render(
        display,
        space=space,
        cmap="magma",
        vmin=0,
        vmax=vmax,
        title=(
            f"{monkey} decoded {name} pre-fixation gaze "
            f"({_scale_note(space, use_log)})\n"
            f"scope {scope}: {len(sessions)} session(s), mazes pooled\n"
            f"{_window_note()}\n"
            f"{SPACES[space]['extent'](bin_size)}, n={n_trials}"
        ),
        cbar="log1p(count)" if use_log else "count",
        stem=f"bin{bin_size:g}_{name}",
        rel_dir=out_rel_dir(visualizer, scope, monkey),
    )


def plot_label_diff(
    maps,
    counts,
    *,
    monkey,
    scope,
    sessions,
    space,
    use_log,
    visualizer,
    bin_size,
):
    """Decoded hierarchical minus decoded sequential, diverging around zero."""
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError(
            f"No trials for monkey={monkey!r}, scope={scope!r} heatmap diff"
        )
    hier = maps[0].astype(float)
    seq = maps[1].astype(float)
    if use_log:
        hier, seq = np.log1p(hier), np.log1p(seq)
    diff = hier - seq
    vmax = float(np.max(np.abs(diff))) if np.any(diff) else 1.0
    return _render(
        diff,
        space=space,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        title=(
            f"{monkey} decoded hierarchical − sequential pre-fixation gaze "
            f"({_scale_note(space, use_log)})\n"
            f"scope {scope}: {len(sessions)} session(s), mazes pooled\n"
            f"{_window_note()}\n"
            f"{SPACES[space]['extent'](bin_size)}, "
            f"n={counts[0]} − {counts[1]}"
        ),
        cbar="log1p(count) (hier − seq)" if use_log else "count (hier − seq)",
        stem=f"bin{bin_size:g}_diff",
        rel_dir=out_rel_dir(visualizer, scope, monkey),
    )


_EYE_CACHE = {}


def _monkey_data(monkey):
    """Eye and behavioural data for `monkey`, loaded once per process.

    A single run covers up to three scopes, and the scopes overlap heavily --
    reloading a monkey's eye npz per scope would dominate the runtime.
    """
    if monkey not in _EYE_CACHE:
        _EYE_CACHE[monkey] = (
            load_eye_data(monkey),
            load_eye_behavioral_data(monkey),
        )
    return _EYE_CACHE[monkey]


def run(
    *,
    visualizer,
    space,
    use_log=False,
    diff=False,
    scope=None,
    monkey=None,
    bins=None,
):
    """Render `visualizer` for every requested scope, monkey and bin size.

    ``scope=None`` covers all three scopes, ``monkey=None`` both monkeys.
    """
    spec = SPACES[space]
    bins = tuple(bins) if bins else spec["bins"]
    scopes = (scope,) if scope else tuple(sorted(SCOPES))

    for scope_name in scopes:
        by_monkey = resolve_scope(scope_name)
        for mk, sessions in sorted(by_monkey.items()):
            if monkey is not None and mk != monkey:
                continue
            print(
                f"{mk}: {len(sessions)} labelled session(s) in scope "
                f"{scope_name!r}: {', '.join(sessions)}"
            )
            eye, behavioral = _monkey_data(mk)
            samples = collect_label_samples(eye, behavioral, sessions, space=space)
            for bin_size in bins:
                maps, counts = label_maps(samples, bin_size=bin_size, space=space)
                shared = dict(
                    monkey=mk,
                    scope=scope_name,
                    sessions=sessions,
                    space=space,
                    use_log=use_log,
                    visualizer=visualizer,
                    bin_size=bin_size,
                )
                if diff:
                    plot_label_diff(maps, counts, **shared)
                else:
                    for label in LABELS:
                        plot_label_heatmap(maps, counts, label, **shared)


def build_parser(doc, *, space):
    """Argument parser shared by the six renderings."""
    parser = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--monkey",
        default=None,
        choices=("Faure", "Nielsen"),
        help="default: every monkey with labelled sessions in scope",
    )
    parser.add_argument(
        "--scope",
        default=None,
        choices=sorted(SCOPES),
        help=f"session scope (default: all of {', '.join(sorted(SCOPES))})",
    )
    if space == "deg":
        parser.add_argument(
            "--bin-deg",
            type=float,
            default=None,
            help=(
                f"Bin width in degrees (default: generate both "
                f"{DEFAULT_BIN_DEG:g} and {COARSE_BIN_DEG:g})"
            ),
        )
    else:
        parser.add_argument(
            "--bin-w",
            type=float,
            default=None,
            help=(
                f"Bin width in unit H (default: generate both "
                f"{DEFAULT_BIN:g} and {COARSE_BIN:g})"
            ),
        )
        parser.add_argument(
            "--bin-deg",
            type=float,
            default=None,
            help=f"Bin width in degrees (converted via 1 / {FIXATION_STEM_DEG:g}°)",
        )
    return parser


def resolve_bins(args, *, space):
    """The bin sizes `args` asks for, or the space's default pair."""
    if space == "deg":
        return [args.bin_deg] if args.bin_deg is not None else None
    if args.bin_w is not None:
        return [args.bin_w]
    if args.bin_deg is not None:
        return [args.bin_deg * DEG_TO_UNIT_H]
    return None


def main(doc, *, visualizer, space, use_log=False, diff=False):
    """Standard entry point: parse `space`'s arguments and run."""
    args = build_parser(doc, space=space).parse_args()
    run(
        visualizer=visualizer,
        space=space,
        use_log=use_log,
        diff=diff,
        scope=args.scope,
        monkey=args.monkey,
        bins=resolve_bins(args, space=space),
    )
