"""Cross-validated unit-H heatmap similarities on a log1p(count/trial) scale.

Gaze is scaled to the unit H via ``data.attractor.to_maze``. Within a
session, maze maps are estimated on independent trial splits and
Pearson-correlated (split-half CV) over bins with gaze in that split.
Those 6×6 matrices are then averaged across sessions.

A maze is included in a session only if it has at least ``MIN_TRIALS``
trials (default 4), so each half has at least two trials.

Usage:
    uv run python -m eye_pre_flash.plotting.similarities.heatmap_log --monkey Faure
    uv run python -m eye_pre_flash.plotting.similarities.heatmap_log --monkey Nielsen
    uv run python -m eye_pre_flash.plotting.similarities.heatmap_log --monkey Faure --bin-w 0.142857
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from data.attractor import to_maze
from data.loader import load_eye_behavioral_data, load_eye_data
from eye_pre_flash.plotting.heatmaps.maze import core as _heatmaps
from data.attractor import FIXATION_EVENT, event_lookup as _event_lookup
from eye_pre_flash.plotting.plot_io import PRE_FIX_END_MS, save_figure
from eye_pre_flash.plotting.similarities.paths import OUT_ROOT
from eye_pre_flash.plotting.similarities.common import BLUE_YELLOW

VISUALIZER = "heatmap_log"
DEG_TO_UNIT_H = 1.0 / _heatmaps.FIXATION_STEM_DEG
HALF_UNIT = _heatmaps.HALF_DEG * DEG_TO_UNIT_H
DEFAULT_BIN = _heatmaps.DEFAULT_BIN_DEG * DEG_TO_UNIT_H
COARSE_BIN = 1.0 * DEG_TO_UNIT_H
PRE_FIX_START_MS = _heatmaps.PRE_FIX_START_MS
PRE_FIX_WINDOW_MS = PRE_FIX_START_MS
MAZES = range(1, 7)
N_SPLITS = 20
MIN_TRIALS = 4


def _behavioral_lookup(behavioral):
    return {
        (str(session), int(trial_id)): i
        for session, trial_id, i in zip(
            behavioral["session"],
            behavioral["trial_indices_all"],
            range(len(behavioral["session"])),
        )
    }


def _fix_start_ms(behavioral, idx):
    return (
        behavioral["fix_start"][idx] - behavioral["geo_present"][idx]
    ) * 1000


def _map_size(bin_w):
    return int(2 * HALF_UNIT / bin_w)


def _accumulate_xy(eyemap, x, y, *, bin_w):
    size = _map_size(bin_w)
    xi = np.floor((x + HALF_UNIT) / bin_w).astype(np.int64)
    yi = np.floor((y + HALF_UNIT) / bin_w).astype(np.int64)
    ok = (xi >= 0) & (xi < size) & (yi >= 0) & (yi < size)
    np.add.at(eyemap, (yi[ok], xi[ok]), 1)


def _pearson(a, b, mask):
    av = np.asarray(a)[mask].ravel()
    bv = np.asarray(b)[mask].ravel()
    if av.size < 2 or av.std() == 0 or bv.std() == 0:
        return np.nan
    return float(np.corrcoef(av, bv)[0, 1])


def _split_half_indices(n, rng):
    order = rng.permutation(n)
    mid = n // 2
    return order[:mid], order[mid:]


def _event_kind(name):
    name = str(name).lower()
    if "saccade" in name:
        return "saccade"
    if name == FIXATION_EVENT:
        return "fixation"
    return None


def _event_kind_mask(t_ms, spans, kind):
    """True on samples inside an event of the given kind.

    `spans` are `data.attractor.event_lookup`'s `(name, onset, offset)` rows.
    """
    mask = np.zeros(t_ms.shape, dtype=bool)
    for name, onset, offset in spans:
        if _event_kind(name) != kind:
            continue
        mask |= (t_ms >= onset) & (t_ms <= offset)
    return mask


def _collect_trials(
    eye, behavioral, *, event_lookup=None, event_kind=None, label_lookup=None
):
    """Pre-fixation unit-H samples keyed by session and maze.

    If `event_kind` is "fixation" or "saccade", samples are additionally
    restricted to that labeled event type (from `event_lookup`, built by
    `data.attractor.event_lookup`).

    With `label_lookup` (``(session, trial_id) -> 0/1``, from
    `classifier.labels.strategy_label_lookup`), trials without a decoded
    strategy label are dropped and each entry becomes ``(x, y, label)`` instead
    of ``(x, y)`` -- what a caller needs to split a maze's
    trials by strategy.
    """
    lookup = _behavioral_lookup(behavioral)
    by_session = {}

    for eye_i in range(len(eye["session"])):
        session = str(eye["session"][eye_i])
        trial_id = int(eye["trial_indices_all"][eye_i])
        beh_i = lookup.get((session, trial_id))
        if beh_i is None:
            continue
        label = None
        if label_lookup is not None:
            label = label_lookup.get((session, trial_id))
            if label is None or not np.isfinite(label):
                continue
        maze = int(behavioral["geo_type"][beh_i])
        if maze not in MAZES:
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
        h = tuple(float(behavioral[f"h{i}"][beh_i]) for i in range(1, 7))

        keep = (
            np.isfinite(t_ms)
            & np.isfinite(x)
            & np.isfinite(y)
            & (t_ms >= fix_ms - PRE_FIX_START_MS)
            & (t_ms <= fix_ms - PRE_FIX_END_MS)
        )
        if event_kind is not None:
            spans = event_lookup.get((session, trial_id), ()) if event_lookup else ()
            keep &= _event_kind_mask(t_ms, spans, event_kind)
        if keep.sum() < 2:
            continue

        x_n, y_n = to_maze(x[keep], y[keep], h)
        mazes = by_session.setdefault(session, {m: [] for m in MAZES})
        mazes[maze].append((x_n, y_n) if label is None else (x_n, y_n, label))

    return by_session


def _trial_count_maps(xy_list, bin_w):
    size = _map_size(bin_w)
    maps = np.zeros((len(xy_list), size, size), dtype=np.int32)
    for i, (x, y) in enumerate(xy_list):
        _accumulate_xy(maps[i], x, y, bin_w=bin_w)
    return maps


def _half_profile(maps, idx, *, use_log=True):
    n = idx.size
    counts = maps[idx].sum(axis=0).astype(np.float64)
    rate = counts / n
    return np.log1p(rate) if use_log else rate


def _session_cv_matrix(
    maze_maps, rng, n_splits=N_SPLITS, min_trials=MIN_TRIALS, *, use_log=True
):
    """6×6 split-half CV Pearson *r* for one session."""
    acc = np.zeros((6, 6), dtype=float)
    counts = np.zeros((6, 6), dtype=float)

    for _ in range(n_splits):
        mean_a = [None] * 6
        mean_b = [None] * 6
        for i, maps in enumerate(maze_maps):
            if maps is None or len(maps) < min_trials:
                continue
            idx_a, idx_b = _split_half_indices(len(maps), rng)
            mean_a[i] = _half_profile(maps, idx_a, use_log=use_log)
            mean_b[i] = _half_profile(maps, idx_b, use_log=use_log)

        present = [m for m in mean_a + mean_b if m is not None]
        if len(present) < 2:
            continue
        active = np.any(np.stack(present, axis=0) > 0, axis=0)

        for i in range(6):
            for j in range(6):
                if mean_a[i] is None or mean_b[j] is None:
                    continue
                if mean_b[i] is None or mean_a[j] is None:
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

    out = np.full((6, 6), np.nan)
    ok = counts > 0
    out[ok] = acc[ok] / counts[ok]
    return out


def maze_similarity_matrix(
    eye,
    behavioral,
    *,
    bin_w=DEFAULT_BIN,
    n_splits=N_SPLITS,
    min_trials=MIN_TRIALS,
    seed=0,
    use_log=True,
    event_lookup=None,
    event_kind=None,
):
    """Session-averaged 6×6 CV Pearson *r*, plus trial counts and n sessions."""
    by_session = _collect_trials(
        eye, behavioral, event_lookup=event_lookup, event_kind=event_kind
    )
    rng = np.random.default_rng(seed)

    session_mats = []
    trial_counts = [0] * 6
    for maze_trials in by_session.values():
        for i, maze in enumerate(MAZES):
            trial_counts[i] += len(maze_trials[maze])
        maze_maps = []
        for maze in MAZES:
            xy_list = maze_trials[maze]
            maze_maps.append(
                None if len(xy_list) < min_trials else _trial_count_maps(xy_list, bin_w)
            )
        mat = _session_cv_matrix(
            maze_maps,
            rng,
            n_splits=n_splits,
            min_trials=min_trials,
            use_log=use_log,
        )
        if np.isfinite(mat).any():
            session_mats.append(mat)

    if not session_mats:
        return np.full((6, 6), np.nan), trial_counts, 0

    corr = np.nanmean(np.stack(session_mats, axis=0), axis=0)
    return corr, trial_counts, len(session_mats)


def plot_similarity(
    monkey="Faure",
    *,
    bin_w=DEFAULT_BIN,
    n_splits=N_SPLITS,
    min_trials=MIN_TRIALS,
    seed=0,
    use_log=True,
    visualizer=VISUALIZER,
    event_kind=None,
):
    eye = load_eye_data(monkey)
    behavioral = load_eye_behavioral_data(monkey)
    event_lookup = None
    if event_kind is not None:
        from data.loader import load_clean_eye_data

        event_lookup = _event_lookup(
            load_clean_eye_data(
                monkey, start_ms=PRE_FIX_START_MS, end_ms=PRE_FIX_END_MS
            )
        )
    corr, counts, n_sessions = maze_similarity_matrix(
        eye,
        behavioral,
        bin_w=bin_w,
        n_splits=n_splits,
        min_trials=min_trials,
        seed=seed,
        use_log=use_log,
        event_lookup=event_lookup,
        event_kind=event_kind,
    )

    finite = corr[np.isfinite(corr)]
    if finite.size:
        vmin = float(finite.min())
        vmax = float(finite.max())
        if vmin == vmax:
            vmax = vmin + 1e-12
    else:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(6.4, 5.6), layout="constrained")
    im = ax.imshow(
        corr,
        origin="upper",
        cmap=BLUE_YELLOW,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    ax.set_xticks(range(6), labels=range(1, 7))
    ax.set_yticks(range(6), labels=range(1, 7))
    ax.set_xlabel("Maze")
    ax.set_ylabel("Maze")
    rate_label = "log1p(count/trial)" if use_log else "count/trial"
    scope_label = f" — {event_kind} samples only" if event_kind else ""
    end_label = "fix_start" if PRE_FIX_END_MS == 0 else f"fix_start − {PRE_FIX_END_MS} ms"
    ax.set_title(
        f"{monkey} pre-fixation heatmap similarities (unit H, CV){scope_label}\n"
        f"split-half Pearson r of {rate_label} | "
        f"avg over {n_sessions} sessions\n"
        f"fix_start − {PRE_FIX_START_MS} ms → {end_label} | {bin_w:g} unit H bins | "
        f"{n_splits} splits | n={','.join(str(n) for n in counts)}",
        fontsize=10,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("correlation coefficient")

    for i in range(6):
        for j in range(6):
            val = corr[i, j]
            if not np.isfinite(val):
                ax.text(j, i, "—", ha="center", va="center", color="white")
                continue
            r, g, b, _ = im.cmap(im.norm(val))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="black" if lum > 0.55 else "white",
                fontsize=9,
            )

    save_figure(
        fig,
        f"similarities_bin{bin_w:g}",
        out_root=OUT_ROOT,
        rel_dir=f"{visualizer}/{monkey}",
    )
    plt.close(fig)
    return fig


def plot_all(monkey="Faure", **kwargs):
    plot_similarity(monkey=monkey, bin_w=DEFAULT_BIN, **kwargs)
    plot_similarity(monkey=monkey, bin_w=COARSE_BIN, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monkey", default="Faure", choices=("Faure", "Nielsen"))
    parser.add_argument(
        "--bin-w",
        type=float,
        default=None,
        help=f"Bin width in unit H (default: generate both {DEFAULT_BIN:g} and {COARSE_BIN:g})",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=N_SPLITS,
        help="Repeated random split-halves per session",
    )
    parser.add_argument(
        "--min-trials",
        type=int,
        default=MIN_TRIALS,
        help="Minimum trials per maze per session to include in CV",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    kwargs = dict(
        n_splits=args.n_splits, min_trials=args.min_trials, seed=args.seed
    )
    if args.bin_w is not None:
        plot_similarity(monkey=args.monkey, bin_w=args.bin_w, **kwargs)
    else:
        plot_all(monkey=args.monkey, **kwargs)


if __name__ == "__main__":
    main()
