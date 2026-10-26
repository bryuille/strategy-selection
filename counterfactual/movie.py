"""Per-trial movies of the post-choice window, to verify fixation splitting
and the exit-list annotation by eye.

Each frame (10 ms steps, quarter speed by default) shows gaze on the H-maze
in degrees with the exit and origin assignment circles, the detected
fixations and saccades, and a timeline strip. The clock and the timeline are
in ms from `flash_three`; the movie starts there, so the choice saccade is
visible, and the analysed window (from `saccade_init`, see
`counterfactual.extract`) is marked on the timeline. While a fixation is active its centroid
marker and its label stay on for the fixation's whole duration; the first
`NEW_FLASH_MS` of every fixation draw a large expanding ring and a "NEW
FIXATION" banner so a split between two abutting fixations is unmistakable.
A fixation inside an exit ball is labelled "counted as <exit>" and the exit
list on the right grows as the items are counted (the choice-saccade landing
is item 0, see `counterfactual.extract`).

Looks and saccades are read from the exit cache (`counterfactual.extract`),
i.e. the `timestamp_preprocessing` package looks the exit list was built
from; nothing is re-detected here. Fixation blocks are the looks' envelopes.

    uv run python -m counterfactual.movie --session june_24_g0 --trial-id 1 2
    uv run python -m counterfactual.movie --session Nov_3_g0 --batch --n-trials 15

Output: ``counterfactual/out/movies/<Monkey>/<session>/trial_<id>.mp4``, or
with ``--batch`` one ``batch_<first>-<last>.mp4`` holding the trials back to
back with a one-second hold at each trial start (``*.mp4`` under ``out/`` is
gitignored).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from data.builder import (  # noqa: E402
    STEM_DEG,
    behavioral_lookup,
    unpack_eye,
)
from data.config import monkey_for_session  # noqa: E402
from data.loader import load_eye_behavioral_data, load_eye_data  # noqa: E402
from counterfactual.config import (  # noqa: E402
    ELSEWHERE,
    EXIT_NAMES,
    EXIT_RADIUS_DEG,
    ORIGIN,
    ORIGIN_RADIUS_DEG,
    OUT_ROOT,
    SESSIONS,
    STATE_NAMES,
)
from counterfactual.exits import exit_landmarks_deg  # noqa: E402
from counterfactual.extract import load  # noqa: E402

FRAME_STEP_MS = 10
DEFAULT_SPEED = 0.25  # 1.0 = real time
TRIAL_HOLD_S = 1.0  # batch mode: hold the first frame of each trial
TRAIL_MS = 150
NEW_FLASH_MS = 150  # expanding ring + banner at every fixation onset
STATE_COLOUR = {
    ELSEWHERE: "0.55",
    ORIGIN: "#7f7f7f",
    1: "#1f77b4",  # LU
    2: "#2ca02c",  # LD
    3: "#d62728",  # RU
    4: "#9467bd",  # RD
}
SACCADE_COLOUR = "#ff7f0e"
GAZE_COLOUR = "k"


# ---- events for one trial -------------------------------------------------------


def window_events(cache, r):
    """Looks ``(onset, offset, cx, cy, state)`` and saccades ``(onset, offset)``
    of cached trial row `r` (ms from geo_present), straight from the exit
    cache: the same looks the exit list was built from."""
    fr = np.flatnonzero(cache["fix_row"] == r)
    fixations = [(float(cache["fix_onset_ms"][k]), float(cache["fix_offset_ms"][k]),
                  float(cache["fix_xy_d"][k][0]), float(cache["fix_xy_d"][k][1]), int(cache["fix_state"][k]))
                 for k in fr]
    sr = np.flatnonzero(cache["sac_row"] == r)
    saccades = sorted((float(cache["sac_onset_ms"][k]), float(cache["sac_offset_ms"][k])) for k in sr)
    return sorted(fixations), saccades


def counted_items(fixations, chosen, landing):
    """``(fixation index -> list position)`` for fixations that enter the exit
    list, mirroring `extract`: item 0 is the chosen exit; the landing fixation
    (first, on the chosen exit) is that item; later exit fixations append."""
    pos = {}
    items = [int(chosen)]
    for k, f in enumerate(fixations):
        if k == 0 and landing:
            pos[k] = 0
            continue
        if 1 <= f[4] <= 4:
            pos[k] = len(items)
            items.append(f[4])
    return pos, items


# ---- drawing ------------------------------------------------------------------------


def strategy_label(value):
    """Dendrogram label text: 0 = H, 1 = S, NaN = unlabelled."""
    if value is None or not np.isfinite(value):
        return "unlabelled"
    return "H (hierarchical)" if int(value) == 0 else "S (sequential)"


def exit_label(code):
    return EXIT_NAMES[code - 1] if 1 <= code <= 4 else STATE_NAMES[code]


def draw_maze(ax, h, chosen, correct):
    h1, h2, h3, h4, h5, h6 = h
    ax.plot([-h1, h4], [0, 0], color="k", lw=2, zorder=1)
    ax.plot([-h1, -h1], [-h3, h2], color="k", lw=2, zorder=1)
    ax.plot([h4, h4], [-h6, h5], color="k", lw=2, zorder=1)
    ax.plot([0, 0], [0, STEM_DEG], color="k", lw=2, zorder=1)
    ax.add_patch(Circle((0, 0), ORIGIN_RADIUS_DEG, fill=False, ls=":", color=STATE_COLOUR[ORIGIN], lw=1))
    for k, (ex, ey) in enumerate(exit_landmarks_deg(h), start=1):
        lw = 2.5 if k == chosen else 1.0
        ax.add_patch(Circle((ex, ey), EXIT_RADIUS_DEG, fill=False, ls="--", color=STATE_COLOUR[k], lw=lw))
        tag = EXIT_NAMES[k - 1] + (" (chosen)" if k == chosen else "") + (" ★" if k == correct else "")
        left = ex < 0
        ax.text(ex + (-1 if left else 1) * (EXIT_RADIUS_DEG + 0.4), ey, tag, ha="right" if left else "left",
                va="center", fontsize=9, color=STATE_COLOUR[k])


def render_trial(fig, writer, *, session, trial_id, maze, strategy, h, chosen, correct, landing, t0, lo, hi,
                 fixations, saccades, samples, speed, hold_s=0.0):
    """Draw one trial's frames into `writer` (already set up on `fig`). `t0`
    is flash three, the clock origin; `lo`..`hi` the analysed window."""
    t_ms, x, y = samples
    pos, items = counted_items(fixations, chosen, landing)
    lim = max(float(max(h)) + EXIT_RADIUS_DEG + 2.0, 14.0)

    fig.clf()
    gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1.6], height_ratios=[6, 1], hspace=0.25, wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax_list = fig.add_subplot(gs[0, 1])
    ax_tl = fig.add_subplot(gs[1, :])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (deg)")
    ax.set_ylabel("y (deg)")
    draw_maze(ax, h, chosen, correct)
    title = ax.set_title("")
    ax_list.axis("off")

    # timeline: fixation blocks coloured by state, saccades in orange
    for k, (on, off, _, _, st) in enumerate(fixations):
        ax_tl.add_patch(Rectangle((on - t0, 0.15), off - on, 0.7, color=STATE_COLOUR[st], alpha=0.85))
        ax_tl.text((on + off) / 2 - t0, 0.5, str(k + 1), ha="center", va="center", fontsize=7, color="w")
        if k in pos and pos[k] > 0:
            ax_tl.text((on + off) / 2 - t0, 1.0, f"→{exit_label(st)}", ha="center", va="bottom", fontsize=7,
                       color=STATE_COLOUR[st], fontweight="bold")
    for on, off in saccades:
        ax_tl.add_patch(Rectangle((on - t0, 0.0), max(off - on, 2), 0.15, color=SACCADE_COLOUR))
    ax_tl.axvline(lo - t0, color="k", lw=1, ls="--")
    ax_tl.text(lo - t0, 1.42, "saccade_init: analysed window starts", ha="left", va="top", fontsize=7)
    ax_tl.set_xlim(0, hi - t0)
    ax_tl.set_ylim(0, 1.5)
    ax_tl.set_yticks([])
    ax_tl.set_xlabel("ms from flash three   (blocks: fixations, numbered; orange: saccades)")
    cursor = ax_tl.axvline(0, color="k", lw=1.5)

    # animated artists
    (trail,) = ax.plot([], [], color=GAZE_COLOUR, lw=1, alpha=0.6, zorder=3)
    (dot,) = ax.plot([], [], "o", color=GAZE_COLOUR, ms=6, zorder=6)
    (centroid,) = ax.plot([], [], "o", ms=16, mfc="none", mew=3, zorder=5)
    (tether,) = ax.plot([], [], lw=1, ls="-", alpha=0.7, zorder=4)
    ring = Circle((0, 0), 0.1, fill=False, lw=3, visible=False, zorder=5)
    ax.add_patch(ring)
    past_markers = []
    for k, (_, _, cx, cy, st) in enumerate(fixations):
        (m,) = ax.plot([cx], [cy], "o", ms=7, color=STATE_COLOUR[st], alpha=0.0, zorder=2)
        txt = ax.text(cx + 0.4, cy + 0.4, str(k + 1), fontsize=7, color=STATE_COLOUR[st], alpha=0.0, zorder=2)
        past_markers.append((m, txt))
    banner = ax.text(0.5, 0.97, "", transform=ax.transAxes, ha="center", va="top", fontsize=15,
                     fontweight="bold", zorder=7)
    status = ax.text(0.5, 0.02, "", transform=ax.transAxes, ha="center", va="bottom", fontsize=10, zorder=7,
                     bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.9))
    list_text = ax_list.text(0.0, 1.0, "", transform=ax_list.transAxes, va="top", ha="left", fontsize=11,
                             family="monospace")

    def list_block(n_counted, active_pos, active_state):
        lines = [f"{session}  trial {trial_id}  maze {maze}",
                 f"strategy: {strategy_label(strategy)}", "",
                 f"chosen exit: {exit_label(chosen)}" + ("" if correct <= 0 else f"   correct: {exit_label(correct)}"),
                 f"landing on chosen: {'yes' if landing else 'NO'}", "", "exit list so far:"]
        for i, code in enumerate(items[:n_counted]):
            mark = " ◀ counted now" if (active_pos is not None and i == active_pos and i > 0) else ""
            tag = "chosen" if i == 0 else f"fix → {exit_label(code)}"
            lines.append(f"  [{i}] {code} {exit_label(code):2s}  {tag}{mark}")
        lines += ["", f"full list: {items}", "", "codes: 1 LU  2 LD  3 RU  4 RD",
                  "(0 origin, -1 elsewhere: not listed)"]
        if active_state is not None and not (1 <= active_state <= 4):
            lines += ["", f"current fixation: {STATE_NAMES[active_state]} (not counted)"]
        return "\n".join(lines)

    frames = np.arange(t0, hi + FRAME_STEP_MS, FRAME_STEP_MS)
    n_hold = int(round(hold_s * writer.fps))
    for i, now in enumerate(frames):
        win = (t_ms >= now - TRAIL_MS) & (t_ms <= now)
        trail.set_data(x[win], y[win])
        cur = (t_ms <= now) & np.isfinite(x)
        if cur.any():
            dot.set_data([x[cur][-1]], [y[cur][-1]])
        cursor.set_xdata([now - t0, now - t0])
        in_sac = any(on <= now <= off for on, off in saccades)
        active = next((k for k, f in enumerate(fixations) if f[0] <= now <= f[1]), None)
        n_counted = 1 + sum(1 for k, p in pos.items() if p > 0 and fixations[k][0] <= now)
        for k, (m, txt) in enumerate(past_markers):
            done = fixations[k][1] < now
            m.set_alpha(0.5 if done else 0.0)
            txt.set_alpha(0.8 if done else 0.0)
        banner.set_text("")
        ring.set_visible(False)
        trail.set_color(SACCADE_COLOUR if in_sac else GAZE_COLOUR)
        if active is None:
            centroid.set_data([], [])
            tether.set_data([], [])
            if now < lo:
                status.set_text(f"before the choice saccade\n(analysed window starts at +{lo - t0:.0f} ms)")
            else:
                status.set_text("saccade" if in_sac else "no fixation (blink / off-screen / < 100 ms)")
            list_text.set_text(list_block(n_counted, None, None))
        else:
            on, off, cx, cy, st = fixations[active]
            col = STATE_COLOUR[st]
            centroid.set_data([cx], [cy])
            centroid.set_color(col)
            if cur.any():
                tether.set_data([cx, x[cur][-1]], [cy, y[cur][-1]])
                tether.set_color(col)
            age = now - on
            if age <= NEW_FLASH_MS:
                ring.set_visible(True)
                ring.set_center((cx, cy))
                ring.set_radius(0.5 + 4.0 * age / NEW_FLASH_MS)
                ring.set_edgecolor(col)
                banner.set_text(f"NEW FIXATION #{active + 1}")
                banner.set_color(col)
            p = pos.get(active)
            if p is None or p == 0:
                counted = ("= choice landing, item 0" if p == 0 else f"{STATE_NAMES[st]}: not counted")
            else:
                counted = f"COUNTED as {exit_label(st)} → item [{p}]"
            status.set_text(f"fixation #{active + 1}/{len(fixations)}  {int(age)} / {int(off - on)} ms\n{counted}")
            list_text.set_text(list_block(n_counted, p, st))
        title.set_text(f"trial {trial_id}  {strategy_label(strategy)}   "
                       f"t = {now - t0:5.0f} ms after flash three   ({speed:g}× speed)")
        writer.grab_frame()
        if i == 0:
            for _ in range(n_hold):
                writer.grab_frame()


# ---- main ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Post-choice movies per trial.")
    parser.add_argument("--session", required=True)
    parser.add_argument("--trial-id", nargs="*", type=int, default=[])
    parser.add_argument("--n-trials", type=int, default=None,
                        help="first N cached trials when no --trial-id (default 3, or 15 with --batch)")
    parser.add_argument("--batch", action="store_true", help="one video with all trials back to back")
    parser.add_argument("--labelled", action="store_true", help="only trials with a dendrogram strategy label")
    parser.add_argument("--start", type=int, default=0, help="skip the first N eligible cached trials")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT / "movies")
    args = parser.parse_args()

    logging.getLogger("matplotlib.animation").setLevel(logging.WARNING)
    session = args.session
    monkey = monkey_for_session(session)
    if session not in SESSIONS[monkey]:
        raise SystemExit(f"{session} is not a publication session of {monkey}: {SESSIONS[monkey]}")
    cache = load(monkey)
    rows = np.flatnonzero(cache["session"].astype(str) == session)
    by_trial = {int(cache["trial_indices_all"][r]): r for r in rows}
    if args.labelled:
        rows = rows[np.isfinite(cache["strategy"][rows])]
    n_trials = args.n_trials or (15 if args.batch else 3)
    trial_ids = args.trial_id or [int(cache["trial_indices_all"][r]) for r in rows[args.start : args.start + n_trials]]

    eye = load_eye_data(monkey)
    behavioral = load_eye_behavioral_data(monkey)
    beh = behavioral_lookup(behavioral)
    eye_ix = {(str(s), int(t)): i for i, (s, t) in enumerate(zip(eye["session"], eye["trial_indices_all"]))}

    fps = max(int(round(1000.0 / FRAME_STEP_MS * args.speed)), 1)
    out_dir = args.out_root / monkey / session
    out_dir.mkdir(parents=True, exist_ok=True)

    def new_writer():
        return FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])

    def trial_kwargs(trial_id, r):
        t_s, x, y, _, _ = unpack_eye(eye, eye_ix[(session, trial_id)])
        h = tuple(float(v) for v in cache["h"][r])
        lo, hi = float(cache["window_lo_ms"][r]), float(cache["window_hi_ms"][r])
        bi = beh[(session, trial_id)]
        t0 = (float(behavioral["flash_three"][bi]) - float(behavioral["geo_present"][bi])) * 1000.0
        if not np.isfinite(t0) or t0 > lo:
            t0 = lo
        fixations, saccades = window_events(cache, r)
        fin = np.isfinite(t_s)
        samples = (np.round(t_s[fin] * 1000.0), x[fin], y[fin])
        return dict(
            session=session, trial_id=trial_id, maze=int(cache["maze_id"][r]),
            strategy=float(cache["strategy"][r]), h=h,
            chosen=int(cache["chosen_exit"][r]), correct=int(cache["correct_exit"][r]),
            landing=bool(cache["landing_on_chosen"][r]), t0=t0, lo=lo, hi=hi,
            fixations=fixations, saccades=saccades, samples=samples, speed=args.speed,
        )

    kept = [(tid, by_trial[tid]) for tid in trial_ids if tid in by_trial]
    for tid in trial_ids:
        if tid not in by_trial:
            print(f"trial {tid}: not in the {monkey} cache (QC / no window); skipped")
    if not kept:
        raise SystemExit("no trials to render")

    fig = plt.figure(figsize=(12.5, 7.5))
    if args.batch:
        out = out_dir / f"batch_{kept[0][0]}-{kept[-1][0]}.mp4"
        writer = new_writer()
        with writer.saving(fig, str(out), args.dpi):
            for tid, r in kept:
                kw = trial_kwargs(tid, r)
                render_trial(fig, writer, hold_s=TRIAL_HOLD_S, **kw)
                print(f"  trial {tid}: {len(kw['fixations'])} fixations, list {cache['exit_seq'][r]}")
        print(f"wrote {out}  ({len(kept)} trials)")
    else:
        for tid, r in kept:
            out = out_dir / f"trial_{tid}.mp4"
            kw = trial_kwargs(tid, r)
            writer = new_writer()
            with writer.saving(fig, str(out), args.dpi):
                render_trial(fig, writer, **kw)
            print(f"wrote {out}  ({len(kw['fixations'])} fixations, {len(kw['saccades'])} saccades, "
                  f"list {cache['exit_seq'][r]})")
    plt.close(fig)


if __name__ == "__main__":
    main()
