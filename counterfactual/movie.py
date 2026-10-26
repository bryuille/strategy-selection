"""Per-trial movies of the counterfactual window, to check the look
assignment and the counterfactual items by eye.

Each frame (10 ms steps, quarter speed by default) shows gaze on the H-maze
in degrees with the exit and origin assignment circles, the looks and
saccades, and a timeline strip. The clock runs from `flash_three` (the start
of the window, `extract.py`) to `trial_end`; `saccade_init` is marked.

- **Gaze** is the `timestamp_preprocessing` package's own sample record
  (`data.looks.sample_quality`: recorded timestamps, no interpolation);
  only valid samples are drawn and the trail never crosses an invalid
  sample or a segment break.
- **Looks** are the window looks of the trial cache (`extract.py`), assigned
  to states exactly as `extract.counterfactual_items` does; **saccades** are
  the package's detected saccades (`data.looks`, epoch ``trial``). Nothing is
  re-detected here.
- A look on an unchosen exit is a counterfactual item. The first look at
  each distinct unchosen exit is scored (Plackett-Luce); repeats are shown
  but add nothing. The side panel lists the items and the H and S rows of
  the model table for the chosen exit (`beliefs.py`).

    uv run python -m counterfactual.movie --session june_24_g0 --trial-id 1 2
    uv run python -m counterfactual.movie --session june_8_g0 june_24_g0 --batch --start 50 --n-trials 20

Output: ``counterfactual/out/movies/<Monkey>/<session>/trial_<id>.mp4``, or with
``--batch`` one ``batch_<first>-<last>.mp4`` per session, the trials back to
back with a one-second hold at each trial start (gitignored).
"""

from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from data.builder import STEM_DEG  # noqa: E402
from data.config import monkey_for_session  # noqa: E402
from data.looks import load_looks, sample_quality, saccades_of, trial_index  # noqa: E402
from counterfactual import beliefs, extract  # noqa: E402
from counterfactual.config import (  # noqa: E402
    ELSEWHERE,
    EXIT_NAMES,
    EXIT_RADIUS_DEG,
    LOOKS_EPOCH,
    MODELS,
    ORIGIN,
    ORIGIN_RADIUS_DEG,
    OUT_ROOT,
    SESSIONS,
)
from counterfactual.task import assign_exit, exit_xy  # noqa: E402

FRAME_STEP_MS = 10
DEFAULT_SPEED = 0.25  # 1.0 = real time
TRIAL_HOLD_S = 1.0  # batch mode: hold the first frame of each trial
TRAIL_MS = 150
NEW_FLASH_MS = 150  # expanding ring + banner at every look onset
STATE_NAMES = {ELSEWHERE: "elsewhere", ORIGIN: "origin", 1: "LU", 2: "LD", 3: "RU", 4: "RD"}
STATE_COLOUR = {ELSEWHERE: "0.55", ORIGIN: "#7f7f7f", 1: "#1f77b4", 2: "#2ca02c", 3: "#d62728", 4: "#9467bd"}
SACCADE_COLOUR = "#ff7f0e"
GAZE_COLOUR = "k"


# ---- events for one trial -----------------------------------------------------------


def trial_events(tr, looks, look_rows, r):
    """Window looks ``(onset, offset, cx, cy, state)`` and saccades
    ``(onset, offset)`` of trial row `r`, ms from geo_present."""
    sl = slice(int(tr["look_start"][r]), int(tr["look_stop"][r]))
    xy = tr["look_xy_d"][sl]
    states = assign_exit(xy, tr["h"][r])
    fixations = [(float(on), float(off), float(p[0]), float(p[1]), int(s)) for on, off, p, s in
                 zip(tr["look_onset_ms"][sl], tr["look_offset_ms"][sl], xy, states)]
    lr = look_rows[(str(tr["session"][r]), int(tr["trial"][r]))]
    ss = saccades_of(looks, lr)
    lo = float(tr["t_lo"][r])
    saccades = [(float(a), float(b)) for a, b in zip(looks["sac_onset_ms"][ss], looks["sac_offset_ms"][ss]) if b >= lo]
    return fixations, sorted(saccades)


def scored_items(fixations, chosen):
    """``look index -> (item position, scored?)`` for looks on unchosen
    exits, in time order (as `extract.counterfactual_items`); a repeat of an
    exit already looked at is not scored."""
    pos, seen = {}, []
    for k, f in enumerate(fixations):
        if 1 <= f[4] <= 4 and f[4] != chosen:
            pos[k] = (len(pos), f[4] not in seen)
            seen.append(f[4])
    return pos


# ---- drawing ------------------------------------------------------------------------


def strategy_label(value):
    if not np.isfinite(value):
        return "unlabelled"
    return "H (hierarchical)" if int(value) == 0 else "S (sequential)"


def draw_maze(ax, h, chosen, correct):
    h1, h2, h3, h4, h5, h6 = h
    ax.plot([-h1, h4], [0, 0], color="k", lw=2, zorder=1)
    ax.plot([-h1, -h1], [-h3, h2], color="k", lw=2, zorder=1)
    ax.plot([h4, h4], [-h6, h5], color="k", lw=2, zorder=1)
    ax.plot([0, 0], [0, STEM_DEG], color="k", lw=2, zorder=1)
    ax.add_patch(Circle((0, 0), ORIGIN_RADIUS_DEG, fill=False, ls=":", color=STATE_COLOUR[ORIGIN], lw=1))
    for k, (ex, ey) in enumerate(exit_xy(h), start=1):
        ax.add_patch(Circle((ex, ey), EXIT_RADIUS_DEG, fill=False, ls="--", color=STATE_COLOUR[k],
                            lw=2.5 if k == chosen else 1.0))
        tag = EXIT_NAMES[k - 1] + (" (chosen)" if k == chosen else "") + (" ★" if k == correct else "")
        left = ex < 0
        ax.text(ex + (-1 if left else 1) * (EXIT_RADIUS_DEG + 0.4), ey, tag, ha="right" if left else "left",
                va="center", fontsize=9, color=STATE_COLOUR[k])


def render_trial(fig, writer, *, session, trial_id, maze, strategy, h, chosen, correct, t0, t_choice, hi,
                 fixations, saccades, samples, q_rows, speed, hold_s=0.0):
    """Draw one trial's frames into `writer`; `t0` (flash three) is the clock origin."""
    t_ms, x, y, seg = samples
    pos = scored_items(fixations, chosen)
    lim = max(float(max(h)) + EXIT_RADIUS_DEG + 2.0, 14.0)

    fig.clf()
    gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1.7], height_ratios=[6, 1], hspace=0.25, wspace=0.08)
    ax, ax_list, ax_tl = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (deg)")
    ax.set_ylabel("y (deg)")
    draw_maze(ax, h, chosen, correct)
    title = ax.set_title("")
    ax_list.axis("off")

    for k, (on, off, _, _, st) in enumerate(fixations):
        a, b = max(on, t0), off
        ax_tl.add_patch(Rectangle((a - t0, 0.15), b - a, 0.7, color=STATE_COLOUR[st], alpha=0.85))
        ax_tl.text((a + b) / 2 - t0, 0.5, str(k + 1), ha="center", va="center", fontsize=7, color="w")
        if k in pos:
            ax_tl.text((a + b) / 2 - t0, 1.0, f"→{STATE_NAMES[st]}", ha="center", va="bottom", fontsize=7,
                       color=STATE_COLOUR[st], fontweight="bold" if pos[k][1] else "normal")
    for on, off in saccades:
        ax_tl.add_patch(Rectangle((on - t0, 0.0), max(off - on, 2), 0.15, color=SACCADE_COLOUR))
    ax_tl.axvline(t_choice - t0, color="k", lw=1, ls="--")
    ax_tl.text(t_choice - t0, 1.42, " saccade_init (choice)", ha="left", va="top", fontsize=7)
    ax_tl.set_xlim(0, hi - t0)
    ax_tl.set_ylim(0, 1.5)
    ax_tl.set_yticks([])
    ax_tl.set_xlabel("ms from flash three   (blocks: looks, numbered; orange: saccades; →: counterfactual)")
    cursor = ax_tl.axvline(0, color="k", lw=1.5)

    (trail,) = ax.plot([], [], color=GAZE_COLOUR, lw=1, alpha=0.6, zorder=3)
    (dot,) = ax.plot([], [], "o", color=GAZE_COLOUR, ms=6, zorder=6)
    (centroid,) = ax.plot([], [], "o", ms=16, mfc="none", mew=3, zorder=5)
    ring = Circle((0, 0), 0.1, fill=False, lw=3, visible=False, zorder=5)
    ax.add_patch(ring)
    past = []
    for k, (_, _, cx, cy, st) in enumerate(fixations):
        (m,) = ax.plot([cx], [cy], "o", ms=7, color=STATE_COLOUR[st], alpha=0.0, zorder=2)
        txt = ax.text(cx + 0.4, cy + 0.4, str(k + 1), fontsize=7, color=STATE_COLOUR[st], alpha=0.0, zorder=2)
        past.append((m, txt))
    banner = ax.text(0.5, 0.97, "", transform=ax.transAxes, ha="center", va="top", fontsize=15,
                     fontweight="bold", zorder=7)
    status = ax.text(0.5, 0.02, "", transform=ax.transAxes, ha="center", va="bottom", fontsize=10, zorder=7,
                     bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.9))
    panel = ax_list.text(0.0, 1.0, "", transform=ax_list.transAxes, va="top", ha="left", fontsize=10,
                         family="monospace")

    others = [e for e in (1, 2, 3, 4) if e != chosen]
    header = [f"{session}  trial {trial_id}  maze {maze}", f"strategy: {strategy_label(strategy)}", "",
              f"chosen {STATE_NAMES[chosen]}   correct {STATE_NAMES[correct]}"
              + ("" if chosen == correct else "  (error)"), "",
              "model row for the chosen exit:", "        " + "    ".join(f"{STATE_NAMES[e]:>3s}" for e in others)]
    header += [f"  q_{m}  " + "  ".join(f"{q_rows[m][e - 1]:5.2f}" for e in others) for m in MODELS]

    def panel_text(now, active):
        lines = header + ["", "counterfactual items:"]
        for k, (p, scored) in pos.items():
            if fixations[k][0] > now:
                break
            mark = " ◀ now" if k == active else ""
            lines.append(f"  [{p}] {STATE_NAMES[fixations[k][4]]}" + ("" if scored else "  (repeat, not scored)") + mark)
        return "\n".join(lines)

    frames = np.arange(t0, hi + FRAME_STEP_MS, FRAME_STEP_MS)
    n_hold = int(round(hold_s * writer.fps))
    for i, now in enumerate(frames):
        cur = np.flatnonzero(t_ms <= now)
        if cur.size:
            last = cur[-1]
            win = (t_ms >= now - TRAIL_MS) & (t_ms <= now) & (seg == seg[last])  # never across a break
            trail.set_data(x[win], y[win])
            dot.set_data([x[last]], [y[last]])
        cursor.set_xdata([now - t0, now - t0])
        in_sac = any(on <= now <= off for on, off in saccades)
        trail.set_color(SACCADE_COLOUR if in_sac else GAZE_COLOUR)
        active = next((k for k, f in enumerate(fixations) if f[0] <= now <= f[1]), None)
        for k, (m, txt) in enumerate(past):
            done = fixations[k][1] < now
            m.set_alpha(0.5 if done else 0.0)
            txt.set_alpha(0.8 if done else 0.0)
        banner.set_text("")
        ring.set_visible(False)
        if active is None:
            centroid.set_data([], [])
            status.set_text("saccade" if in_sac else "no look (saccade / invalid samples / < 75 ms stable)")
        else:
            on, off, cx, cy, st = fixations[active]
            col = STATE_COLOUR[st]
            centroid.set_data([cx], [cy])
            centroid.set_color(col)
            if 0 <= now - on <= NEW_FLASH_MS:
                ring.set_visible(True)
                ring.set_center((cx, cy))
                ring.set_radius(0.5 + 4.0 * (now - on) / NEW_FLASH_MS)
                ring.set_edgecolor(col)
                banner.set_text(f"NEW LOOK #{active + 1}")
                banner.set_color(col)
            if st == chosen:
                what = "on the chosen exit: not counterfactual"
            elif active in pos:
                p, scored = pos[active]
                what = f"COUNTERFACTUAL item [{p}] → {STATE_NAMES[st]}" + ("" if scored else " (repeat)")
            else:
                what = f"{STATE_NAMES[st]}: not counted"
            status.set_text(f"look #{active + 1}/{len(fixations)}  {max(now - on, 0):.0f} / {off - on:.0f} ms\n{what}")
        panel.set_text(panel_text(now, active))
        phase = "before the choice" if now < t_choice else "after the choice"
        title.set_text(f"trial {trial_id}   t = {now - t0:5.0f} ms after flash three ({phase}, {speed:g}× speed)")
        writer.grab_frame()
        if i == 0:
            for _ in range(n_hold):
                writer.grab_frame()


# ---- main -----------------------------------------------------------------------------


def render_session(session, args):
    monkey = monkey_for_session(session)
    if session not in SESSIONS[monkey]:
        raise SystemExit(f"{session} is not a session of {monkey}: {SESSIONS[monkey]}")
    tr, q = extract.load(monkey), beliefs.load(monkey)
    looks = load_looks(monkey, LOOKS_EPOCH, SESSIONS[monkey])
    look_rows = trial_index(looks)
    rows = np.flatnonzero(tr["session"].astype(str) == session)
    if args.labelled:
        rows = rows[np.isfinite(tr["label"][rows])]
    by_trial = {int(tr["trial"][r]): r for r in rows}
    n = args.n_trials or (20 if args.batch else 3)
    trial_ids = args.trial_id or [int(tr["trial"][r]) for r in rows[args.start: args.start + n]]
    kept = [(t, by_trial[t]) for t in trial_ids if t in by_trial]
    if not kept:
        raise SystemExit(f"{session}: no trials to render")

    def kwargs(trial_id, r):
        s = sample_quality(monkey, session, trial_id)
        ok = s["valid"] & np.isfinite(s["t"])
        fixations, saccades = trial_events(tr, looks, look_rows, r)
        return dict(session=session, trial_id=trial_id, maze=int(tr["maze"][r]), strategy=float(tr["label"][r]),
                    h=tuple(float(v) for v in tr["h"][r]), chosen=int(tr["chosen"][r]),
                    correct=int(tr["correct"][r]), t0=float(tr["t_lo"][r]), t_choice=float(tr["t_choice"][r]),
                    hi=float(tr["t_hi"][r]), fixations=fixations, saccades=saccades,
                    samples=(s["t"][ok], s["x"][ok], s["y"][ok], s["segment"][ok]),
                    q_rows={m: q[f"q_{m}"][r] for m in MODELS}, speed=args.speed)

    fps = max(int(round(1000.0 / FRAME_STEP_MS * args.speed)), 1)
    out_dir = OUT_ROOT / "movies" / monkey / session
    out_dir.mkdir(parents=True, exist_ok=True)
    writer_for = lambda: FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    fig = plt.figure(figsize=(13, 7.5))
    if args.batch:
        out = out_dir / f"batch_{kept[0][0]}-{kept[-1][0]}.mp4"
        writer = writer_for()
        with writer.saving(fig, str(out), args.dpi):
            for t, r in kept:
                render_trial(fig, writer, hold_s=TRIAL_HOLD_S, **kwargs(t, r))
        print(f"wrote {out}  ({len(kept)} trials: {kept[0][0]}..{kept[-1][0]})")
    else:
        for t, r in kept:
            out = out_dir / f"trial_{t}.mp4"
            writer = writer_for()
            with writer.saving(fig, str(out), args.dpi):
                render_trial(fig, writer, **kwargs(t, r))
            print(f"wrote {out}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Counterfactual-window movies per trial.")
    p.add_argument("--session", nargs="+", required=True)
    p.add_argument("--trial-id", nargs="*", type=int, default=[])
    p.add_argument("--n-trials", type=int, help="trials when no --trial-id (default 3, or 20 with --batch)")
    p.add_argument("--start", type=int, default=0, help="skip the first N eligible trials of the session")
    p.add_argument("--batch", action="store_true", help="one video per session, trials back to back")
    p.add_argument("--labelled", action="store_true", help="only trials with a dendrogram label")
    p.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    p.add_argument("--dpi", type=int, default=120)
    args = p.parse_args()
    logging.getLogger("matplotlib.animation").setLevel(logging.WARNING)
    for session in args.session:
        render_session(session, args)


if __name__ == "__main__":
    main()
