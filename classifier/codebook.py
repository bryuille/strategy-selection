"""Codebook visuals: the regions of one branch drawn over every maze.

    uv run python -m classifier.codebook                  # lor → out/dendro/lor/lor_by_maze.png
    uv run python -m classifier.codebook --branch regions exits lor
    uv run python -m classifier.codebook --labels svm --monkey Faure
    uv run python -m classifier.codebook --branch regions --minority

One panel per maze (1-6), in degrees: the maze skeleton, the branch's region
boundaries, and every in-window fixation centroid of the scope's sessions
coloured by the region `classifier.features` assigns it (grey: unassigned, or
"elsewhere" for `regions`). The panel title gives the maze's arm lengths and
the unassigned share, so a codebook can be checked against the data it is
applied to.

`--minority` writes one figure per monkey (`<branch>_minority_<Monkey>.png`):
the same outlines, but every fixation of the maze's majority strategy is grey
and only the minority strategy's fixations are coloured (blue H, red S).
"""

from __future__ import annotations

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from classifier import config as cfg  # noqa: E402
from classifier import features as F  # noqa: E402

LIM_DEG = 18.0
GREY = "#c8c6bd"
INK = "#1f1f1e"
REGION_COLOURS = {"origin LU": "#f28cb1", "origin LD": "#1b8a3a", "origin RU": "#5b3fb5", "origin RD": "#d62728",
                  "exit LU": "#2a78d6", "exit LD": "#eb6834", "exit RU": "#2a9d8f", "exit RD": "#e9a800",
                  "elsewhere": GREY}
LOR_COLOURS = {"origin": "#1f77b4", "left": "#ff7f0e", "right": "#2ca02c"}
BALLS_COLOURS = {"origin": "#7b4fb5", **{k: REGION_COLOURS[k] for k in F.BALLS_NAMES[1:]}}
CAPSULE_COLOURS = {"origin": "#7b4fb5", "elsewhere": GREY,
                   **{f"capsule {n.split()[1]}": REGION_COLOURS[n] for n in F.REGION_NAMES[4:8]}}
BRANCHES = ("regions", "exits", "lor", "balls", "capsules", "capsules_tight")


def load_fixations(labels, monkeys):
    """Fixation centroids (deg), maze id, arm lengths, strategy label (NaN when
    the trial is unlabelled) and a trial key, pooled over `monkeys`."""
    scope = cfg.LABEL_SCOPES[labels]
    xy, maze, h, y, trial = [], [], [], [], []
    for m in monkeys:
        d = F.load_fix(m, labels=labels)
        rows = np.asarray(d["fix_row"], dtype=int)
        sess = np.asarray(d["session"], dtype=str)[rows]
        tid = np.asarray(d["trial_indices_all"], dtype=int)[rows]
        lookup = F.label_lookup(scope["sessions"][m], scope["stem"])
        xy.append(np.asarray(d["fix_xy_d"], dtype=float))
        maze.append(np.asarray(d["maze_id"], dtype=int)[rows])
        h.append(np.asarray(d["h"], dtype=float)[rows])
        y.append(np.asarray([lookup.get((s, int(t)), np.nan) for s, t in zip(sess, tid)]))
        trial.append(np.asarray([f"{s}|{int(t)}" for s, t in zip(sess, tid)]))
    return (np.vstack(xy), np.concatenate(maze), np.vstack(h),
            np.concatenate(y), np.concatenate(trial))


def minority_of(y, trial, sel):
    """``(label, n_H, n_S)``: the strategy with fewer labelled trials in `sel`
    (0 = H, 1 = S), or ``None`` on a tie. Counts are trials, not fixations."""
    seen = {}
    for key, lab in zip(trial[sel], y[sel]):
        if key not in seen and np.isfinite(lab):
            seen[key] = int(lab)
    n_h = sum(v == 0 for v in seen.values())
    n_s = sum(v == 1 for v in seen.values())
    minor = 0 if n_h < n_s else 1 if n_s < n_h else None
    return minor, n_h, n_s


def assign(branch, xy, h):
    """``(region id per fixation, names)``; -1 is unassigned / grey."""
    if branch == "lor":
        return F.lor_index(xy, h), F.LOR_NAMES
    if branch == "balls":
        return F.balls_index(xy, h), F.BALLS_NAMES
    if branch.startswith("capsules"):
        origin = cfg.CAPSULE_ORIGIN_TIGHT_DEG if branch == "capsules_tight" else cfg.CAPSULE_ORIGIN_DEG
        idx = F.capsule_index(xy, h, origin_deg=origin)
        idx = np.where(idx == F.CAPSULE_NAMES.index("elsewhere"), -1, idx)
        return idx, F.CAPSULE_NAMES
    idx = F.region_index(xy, h)
    if branch == "exits":
        idx = np.where(np.isin(idx, list(F.EXIT_IDS)), idx, -1)
    else:
        idx = np.where(idx == F.REGION_NAMES.index("elsewhere"), -1, idx)
    return idx, F.REGION_NAMES


def draw_skeleton(ax, h):
    h1, h2, h3, h4, h5, h6 = h
    kw = dict(color=INK, lw=2.2, solid_capstyle="round", zorder=3)
    ax.plot([-h1, h4], [0, 0], **kw)
    ax.plot([-h1, -h1], [-h3, h2], **kw)
    ax.plot([h4, h4], [-h6, h5], **kw)
    ax.plot([0, 0], [0, cfg.STEM_DEG], **kw)


H_COLOUR, S_COLOUR = "#1f77b4", "#d62728"  # the same blue / red as the weight panels
STRATEGY = {0: "H", 1: "S"}


def draw_regions(ax, branch, h, *, mono=False):
    exits = F._exit_landmarks(h[None, :])[0]  # LU, LD, RU, RD
    if branch in ("regions", "exits", "balls"):
        for (ex, ey), name in zip(exits, F.REGION_NAMES[4:8]):
            ax.add_patch(Circle((ex, ey), cfg.EXIT_BALL_DEG, fill=False, lw=1.4,
                                color=INK if mono else REGION_COLOURS[name], zorder=4))
            ax.annotate(name.split()[1], (ex, ey), xytext=(-10 if ex < 0 else 10, 0), textcoords="offset points",
                        ha="right" if ex < 0 else "left", va="center", fontsize=7, color=INK)
        if branch == "regions":
            r = cfg.ORIGIN_DISC_DEG
            ax.add_patch(Circle((0, 0), r, fill=False, lw=1.4, color="0.3", zorder=4))
            ax.plot([-r, r], [0, 0], color="0.3", lw=0.8, zorder=4)
            ax.plot([0, 0], [-r, r], color="0.3", lw=0.8, zorder=4)
        if branch == "balls":
            ax.add_patch(Circle((0, 0), cfg.BALLS_ORIGIN_DEG, fill=False, lw=1.4,
                                color=INK if mono else BALLS_COLOURS["origin"], zorder=4))
        return
    if branch.startswith("capsules"):
        origin = cfg.CAPSULE_ORIGIN_TIGHT_DEG if branch == "capsules_tight" else cfg.CAPSULE_ORIGIN_DEG
        theta = np.linspace(-np.pi / 2, np.pi / 2, 40)
        for (ex, ey), name in zip(exits, F.REGION_NAMES[4:8]):
            e = np.array([ex, ey])
            u = e / np.linalg.norm(e)
            n = np.array([-u[1], u[0]])
            cap = cfg.CAPSULE_DEG * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * n)
            inner = cfg.CAPSULE_DEG * (-np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * n)
            outline = np.vstack([e + cap, inner[::-1], (e + cap)[:1]])
            ax.plot(outline[:, 0], outline[:, 1], color=INK if mono else REGION_COLOURS[name], lw=1.2, zorder=4)
            ax.annotate(name.split()[1], (ex, ey), xytext=(-10 if ex < 0 else 10, 0), textcoords="offset points",
                        ha="right" if ex < 0 else "left", va="center", fontsize=7, color=INK)
        ax.add_patch(Circle((0, 0), origin, fill=False, lw=1.4, color=INK if mono else CAPSULE_COLOURS["origin"],
                            zorder=5))
        return
    # lor: the perimeter is the exit quadrilateral grown by LOR_PERIMETER_DEG
    g = np.linspace(-LIM_DEG, LIM_DEG, 721)
    gx, gy = np.meshgrid(g, g)
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    quad = np.broadcast_to(F.exit_quad(h[None, :])[0], (len(pts), 4, 2))
    dist = F.dist_to_polygon(pts, quad).reshape(gx.shape)
    ax.contour(gx, gy, dist, levels=[cfg.LOR_PERIMETER_DEG], colors=[INK], linewidths=1.4, zorder=4)
    col = dist[:, np.argmin(np.abs(g))] <= cfg.LOR_PERIMETER_DEG  # the x = 0 divider, clipped to the perimeter
    ax.plot([0, 0], [g[col].min(), g[col].max()], color=INK, lw=1.0, zorder=4)
    ax.add_patch(Circle((0, 0), cfg.LOR_ORIGIN_DEG, fill=False, lw=1.4,
                        color=INK if mono else LOR_COLOURS["origin"], zorder=5))
    for (ex, ey), name in zip(exits, ("LU", "LD", "RU", "RD")):
        ax.plot(ex, ey, "x", color=INK, ms=6, mew=1.2, zorder=5)
        ax.annotate(name, (ex, ey), xytext=(-8 if ex < 0 else 8, 0), textcoords="offset points",
                    ha="right" if ex < 0 else "left", va="center", fontsize=7, color=INK)


def describe(branch):
    if branch == "lor":
        return (f"origin ball {cfg.LOR_ORIGIN_DEG:g}°, then left (x < 0) / right inside the maze, i.e. within "
                f"{cfg.LOR_PERIMETER_DEG:g}° of the quadrilateral the exits span\n(the perimeter hugs a "
                f"{cfg.LOR_PERIMETER_DEG:g}° ball at each exit); outside the perimeter unassigned")
    if branch == "exits":
        return f"exit balls {cfg.EXIT_BALL_DEG:g}° at each maze's exits; everything else unassigned"
    if branch == "balls":
        return (f"msp's r0.5 codebook in degrees: origin ball {cfg.BALLS_ORIGIN_DEG:g}° + exit balls "
                f"{cfg.EXIT_BALL_DEG:g}° at each maze's exits; everything else unassigned")
    if branch.startswith("capsules"):
        origin = cfg.CAPSULE_ORIGIN_TIGHT_DEG if branch == "capsules_tight" else cfg.CAPSULE_ORIGIN_DEG
        return (f"origin ball {origin:g}° plus a capsule per exit: every point within {cfg.CAPSULE_DEG:g}° of the "
                f"segment\norigin → exit (the two balls joined by parallel tangents); everything else 'elsewhere'")
    return (f"origin disc {cfg.ORIGIN_DISC_DEG:g}° split into quadrants, exit balls {cfg.EXIT_BALL_DEG:g}° at each "
            f"maze's exits, everything else 'elsewhere'")


def figure(branch, xy, maze, h, *, scope_label, dot_size=2.5):
    idx, names = assign(branch, xy, h)
    colours = {"lor": LOR_COLOURS, "balls": BALLS_COLOURS, "capsules": CAPSULE_COLOURS,
               "capsules_tight": CAPSULE_COLOURS}.get(branch, REGION_COLOURS)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.6), layout="constrained")
    for ax, m in zip(axes.ravel(), cfg.MAZES):
        sel = maze == m
        hm = np.unique(h[sel], axis=0)
        if len(hm) != 1:
            print(f"  maze {m}: {len(hm)} distinct arm-length rows; drawing the first")
        hm = hm[0]
        ax.set_aspect("equal")
        ax.set_xlim(-LIM_DEG, LIM_DEG)
        ax.set_ylim(-LIM_DEG, LIM_DEG)
        ax.axhline(0, color="0.9", lw=0.6, zorder=0)
        ax.axvline(0, color="0.9", lw=0.6, zorder=0)
        un = sel & (idx < 0)
        ax.scatter(xy[un, 0], xy[un, 1], s=dot_size, color=GREY, alpha=0.5, lw=0, zorder=1)
        for k, name in enumerate(names):
            if name not in colours or colours[name] == GREY:
                continue
            s = sel & (idx == k)
            ax.scatter(xy[s, 0], xy[s, 1], s=dot_size, color=colours[name], alpha=0.7, lw=0, zorder=2)
        draw_skeleton(ax, hm)
        draw_regions(ax, branch, hm)
        h1, h2, h3, h4, h5, h6 = hm
        ax.set_title(f"maze {m}   (arms: left {h1:g}°, right {h4:g}°)\n"
                     f"exit heights LU {h2:g} / LD {h3:g} / RU {h5:g} / RD {h6:g}°\n{sel.sum()} fixations, "
                     f"{'elsewhere' if branch == 'regions' or branch.startswith('capsules') else 'unassigned'} "
                     f"{un.sum() / max(sel.sum(), 1):.0%}",
                     fontsize=8.5)
        ax.set_xlabel("x (deg)", fontsize=8)
        ax.set_ylabel("y (deg)", fontsize=8)
        ax.tick_params(labelsize=7)
    handles = [Line2D([], [], marker="o", ls="", color=colours[n], label=n) for n in names if n in colours
               and colours[n] != GREY]
    handles.append(Line2D([], [], marker="o", ls="", color=GREY,
                          label="elsewhere" if branch == "regions" or branch.startswith("capsules") else "unassigned"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle(f"`{branch}` codebook (degrees): {describe(branch)}.\n"
                 f"Dots: every in-window fixation centroid ({scope_label}), coloured by assigned region.",
                 fontsize=10)
    return fig


def minority_figure(branch, xy, maze, h, y, trial, *, monkey, scope_label):
    """The codebook with every fixation grey except those of the maze's minority
    strategy (fewer labelled trials): blue for H, red for S, drawn on top."""
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.6), layout="constrained")
    for ax, m in zip(axes.ravel(), cfg.MAZES):
        sel = maze == m
        hm = np.unique(h[sel], axis=0)[0]
        minor, n_h, n_s = minority_of(y, trial, sel)
        ax.set_aspect("equal")
        ax.set_xlim(-LIM_DEG, LIM_DEG)
        ax.set_ylim(-LIM_DEG, LIM_DEG)
        ax.axhline(0, color="0.9", lw=0.6, zorder=0)
        ax.axvline(0, color="0.9", lw=0.6, zorder=0)
        majority = sel & (y != minor)
        ax.scatter(xy[majority, 0], xy[majority, 1], s=2.5, color=GREY, alpha=0.45, lw=0, zorder=1)
        highlight = sel & (y == minor) if minor is not None else np.zeros(len(y), dtype=bool)
        colour = H_COLOUR if minor == 0 else S_COLOUR
        ax.scatter(xy[highlight, 0], xy[highlight, 1], s=7, color=colour, alpha=0.85, lw=0, zorder=2)
        draw_skeleton(ax, hm)
        draw_regions(ax, branch, hm, mono=True)
        n_minor = (n_h, n_s)[minor] if minor is not None else None
        who = "tie" if minor is None else (f"no {STRATEGY[minor]} trials" if n_minor == 0
                                            else f"minority {STRATEGY[minor]}")
        ax.set_title(f"maze {m}   {who}   ({n_h} H, {n_s} S trials)\n"
                     f"{highlight.sum()} of {sel.sum()} fixations highlighted", fontsize=8.5)
        ax.set_xlabel("x (deg)", fontsize=8)
        ax.set_ylabel("y (deg)", fontsize=8)
        ax.tick_params(labelsize=7)
    handles = [Line2D([], [], marker="o", ls="", color=GREY, label="majority strategy"),
               Line2D([], [], marker="o", ls="", color=H_COLOUR, label="minority H"),
               Line2D([], [], marker="o", ls="", color=S_COLOUR, label="minority S")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"{monkey} — `{branch}` codebook (degrees): {describe(branch)}.\n"
                 f"Grey: fixations of the maze's majority strategy. Colour: the minority strategy "
                 f"({scope_label}).", fontsize=10)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branch", nargs="*", default=["lor"], choices=BRANCHES)
    ap.add_argument("--labels", choices=list(cfg.LABEL_SCOPES), default=cfg.DEFAULT_LABELS)
    ap.add_argument("--monkey", nargs="*", default=list(cfg.MONKEYS), choices=cfg.MONKEYS)
    ap.add_argument("--minority", action="store_true",
                    help="one figure per monkey: grey majority-strategy fixations, coloured minority "
                         "(blue H, red S), counted per maze")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()
    scope = cfg.LABEL_SCOPES[args.labels]["label"]
    tag = "" if args.labels == cfg.DEFAULT_LABELS else f"_{args.labels}"

    def out_dir(branch):
        path = cfg.OUT_ROOT / args.labels / branch
        path.mkdir(parents=True, exist_ok=True)
        return path

    if args.minority:
        for monkey in args.monkey:
            xy, maze, h, y, trial = load_fixations(args.labels, [monkey])
            for branch in args.branch:
                fig = minority_figure(branch, xy, maze, h, y, trial, monkey=monkey, scope_label=scope)
                path = out_dir(branch) / f"{branch}_minority_{monkey}{tag}.png"
                fig.savefig(path, dpi=args.dpi)
                plt.close(fig)
                print(f"{branch} {monkey}: {path}")
        return
    xy, maze, h, _, _ = load_fixations(args.labels, args.monkey)
    who = "both monkeys" if set(args.monkey) == set(cfg.MONKEYS) else ", ".join(args.monkey)
    for branch in args.branch:
        fig = figure(branch, xy, maze, h, scope_label=f"{who}, {scope}")
        path = out_dir(branch) / f"{branch}_by_maze{tag}.png"
        fig.savefig(path, dpi=args.dpi)
        plt.close(fig)
        idx, names = assign(branch, xy, h)
        share = {n: f"{(idx == k).mean():.1%}" for k, n in enumerate(names) if n != "elsewhere"}
        print(f"{branch}: {path}  shares {share}  unassigned {(idx < 0).mean():.1%}")


if __name__ == "__main__":
    main()
