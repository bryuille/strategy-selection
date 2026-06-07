from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from decoder import compute_dv_traces, prepare_decoder_data

MAZES = range(1, 7)
POST_FLASH3_MS = 300
OUTPUT_PATH = Path("figures/dv_by_maze.png")

COLOR_LEFT = "#c0392b"
COLOR_RIGHT = "#2980b9"


def mean_sem(traces):
    mean = np.nanmean(traces, axis=0)
    n = np.sum(~np.isnan(traces), axis=0)
    sem = np.nanstd(traces, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return mean, sem


def maze_timing_by_choice(geo_type, flash2_ms, flash3_ms, labels):
    timing = {}
    for maze in MAZES:
        base = (geo_type == maze) & ~np.isnan(labels)
        if not np.any(base):
            timing[maze] = None
            continue

        sides = {}
        for choice_val, key in ((0, "left"), (1, "right")):
            mask = base & (labels == choice_val)
            if not np.any(mask):
                sides[key] = None
                continue
            sides[key] = {
                "flash2": int(np.round(np.nanmedian(flash2_ms[mask]))),
                "flash3": int(np.round(np.nanmedian(flash3_ms[mask]))),
            }

        crop_candidates = [
            sides[k]["flash3"] + POST_FLASH3_MS
            for k in ("left", "right")
            if sides.get(k) is not None
        ]

        timing[maze] = {
            "left": sides.get("left"),
            "right": sides.get("right"),
            "crop_end": int(max(crop_candidates)) if crop_candidates else 0,
        }
    return timing


def draw_flash_markers(ax, flash_ms, color, y_half):
    if y_half == "bottom":
        ymin, ymax = 0.0, 0.5
    else:
        ymin, ymax = 0.5, 1.0

    for t in (flash_ms["flash2"], flash_ms["flash3"]):
        ax.axvline(t, ymin=ymin, ymax=ymax, color=color, linewidth=1.8, linestyle="-")


def plot_maze_dv(dv, labels, geo_type, flash2_ms, flash3_ms, save_path=OUTPUT_PATH):
    timing = maze_timing_by_choice(geo_type, flash2_ms, flash3_ms, labels)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    axes = axes.ravel()

    for maze, ax in zip(MAZES, axes):
        mask = (geo_type == maze) & ~np.isnan(labels)
        maze_time = timing[maze]

        if not np.any(mask) or maze_time is None:
            ax.set_title(f"Maze {maze} (no trials)")
            ax.axis("off")
            continue

        crop_end = min(maze_time["crop_end"], dv.shape[1])
        time_ms = np.arange(crop_end)

        for choice, color, name in [(0, COLOR_LEFT, "Left"), (1, COLOR_RIGHT, "Right")]:
            choice_mask = mask & (labels == choice)
            if not np.any(choice_mask):
                continue
            mean, sem = mean_sem(dv[choice_mask, :crop_end])
            ax.plot(time_ms, mean, color=color, label=f"{name} (n={choice_mask.sum()})")
            ax.fill_between(time_ms, mean - sem, mean + sem, color=color, alpha=0.2)

        ax.axvline(0, color="gray", linewidth=1, linestyle=":", alpha=0.8)

        if maze_time["left"] is not None:
            draw_flash_markers(ax, maze_time["left"], COLOR_LEFT, "bottom")
        if maze_time["right"] is not None:
            draw_flash_markers(ax, maze_time["right"], COLOR_RIGHT, "top")

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xlim(0, crop_end)
        ax.set_title(f"Maze {maze}")
        ax.set_xlabel("Time (ms from flash 1)")

    handles = [
        plt.Line2D([0], [0], color=COLOR_LEFT, linewidth=2, label="Left choice flashes"),
        plt.Line2D([0], [0], color=COLOR_RIGHT, linewidth=2, label="Right choice flashes"),
    ]
    axes[0].legend(handles=handles, fontsize=7, loc="upper left")
    axes[0].set_ylabel("Decision variable")
    fig.suptitle("Decoder DV dynamics by maze", fontsize=13)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    plt.close(fig)


def main():
    X, labels, trial_mask, geo_type, flash2_ms, flash3_ms = prepare_decoder_data()
    _, dv, valid, best_lambda = compute_dv_traces(X, labels, trial_mask)

    print(f"Decoder lambda: {best_lambda:.2e}")
    plot_maze_dv(
        dv,
        labels[valid],
        geo_type[valid],
        flash2_ms[valid],
        flash3_ms[valid],
    )


if __name__ == "__main__":
    main()
