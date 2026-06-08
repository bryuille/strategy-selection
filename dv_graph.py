from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from decoder import (
    compute_dv_traces,
    prepare_decoder_data,
)

OUTPUT_PATH = Path("figures/dv_traces.png")

BG_HIERARCHICAL = "#FFFFAB"
BG_SEQUENTIAL = "#FFE4DC"

TRACE_COLORS = {
    BG_HIERARCHICAL: ("#E04838", "#E07820"),
    BG_SEQUENTIAL: ("#5EB0E8", "#1E4088"),
}

PANEL_EXITS = ((1, 3), (2, 4))

# used for comparing hierarchical and sequential strategies
GAP_THRESHOLD = 0.3
TIME_THRESHOLD = 200


def sigmoid_dv(dv):
    return 1.0 / (1.0 + np.exp(-dv))


def path_type_for(maze, exit_idx):
    return (maze - 1) * 4 + exit_idx


def exit_color(bg, exit_idx):
    left, right = TRACE_COLORS[bg]
    return left if exit_idx in (1, 2) else right


def plot_trace(ax, traces, color, zorder=2):
    mean = np.nanmean(traces, axis=0)
    half_sd = 0.5 * np.nanstd(traces, axis=0, ddof=1)
    t = np.arange(len(mean))
    is_null = color == "black"
    fill = "0.75" if is_null else color
    ax.plot(
        mean,
        t,
        color=color,
        linewidth=1.0 if is_null else 1.5,
        alpha=0.4 if is_null else 1.0,
        zorder=zorder + 1,
    )
    ax.fill_betweenx(
        t,
        mean - half_sd,
        mean + half_sd,
        color=fill,
        alpha=0.45 if is_null else 0.35,
        zorder=zorder,
    )


def plot_maze_column(
    ax_top,
    ax_bottom,
    dv_prob,
    path_type_by_row,
    flash2_by_row,
    flash3_by_row,
    maze,
    shuffle_traces,
):
    for ax, exits, flip in (
        (ax_top, (1, 3), False),
        (ax_bottom, (2, 4), True),
    ):
        pt_a, pt_b = path_type_for(maze, exits[0]), path_type_for(maze, exits[1])
        mask_a, mask_b = path_type_by_row == pt_a, path_type_by_row == pt_b

        end_a = min(int(np.nanmax(flash3_by_row[mask_a]) + 200), dv_prob.shape[1])
        end_b = min(int(np.nanmax(flash3_by_row[mask_b]) + 200), dv_prob.shape[1])
        t_max = max(end_a, end_b)

        mean_a = np.nanmean(dv_prob[mask_a], axis=0)
        mean_b = np.nanmean(dv_prob[mask_b], axis=0)
        pre_flash_early = (
            int(
                min(
                    np.nanmedian(flash3_by_row[mask_a]),
                    np.nanmedian(flash3_by_row[mask_b]),
                )
            )
            - TIME_THRESHOLD
        )
        gap = np.abs(mean_a[:pre_flash_early] - mean_b[:pre_flash_early])
        bg = (
            BG_HIERARCHICAL
            if pre_flash_early > 0 and np.nanmax(gap) > GAP_THRESHOLD
            else BG_SEQUENTIAL
        )
        ax.set_facecolor(bg)

        for exit_idx, mask, end in zip(exits, (mask_a, mask_b), (end_a, end_b)):
            color = exit_color(bg, exit_idx)
            plot_trace(ax, dv_prob[mask, :end], color)

            xmin, xmax = (0.0, 0.5) if exit_idx in (1, 2) else (0.5, 1.0)
            for flash_t in (
                np.nanmedian(flash2_by_row[mask]),
                np.nanmedian(flash3_by_row[mask]),
            ):
                ax.hlines(
                    flash_t,
                    xmin,
                    xmax,
                    colors=color,
                    linewidth=1.0,
                    alpha=0.85,
                    zorder=4,
                )
                is_left = xmax == 0.5
                ax.text(
                    -0.02 if is_left else 1.02,
                    flash_t,
                    f"{int(flash_t)}",
                    color=color,
                    fontsize=5,
                    va="center",
                    ha="right" if is_left else "left",
                    transform=ax.get_yaxis_transform(),
                    zorder=5,
                    clip_on=False,
                )

        if shuffle_traces is not None:
            plot_trace(ax, shuffle_traces[:, :t_max], "black", zorder=0)

        ax.axvline(0.5, color="0.55", linewidth=0.7, linestyle="--", zorder=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, t_max)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
        if flip:
            ax.invert_yaxis()

    ax_top.set_title(f"Maze {maze}", fontsize=9, pad=4)


def plot_dv_by_maze(
    dv,
    trial_mask,
    path_type,
    flash2_ms,
    flash3_ms,
    shuffle_traces=None,
    save_path=OUTPUT_PATH,
):
    eligible = np.where(trial_mask)[0]
    path_type_by_row = path_type[eligible]
    flash2_by_row = flash2_ms[eligible]
    flash3_by_row = flash3_ms[eligible]
    dv_prob = sigmoid_dv(dv)

    fig, axes = plt.subplots(2, 6, figsize=(14, 5.8))
    fig.subplots_adjust(
        left=0.03, right=0.97, top=0.9, bottom=0.12, hspace=0.42, wspace=0.3
    )

    for col, maze in enumerate(range(1, 7)):
        plot_maze_column(
            axes[0, col],
            axes[1, col],
            dv_prob,
            path_type_by_row,
            flash2_by_row,
            flash3_by_row,
            maze,
            sigmoid_dv(shuffle_traces),
        )

    axes[0, 0].set_ylabel("Time (ms)", fontsize=8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    print("Loading graph...")
    X, labels, trial_mask, _geo, path_type, flash2_ms, flash3_ms = (
        prepare_decoder_data()
    )
    _, dv, _, best_lambda = compute_dv_traces(X, labels, trial_mask)
    shuffle_traces = compute_dv_traces(
        X, labels, trial_mask, random_state=0, alpha=best_lambda, n_shuffles=3
    )
    plot_dv_by_maze(dv, trial_mask, path_type, flash2_ms, flash3_ms, shuffle_traces)


if __name__ == "__main__":
    main()
