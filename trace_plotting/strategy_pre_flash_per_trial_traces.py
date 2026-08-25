import matplotlib.pyplot as plt
import numpy as np

from decoders.common import (
    N_SHUFFLE_ITERS,
    compute_dv_traces,
    compute_shuffle_dv_traces,
    fit_decoder,
)
from decoders.strategy import (
    get_initial_mean,
    prepare_decoder_data,
    prepare_pre_flash_data,
)
from trace_plotting.common import (
    OUT_DIR,
    PRE_FLASH_WINDOW_MS,
    align_pre_flash_to_flash_one,
    plot_flash_label,
    trace_stats,
)
from utils import path_type_for, sigmoid

OUTPUT_PATH = OUT_DIR / "strategy_pre_flash_per_trial_traces.png"

BG_HIERARCHICAL = "#FFFFAB"
BG_SEQUENTIAL = "#FFE4DC"

TRACE_COLOR = {
    BG_HIERARCHICAL: "#E04838",
    BG_SEQUENTIAL: "#5EB0E8",
}

MAZE_BG = {
    1: BG_HIERARCHICAL,
    2: BG_HIERARCHICAL,
    3: BG_HIERARCHICAL,
    4: BG_SEQUENTIAL,
    5: BG_SEQUENTIAL,
    6: BG_SEQUENTIAL,
}

N_TRIALS = 6


########## Helpers


def maze_mask(path_type, maze):
    return np.isin(
        path_type, [path_type_for(maze, exit_idx) for exit_idx in (1, 2, 3, 4)]
    )


########## Plotting


def plot_single_trace(ax, trace, shuffle_mean, shuffle_half_sd, color, bg):
    t = np.arange(PRE_FLASH_WINDOW_MS)
    ax.set_facecolor(bg)
    ax.axvline(0.5, color="0.55", linewidth=0.7, linestyle="--", zorder=0)

    ax.fill_betweenx(
        t,
        shuffle_mean[:PRE_FLASH_WINDOW_MS] - shuffle_half_sd[:PRE_FLASH_WINDOW_MS],
        shuffle_mean[:PRE_FLASH_WINDOW_MS] + shuffle_half_sd[:PRE_FLASH_WINDOW_MS],
        color="0.75",
        alpha=0.45,
        zorder=0,
    )
    ax.plot(
        shuffle_mean[:PRE_FLASH_WINDOW_MS],
        t,
        color="black",
        linewidth=0.8,
        alpha=0.4,
        zorder=1,
    )
    ax.plot(
        trace[:PRE_FLASH_WINDOW_MS],
        t,
        color=color,
        linewidth=1.2,
        alpha=0.9,
        zorder=2,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, PRE_FLASH_WINDOW_MS)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([])
    ax.tick_params(labelsize=6)


def plot_strategy_per_trial_traces(
    traces,
    path_type,
    shuffle_traces,
    save_path=OUTPUT_PATH,
):
    fig, axes = plt.subplots(N_TRIALS, 6, figsize=(14, N_TRIALS * 2.0))
    fig.subplots_adjust(
        left=0.05, right=0.97, top=0.94, bottom=0.04, wspace=0.25, hspace=0.35
    )

    for col, maze in enumerate(range(1, 7)):
        mask = maze_mask(path_type, maze)

        maze_traces = traces[mask, :PRE_FLASH_WINDOW_MS]
        bg = MAZE_BG[maze]
        color = TRACE_COLOR[bg]

        shuffle_mean, shuffle_half_sd = trace_stats(
            shuffle_traces[mask, :PRE_FLASH_WINDOW_MS]
        )

        n_available = len(maze_traces)
        sample_idx = np.arange(min(N_TRIALS, n_available))

        for row in range(N_TRIALS):
            ax = axes[row, col]
            if row < n_available:
                plot_single_trace(
                    ax,
                    maze_traces[sample_idx[row]],
                    shuffle_mean,
                    shuffle_half_sd,
                    color,
                    bg,
                )
                if col == 0:
                    ax.set_ylabel(f"{row + 1}", fontsize=7, labelpad=2)
            else:
                ax.set_visible(False)

            if row == 0:
                ax.set_title(f"Maze {maze}", fontsize=9, pad=4)

            plot_flash_label(ax, PRE_FLASH_WINDOW_MS, color, "flash_one")
            plot_flash_label(ax, 0, color, f"-{PRE_FLASH_WINDOW_MS}")

    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved {save_path}")


########## Runtime


def generate_traces():
    X, y, trial_mask, _, _, _ = prepare_decoder_data()
    X_mean = get_initial_mean(X)
    clf, _, best_lambda = fit_decoder(X_mean, y, trial_mask)

    X_pre_flash, path_type, flash1_ms, trial_mask = prepare_pre_flash_data()

    dv = compute_dv_traces(clf, X_pre_flash, trial_mask)
    shuffle_raw = compute_shuffle_dv_traces(
        X_mean,
        X_pre_flash,
        y,
        trial_mask,
        alpha=best_lambda,
        n_shuffles=N_SHUFFLE_ITERS,
    )
    flash1_valid = flash1_ms[trial_mask]
    dv = align_pre_flash_to_flash_one(dv, flash1_valid)
    shuffle_raw = align_pre_flash_to_flash_one(shuffle_raw, flash1_valid)
    mask = trial_mask
    return (
        sigmoid(dv),
        path_type[mask],
        sigmoid(shuffle_raw),
    )


def main():
    print("Generating traces...")
    traces, path_type, shuffle_traces = generate_traces()
    print("Plotting....")
    plot_strategy_per_trial_traces(traces, path_type, shuffle_traces)


if __name__ == "__main__":
    main()
