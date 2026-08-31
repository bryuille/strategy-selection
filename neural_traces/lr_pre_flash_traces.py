import matplotlib.pyplot as plt
import numpy as np

from neural_traces.common import (
    OUT_DIR,
    PRE_FLASH_WINDOW_MS,
    align_pre_flash_to_flash_one,
    plot_flash_label,
    plot_plot,
    trace_stats,
)
from neural_traces.decoders.common import (
    N_SHUFFLE_ITERS,
    compute_dv_traces,
    compute_shuffle_dv_traces,
    fit_decoder,
)
from neural_traces.decoders.lr import (
    get_endpoint_mean,
    prepare_decoder_data,
    prepare_pre_flash_data,
)
from utils import path_type_for, sigmoid

OUTPUT_PATH = OUT_DIR / "lr_pre_flash_traces.png"

BG_HIERARCHICAL = "#FFFFAB"
BG_SEQUENTIAL = "#FFE4DC"

TRACE_COLOR = {
    BG_HIERARCHICAL: "#E04838",
    BG_SEQUENTIAL: "#5EB0E8",
}

DISTANCE_THRESHOLD = 0.19


########## Helpers


def maze_mask(path_type, maze):
    return np.isin(
        path_type, [path_type_for(maze, exit_idx) for exit_idx in (1, 2, 3, 4)]
    )


def get_background_color(mean):
    if np.abs(mean[PRE_FLASH_WINDOW_MS - 1] - 0.5) > DISTANCE_THRESHOLD:
        return BG_HIERARCHICAL
    return BG_SEQUENTIAL


########## Plotting


def plot_lr_traces(
    traces,
    path_type,
    shuffle_traces,
    save_path=OUTPUT_PATH,
):
    fig, axes = plt.subplots(1, 6, figsize=(14, 2.8))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.15, wspace=0.3)

    for col, maze in enumerate(range(1, 7)):
        ax = axes[col]
        mask = maze_mask(path_type, maze)
        mean, half_sd = trace_stats(traces[mask, :PRE_FLASH_WINDOW_MS])
        shuffle_mean, shuffle_half_sd = trace_stats(
            shuffle_traces[mask, :PRE_FLASH_WINDOW_MS]
        )
        bg = get_background_color(mean)

        plot_plot(
            ax,
            [(mean, half_sd, TRACE_COLOR[bg])],
            t_max=PRE_FLASH_WINDOW_MS,
            bg_color=bg,
            shuffle=(shuffle_mean, shuffle_half_sd),
        )
        plot_flash_label(ax, PRE_FLASH_WINDOW_MS, TRACE_COLOR[bg], "flash_one")
        plot_flash_label(ax, 0, TRACE_COLOR[bg], f"-{PRE_FLASH_WINDOW_MS}")
        ax.set_title(f"Maze {maze}", fontsize=9, pad=4)

    axes[0].set_ylabel("Time (ms)", fontsize=8)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved {save_path}")


########## Runtime


def generate_traces():
    X_trial, y, trial_mask, _, _, flash3_ms = prepare_decoder_data()
    clf, _, best_lambda = fit_decoder(
        get_endpoint_mean(X_trial, flash3_ms), y, trial_mask
    )
    X, path_type, flash1_ms, trial_mask = prepare_pre_flash_data()

    dv = compute_dv_traces(clf, X, trial_mask)
    shuffle_raw = compute_shuffle_dv_traces(
        get_endpoint_mean(X_trial, flash3_ms),
        X,
        y,
        trial_mask,
        random_state=0,
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
    plot_lr_traces(traces, path_type, shuffle_traces)


if __name__ == "__main__":
    main()
