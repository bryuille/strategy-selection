import matplotlib.pyplot as plt
import numpy as np

from decoders.common import compute_dv_traces, compute_shuffle_dv_traces, fit_decoder
from decoders.lr import prepare_decoder_data, prepare_pre_flash_data
from plotting.shared.plots import plot_plot, trace_stats
from utils import path_type_for, sigmoid_dv

OUTPUT_PATH = "./figures/lr_pre_flash_traces.png"
N_SHUFFLES = 5
T_MAX = 700

BG_HIERARCHICAL = "#FFFFAB"
BG_SEQUENTIAL = "#FFE4DC"

TRACE_COLOR = {
    BG_HIERARCHICAL: "#E04838",
    BG_SEQUENTIAL: "#5EB0E8",
}

DISTANCE_THRESHOLD = 0.2


########## Helpers


def maze_mask(path_type, maze):
    return np.isin(
        path_type, [path_type_for(maze, exit_idx) for exit_idx in (1, 2, 3, 4)]
    )


def get_background_color(mean):
    if np.abs(mean[T_MAX - 1] - 0.5) > DISTANCE_THRESHOLD:
        return BG_HIERARCHICAL
    return BG_SEQUENTIAL


def plot_flash_label(ax, flash_t, color, label):
    ax.text(
        -0.02,
        flash_t,
        label,
        color=color,
        fontsize=5,
        va="top",
        ha="right",
        transform=ax.get_yaxis_transform(),
        zorder=5,
        clip_on=False,
    )


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
        mean, half_sd = trace_stats(traces[mask, :T_MAX])
        shuffle_mean, shuffle_half_sd = trace_stats(shuffle_traces[mask, :T_MAX])
        bg = get_background_color(mean)

        plot_plot(
            ax,
            [(mean, half_sd, TRACE_COLOR[bg])],
            t_max=T_MAX,
            bg_color=bg,
            shuffle=(shuffle_mean, shuffle_half_sd),
        )
        plot_flash_label(ax, T_MAX, TRACE_COLOR[bg], f"{T_MAX}")
        ax.set_title(f"Maze {maze}", fontsize=9, pad=4)

    axes[0].set_ylabel("Time (ms)", fontsize=8)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved {save_path}")


########## Runtime


def generate_traces():
    X_fit, labels, trial_mask, _, _, _ = prepare_decoder_data()
    clf, _, best_lambda = fit_decoder(X_fit, labels, trial_mask)

    X, path_type, _, trial_mask = prepare_pre_flash_data()
    dv = compute_dv_traces(clf, X, trial_mask)
    shuffle_raw = compute_shuffle_dv_traces(
        X,
        labels,
        trial_mask,
        alpha=best_lambda,
        n_shuffles=N_SHUFFLES,
    )
    mask = trial_mask
    return (
        sigmoid_dv(dv),
        path_type[mask],
        sigmoid_dv(shuffle_raw),
    )


def main():
    print("Generating traces...")
    traces, path_type, shuffle_traces = generate_traces()
    print("Plotting....")
    plot_lr_traces(traces, path_type, shuffle_traces)


if __name__ == "__main__":
    main()
