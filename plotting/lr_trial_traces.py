from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from decoders.common import compute_dv_traces, compute_shuffle_dv_traces, fit_decoder
from decoders.lr import get_endpoint_mean, prepare_decoder_data
from plotting.shared.plots import plot_plot, trace_stats
from utils import path_type_for, sigmoid_dv

OUTPUT_PATH = "./figures/lr_trial_traces.png"
N_SHUFFLES = 5

BG_HIERARCHICAL = "#FFFFAB"
BG_SEQUENTIAL = "#FFE4DC"

TRACE_COLORS = {
    BG_HIERARCHICAL: ("#E04838", "#E07820"),
    BG_SEQUENTIAL: ("#5EB0E8", "#1E4088"),
}

GAP_THRESHOLD = 0.3
TIME_THRESHOLD = 200

AXIS_CONFIGS = [
    {"exits": (1, 3), "flip_y": False},
    {"exits": (2, 4), "flip_y": True},
]


########## Helpers


def get_background_color(mean_a, mean_b, cutoff):
    if cutoff <= 0:
        return BG_SEQUENTIAL
    gap = np.abs(mean_a[:cutoff] - mean_b[:cutoff])
    return BG_HIERARCHICAL if np.nanmax(gap) > GAP_THRESHOLD else BG_SEQUENTIAL


def path_color(bg, exit_idx):
    left, right = TRACE_COLORS[bg]
    return left if exit_idx in (1, 2) else right


########## Plotting


def plot_maze_column(
    ax_top,
    ax_bottom,
    traces,
    path_type,
    flash2_ms,
    flash3_ms,
    maze,
    shuffle_traces=None,
    end_padding_ms=200,
):
    shuffle = trace_stats(shuffle_traces) if shuffle_traces is not None else None

    for config, ax in zip(AXIS_CONFIGS, (ax_top, ax_bottom)):
        exits = config["exits"]
        pt_a, pt_b = path_type_for(maze, exits[0]), path_type_for(maze, exits[1])
        mask_a, mask_b = path_type == pt_a, path_type == pt_b

        end_a = min(int(np.nanmax(flash3_ms[mask_a]) + end_padding_ms), traces.shape[1])
        end_b = min(int(np.nanmax(flash3_ms[mask_b]) + end_padding_ms), traces.shape[1])
        t_max = max(end_a, end_b)

        # bg calculation
        mean_a, _ = trace_stats(traces[mask_a])
        mean_b, _ = trace_stats(traces[mask_b])
        bg_cutoff = (
            int(min(np.nanmedian(flash3_ms[mask_a]), np.nanmedian(flash3_ms[mask_b])))
            - TIME_THRESHOLD
        )
        bg = get_background_color(mean_a, mean_b, bg_cutoff)

        plot_data, flashes = [], []
        for exit_idx, mask, end in zip(exits, (mask_a, mask_b), (end_a, end_b)):
            color = path_color(bg, exit_idx)
            mean, half_sd = trace_stats(traces[mask, :end])
            plot_data.append((mean, half_sd, color))
            flashes += [
                (np.nanmedian(flash2_ms[mask]), color, exit_idx),
                (np.nanmedian(flash3_ms[mask]), color, exit_idx),
            ]

        plot_plot(
            ax,
            plot_data,
            flashes,
            t_max=t_max,
            bg_color=bg,
            shuffle=shuffle,
            flip_y=config["flip_y"],
        )

    ax_top.set_title(f"Maze {maze}", fontsize=9, pad=4)


def plot_lr_traces(
    traces,
    path_type,
    flash2_ms,
    flash3_ms,
    shuffle_traces,
    save_path=OUTPUT_PATH,
):
    fig, axes = plt.subplots(2, 6, figsize=(14, 5.8))
    fig.subplots_adjust(
        left=0.03, right=0.97, top=0.9, bottom=0.12, hspace=0.42, wspace=0.3
    )

    for col, maze in enumerate(range(1, 7)):
        plot_maze_column(
            axes[0, col],
            axes[1, col],
            traces,
            path_type,
            flash2_ms,
            flash3_ms,
            maze,
            shuffle_traces=shuffle_traces,
        )

    axes[0, 0].set_ylabel("Time (ms)", fontsize=8)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved {save_path}")


########## Runtime


def generate_traces():
    X, y, trial_mask, path_type, flash2_ms, flash3_ms = prepare_decoder_data()
    X_mean = get_endpoint_mean(X)
    clf, trial_mask, best_lambda = fit_decoder(X_mean, y, trial_mask)
    dv = compute_dv_traces(clf, X, trial_mask)
    shuffle_raw = compute_shuffle_dv_traces(
        X_mean,
        X,
        y,
        trial_mask,
        random_state=0,
        alpha=best_lambda,
        n_shuffles=N_SHUFFLES,
    )
    mask = trial_mask
    return (
        sigmoid_dv(dv),
        path_type[mask],
        flash2_ms[mask],
        flash3_ms[mask],
        sigmoid_dv(shuffle_raw),
    )


def main():
    print("Generating traces...")
    traces, path_type, flash2_ms, flash3_ms, shuffle_traces = generate_traces()
    print("Plotting....")
    plot_lr_traces(traces, path_type, flash2_ms, flash3_ms, shuffle_traces)


if __name__ == "__main__":
    main()
