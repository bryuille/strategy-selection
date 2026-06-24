import matplotlib.pyplot as plt
import numpy as np

from decoders.common import (
    compute_dv_traces,
    compute_shuffle_dv_traces,
    fit_decoder,
)
from decoders.strategy import (
    get_initial_mean,
    prepare_decoder_data,
    prepare_pre_flash_data,
)
from plotting.common import trace_stats
from utils import path_type_for, sigmoid_dv

OUTPUT_PATH = "./figures/strategy_pre_flash_per_trial_traces.png"
N_SHUFFLES = 5

BG_HIERARCHICAL = "#FFFFAB"
BG_SEQUENTIAL = "#FFE4DC"

TRACE_COLOR = {
    BG_HIERARCHICAL: "#E04838",
    BG_SEQUENTIAL: "#5EB0E8",
}

N_TRIALS = 6


########## Helpers


def maze_mask(path_type, maze):
    return np.isin(
        path_type, [path_type_for(maze, exit_idx) for exit_idx in (1, 2, 3, 4)]
    )


def get_background_color(mean, t_max):
    if (1 - mean[t_max - 1]) > 0.65:
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


def plot_single_trace(ax, trace, shuffle_mean, shuffle_half_sd, color, bg, t_max):
    t = np.arange(t_max)
    ax.set_facecolor(bg)
    ax.axvline(0.5, color="0.55", linewidth=0.7, linestyle="--", zorder=0)

    # shuffle band
    ax.fill_betweenx(
        t,
        shuffle_mean[:t_max] - shuffle_half_sd[:t_max],
        shuffle_mean[:t_max] + shuffle_half_sd[:t_max],
        color="0.75",
        alpha=0.45,
        zorder=0,
    )
    ax.plot(
        shuffle_mean[:t_max],
        t,
        color="black",
        linewidth=0.8,
        alpha=0.4,
        zorder=1,
    )

    ax.plot(trace[:t_max], t, color=color, linewidth=1.2, alpha=0.9, zorder=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, t_max)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([])
    ax.tick_params(labelsize=6)


def plot_strategy_per_trial_traces(
    traces,
    path_type,
    shuffle_traces,
    t_max,
    save_path=OUTPUT_PATH,
):
    T_MAX = t_max
    fig, axes = plt.subplots(N_TRIALS, 6, figsize=(14, N_TRIALS * 2.0))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.94, bottom=0.04, wspace=0.25, hspace=0.35)

    rng = np.random.default_rng(seed=42)

    for col, maze in enumerate(range(1, 7)):
        mask = maze_mask(path_type, maze)

        maze_traces = traces[mask, :T_MAX]
        mean, _ = trace_stats(maze_traces)
        bg = get_background_color(mean, T_MAX)
        color = TRACE_COLOR[bg]

        shuffle_mean, shuffle_half_sd = trace_stats(shuffle_traces[mask, :T_MAX])

        n_trials = min(N_TRIALS, len(maze_traces))
        sample_idx = rng.choice(len(maze_traces), size=n_trials, replace=False)

        for row in range(N_TRIALS):
            ax = axes[row, col]
            if row < n_trials:
                plot_single_trace(
                    ax,
                    maze_traces[sample_idx[row]],
                    shuffle_mean,
                    shuffle_half_sd,
                    color,
                    bg,
                    T_MAX,
                )
                if col == 0:
                    ax.set_ylabel(f"{row + 1}", fontsize=7, labelpad=2)
            else:
                ax.set_visible(False)

            if row == 0:
                ax.set_title(f"Maze {maze}", fontsize=9, pad=4)

            plot_flash_label(ax, T_MAX, color, f"{T_MAX}")
            plot_flash_label(ax, 0, color, "geo_pres")

    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved {save_path}")


########## Runtime


def generate_traces():
    X, y, trial_mask, _, _, _ = prepare_decoder_data()
    X_mean = get_initial_mean(X)
    clf, _, best_lambda = fit_decoder(X_mean, y, trial_mask)

    X_pre_flash, path_type, flash1_ms, trial_mask = prepare_pre_flash_data()

    t_max = int(flash1_ms[0])

    dv = compute_dv_traces(clf, X_pre_flash, trial_mask)
    shuffle_raw = compute_shuffle_dv_traces(
        X_mean,
        X_pre_flash,
        y,
        trial_mask,
        alpha=best_lambda,
        n_shuffles=N_SHUFFLES,
    )
    mask = trial_mask
    return (
        sigmoid_dv(dv),
        path_type[mask],
        sigmoid_dv(shuffle_raw),
        t_max,
    )


def main():
    print("Generating traces...")
    traces, path_type, shuffle_traces, t_max = generate_traces()
    print("Plotting....")
    plot_strategy_per_trial_traces(traces, path_type, shuffle_traces, t_max)


if __name__ == "__main__":
    main()
