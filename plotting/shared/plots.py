import numpy as np


def trace_stats(traces):
    mean = np.nanmean(traces, axis=0)
    half_sd = 0.5 * np.nanstd(traces, axis=0, ddof=1)
    return mean, half_sd


def plot_trace(ax, mean, half_sd, color, zorder=2, *, end=None):
    mean = mean[:end]
    half_sd = half_sd[:end]
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


def plot_flash_marker(ax, flash_t, color, xmin, xmax):
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


def plot_plot(
    ax,
    data,
    flashes=(),
    *,
    t_max,
    bg_color="blue",
    shuffle=None,
    flip_y=False,
):
    for mean, half_sd, color in data:
        plot_trace(ax, mean, half_sd, color, end=t_max)
    if shuffle is not None:
        shuffle_mean, shuffle_half_sd = shuffle
        plot_trace(ax, shuffle_mean, shuffle_half_sd, "black", zorder=0, end=t_max)
    for flash_t, color, exit_idx in flashes:
        xmin, xmax = (0.0, 0.5) if exit_idx in (1, 2) else (0.5, 1.0)
        plot_flash_marker(ax, flash_t, color, xmin, xmax)

    ax.set_facecolor(bg_color)
    ax.axvline(0.5, color="0.55", linewidth=0.7, linestyle="--", zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, t_max)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([])
    ax.tick_params(labelsize=7)
    if flip_y:
        ax.invert_yaxis()
