"""Log-scale pooled pre-fixation gaze heatmaps by decoded label (unit H).

Same raw summed counts as ``heatmap_label.sum_norm``, displayed as
``log1p(count)``. Normalized to unit H via ``data.attractor.to_maze``.

Usage:
    uv run python -m eye_pre_flash.plotting.heatmaps.label.sum_log_norm
    uv run python -m eye_pre_flash.plotting.heatmaps.label.sum_log_norm --scope publication
    uv run python -m eye_pre_flash.plotting.heatmaps.label.sum_log_norm --monkey Nielsen --bin-w 0.142857
"""

from __future__ import annotations

from eye_pre_flash.plotting.heatmaps.label import core

VISUALIZER = "sum_log_norm"


def main():
    core.main(__doc__, visualizer=VISUALIZER, space="unith", use_log=True)


if __name__ == "__main__":
    main()
