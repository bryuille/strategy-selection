"""Log-scale pooled pre-fixation gaze heatmaps by decoded neural label.

Same raw summed counts as ``heatmap_label.sum``, displayed as
``log1p(count)``. Screen degrees, no unit-H warp.

Usage:
    uv run python -m eye_pre_flash.plotting.heatmaps.label.sum_log
    uv run python -m eye_pre_flash.plotting.heatmaps.label.sum_log --scope all
    uv run python -m eye_pre_flash.plotting.heatmaps.label.sum_log --monkey Faure --bin-deg 1
"""

from __future__ import annotations

from eye_pre_flash.plotting.heatmaps.label import core

VISUALIZER = "sum_log"


def main():
    core.main(__doc__, visualizer=VISUALIZER, space="deg", use_log=True)


if __name__ == "__main__":
    main()
