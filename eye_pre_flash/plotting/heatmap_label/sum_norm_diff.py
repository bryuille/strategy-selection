"""Difference of pooled unit-H pre-fixation gaze heatmaps by decoded label.

Same raw summed counts as ``heatmap_label.sum_norm``. Decoded hierarchical
minus decoded sequential, diverging around zero.

Usage:
    uv run python -m eye_pre_flash.plotting.heatmap_label.sum_norm_diff
    uv run python -m eye_pre_flash.plotting.heatmap_label.sum_norm_diff --scope all
    uv run python -m eye_pre_flash.plotting.heatmap_label.sum_norm_diff --monkey Faure --bin-w 0.142857
"""

from __future__ import annotations

from eye_pre_flash.plotting.heatmap_label import core

VISUALIZER = "sum_norm_diff"


def main():
    core.main(__doc__, visualizer=VISUALIZER, space="unith", diff=True)


if __name__ == "__main__":
    main()
