"""Pooled pre-fixation gaze heatmaps by decoded neural label (unit H).

Same two label pools as ``heatmap_label.sum``, normalized to the unit H
coordinate frame via ``data.attractor.to_maze`` -- so pooling all six mazes
into one map warps every maze onto the same H first.

Usage:
    uv run python -m eye_pre_flash.plotting.heatmaps.label.sum_norm
    uv run python -m eye_pre_flash.plotting.heatmaps.label.sum_norm --scope allplus
    uv run python -m eye_pre_flash.plotting.heatmaps.label.sum_norm --monkey Faure --bin-w 0.142857
"""

from __future__ import annotations

from eye_pre_flash.plotting.heatmaps.label import core

VISUALIZER = "sum_norm"


def main():
    core.main(__doc__, visualizer=VISUALIZER, space="unith")


if __name__ == "__main__":
    main()
