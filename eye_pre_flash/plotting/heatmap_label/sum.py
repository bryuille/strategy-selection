"""Pooled pre-fixation gaze heatmaps by decoded neural label.

Sums raw sample counts over the trials the neural clustering labelled
hierarchical and the trials it labelled sequential, all six mazes mixed into
both pools. Screen degrees, no unit-H warp, no per-trial or L1 normalization.

The maze-grouped counterpart is ``heatmap_maze.sum``; see
``heatmap_label.core`` for what the scopes mean.

Usage:
    uv run python -m eye_pre_flash.plotting.heatmap_label.sum
    uv run python -m eye_pre_flash.plotting.heatmap_label.sum --scope publication
    uv run python -m eye_pre_flash.plotting.heatmap_label.sum --monkey Faure --bin-deg 1
"""

from __future__ import annotations

from eye_pre_flash.plotting.heatmap_label import core

VISUALIZER = "sum"


def main():
    core.main(__doc__, visualizer=VISUALIZER, space="deg")


if __name__ == "__main__":
    main()
