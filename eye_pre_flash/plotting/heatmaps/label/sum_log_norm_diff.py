"""Log-scale difference of pooled unit-H heatmaps by decoded label.

Same raw summed counts as ``heatmap_label.sum_log_norm``, displayed as
``log1p(count_hier) − log1p(count_seq)``.

Usage:
    uv run python -m eye_pre_flash.plotting.heatmaps.label.sum_log_norm_diff
    uv run python -m eye_pre_flash.plotting.heatmaps.label.sum_log_norm_diff --scope all
    uv run python -m eye_pre_flash.plotting.heatmaps.label.sum_log_norm_diff --monkey Faure --bin-w 0.142857
"""

from __future__ import annotations

from eye_pre_flash.plotting.heatmaps.label import core

VISUALIZER = "sum_log_norm_diff"


def main():
    core.main(__doc__, visualizer=VISUALIZER, space="unith", use_log=True, diff=True)


if __name__ == "__main__":
    main()
