import logging
from abc import abstractmethod
from typing import Callable

import numpy as np
import torch
from torchmetrics import Metric


class OverlappingRatio(Metric):
    is_differentiable = False
    higher_is_better = False

    def __init__(self, **kwargs):
        """Metric to calculate the overlapping ratio among prediction timespans.

        Args:
            kwargs: Additional arguments for Metric.
        """
        super().__init__(**kwargs)
        self.add_state("overlapping_ratio", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_segment_map: dict, target_segment_map: dict):
        """Update.

        Note that we only calculate the top 1 timespan prediction for each diagram.

        Args:
            pred_segment_map (dict): {diagram_index: [(start_frame, end_frame), ...]}
            target_segment_map (dict): {diagram_index: [(start_frame, end_frame), ...]}
        """
        timespan_list = []
        for diagram_index in pred_segment_map:
            pass

    def compute(self):
        if self.count == 0:
            return 0.0
        return self.overlapping_duration / self.count


if __name__ == "__main__":
    overlapping_duration = OverlappingRatio()
    pred_segment_map = {
        0: [(0, 10), (20, 30)],
        1: [(0, 10), (20, 30)],
    }
    target_segment_map = {
        0: [(0, 10), (20, 30)],
        1: [(0, 10), (20, 30)],
    }
    overlapping_duration.update(pred_segment_map, target_segment_map)
    print(overlapping_duration.compute())
