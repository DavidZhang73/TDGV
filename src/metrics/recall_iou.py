import logging
from abc import abstractmethod
from typing import Callable

import numpy as np
import torch
from torchmetrics import Metric

from src.metrics.utils import frame_list_to_segment_map, iou, iou_frame


class RecallAtIoUBase(Metric):
    is_differentiable = False
    higher_is_better = True

    def __init__(
        self,
        k: int | list[int],
        iou_threshold: float | list[float],
        count_missing: bool = True,
        return_mIoU: bool = True,
        return_avgIoU: bool = True,
        **kwargs,
    ):
        """Base class for Recall@IoU metric.

        Args:
            k (int | list[int]): Top k predictions to consider.
            iou_threshold (float | list[float]): IoU threshold.
            count_missing (bool, optional): Whether to count missing diagram predictions in denominator. Defaults to
                                            True.
            return_mIoU (bool, optional): Whether to return mean IoU. Defaults to True.
            return_avgIoU (bool, optional): Whether to return average of all  IoUs for each k. Defaults to True.
            kwargs: Additional arguments for Metric.
        """
        super().__init__(**kwargs)
        if isinstance(k, int):
            k = [k]
        if isinstance(iou_threshold, float):
            iou_threshold = [iou_threshold]
        self.k = k
        self.iou_threshold = iou_threshold
        self.count_missing = count_missing
        self.return_mIoU = return_mIoU
        self.return_avgIoU = return_avgIoU
        for _k in self.k:
            for _iou_threshold in self.iou_threshold:
                self.add_state(f"{_k}_{_iou_threshold}", torch.tensor(0.0), dist_reduce_fx="sum")
        if self.return_mIoU:
            self.add_state("mIoU", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")

    @abstractmethod
    def iou_func(self, a_segment, b_target):
        pass

    @abstractmethod
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        pass

    def _update(self, pred_segment_map: dict, target_segment_map: dict):
        for target_digram_index, target_segments in target_segment_map.items():
            if target_digram_index not in pred_segment_map:
                if self.count_missing:
                    self.count += len(target_segments)
                else:
                    logging.warning(f"Diagram {target_digram_index} not found in pred_segment_map.")
                continue
            for target_segment in target_segments:
                self.count += 1
                iou_rank_list = [
                    self.iou_func(pred_segment, target_segment)
                    for pred_segment in pred_segment_map[target_digram_index]
                ]
                if len(iou_rank_list) == 0:
                    continue
                if self.return_mIoU:
                    self.mIoU += iou_rank_list[0]
                for _k in self.k:
                    top_k_iou_rank_list = iou_rank_list[: min(_k, len(iou_rank_list))]
                    for _iou_threshold in self.iou_threshold:
                        if len([iou for iou in top_k_iou_rank_list if iou >= _iou_threshold]):
                            accumulator = getattr(self, f"{_k}_{_iou_threshold}")
                            accumulator += 1

    def compute(self):
        ret = {}
        for _k in self.k:
            avg_iou = 0.0
            for _iou_threshold in self.iou_threshold:
                accumulator = getattr(self, f"{_k}_{_iou_threshold}")
                value = accumulator / self.count
                logging.info(f"Recall@{_k},IoU@{_iou_threshold}={value}")
                ret[f"R@{_k}_IoU@{_iou_threshold}"] = value
                if self.return_avgIoU:
                    avg_iou += value
            if self.return_avgIoU:
                logging.info(f"Recall@{_k},avgIoU={avg_iou / len(self.iou_threshold)}")
                ret[f"R@{_k}_avgIoU"] = avg_iou / len(self.iou_threshold)
        if self.return_mIoU:
            logging.info(f"mIoU={self.mIoU / self.count}")
            ret["mIoU"] = self.mIoU / self.count
        return ret


class RecallAtIoUSegmentMap(RecallAtIoUBase):
    def iou_func(self, a_segment, b_target):
        return iou(a_segment, b_target)

    def update(self, preds: dict, target: dict):
        """Update.

        Args:
            preds (dict): {diagram_index: [(start_frame, end_frame), ...]}
            target (dict): {diagram_index: [(start_frame, end_frame), ...]}
        """
        self._update(preds, target)


class RecallAtIoUFrameList(RecallAtIoUBase):
    def __init__(
        self,
        segment_fn: Callable = frame_list_to_segment_map,
        *args,
        **kwargs,
    ):
        """Recall@IoU metric for frame list.

        Args:
            segment_fn (Callable, optional): Segment function. Defaults to frame_list_to_segment_map.
            args: Additional arguments for RecallAtIoUBase.
            kwargs: Additional arguments for RecallAtIoUBase.
        """
        super().__init__(*args, **kwargs)
        self.segment_fn = segment_fn

    def iou_func(self, a_segment, b_target):
        return iou_frame(a_segment, b_target)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update.

        Args:
            preds (torch.Tensor): (batch_size, T) or (T, ), torch.int64
            target (torch.Tensor): (batch_size, T) or (T, ), torch.int64
        """
        assert len(preds.shape) == 1 or len(preds.shape) == 2, f"Invalid shape: {preds.shape}"
        assert preds.shape == target.shape, f"Invalid shape: preds {preds.shape} != target {target.shape}"
        assert preds.dtype == torch.int64, f"Invalid preds dtype: {preds.dtype}"
        assert target.dtype == torch.int64, f"Invalid target dtype: {target.dtype}"
        if len(preds.shape) == 1:
            preds = preds.unsqueeze(0)
            target = target.unsqueeze(0)
        batch_size = target.shape[0]
        for i in range(batch_size):
            target_segment_map = self.segment_fn(target[i])
            pred_segment_map = self.segment_fn(preds[i])
            self._update(pred_segment_map, target_segment_map)
