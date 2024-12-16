"""Utils for metrics.

Ground truth and prediction data structures:
For each video:
1. FrameList: index list, each element is an integer, 0 means background for that corresponding frame, 1 means diagram 1
    and so on.
2. SegmentMap: segment map, {diagram_index: [(start_frame, end_frame), ...]}
"""

from collections import defaultdict


def frame_list_to_segment_map(frame_list):
    """Convert frame_list to segment_map.

    Args:
        frame_list: (T, ), torch.int64 or int, 0 for background, 1 for diagram 1 and so on.
        return: segment_map, int, {diagram_index: [(start_frame, end_frame), ...]}, ignore background.
    """
    ret = defaultdict(list)
    start = -1
    end = -1
    current_index = -1
    for i, index in enumerate(frame_list):
        index = index.item()
        if index == 0:
            if current_index != -1:
                # end the segment when
                ret[current_index].append((start, end))
                current_index = -1
            continue
        if current_index == -1:
            # start a new segment
            current_index = index
            start = i
            end = i
        elif index == current_index:
            # continue the segment
            end = i
        else:
            # end the segment
            ret[current_index].append((start, end))
            current_index = index
            start = i
            end = i
    # add the last segment if exists
    if current_index != -1:
        ret[current_index].append((start, end))
    return dict(ret)


def segment_map_to_frame_list(segment_map, frame_count: int = None):
    """Convert segment_map to frame_list.

    Args:
    segment_map: int, {diagram_index: [(start_frame, end_frame), ...]}
    frame_count: int, if None, use the max end_frame in segment_map.
    return: frame_list, (T, ), torch.int64, 0 for background, 1 for diagram 1 and so on.
    """
    if frame_count is None:
        frame_count = max(segment[1] + 1 for segment_list in segment_map.values() for segment in segment_list)
    ret = torch.zeros(frame_count, dtype=torch.int64)
    for diagram_index, segment_list in segment_map.items():
        for segment in segment_list:
            start, end = segment
            ret[start : end + 1] = diagram_index
    return ret


def iou_frame(a_segment: tuple[int, int], b_target: tuple[int, int]):
    """Return the Intersection over Union (IoU) of two segments."""
    pred_start, pred_end = a_segment
    target_start, target_end = b_target
    intersection = max(0, min(pred_end, target_end) - max(pred_start, target_start) + 1)
    union = max(pred_end, target_end) - min(pred_start, target_start) + 1
    return intersection / union


def iou(a_segment: tuple[float, float], b_target: tuple[float, float]):
    """Return the Intersection over Union (IoU) of two segments."""
    pred_start, pred_end = a_segment
    target_start, target_end = b_target
    intersection = max(0, min(pred_end, target_end) - max(pred_start, target_start))
    union = max(pred_end, target_end) - min(pred_start, target_start)
    return intersection / union
