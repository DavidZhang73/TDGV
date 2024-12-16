"""Video Clip Samplers."""

import logging
from collections import defaultdict
from typing import Literal


class IAWVideoFeatureClipSamplerBase:
    """Base class for video feature clip sampler."""

    def __init__(
        self,
        frames_per_second: int = 10,
        frames_per_feature: float = 16.0,
        drop_small_segment_time_threshold: float = 0.8,
    ) -> None:
        """Init video feature clip sampler.

        Args:
            frames_per_second (int, optional): FPS used when extracting video features, here is used to calibrate the
                start end feature index. Defaults to 10.
            frames_per_feature (float, optional): FPF used when extracting video features, this is dependent on the video
                encoder. Here is used to calibrate the start end feature index. Defaults to 16.
            drop_small_segment_time_threshold (float, optional): when the timespan is smaller than this threshold, the
                corresponding segment will be dropped. Typical value is `frames_per_feature/frames_per_second/2`.
                Defaults to 0.8 for 16-frames video encoders at 10 FPS.
        """
        self.frames_per_second = frames_per_second
        self.frames_per_feature = frames_per_feature
        self.second_per_feature = frames_per_feature / frames_per_second
        self.drop_small_segment_time_threshold = drop_small_segment_time_threshold

    def _get_dataset_map(self, dataset: list) -> dict:
        """Get a map from furniture_video key to video duration and ground truth.

        This part of the code is dependent on the data structure of the IAW JSON file.

        Args:
            dataset (list): this is the content from the IAW dataset json file.

        Returns:
            dict: dict with keys as furniture_video key and values as a dict with keys:
                - video_duration: video duration in seconds.
                - ground_truth: dict of ground truth with keys as action and values as list of
                    (start timestamp, end timestamp) tuples w.r.t the video.
        """
        ret = {}
        for furniture in dataset:
            furniture_id = furniture["id"]
            for video in furniture["videoList"]:
                video_id = video["url"].split("https://www.youtube.com/watch?v=")[-1]
                key = f"{furniture_id}_{video_id}"

                video_duration = video["duration"]

                ret[key] = dict(video_duration=video_duration, ground_truth=defaultdict(list))
                for annotation in video["annotation"]:
                    start = annotation["start"]
                    end = annotation["end"]
                    action = annotation["action"]
                    ret[key]["ground_truth"][action].append((start, end))

        return ret

    def _assert_duration_and_num_features(self, key, video_duration, num_features, strict: bool = False):
        """Assert video duration and number of features match.

        Args:
            key (str): video key.
            video_duration (float): video duration in seconds.
            num_features (int): number of features in the video.
            strict (bool, optional): raise error if not match. Defaults to False.
        """
        if (
            abs(video_duration * self.frames_per_second - num_features * self.frames_per_feature)
            > 2 * self.frames_per_feature
        ):
            msg = (
                f"{key} video duration and number of features do not match: "
                f"{video_duration * self.frames_per_second} != {num_features * self.frames_per_feature}"
            )
            if strict:
                raise ValueError(msg)
            else:
                logging.warning(msg)

    def _generate_clip_ground_truth(
        self,
        clip_start_index: int,
        clip_end_index: int,
        video_duration: float,
        num_features: int,
        window_size: int,
        ground_truth: dict[int, list[tuple[float, float]]],
    ) -> dict[Literal["gt_discrete", "gt_continuous"], dict[int, list[tuple[int, int] | tuple[float, float]]]]:
        """Generate the corresponding ground truth for the given clip.

        Args:
            clip_start_index (int): start index of the clip. Included in the clip, closed left.
            clip_end_index (int): end index of the clip. **NOT** included in the clip, open right.
            video_duration (float): video duration in seconds.
            num_features (int): number of features in the video feature.
            window_size (int): number of features in the clip.
            ground_truth (dict[int, list[tuple[float, float]]]): dict with keys as action and values as a list of
                (start timestamp, end timestamp) tuples w.r.t the video.

        Returns:
            dict[Literal["gt_discrete", "gt_continuous"], dict[int, list[tuple[int, int] | tuple[float, float]]]]:
                dict with keys as "gt_discrete" and "gt_continuous" and values as a dict with keys as action and values
                as a list of (start index, end index) or (start percentage, end percentage) tuples w.r.t the clip
                (not the video).
        """
        gt_discrete = defaultdict(list)
        gt_continuous = defaultdict(list)
        drop_count = 0
        clip_start_timestamp = clip_start_index * self.second_per_feature
        clip_end_timestamp = min(clip_end_index * self.second_per_feature, video_duration)
        clip_duration = clip_end_timestamp - clip_start_timestamp
        for diagram_index, gt_timestamps in ground_truth.items():
            for start_timestamp, end_timestamp in gt_timestamps:
                start_index = round(start_timestamp / self.second_per_feature)
                end_index = min(int(end_timestamp / self.second_per_feature) + 1, num_features)  # open right
                if not (end_index <= clip_start_index or start_index >= clip_end_index):
                    # The following are relative to the clip, not the video.
                    _start_index = max(0, start_index - clip_start_index)
                    _end_index = min(end_index - clip_start_index, window_size)
                    _start_timestamp = max(0, start_timestamp - clip_start_timestamp)
                    _end_timestamp = min(end_timestamp - clip_start_timestamp, clip_duration)
                    if _end_index < _start_index:
                        raise ValueError(f"segment end < start: {_end_index} < {_start_index}")
                    elif _end_index == _start_index:
                        if _end_timestamp - _start_timestamp <= self.drop_small_segment_time_threshold:
                            drop_count += 1
                            continue
                        if _end_index == window_size:
                            _start_index -= 1
                            _start_timestamp -= self.second_per_feature
                        else:
                            _end_index += 1
                            _end_timestamp += self.second_per_feature
                    elif _end_timestamp - _start_timestamp <= self.drop_small_segment_time_threshold:
                        drop_count += 1
                        continue
                    gt_discrete[diagram_index].append((_start_index, _end_index))
                    gt_continuous[diagram_index].append(
                        (_start_timestamp / clip_duration, _end_timestamp / clip_duration)
                    )
        if drop_count:
            logging.info(
                f"Dropped {drop_count} segments because they are too short w.r.t the threshold "
                f"{self.drop_small_segment_time_threshold} in seconds."
            )
        return dict(
            gt_discrete=gt_discrete,
            gt_continuous=gt_continuous,
        )

    def __call__(self, dataset: list, video_num_features_map: dict[str, int]) -> list[dict]:
        """Sample video clips from the dataset.

        Args:
            dataset (list): this is the content from the IAW dataset json file.
            video_num_features_map (dict[str, int]): dict with keys as furniture_video key and values as video duration
                in number of frames.

        Returns:
            list: list of dict with keys:
                - key: video key.
                - video_duration: video duration in seconds.
                - clip_start_index: start index of the clip. Included in the clip, closed left.
                - clip_end_index: end index of the clip. **NOT** included in the clip, open right.
                - gt_discrete: dict of ground truth with keys as action and values as a list of
                    (start index:int, end index:int) tuples w.r.t the clip (not the video).
                - gt_continuous: dict of ground truth with keys as action and values as a list of
                    (start percentage:float, end percentage:float) tuples w.r.t the clip (not the video).
        """
        raise NotImplementedError("__call__ is not implemented.")


class IAWVideoFeatureFullVideoClipSampler(IAWVideoFeatureClipSamplerBase):
    def __call__(
        self,
        dataset: list,
        video_num_features_map: dict[str, int],
    ) -> list[dict]:
        dataset_map = self._get_dataset_map(dataset)
        ret = []
        for key, video_data in dataset_map.items():
            video_duration = video_data["video_duration"]
            num_features = video_num_features_map[key]
            self._assert_duration_and_num_features(key, video_duration, num_features)
            gt = self._generate_clip_ground_truth(
                clip_start_index=0,
                clip_end_index=num_features,
                video_duration=video_duration,
                num_features=num_features,
                window_size=num_features,
                ground_truth=video_data["ground_truth"],
            )
            ret.append(
                dict(
                    key=key,
                    video_duration=video_duration,
                    clip_start_index=0,
                    clip_end_index=num_features,
                    **gt,
                )
            )
        return ret


class IAWVideoFeatureSlideWindowClipSampler(IAWVideoFeatureClipSamplerBase):
    def __init__(
        self,
        window_size: int | list[int],
        stride: int | list[int],
        drop_empty_clip: bool = True,
        last_clip_strategy: Literal["drop", "backpad"] = "backpad",
        **kwargs,
    ) -> None:
        """Slide window clip sampler.

        Args:
            window_size (int | list[int]): number of features in a clip.
            stride (int | list[int]): number of features to skip.
            drop_empty_clip (bool, optional): drop empty clips. Defaults to True.
            last_clip_strategy (Literal["drop", "backpad"], optional): strategy for the last clip.
                Defaults to "backpad".
            kwargs: other arguments for IAWVideoFeatureClipSamplerBase.
        """
        super().__init__(**kwargs)
        if isinstance(window_size, int):
            window_size = [window_size]
        if isinstance(stride, int):
            stride = [stride]
        self.window_size = window_size
        self.stride = stride
        self.drop_empty_clip = drop_empty_clip
        self.last_clip_strategy = last_clip_strategy

    def __call__(self, dataset: list, video_num_features_map: dict[str, int]) -> list[dict]:
        ret = []
        full_video_keys = set()
        dataset_map = self._get_dataset_map(dataset)
        for window_size, stride in zip(self.window_size, self.stride):
            for key, video_data in dataset_map.items():
                # skip full video because it is already in the list
                if key in full_video_keys:
                    continue
                video_duration = video_data["video_duration"]
                num_features = video_num_features_map[key]
                ground_truth = video_data["ground_truth"]
                # video is shorter than window size, return full video
                if num_features < window_size:
                    gt = self._generate_clip_ground_truth(
                        0, num_features, video_duration, num_features, num_features, ground_truth
                    )
                    ret.append(
                        dict(
                            key=key,
                            clip_start_index=0,
                            clip_end_index=num_features,
                            video_duration=video_duration,
                            **gt,
                        )
                    )
                    full_video_keys.add(key)
                    continue
                # slide window
                clip_start_index = 0
                clip_end_index = window_size
                while clip_end_index <= num_features:
                    gt = self._generate_clip_ground_truth(
                        clip_start_index, clip_end_index, video_duration, num_features, window_size, ground_truth
                    )
                    # drop empty clip
                    if not (self.drop_empty_clip and len(gt["gt_continuous"]) == 0):
                        ret.append(
                            dict(
                                key=key,
                                clip_start_index=clip_start_index,
                                clip_end_index=clip_end_index,
                                video_duration=video_duration,
                                **gt,
                            )
                        )
                    clip_start_index += stride
                    clip_end_index += stride
                # last clip - backpad
                if clip_start_index < num_features:
                    if self.last_clip_strategy == "backpad":
                        gt = self._generate_clip_ground_truth(
                            num_features - window_size,
                            num_features,
                            video_duration,
                            num_features,
                            window_size,
                            ground_truth,
                        )
                        # drop empty clip
                        if not (self.drop_empty_clip and len(gt["gt_continuous"]) == 0):
                            ret.append(
                                dict(
                                    key=key,
                                    clip_start_index=num_features - window_size,
                                    clip_end_index=num_features,
                                    video_duration=video_duration,
                                    **gt,
                                )
                            )
                    # last clip - drop
                    elif self.last_clip_strategy == "drop":
                        continue
        return ret
