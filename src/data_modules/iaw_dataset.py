"""Pytorch Dataset for the Ikea Assembly in the Wild (IAW) dataset with features only."""

import json
from typing import Callable

import torch
from torch.utils.data import Dataset


class IAWVideoFeatureDataset(Dataset):
    """IAW video feature dataset."""

    def __init__(
        self,
        dataset_pathname: str,
        video_feature_pathname: str,
        video_feature_sampler: Callable,
        one_step_diagram_per_sample: bool = False,
    ):
        self.dataset_pathname = dataset_pathname
        self.video_feature_pathname = video_feature_pathname
        self.video_feature_sampler = video_feature_sampler

        with open(dataset_pathname) as f:
            self.dataset = json.load(f)
        self.video_data = torch.load(video_feature_pathname, map_location="cpu")
        sampled_dataset = video_feature_sampler(
            dataset=self.dataset,
            video_num_features_map={key: vf.shape[0] for key, vf in self.video_data.items()},
        )
        if one_step_diagram_per_sample:
            self.sampled_dataset = []
            for sample in sampled_dataset:
                for diagram_index in sample["gt_discrete"].keys():
                    self.sampled_dataset.append(
                        {
                            **sample,
                            "gt_discrete": {diagram_index: sample["gt_discrete"][diagram_index]},
                            "gt_continuous": {diagram_index: sample["gt_continuous"][diagram_index]},
                        }
                    )
        else:
            self.sampled_dataset = sampled_dataset

    def __getitem__(self, index) -> dict:
        sample = self.sampled_dataset[index]
        key = sample["key"]
        # load video features
        clip_start_index = sample["clip_start_index"]
        clip_end_index = sample["clip_end_index"]
        sample["clip_features"] = self.video_data[key][clip_start_index:clip_end_index]
        return sample

    def __len__(self) -> int:
        return len(self.sampled_dataset)
