"""PytorchLightning LightningDataModule for the Ikea Assembly in the Wild (IAW) dataset with features only."""

import os
from typing import Literal

import pytorch_lightning as pl
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, default_collate

from src.data_modules.iaw_dataset import IAWVideoFeatureDataset
from src.data_modules.iaw_sampler import (
    IAWVideoFeatureClipSamplerBase,
    IAWVideoFeatureFullVideoClipSampler,
    IAWVideoFeatureSlideWindowClipSampler,
)


class IAWVideoFeatureDataModule(pl.LightningDataModule):
    """IAW DataModule for video features. (Image features can be loaded in training stage, use the key as index)."""

    def __init__(
        self,
        dataset_path: str,
        dataset_train_name: str = "split/train.json",
        dataset_val_name: str = "split/val.json",
        dataset_test_name: str = "split/test.json",
        one_step_diagram_per_sample: bool = False,
        # Video
        video_size: int = 224,
        video_fps: int = 30,
        video_encoder: Literal[
            "i3d",  # for youcook2
            "slowfast_r50",
            "videomae_vit_b",
            "videomae_vit_b_ssv2",
            "videomae_vit_l",
            "videomae_vit_l_resize",
            "videomae_vit_h",
            "videomaev2_vit_g",
            "videomaev2_vit_g_resize",
            "clip_vit_b_16",
            "languagebind_vit_l_14",
        ] = "slowfast_r50",
        video_dim: int = 2304,
        video_feature_train_name: str = "feature/video/train_video_{size}_{fps}_{encoder}_{dim}.pt",
        video_feature_val_name: str = "feature/video/val_video_{size}_{fps}_{encoder}_{dim}.pt",
        video_feature_test_name: str = "feature/video/test_video_{size}_{fps}_{encoder}_{dim}.pt",
        video_train_sampler: IAWVideoFeatureClipSamplerBase = IAWVideoFeatureSlideWindowClipSampler(
            window_size=128,
            stride=64,
            drop_empty_clip=True,
            last_clip_strategy="backpad",
            frames_per_second=30,
            frames_per_feature=32,
            drop_small_segment_time_threshold=0.5,
        ),
        video_full_sampler: IAWVideoFeatureClipSamplerBase = IAWVideoFeatureFullVideoClipSampler(
            frames_per_second=30,
            frames_per_feature=32,
            drop_small_segment_time_threshold=0.5,
        ),
        # Data Loader
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        shuffle: bool = True,
        prefetch_factor: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        # Build path
        self.dataset_train_pathname = os.path.join(dataset_path, dataset_train_name)
        self.dataset_val_pathname = os.path.join(dataset_path, dataset_val_name)
        self.dataset_test_pathname = os.path.join(dataset_path, dataset_test_name)
        self.video_feature_train_pathname = os.path.join(
            dataset_path,
            video_feature_train_name.format(size=video_size, fps=video_fps, encoder=video_encoder, dim=video_dim),
        )
        self.video_feature_val_pathname = os.path.join(
            dataset_path,
            video_feature_val_name.format(size=video_size, fps=video_fps, encoder=video_encoder, dim=video_dim),
        )
        self.video_feature_test_pathname = os.path.join(
            dataset_path,
            video_feature_test_name.format(size=video_size, fps=video_fps, encoder=video_encoder, dim=video_dim),
        )
        # Init dataset
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.prefetch_factor = prefetch_factor

    def setup(self, stage: str | None = None):
        # Build Datasets
        if stage in ["fit", "predict"] or stage is None:
            assert os.path.isfile(self.dataset_train_pathname), f"{self.dataset_train_pathname} does not exist."
            assert os.path.isfile(
                self.video_feature_train_pathname
            ), f"{self.video_feature_train_pathname} does not exist."
            self.train_dataset = IAWVideoFeatureDataset(
                dataset_pathname=self.dataset_train_pathname,
                video_feature_pathname=self.video_feature_train_pathname,
                video_feature_sampler=self.hparams.video_train_sampler
                if stage == "fit"
                else self.hparams.video_full_sampler,
                one_step_diagram_per_sample=self.hparams.one_step_diagram_per_sample,
            )
        if stage in ["fit", "validate", "predict"] or stage is None:
            assert os.path.isfile(self.dataset_val_pathname), f"{self.dataset_val_pathname} does not exist."
            assert os.path.isfile(self.video_feature_val_pathname), f"{self.video_feature_val_pathname} does not exist."
            self.val_dataset = IAWVideoFeatureDataset(
                dataset_pathname=self.dataset_val_pathname,
                video_feature_pathname=self.video_feature_val_pathname,
                video_feature_sampler=self.hparams.video_full_sampler,
                one_step_diagram_per_sample=self.hparams.one_step_diagram_per_sample,
            )
        if stage in ["test", "predict"] or stage is None:
            assert os.path.isfile(self.dataset_test_pathname), f"{self.dataset_test_pathname} does not exist."
            assert os.path.isfile(
                self.video_feature_test_pathname
            ), f"{self.video_feature_test_pathname} does not exist."
            self.test_dataset = IAWVideoFeatureDataset(
                dataset_pathname=self.dataset_test_pathname,
                video_feature_pathname=self.video_feature_test_pathname,
                video_feature_sampler=self.hparams.video_full_sampler,
                one_step_diagram_per_sample=self.hparams.one_step_diagram_per_sample,
            )

    @staticmethod
    def collate_fn(batch):
        ret = dict()
        for key in batch[0].keys():
            if key in [
                "key",
                "video_duration",
                "clip_start_index",
                "clip_end_index",
                "gt_discrete",
                "gt_continuous",
                "clip_features",
            ]:
                ret[key] = [item[key] for item in batch]
            else:
                ret[key] = default_collate([sample[key] for sample in batch])
        return ret

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
            shuffle=self.hparams.shuffle,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=False,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=False,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        return iter(
            CombinedLoader(
                [
                    DataLoader(
                        dataset,
                        batch_size=1,
                        num_workers=self.hparams.num_workers,
                        pin_memory=self.hparams.pin_memory,
                        drop_last=False,
                        shuffle=False,
                        prefetch_factor=self.prefetch_factor,
                        collate_fn=self.collate_fn,
                    )
                    for dataset in [self.train_dataset, self.val_dataset, self.test_dataset]
                ],
                "sequential",
            )
        )
