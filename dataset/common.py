import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from os.path import join
from .superpixel import SuperpixelDataset

class SuperpixelDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset = config["dataset"]
        self.test_dataset = config["test_dataset"]
        self.batch_size = config["batch_size"]
        self.num_workers = 8  # config["gpus"]*2
        self.image_transform = (
            config["image_transform"] if config["image_transform"] != "None" else None
        )
        self.point_cloud_transform = config["point_cloud_transform"]
        self.aug_rotate = config["rotate"]
        self.aug_pc_norm = config["pc_normal"]
        self.data_dirs = {
            "train": join(
                config["data_dir"], "tile_128", "train", f"{config['dataset']}"
            ),
            "val": join(config["data_dir"], "tile_128", "val", f"{config['dataset']}"),
            "test": join(
                config["test_data_dir"], "tile_128", "test", f"{config['test_dataset']}"
            ),
        }

    def setup(self, stage=None):
        # Create datasets for train, validation, and test
        self.datasets = {}
        for split in ["train", "val", "test"]:
            data_dir = self.data_dirs[split]
            superpixel_files = [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".npz")
            ]
            if split == "test":
                img_mean = self.config[f"{self.config['test_dataset']}_img_mean"]
                img_std = self.config[f"{self.config['test_dataset']}_img_std"]
            else:
                img_mean = self.config[f"{self.config['dataset']}_img_mean"]
                img_std = self.config[f"{self.config['dataset']}_img_std"]
            self.datasets[split] = SuperpixelDataset(
                self.dataset if split == "test" else self.test_dataset,
                superpixel_files,
                rotate=None,
                pc_normal=self.aug_pc_norm,
                image_transform=None,
                point_cloud_transform=None,
                img_mean=img_mean,
                img_std=img_std,
            )
            if split == "train":
                if not (
                    self.image_transform is None or self.point_cloud_transform is False
                ):
                    aug_pc_dataset = SuperpixelDataset(
                        self.dataset,
                        superpixel_files,
                        rotate=self.aug_rotate,
                        pc_normal=self.aug_pc_norm,
                        image_transform=None,
                        point_cloud_transform=self.point_cloud_transform,
                        img_mean=img_mean,
                        img_std=img_std,
                    )
                    aug_img_dataset = SuperpixelDataset(
                        self.dataset,
                        superpixel_files,
                        rotate=self.aug_rotate,
                        pc_normal=self.aug_pc_norm,
                        image_transform=self.image_transform,
                        point_cloud_transform=None,
                        img_mean=img_mean,
                        img_std=img_std,
                    )
                    self.datasets["train"] = torch.utils.data.ConcatDataset(
                        [self.datasets["train"], aug_pc_dataset, aug_img_dataset]
                    )

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        # Implement custom collate function if necessary
        batch = [b for b in batch if b is not None]  # Remove None samples if any

        images = torch.stack(
            [item["images"] for item in batch]
        )  # Shape: (batch_size, num_seasons, num_channels, 128, 128)
        point_clouds = torch.stack(
            [item["point_cloud"] for item in batch]
        )  # Shape: (batch_size, num_points, 3)
        pc_feats = torch.stack(
            [item["pc_feat"] for item in batch]
        )  # Shape: (batch_size, num_points, 3)
        labels = torch.stack(
            [item["label"] for item in batch]
        )  # Shape: (batch_size, num_classes)
        per_pixel_labels = torch.stack(
            [item["per_pixel_labels"] for item in batch]
        )  # Shape: (batch_size, num_classes, 128, 128)
        nodata_masks = torch.stack(
            [item["mask"] for item in batch]
        )  # Shape: (batch_size, 128, 128)

        return {
            "images": images,
            "point_cloud": point_clouds,
            "pc_feat": pc_feats,
            "label": labels,
            "per_pixel_labels": per_pixel_labels,
            "mask": nodata_masks,
        }
