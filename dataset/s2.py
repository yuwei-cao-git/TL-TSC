import torch
from torch.utils.data import Dataset
import numpy as np
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from os.path import join
from .augment import image_augment
import torchvision.transforms.v2 as transforms


class S2Dataset(Dataset):
    def __init__(
        self,
        superpixel_files,
        image_transform=None,
        img_mean=None,
        img_std=None
    ):
        self.superpixel_files = superpixel_files
        self.image_transform = image_transform
        self.transforms = transforms.Compose(
            [
                transforms.ToImage(), 
                transforms.ToDtype(torch.float32, scale=False),
                transforms.Normalize(mean=img_mean, std=img_std)
            ]
        )

    def __len__(self):
        return len(self.superpixel_files)

    def __getitem__(self, idx):
        data = np.load(self.superpixel_files[idx], allow_pickle=True)
        # Load data from the .npz file
        superpixel_images = data[
            "superpixel_images"
        ].astype(np.float32)  # Shape: (num_seasons, num_channels, 128, 128)
        label = data["label"]  # Shape: (num_classes,)
        per_pixel_labels = data["per_pixel_labels"]  # Shape: (num_classes, 128, 128)
        nodata_mask = data["nodata_mask"]  # Shape: (128, 128)

        superpixel_images = torch.from_numpy(
            superpixel_images
        ).float()  # Shape: (num_seasons, num_channels, 128, 128)
        per_pixel_labels = torch.from_numpy(
            per_pixel_labels
        ).float()  # Shape: (num_classes, 128, 128)
        nodata_mask = torch.from_numpy(nodata_mask).bool()
        
        superpixel_images = self.transforms(superpixel_images)

        # Apply transforms if needed
        if self.image_transform != None:
            superpixel_images = image_augment(superpixel_images, self.image_transform, 128)

        label = torch.from_numpy(label).float()  # Shape: (num_classes,)

        sample = {
            "images": superpixel_images,  # Padded images of shape [num_seasons, num_channels, 128, 128]
            "mask": nodata_mask,  # Padded masks of shape [num_seasons, 128, 128]
            "per_pixel_labels": per_pixel_labels,  # Tensor: (num_classes, 128, 128)
            "label": label,
        }
        return sample


class S2DataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config["batch_size"]
        self.num_workers = config["gpus"]*2
        self.image_transform = (
            config["image_transform"] if config["image_transform"] != "None" else None
        )
        self.data_dirs = {
            "train": join(
                config["data_dir"],
                "tile_128",
                "train",
                f"{config['dataset']}"
            ),
            "val": join(
                config["data_dir"],
                "tile_128",
                "val",
                f"{config['dataset']}"
            ),
            "test": join(
                config["data_dir"],
                "tile_128",
                "test",
                f"{config['dataset']}"
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
            img_mean=self.config[f"{self.config['dataset']}_img_mean"]
            img_std=self.config[f"{self.config['dataset']}_img_std"]
            self.datasets[split] = S2Dataset(
                superpixel_files,
                image_transform=None,
                img_mean=img_mean,
                img_std=img_std
            )
            if split == "train":
                if self.image_transform:
                    aug_img_dataset = S2Dataset(
                        superpixel_files,
                        image_transform=self.image_transform,
                        img_mean=img_mean,
                        img_std=img_std
                    )
                    self.datasets["train"] = torch.utils.data.ConcatDataset(
                        [self.datasets["train"], aug_img_dataset]
                    )

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False
        )