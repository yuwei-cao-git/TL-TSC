import torch
from torch.utils.data import Dataset
import numpy as np
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from os.path import join
import torchvision.transforms.v2 as transforms
from .augment import (
    pointCloudTransform,
    image_augment,
)

class BalancedDataset(Dataset):
    def __init__(
        self,
        dataset_files,
        data2use,
        tile_size,
        dataset='ovf',
        image_transform=None,
        point_cloud_transform=None,
    ):
        self.dataset = dataset
        self.dataset_files = dataset_files
        self.images_list = data2use
        self.image_transform = image_transform
        self.point_cloud_transform = point_cloud_transform
        self.tile_size = tile_size

        # Create a transform to resize and normalize the input images
        self.transforms = transforms.Compose(
            [
                transforms.ToImage(),  # Convert to tensor and handle HWC to CHW
                transforms.ToDtype(torch.float32, scale=True) if self.dataset=="rmf" else transforms.ToTensor(),  # Convert to float32 and scale to [0, 1]
            ]
        )

    def __len__(self):
        return len(self.dataset_files)

    def __getitem__(self, idx):
        # Load data from the .npz file
        data = np.load(self.dataset_files[idx], allow_pickle=True)
        # Select the images based on the data2use list
        if self.dataset =="ovf":
            images = [np.nan_to_num(np.where(np.logical_or(np.isinf(data[k]), data[k] == 255.0), np.nan, data[k]), nan=1.0) for k in self.images_list]
            images = [self.transforms(img) for img in images]
        else:
            images = [self.transforms(data[image_key]) for image_key in self.images_list]
        # images = torch.stack( images, axis=0)  # Shape: (num_seasons, num_channels, tile_size, tile_size)

        # Apply transforms if needed
        if self.image_transform != None:
            images = [
                image_augment(image, self.image_transform, tile_size=self.tile_size)
                for image in images
            ]

        per_pixel_labels = data["pixel_labels"].transpose(2, 0, 1)
        per_pixel_labels = torch.from_numpy(
            per_pixel_labels
        ).float()  # Shape: (num_classes, tile_size, tile_size)
        nodata_mask = data[
            "valid_mask"
        ]  # Shape: (32, 32)  # Assuming 'valid_mask' is the correct key
        nodata_mask = torch.from_numpy(nodata_mask).bool()

        point_cloud = data["point_cloud"]  # Shape: (7168, 6)

        plot_label = data["plot_label"]  # Shape: (num_classes,)

        # Apply point cloud transforms if any
        xyz, pc_feat, label = pointCloudTransform(
            xyz=point_cloud[:, :3],
            pc_feat=point_cloud[:, 3:],
            target=plot_label,
        )

        # After applying transforms
        xyz = torch.from_numpy(xyz).float()  # Shape: (7168, 3)
        pc_feat = torch.from_numpy(pc_feat).float()  # Shape: (7168, 3)
        label = torch.from_numpy(label).float()  # Shape: (num_classes,)

        sample = {
            "images": images,  # Padded images of shape [num_seasons, num_channels, tile_size, tile_size]
            "nodata_mask": nodata_mask,  # Padded masks of shape [num_seasons, tile_size, tile_size]
            "per_pixel_labels": per_pixel_labels,  # Tensor: (num_classes, tile_size, tile_size)
            "point_cloud": xyz,
            "pc_feat": pc_feat,
            "label": label,
        }
        return sample


class BalancedDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset = config["dataset"]
        self.batch_size = config["batch_size"]
        self.tile_size = config["tile_size"]
        self.image_transform = (
            config["img_transforms"] if config["img_transforms"] != "None" else None
        )
        self.point_cloud_transform = config["pc_transforms"]
        self.data_dirs = {
            "train": join(
                config["data_dir"],
                f"tile_{self.tile_size}",
                "train",
            ),
            "val": join(
                config["data_dir"],
                f"tile_{self.tile_size}",
                "val",
            ),
            "test": join(
                config["data_dir"],
                f"tile_{self.tile_size}",
                "test",
            ),
        }
        self.dataset2use = config["seasons"]

    def setup(self, stage=None):
        # Create datasets for train, validation, and test
        self.datasets = {}
        if stage == "fit":
            train_superpixel_files = [
                os.path.join(self.data_dirs["train"], f)
                for f in os.listdir(self.data_dirs["train"])
                if f.endswith(".npz")
            ]
            val_superpixel_files = [
                os.path.join(self.data_dirs["val"], f)
                for f in os.listdir(self.data_dirs["val"])
                if f.endswith(".npz")
            ]
            self.train_datasets = BalancedDataset(
                train_superpixel_files,
                data2use=self.dataset2use,
                dataset=self.dataset,
                tile_size=self.tile_size,
                point_cloud_transform=None,
            )
            if not (self.image_transform is None or self.point_cloud_transform is False):
                aug_dataset = BalancedDataset(
                    train_superpixel_files,
                    data2use=self.dataset2use,
                    tile_size=self.tile_size,
                    dataset=self.dataset,
                    image_transform=self.image_transform,
                    point_cloud_transform=self.point_cloud_transform,
                )
                self.train_datasets = torch.utils.data.ConcatDataset(
                    [self.train_datasets, aug_dataset]
                )
                self.val_datasets = BalancedDataset(
                    val_superpixel_files,
                    data2use=self.dataset2use,
                    dataset=self.dataset,
                    tile_size=self.tile_size,
                    image_transform=None,
                    point_cloud_transform=None,
                )
        if stage == "test":
            test_superpixel_files = [
                os.path.join(self.data_dirs["test"], f)
                for f in os.listdir(self.data_dirs["test"])
                if f.endswith(".npz")
            ]
            self.test_datasets = BalancedDataset(
                test_superpixel_files,
                data2use=self.dataset2use,
                tile_size=self.tile_size,
                dataset=self.dataset,
                image_transform=None,
                point_cloud_transform=None,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_datasets,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_datasets,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=8
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_datasets,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8
        )

    """
    def collate_fn(self, batch):
        # Implement custom collate function if necessary
        batch = [b for b in batch if b is not None]  # Remove None samples if any

        images = torch.stack(
            [item["images"] for item in batch]
        )  # Shape: (batch_size, num_seasons, num_channels, tile_size, tile_size)
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
        )  # Shape: (batch_size, num_classes, tile_size, tile_size)
        nodata_masks = torch.stack(
            [item["nodata_mask"] for item in batch]
        )  # Shape: (batch_size, tile_size, tile_size)

        return {
            "images": images,
            "point_cloud": point_clouds,
            "pc_feat": pc_feats,
            "label": labels,
            "per_pixel_labels": per_pixel_labels,
            "nodata_mask": nodata_masks,
        }
"""
