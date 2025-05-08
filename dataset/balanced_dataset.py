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
        self.tile_size = config["tile_size"]
        self.batch_size = config["batch_size"]
        self.dataset2use = config["test_seasons"]
        self.data_dir_root = config["data_dir"]
        self.image_transform = config.get("image_transform", None)
        self.point_cloud_transform = config.get("point_cloud_transform", None)
        self.train_dataset = config.get("train_on", "rmf")
        self.test_dataset = config.get("test_on", "rmf")

    def setup(self, stage=None):
        if stage == "fit":
            if len(self.train_dataset)==1:
                self.train_datasets = self.load_single_dataset("train", self.train_dataset[0])
                self.val_datasets = self.load_single_dataset("val", self.train_dataset[0])
            else:
                rmf_train = self.load_single_dataset("train", "rmf")
                ovf_train = self.load_single_dataset("train", "ovf")
                self.train_datasets = torch.utils.data.ConcatDataset([rmf_train, ovf_train])
                rmf_val = self.load_single_dataset("val", "rmf")
                ovf_val = self.load_single_dataset("val", "ovf")
                self.train_datasets = torch.utils.data.ConcatDataset([rmf_train, ovf_train])

        if stage == "test":
            self.test_datasets = self.load_single_dataset("test", self.test_dataset)

    def load_single_dataset(self, split, dataset_name):
        path = os.path.join(self.data_dir_root, f"{dataset_name}_tl_dataset", f"tile_{self.tile_size}", split)
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".npz")]
        if split=="train":
            no_aug_dataset=BalancedDataset(
                    dataset_files=files,
                    data2use=self.dataset2use,
                    tile_size=self.tile_size,
                    dataset=dataset_name,
                    image_transform=None,
                    point_cloud_transform=None,
                )
            if not (self.image_transform is None or self.point_cloud_transform is False):
                aug_img_dataset=BalancedDataset(
                    dataset_files=files,
                    data2use=self.dataset2use,
                    tile_size=self.tile_size,
                    dataset=dataset_name,
                    image_transform=self.image_transform if split == "train" else None,
                    point_cloud_transform=None,
                )
                aug_pc_dataset=BalancedDataset(
                    dataset_files=files,
                    data2use=self.dataset2use,
                    tile_size=self.tile_size,
                    dataset=dataset_name,
                    image_transform=self.image_transform if split == "train" else None,
                    point_cloud_transform=None,
                )
                return torch.utils.data.ConcatDataset(
                    [no_aug_dataset, aug_img_dataset, aug_pc_dataset]
                )
            else:
                return no_aug_dataset
        else:
            return BalancedDataset(
                dataset_files=files,
                data2use=self.dataset2use,
                tile_size=self.tile_size,
                dataset=dataset_name,
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
