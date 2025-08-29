import torch
from torch.utils.data import Dataset
import numpy as np
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from .augment import (
    pointCloudTransform,
    image_augment,
)
import open3d as o3d

class BalancedDataset(Dataset):
    def __init__(
        self,
        dataset_files,
        data2use,
        tile_size,
        image_transform=None,
        point_cloud_transform=None,
        img_mean=None,
        img_std=None,
    ):
        self.dataset_files = dataset_files
        self.images_list = data2use
        self.tile_size = tile_size
        self.image_aug = image_transform
        self.point_cloud_aug = point_cloud_transform

        # Create a transform to resize and normalize the input images
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(), 
                transforms.Normalize(mean=img_mean, std=img_std)
            ]
        )

    def __len__(self):
        return len(self.dataset_files)

    def __getitem__(self, idx):
        # Load data from the .npz file
        data = np.load(self.dataset_files[idx], allow_pickle=True)
        
        # Select the images based on the data2use list
        # images = [self.transforms(data[image_key]) for image_key in self.images_list]
        images = [self.transforms(data[image_key]) for image_key in self.images_list]
        
        # on-the-fly augment
        if self.image_aug != None:
            images = [image_augment(image, self.image_aug, tile_size=self.tile_size) for image in images]
        
        # per-pixel class-proportion labels
        per_pixel_labels = data["pixel_labels"].transpose(2, 0, 1)
        per_pixel_labels = torch.from_numpy(per_pixel_labels).float()  # (C,H,W)
        
        # validity mask
        mask = torch.from_numpy(data["valid_mask"]).bool() # (H,W)
        
        # point cloud + plot label
        point_cloud = data["point_cloud"]  # Shape: (7168, 6)
        plot_label = data["plot_label"]  # Shape: (num_classes,)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])

        # Estimate normals (change radius/knn depending on density)
        #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=16))
        #feats = np.asarray(pcd.normals)  # Shape: (N, 3)
        #feats = np.hstack((point_cloud[:, 3:],feats))

        # Apply point cloud augment
        xyz, pc_feat, label = pointCloudTransform(
            xyz=point_cloud[:, :3],
            pc_feat=point_cloud[:, 3:],
            target=plot_label,
        )
        xyz = torch.from_numpy(xyz).float()  # Shape: (7168, 3)
        pc_feat = torch.from_numpy(pc_feat).float()  # Shape: (7168, 3)
        label = torch.from_numpy(label).float()  # Shape: (num_classes,)

        return {
            "images": images,  # Padded images of shape [num_seasons, num_channels, tile_size, tile_size]
            "mask": mask,  
            "per_pixel_labels": per_pixel_labels,
            "point_cloud": xyz,
            "pc_feat": pc_feat,
            "label": label,
        }


class BalancedDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.img_mean = config[f"{self.cfg['dataset']}_img_mean"]
        self.img_std  = config[f"{self.cfg['dataset']}_img_std"]
        self.batch_size = config["batch_size"]

    def load_dataset(self, split, dataset_name, apply_pc_aug=False, apply_img_aug=False):
        files = sorted(
            os.path.join(self.cfg["data_dir"], f"{dataset_name}_tl_dataset", f"tile_{self.cfg['tile_size']}", split, f)
            for f in os.listdir(os.path.join(self.cfg["data_dir"], f"{dataset_name}_tl_dataset", f"tile_{self.cfg['tile_size']}", split))
            if f.endswith(".npz")
        )
        return BalancedDataset(
            dataset_files=files,
            data2use=self.cfg[f"{dataset_name}_season_map"],
            tile_size=self.cfg["tile_size"],
            image_transform=(self.cfg.get("image_transform") if (split=="train" and apply_img_aug) else None),
            point_cloud_transform=(self.cfg.get("point_cloud_transform") if (split=="train" and apply_pc_aug) else None),
            img_mean=self.img_mean,
            img_std=self.img_std,
        )
        
    def setup(self, stage=None):
        if stage == "fit":
            self.train_datasets = torch.utils.data.ConcatDataset([self.load_dataset("train", self.cfg['dataset']), self.load_dataset("train", self.cfg['dataset'], apply_img_aug=True), self.load_dataset("train", self.cfg['dataset'], apply_pc_aug=True)])
            self.val_datasets = self.load_dataset("val", self.cfg['dataset'])
        if stage == "test":
            self.test_datasets = self.load_dataset("test", self.cfg['dataset'])
            
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
