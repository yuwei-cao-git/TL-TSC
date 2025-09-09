import torch
from torch.utils.data import Dataset
import numpy as np
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset
from os.path import join
from .augment import pointCloudTransform, image_augment, normalize_point_cloud, center_point_cloud
import torchvision.transforms.v2 as transforms
import open3d as o3d
from pytorch_lightning.utilities.combined_loader import CombinedLoader
import yaml

class SuperpixelDataset(Dataset):
    def __init__(self, superpixel_files, rotate=None, pc_normal=None,
                image_transform=None, point_cloud_transform=None,
                img_mean=None, img_std=None, sampling=False,
                region_key: str = "A"):
        
        self.superpixel_files = superpixel_files
        self.image_transform = image_transform
        self.point_cloud_transform = point_cloud_transform
        self.rotate = rotate
        self.normal = pc_normal
        self.sampling = sampling
        self.region_key = region_key

        self.transforms = transforms.Compose(
            [
                transforms.ToImage(), 
                transforms.ToDtype(torch.float32, scale=True),
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
        ].astype(np.float32) # / 65535.0  # Shape: (num_seasons, num_channels, 128, 128)
        coords = data["point_cloud"]  # Shape: (7168, 3)
        label = data["label"]  # Shape: (num_classes,)
        nodata_mask = data["nodata_mask"]  # Shape: (128, 128)

        superpixel_images = torch.from_numpy(
            superpixel_images
        ).float()  # Shape: (num_seasons, num_channels, 128, 128)
        nodata_mask = torch.from_numpy(nodata_mask).bool()
        
        superpixel_images = self.transforms(superpixel_images)

        # Apply transforms if needed
        if self.image_transform != None:
            superpixel_images = image_augment(superpixel_images, self.image_transform, 128)
        
        centred_coords = center_point_cloud(coords)
        if self.normal:
            # Convert numpy array to Open3D PointCloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords)

            # Estimate normals (change radius/knn depending on density)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=16))
            feats = np.asarray(pcd.normals)  # Shape: (N, 3)
        else:
            norm_coords = normalize_point_cloud(coords)
            feats = norm_coords
        
        # Apply point cloud transforms if any
        if self.point_cloud_transform:
            centred_coords, feats, label = pointCloudTransform(
                centred_coords, pc_feat=feats, target=label, rot=self.rotate
            )

        # After applying transforms
        feats = torch.from_numpy(feats).float()  # Shape: (7168, 3)
        centred_coords = torch.from_numpy(centred_coords).float()  # Shape: (7168, 3)
        label = torch.from_numpy(label).float()  # Shape: (num_classes,)

        sample = {
            "images": superpixel_images,  # Padded images of shape [num_seasons, num_channels, 128, 128]
            "mask": nodata_mask,  # Padded masks of shape [num_seasons, 128, 128]
            "point_cloud": centred_coords,
            "pc_feat": feats,
            "label": label,
            "region": self.region_key, 
        }
        return sample


class RegionSpec:
    def __init__(self, name, root_dir, img_mean, img_std,
                dataset_tag: str):             # e.g., "RMF" or "OVF"
        self.name = name              # 'A' or 'B'
        self.root_dir = root_dir      # base path for this region
        self.img_mean = img_mean
        self.img_std = img_std
        self.dataset_tag = dataset_tag  # used in path building

class RegionDataModule(LightningDataModule):
    def __init__(self, config, spec: RegionSpec):
        super().__init__()
        self.config = config
        self.spec = spec
        self.batch_size = config["batch_size"]
        self.num_workers = config["gpus"] * 2
        self.image_transform = (config["image_transform"]
                                if config["image_transform"] != "None" else None)
        self.point_cloud_transform = config["point_cloud_transform"]
        self.aug_rotate = config["rotate"]
        self.aug_pc_norm = config["pc_normal"]
        self.fps = config["fps"]

        self.data_dirs = {
            "train": join(spec.root_dir, "tile_128", "train", spec.dataset_tag),
            "val":   join(spec.root_dir, "tile_128", "val",   spec.dataset_tag),
            "test":  join(spec.root_dir, "tile_128", "test",  spec.dataset_tag),
        }
        self.datasets = {}

    def setup(self, stage=None):
        for split in ["train", "val", "test"]:
            data_dir = self.data_dirs[split]
            files = [join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]

            base = SuperpixelDataset(
                files,
                rotate=None,
                pc_normal=self.aug_pc_norm,
                image_transform=None,
                point_cloud_transform=None,
                img_mean=self.spec.img_mean,
                img_std=self.spec.img_std,
                sampling=self.fps,
                region_key=self.spec.name,           # <<< important
            )

            if split == "train" and (self.image_transform or self.point_cloud_transform):
                ds_pc = SuperpixelDataset(
                    files,
                    rotate=self.aug_rotate,
                    pc_normal=self.aug_pc_norm,
                    image_transform=None,
                    point_cloud_transform=self.point_cloud_transform,
                    img_mean=self.spec.img_mean,
                    img_std=self.spec.img_std,
                    sampling=self.fps,
                    region_key=self.spec.name,
                )
                ds_img = SuperpixelDataset(
                    files,
                    rotate=self.aug_rotate,
                    pc_normal=self.aug_pc_norm,
                    image_transform=self.image_transform,
                    point_cloud_transform=None,
                    img_mean=self.spec.img_mean,
                    img_std=self.spec.img_std,
                    sampling=self.fps,
                    region_key=self.spec.name,
                )
                self.datasets[split] = ConcatDataset([base, ds_pc, ds_img])
            else:
                self.datasets[split] = base

    def _collate(self, batch):
        batch = [b for b in batch if b is not None]
        return {
            "images": torch.stack([b["images"] for b in batch]),
            "point_cloud": torch.stack([b["point_cloud"] for b in batch]),
            "pc_feat": torch.stack([b["pc_feat"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]),
            "mask": torch.stack([b["mask"] for b in batch]),
            "region": self.spec.name,  # single string per batch
        }

    def train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                        shuffle=True, num_workers=self.num_workers,
                        drop_last=True, collate_fn=self._collate)

    def val_dataloader(self):
        return DataLoader(self.datasets["val"], batch_size=self.batch_size,
                        shuffle=False, num_workers=self.num_workers,
                        drop_last=True, collate_fn=self._collate)

    def test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                        shuffle=False, num_workers=self.num_workers,
                        drop_last=False, collate_fn=self._collate)
        
        

class MultiRegionDataModule(LightningDataModule):
    def __init__(self, dm_A: RegionDataModule, dm_B: RegionDataModule):
        super().__init__()
        self.dm_A = dm_A
        self.dm_B = dm_B

    def setup(self, stage=None):
        self.dm_A.setup(stage)
        self.dm_B.setup(stage)

    def train_dataloader(self):
        return CombinedLoader(
            {"A": self.dm_A.train_dataloader(), "B": self.dm_B.train_dataloader()},
            mode="max_size_cycle"
        )

    def val_dataloader(self):
        return CombinedLoader(
            {"A": self.dm_A.val_dataloader(), "B": self.dm_B.val_dataloader()},
            mode="max_size_cycle"
        )

    def test_dataloader(self):
        return CombinedLoader(
            {"A": self.dm_A.test_dataloader(), "B": self.dm_B.test_dataloader()},
            mode="max_size_cycle"
        )

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_multi_region_dm(cfg_path_A: str, cfg_path_B: str):
    cfg_A = load_yaml(cfg_path_A)   # e.g., config_rmf.yaml (Region A, 9 classes)
    cfg_B = load_yaml(cfg_path_B)   # e.g., config_ovf.yaml (Region B, 6-class head)

    # Region A (no aggregation)
    spec_A = RegionSpec(
        name='A',
        root_dir=cfg_A["data_dir"],
        dataset_tag=cfg_A["dataset"],    # 'RMF' (for example)
        img_mean=cfg_A[f"{cfg_A['dataset']}_img_mean"],
        img_std=cfg_A[f"{cfg_A['dataset']}_img_std"]
    )
    # Region B (aggregate 11->6 if NPZ labels are 11-class)
    spec_B = RegionSpec(
        name='B',
        root_dir=cfg_B["data_dir"],
        dataset_tag=cfg_B["dataset"],    # 'OVF' (for example)
        img_mean=cfg_B[f"{cfg_B['dataset']}_img_mean"],
        img_std=cfg_B[f"{cfg_B['dataset']}_img_std"]
    )

    dm_A = RegionDataModule(cfg_A, spec_A)
    dm_B = RegionDataModule(cfg_B, spec_B)
    return MultiRegionDataModule(dm_A, dm_B), cfg_A, cfg_B

