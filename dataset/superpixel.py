import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from os.path import join
from .augment import pointCloudTransform, image_augment
import torchvision.transforms.v2 as transforms
import open3d as o3d


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point
class SuperpixelDataset(Dataset):
    def __init__(
        self,
        superpixel_files,
        rotate=None,
        pc_normal=None,
        image_transform=None,
        point_cloud_transform=None,
        img_mean=None,
        img_std=None,
        sampling=False
    ):
        self.superpixel_files = superpixel_files
        self.image_transform = image_transform
        self.point_cloud_transform = point_cloud_transform
        self.rotate = rotate
        self.normal = pc_normal
        self.sampling = sampling

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
        ]  # Shape: (num_seasons, num_channels, 128, 128)
        coords = data["point_cloud"]  # Shape: (7168, 3)
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
        if self.normal:
            min_coords = np.min(coords, axis=0)
            max_coords = np.max(coords, axis=0)

            center = (min_coords + max_coords) / 2.0
            coords_centered = coords - center

            z_range = max_coords[2] - min_coords[2] + 1e-8  # avoid divide-by-zero
            norm_coords = coords_centered / z_range
            
            if self.sampling:
                norm_coords = farthest_point_sample(norm_coords, 1024)
            
            # Convert numpy array to Open3D PointCloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(norm_coords)

            # Estimate normals (change radius/knn depending on density)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=16))
            feats = np.asarray(pcd.normals)  # Shape: (N, 3)
        else:
            xyz = coords - np.mean(coords, axis=0)
            m = np.max(np.linalg.norm(xyz, axis=1, keepdims=True))
            norm_coords = xyz / m
            if self.sampling:
                norm_coords = farthest_point_sample(xyz, 1024)
            feats = xyz
        
        # Apply point cloud transforms if any
        if self.point_cloud_transform:
            norm_coords, feats, label = pointCloudTransform(
                norm_coords, pc_feat=feats, target=label, rot=self.rotate
            )

        # After applying transforms
        feats = torch.from_numpy(feats).float()  # Shape: (7168, 3)
        norm_coords = torch.from_numpy(norm_coords).float()  # Shape: (7168, 3)
        label = torch.from_numpy(label).float()  # Shape: (num_classes,)

        sample = {
            "images": superpixel_images,  # Padded images of shape [num_seasons, num_channels, 128, 128]
            "mask": nodata_mask,  # Padded masks of shape [num_seasons, 128, 128]
            "per_pixel_labels": per_pixel_labels,  # Tensor: (num_classes, 128, 128)
            "point_cloud": norm_coords,
            "pc_feat": feats,
            "label": label,
        }
        return sample


class SuperpixelDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config["batch_size"]
        self.num_workers = config["gpus"]*2
        self.image_transform = (
            config["image_transform"] if config["image_transform"] != "None" else None
        )
        self.point_cloud_transform = config["point_cloud_transform"]
        self.aug_rotate = config["rotate"]
        self.aug_pc_norm = config["pc_normal"]
        self.fps = config["fps"]
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
            self.datasets[split] = SuperpixelDataset(
                superpixel_files,
                rotate=None,
                pc_normal=self.aug_pc_norm,
                image_transform=None,
                point_cloud_transform=None,
                img_mean=img_mean,
                img_std=img_std,
                sampling=self.fps
            )
            if split == "train":
                if not (
                    self.image_transform is None or self.point_cloud_transform is False
                ):
                    aug_pc_dataset = SuperpixelDataset(
                        superpixel_files,
                        rotate=self.aug_rotate,
                        pc_normal=self.aug_pc_norm,
                        image_transform=None,
                        point_cloud_transform=self.point_cloud_transform,
                        img_mean=img_mean,
                        img_std=img_std,
                        sampling=self.fps
                    )
                    aug_img_dataset = SuperpixelDataset(
                        superpixel_files,
                        rotate=self.aug_rotate,
                        pc_normal=self.aug_pc_norm,
                        image_transform=self.image_transform,
                        point_cloud_transform=None,
                        img_mean=img_mean,
                        img_std=img_std,
                        sampling=self.fps
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
            drop_last=False,
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
