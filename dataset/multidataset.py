from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from pytorch_lightning import LightningDataModule
import os
from os.path import join
import torch
import numpy as np
from torch.utils.data import Dataset
from .augment import pointCloudTransform, image_augment, normalize_point_cloud, center_point_cloud
import torchvision.transforms.v2 as transforms
import open3d as o3d
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
class SuperpixelDataset(Dataset):
    def __init__(
        self,
        superpixel_files,
        source,  # "ovf" or "rmf"
        rotate=None,
        pc_normal=None,
        image_transform=None,
        point_cloud_transform=None,
        img_mean=None,
        img_std=None
    ):
        self.superpixel_files = superpixel_files
        self.image_transform = image_transform
        self.point_cloud_transform = point_cloud_transform
        self.rotate = rotate
        self.normal = pc_normal
        self.source = source

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

        superpixel_images = data["superpixel_images"].astype(np.float32) / 10000.0
        coords = data["point_cloud"]
        label = data["label"]
        per_pixel_labels = data["per_pixel_labels"]
        nodata_mask = data["nodata_mask"]

        superpixel_images = torch.from_numpy(superpixel_images).float()
        per_pixel_labels = torch.from_numpy(per_pixel_labels).float()
        nodata_mask = torch.from_numpy(nodata_mask).bool()

        superpixel_images = self.transforms(superpixel_images)

        if self.image_transform is not None:
            superpixel_images = image_augment(superpixel_images, self.image_transform, 128)

        normalized_coords = normalize_point_cloud(coords)

        if self.normal:
            if self.point_cloud_transform:
                normalized_coords, _, label = pointCloudTransform(
                    normalized_coords, pc_feat=None, target=label, rot=self.rotate
                )
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(normalized_coords)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=16))
            feats = np.asarray(pcd.normals)
        else:
            feats = center_point_cloud(coords)
            if self.point_cloud_transform:
                normalized_coords, feats, label = pointCloudTransform(
                    normalized_coords, pc_feat=feats, target=label, rot=self.rotate
                )

        feats = torch.from_numpy(feats).float()
        normalized_coords = torch.from_numpy(normalized_coords).float()
        label = torch.from_numpy(label).float()

        return {
            "images": superpixel_images,
            "mask": nodata_mask,
            "per_pixel_labels": per_pixel_labels,
            "point_cloud": normalized_coords,
            "pc_feat": feats,
            "label": label,
            "source": self.source,   # <- tag the sample's origin
        }


class MultiSourceDataModule(LightningDataModule):
    def __init__(self,
                 train_sources=("ovf","rmf"),
                 val_sources=("ovf",),
                 test_sources=("ovf",)):
        super().__init__()
        ovf_cfg = load_config("./configs/config_ovf_5class.yaml")
        rmf_cfg = load_config("./configs/config_rmf_5class.yaml")
        
        self.ovf_cfg, self.rmf_cfg = ovf_cfg, rmf_cfg
        self.train_sources, self.val_sources, self.test_sources = train_sources, val_sources, test_sources

        self.batch_size = ovf_cfg["batch_size"]
        self.num_workers = ovf_cfg["gpus"] * 2

        self.image_transform = ovf_cfg["image_transform"] if ovf_cfg["image_transform"] != "None" else None
        self.point_cloud_transform = ovf_cfg["point_cloud_transform"]
        self.aug_rotate = ovf_cfg["rotate"]
        self.aug_pc_norm = ovf_cfg["pc_normal"]

        self.data_dirs = {
            "ovf": { sp: join(ovf_cfg["data_dir"], "tile_128", sp, ovf_cfg["dataset"]) for sp in ["train","val","test"] },
            "rmf": { sp: join(rmf_cfg["data_dir"], "tile_128", sp, rmf_cfg["dataset"]) for sp in ["train","val","test"] },
        }
        self.ovf_mean = ovf_cfg[f"{ovf_cfg['dataset']}_img_mean"]; self.ovf_std = ovf_cfg[f"{ovf_cfg['dataset']}_img_std"]
        self.rmf_mean = rmf_cfg[f"{rmf_cfg['dataset']}_img_mean"]; self.rmf_std = rmf_cfg[f"{rmf_cfg['dataset']}_img_std"]

    def _load_files(self, source, split):
        root = self.data_dirs[source][split]
        if not os.path.isdir(root): return []
        return [join(root, f) for f in os.listdir(root) if f.endswith(".npz")]

    def _build_dataset_for(self, source, files, split, aug=False):
        mean, std = (self.ovf_mean, self.ovf_std) if source == "ovf" else (self.rmf_mean, self.rmf_std)
        # base
        base = SuperpixelDataset(files, source=source,
                                 rotate=None, pc_normal=self.aug_pc_norm,
                                 image_transform=None, point_cloud_transform=None,
                                 img_mean=mean, img_std=std)
        datasets = [base]
        # optional aug for train
        if split == "train" and aug:
            datasets += [
                SuperpixelDataset(files, source=source,
                                  rotate=self.aug_rotate, pc_normal=self.aug_pc_norm,
                                  image_transform=None, point_cloud_transform=self.point_cloud_transform,
                                  img_mean=mean, img_std=std),
                SuperpixelDataset(files, source=source,
                                  rotate=self.aug_rotate, pc_normal=self.aug_pc_norm,
                                  image_transform=self.image_transform, point_cloud_transform=None,
                                  img_mean=mean, img_std=std),
            ]
        return datasets

    def _make_split(self, split, include_sources):
        datasets, src_tags = [], []
        for src in include_sources:
            files = self._load_files(src, split)
            aug = not (self.image_transform is None and self.point_cloud_transform is False)
            parts = self._build_dataset_for(src, files, split, aug=aug)
            datasets.extend(parts)
            for d in parts: src_tags += [src] * len(d)
        concat = ConcatDataset(datasets) if len(datasets) else datasets
        return concat, src_tags

    def setup(self, stage=None):
        self.train_concat, self.train_src = self._make_split("train", self.train_sources)
        self.val_concat,   self.val_src   = self._make_split("val",   self.val_sources)
        self.test_concat,  self.test_src  = self._make_split("test",  self.test_sources)

        # Balanced sampler only for train (OVF+RMF)
        self.train_sampler = None
        if isinstance(self.train_concat, ConcatDataset) and len(self.train_src):
            counts = {}
            for s in self.train_src: counts[s] = counts.get(s, 0) + 1
            weights = [1.0 / counts[s] for s in self.train_src]
            self.train_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    def train_dataloader(self):
        return DataLoader(self.train_concat, batch_size=self.batch_size,
                          shuffle=(self.train_sampler is None), sampler=self.train_sampler,
                          num_workers=self.num_workers, drop_last=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_concat, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          drop_last=True, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_concat, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          drop_last=False, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        batch = [b for b in batch if b is not None]
        return {
            "images": torch.stack([b["images"] for b in batch]),
            "point_cloud": torch.stack([b["point_cloud"] for b in batch]),
            "pc_feat": torch.stack([b["pc_feat"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]),
            "per_pixel_labels": torch.stack([b["per_pixel_labels"] for b in batch]),
            "mask": torch.stack([b["mask"] for b in batch]),
            "source": [b["source"] for b in batch],
        }
