import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .blocks import MambaFusionBlock, FusionBlock
from .unet import UNet
from .ResUnet import ResUnet
from .pointnext import PointNextModel

from torchmetrics.regression import R2Score
from torchmetrics.functional import r2_score
from torchmetrics.classification import (
    ConfusionMatrix,
)
from .loss import apply_mask, calc_loss


class FusionModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.n_bands = 12
        total_input_channels = (
            self.n_bands * 4
        )  # If no MF module, concatenating all seasons directly
        self.spatial_attention = self.config["spatial_attention"]
        if self.spatial_attention:
            self.mf_module = FusionBlock(n_inputs=4, in_ch=self.n_bands, n_filters=64)
            total_input_channels = 64
        # Using standard UNet
        self.s2_model = UNet(
            n_channels=total_input_channels, n_classes=self.config["n_classes"]
        )
        self.pc_model = PointNextModel(self.config, in_dim=3)

        # Fusion and classification layers with additional linear layer
        self.fuse_head = MambaFusionBlock(
            in_img_chs=512,
            in_pc_chs=(self.config["emb_dims"]),
            dim=self.config["fusion_dim"],
            hidden_ch=self.config["linear_layers_dims"],
            num_classes=self.config["n_classes"],
            drop=self.config["dp_fuse"],
            last_feat_size=self.config["dp_fuse"]/16,  # tile_size=64, last_feat_size=4
        )

        # Define loss functions
        if self.config["weighted_loss"]:
            # Loss function and other parameters
            self.weights = self.config["train_weights"]  # Initialize on CPU
        self.criterion = nn.MSELoss()

        # Metrics
        self.train_r2 = R2Score()

        self.val_r2 = R2Score()

        self.test_r2 = R2Score()

        self.confmat = ConfusionMatrix(
            task="multiclass", num_classes=self.config["n_classes"]
        )

        # Optimizer and scheduler settings
        self.optimizer_type = self.config["optimizer"]
        self.scheduler_type = self.config["scheduler"]

        # Learning rates for different parts
        self.img_lr = self.config.get("img_lr")
        self.pc_lr = self.config.get("pc_lr")
        self.fusion_lr = self.config.get("fuse_lr")

        # Loss weights
        self.pc_loss_weight = self.config.get("pc_loss_weight", 2.0)
        self.img_loss_weight = self.config.get("img_loss_weight", 1.0)
        self.fuse_loss_weight = self.config.get("fuse_loss_weight", 2.0)

        self.best_test_r2 = 0.0
        self.best_test_outputs = None
        self.validation_step_outputs = []

    def forward(self, images, pc_feat, xyz):
        image_outputs = None
        img_emb = None
        point_outputs = None
        pc_emb = None

        # Process images
        if (
            self.spatial_attention
        ):  # Apply the MF module first to extract features from input
            stacked_features = self.mf_module(images)
        else:
            # batch_size, num_seasons, num_channels, width, height = images.shape
            # stacked_features = images.reshape(batch_size, num_seasons * num_channels, width, height)
            stacked_features = torch.cat(images, dim=1)
        image_outputs, img_emb = self.s2_model(
            stacked_features
        )  # shape: image_outputs: torch.Size([8, 9, 64, 64]), img_emb: torch.Size([8, 512, tile_size/2/2/2, 4])
        # Process point clouds
        point_outputs, pc_emb = self.pc_model(
            pc_feat, xyz
        )  # torch.Size([8, 9]), torch.Size([8, 768, 28])

        # Fusion and classification
        class_output = self.fuse_head(img_emb, pc_emb)
        return image_outputs, point_outputs, class_output

    def forward_and_metrics(
        self, images, img_masks, pc_feat, point_clouds, labels, pixel_labels, stage
    ):
        """
        Forward operations, computes the masked loss, R² score, and logs the metrics.

        Args:
        - images: Image data
        - img_masks: Masks for images
        - pc_feat: Point cloud features
        - point_clouds: Point cloud coordinates
        - labels: Ground truth labels for classification
        - pixel_labels: Ground truth labels for per-pixel predictions
        - stage: One of 'train', 'val', or 'test', used to select appropriate metrics and logging

        Returns:
        - loss: The computed loss
        """
        # Permute point cloud data if available
        pc_feat = pc_feat.permute(0, 2, 1) if pc_feat is not None else None
        point_clouds = (
            point_clouds.permute(0, 2, 1) if point_clouds is not None else None
        )

        # Forward pass
        pixel_preds, pc_preds, fuse_preds = self.forward(images, pc_feat, point_clouds)

        loss = 0
        logs = {}

        # Select appropriate metric instances based on the stage
        if stage == "train":
            r2_metric = self.train_r2
        elif stage == "val":
            r2_metric = self.val_r2
        else:  # stage == "test"
            r2_metric = self.test_r2

        """
        # Compute point cloud loss
        if self.config["weighted_loss"] and stage == "train":
            self.weights = self.weights.to(pc_preds.device)
            loss_point = self.pc_loss_weight * calc_loss(labels, pc_preds, self.weights)
            # loss_point = self.pc_loss_weight * calc_pinball_loss(labels, pc_preds)
        else:
            loss_point = self.pc_loss_weight * self.criterion(pc_preds, labels)
        loss += loss_point
        

        # Compute R² metric
        pc_preds_rounded = torch.round(pc_preds, decimals=2)
        pc_r2 = r2_metric(pc_preds_rounded.view(-1), labels.view(-1))

        # Log metrics
        logs.update(
            {
                f"pc_{stage}_loss": loss_point,
                f"pc_{stage}_r2": pc_r2,
            }
        )
        """
        # Image stream
        # Apply mask to predictions and labels
        valid_pixel_preds, valid_pixel_true = apply_mask(
            pixel_preds, pixel_labels, img_masks
        )

        # Compute pixel-level loss
        # loss_pixel = self.criterion(valid_pixel_preds, valid_pixel_true)
        if self.config["weighted_loss"] and stage == "train":
            self.weights = self.weights.to(pc_preds.device)
            loss_pixel = self.img_loss_weight * calc_loss(
                valid_pixel_true, valid_pixel_preds, self.weights
            )
            # loss_pixel = self.img_loss_weight * calc_pinball_loss(valid_pixel_true, valid_pixel_preds)
        else:
            loss_pixel = self.img_loss_weight * self.criterion(
                valid_pixel_preds, valid_pixel_true
            )
        loss += loss_pixel

        # Compute R² metric
        valid_pixel_preds_rounded = torch.round(valid_pixel_preds, decimals=2)
        pixel_r2 = r2_metric(
            valid_pixel_preds_rounded.view(-1), valid_pixel_true.view(-1)
        )

        # Log metrics
        logs.update(
            {
                f"pixel_{stage}_loss": loss_pixel,
                f"pixel_{stage}_r2": pixel_r2,
            }
        )

        # Fusion stream
        # Compute fusion loss
        if self.config["weighted_loss"] and stage == "train":
            loss_fuse = self.fuse_loss_weight * calc_loss(
                labels, fuse_preds, self.weights
            )
            # loss_fuse = self.fuse_loss_weight * calc_pinball_loss(labels, fuse_preds)
        else:
            loss_fuse = self.fuse_loss_weight * self.criterion(fuse_preds, labels)
        loss += loss_fuse

        # Compute R² metric
        fuse_preds_rounded = torch.round(fuse_preds, decimals=2)
        fuse_r2 = r2_metric(fuse_preds_rounded.view(-1), labels.view(-1))

        # Log metrics
        logs.update({f"fuse_{stage}_loss": loss_fuse, f"fuse_{stage}_r2": fuse_r2})
        if stage == "val":
            self.validation_step_outputs.append(
                {"val_target": labels, "val_pred": fuse_preds}
            )

        # Compute RMSE
        rmse = torch.sqrt(loss_fuse)
        logs.update(
            {
                f"{stage}_loss": loss,
                f"{stage}_rmse": rmse,
            }
        )

        # Log all metrics
        for key, value in logs.items():
            self.log(
                key,
                value,
                on_step="loss" in key,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        if stage == "test":
            return labels, fuse_preds, loss
        else:
            return loss

    def training_step(self, batch, batch_idx):
        images = batch["images"] if "images" in batch else None
        point_clouds = batch["point_cloud"] if "point_cloud" in batch else None
        labels = batch["label"]
        per_pixel_labels = (
            batch["per_pixel_labels"] if "per_pixel_labels" in batch else None
        )
        image_masks = batch["nodata_mask"] if "nodata_mask" in batch else None
        pc_feat = batch["pc_feat"] if "pc_feat" in batch else None

        loss = self.forward_and_metrics(
            images,
            image_masks,
            pc_feat,
            point_clouds,
            labels,
            per_pixel_labels,
            stage="train",
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["images"] if "images" in batch else None
        point_clouds = batch["point_cloud"] if "point_cloud" in batch else None
        labels = batch["label"]
        per_pixel_labels = (
            batch["per_pixel_labels"] if "per_pixel_labels" in batch else None
        )
        image_masks = batch["nodata_mask"] if "nodata_mask" in batch else None
        pc_feat = batch["pc_feat"] if "pc_feat" in batch else None

        loss = self.forward_and_metrics(
            images,
            image_masks,
            pc_feat,
            point_clouds,
            labels,
            per_pixel_labels,
            stage="val",
        )
        return loss

    def on_validation_epoch_end(self):
        sys_r2 = self.val_r2.compute()
        test_true = torch.cat(
            [output["val_target"] for output in self.validation_step_outputs], dim=0
        )
        test_pred = torch.cat(
            [output["val_pred"] for output in self.validation_step_outputs], dim=0
        )

        last_epoch_val_r2 = r2_score(
            torch.round(test_pred.flatten(), decimals=1), test_true.flatten()
        )
        self.log("ave_val_r2", last_epoch_val_r2, sync_dist=True)
        self.log("sys_r2", sys_r2, sync_dist=True)

        print(f"average r2 score at epoch {self.current_epoch}: {last_epoch_val_r2}")
        if last_epoch_val_r2 > self.best_test_r2:
            self.best_test_r2 = last_epoch_val_r2
            self.best_test_outputs = {
                "preds_all": test_pred,
                "true_labels_all": test_true,
            }

            cm = self.confmat(
                torch.argmax(test_pred, dim=1), torch.argmax(test_true, dim=1)
            )
            print("Confusion Matrix at best R²:")
            print(cm)
            print(f"R2 Score:{sys_r2}")
            print(
                f"r2_score per class check: {r2_score(torch.round(test_pred, decimals=1), test_true, multioutput='raw_values')}"
            )

            self.save_to_file(test_true, test_pred, self.config["classes"])

        self.validation_step_outputs.clear()
        self.val_r2.reset()

    def test_step(self, batch, batch_idx):
        images = batch["images"] if "images" in batch else None
        point_clouds = batch["point_cloud"] if "point_cloud" in batch else None
        labels = batch["label"]
        per_pixel_labels = (
            batch["per_pixel_labels"] if "per_pixel_labels" in batch else None
        )
        image_masks = batch["nodata_mask"] if "nodata_mask" in batch else None
        pc_feat = batch["pc_feat"] if "pc_feat" in batch else None

        labels, fuse_preds, loss = self.forward_and_metrics(
            images,
            image_masks,
            pc_feat,
            point_clouds,
            labels,
            per_pixel_labels,
            stage="test",
        )

        self.save_to_file(labels, fuse_preds, self.config["classes"])
        return loss

    def save_to_file(self, labels, outputs, classes):
        # Convert tensors to numpy arrays or lists as necessary
        labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
        outputs = (
            outputs.cpu().numpy() if isinstance(outputs, torch.Tensor) else outputs
        )
        num_samples = labels.shape[0]
        data = {"SampleID": np.arange(num_samples)}

        # Add true and predicted values for each class
        for i, class_name in enumerate(classes):
            data[f"True_{class_name}"] = labels[:, i]
            data[f"Pred_{class_name}"] = outputs[:, i]

        df = pd.DataFrame(data)

        output_dir = os.path.join(
            self.config["save_dir"],
            self.config["log_name"],
            "outputs",
        )
        os.makedirs(output_dir, exist_ok=True)
        # Save DataFrame to a CSV file
        df.to_csv(
            os.path.join(output_dir, "test_outputs.csv"),
            mode="a",
        )

    def configure_optimizers(self):
        params = []

        # Include parameters from the image model
        image_params = list(self.s2_model.parameters())
        params.append({"params": image_params, "lr": self.img_lr})

        # Include parameters from the point cloud model
        point_params = list(self.pc_model.parameters())
        params.append({"params": point_params, "lr": self.fusion_lr})

        # Include parameters from the fusion layers
        fusion_params = list(self.fuse_head.parameters())
        params.append({"params": fusion_params, "lr": self.fusion_lr})

        # Choose the optimizer based on input parameter
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                params,
                betas=(0.9, 0.999),
                eps=1e-08,
            )
        elif self.optimizer_type == "adamW":
            optimizer = torch.optim.AdamW(
                params, weight_decay=self.config["weight_decay"]
            )
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                params,
                momentum=self.config["momentum"],
                weight_decay=self.config["weight_decay"],
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

        # Configure the scheduler based on the input parameter
        if self.scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=self.config["patience"], factor=0.5
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",  # Reduce learning rate when 'val_loss' plateaus
                },
            }
        elif self.scheduler_type == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.config["step_size"], gamma=0.1
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=50,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.scheduler_type == "cosinewarmup":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=30
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer
