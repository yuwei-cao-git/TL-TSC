import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .decoder import MambaFusionDecoder
from .pointnext import PointNextModel

from torchmetrics.regression import R2Score
from torchmetrics.functional import r2_score
from torchmetrics.classification import (
    ConfusionMatrix,
)
from .loss import apply_mask, calc_masked_loss


class FusionModel(pl.LightningModule):
    def __init__(self, config, n_classes):
        super().__init__()
        
        self.save_hyperparameters(config)
        
        self.cfg = config
        self.lr = self.cfg["lr"]
        
        # seasonal s2 data fusion block
        self.ms_fusion = self.cfg["use_ms"]
        if self.ms_fusion:  # mid fusion
            from .seasonal_fusion import FusionBlock
            self.mf_module = FusionBlock(n_inputs=4, in_ch=self.cfg["n_bands"], n_filters=64)
            total_input_channels = 64
        else:  # early-fusion
            total_input_channels = (
                self.cfg["n_bands"] * 4
            )  # Concatenating all seasonal data directly

        # Image stream backbone
        if self.cfg["network"] == "Unet":
            from .unet import UNet

            self.s2_model = UNet(
                n_channels=total_input_channels,
                n_classes=n_classes,
                decoder=(self.cfg["head"] == "no_pc_head" or self.cfg["head"] == "all_head")
            )
        elif self.cfg["network"] == "ResUnet":
            from .ResUnet import ResUnet

            self.s2_model = ResUnet(
                n_channels=total_input_channels,
                n_classes=n_classes,
                decoder=(self.cfg["head"] == "no_pc_head" or self.cfg["head"] == "all_head")
            )
        elif self.cfg["network"] == "ResNet":
            from .resnet import FCNResNet50

            self.s2_model = FCNResNet50(
                n_channels=total_input_channels,
                n_classes=n_classes,
                upsample_method='deconv',
                pretrained=False,
                decoder=(self.cfg["head"] == "no_pc_head" or self.cfg["head"] == "all_head")
            )

        # PC stream backbone
        self.pc_model = PointNextModel(self.cfg, 
                                    in_dim=3 if self.cfg["dataset"]=="rmf" else 6, 
                                    n_classes=n_classes, 
                                    decoder=self.cfg["head"] == "no_img_head" or self.cfg["head"] == "all_head"
                                )

        # Late Fusion and classification layers with additional MLPs
        if self.cfg["network"] == "ResNet":
            img_chs=2048
        elif self.cfg["network"] == "ResUnet":
            img_chs=1024
        else:
            img_chs=512
        self.fuse_head = MambaFusionDecoder(
            in_img_chs=img_chs,
            in_pc_chs=(self.cfg["emb_dims"]),
            dim=self.cfg["fusion_dim"],
            hidden_ch=self.cfg["linear_layers_dims"],
            num_classes=n_classes,
            drop=self.cfg["dp_fuse"],
            last_feat_size=(self.cfg["tile_size"]
            // 8) if self.cfg["network"] == "ResNet" else (self.cfg["tile_size"]
            // 16),
            return_feature=False
        )

        # Define loss functions
        self.weights = self.cfg["class_weights"]
        
        # multi-task loss weight
        if self.cfg["multitasks_uncertain_loss"]:
            from .loss import AutomaticWeightedLoss
            self.awl = AutomaticWeightedLoss(3 if self.cfg["head"]=='all_head' else 2)
            
        # use it during test/val/not uncertainty weighted loss - equal loss
        self.loss_func = self.cfg["loss_func"]
        if self.cfg["head"] == "no_img_head" or self.cfg["head"] == "all_head":
            self.pc_loss_weight = self.cfg.get("pc_loss_weight", 2.0) # /0.005
        if self.cfg["head"] == "no_pc_head" or self.cfg["head"] == "all_head":
            self.img_loss_weight = self.cfg.get("img_loss_weight", 1.0) # /0.15
        self.fuse_loss_weight = self.cfg.get("fuse_loss_weight", 2.0) # /0.005
        
        self.criterion = nn.MSELoss()

        # Metrics
        self.train_r2 = R2Score()

        self.val_r2 = R2Score()

        self.test_r2 = R2Score()

        self.confmat = ConfusionMatrix(
            task="multiclass", num_classes=n_classes
        )

        # Optimizer and scheduler settings
        self.optimizer_type = self.cfg["optimizer"]
        self.scheduler_type = self.cfg["scheduler"]

        self.best_test_r2 = 0.0
        self.best_test_outputs = None
        self.validation_step_outputs = []
    
    def forward(self, images, pc_feat, xyz):
        image_outputs = None
        img_emb = None
        point_outputs = None
        pc_emb = None

        # Process images
        if self.ms_fusion:  # Apply the MF module first to extract features from input
            stacked_features = self.mf_module(images)
        else:
            if self.cfg["dataset"] == 'rmf':
                stacked_features = torch.cat(images, dim=1)
            else:
                B, _, _, H, W = images.shape
                stacked_features = images.view(B, -1, H, W)
        if self.cfg["head"] in ["no_pc_head", "all_head"]:
            image_outputs, img_emb = self.s2_model(stacked_features) # torch.Size([bs, 9, 64, 64]), torch.Size([bs, 512, tile_size/16, tile_size/16])
        else:
            img_emb = self.s2_model(stacked_features)  
        # Process point clouds
        if self.cfg["head"] in ["no_img_head", "all_head"]:
            point_outputs, pc_emb = self.pc_model(pc_feat, xyz)  # torch.Size([bs, 9]), torch.Size([bs, 768, 28])
        else:
            pc_emb = self.pc_model(pc_feat, xyz)  # torch.Size([bs, 768, 28])

        # Fusion and classification
        class_output = self.fuse_head(img_emb, pc_emb) # torch.Size([8, 1794, 8, 8])
        
        if self.cfg["head"] == "all_head":
            return image_outputs, point_outputs, class_output
        elif self.cfg["head"] == "fuse_head":
            return None, None, class_output
        elif self.cfg["head"] == "no_img_head":
            return None, point_outputs, class_output
        else:
            return image_outputs, None, class_output

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
        loss_point = torch.tensor(0.0, device=labels.device)
        loss_pixel = torch.tensor(0.0, device=labels.device)

        # Permute point cloud data if available
        pc_feat = pc_feat.permute(0, 2, 1) if pc_feat is not None else None
        point_clouds = (point_clouds.permute(0, 2, 1) if point_clouds is not None else None)

        # Forward pass
        pixel_preds, pc_preds, fuse_preds = self.forward(images, pc_feat, point_clouds)

        logs = {}
        # Select appropriate metric instances based on the stage
        if stage == "train":
            r2_metric = self.train_r2
        elif stage == "val":
            r2_metric = self.val_r2
        else:  # stage == "test"
            r2_metric = self.test_r2
        
        # PC stream
        if pc_preds != None:
            # Compute point cloud loss
            if stage == "train":
                if self.cfg["weighted_loss"]:
                    self.weights = self.weights.to(pc_preds.device)
                else:
                    self.weight = torch.ones(9, dtype=float).to(pc_preds.device)
                loss_point = calc_masked_loss(self.loss_func, pc_preds, labels, self.weights)
            else:
                loss_point = self.criterion(pc_preds, labels)

            # Compute R² metric
            pc_preds_rounded = torch.round(pc_preds, decimals=2)
            pc_r2 = r2_metric(pc_preds_rounded.view(-1), labels.view(-1))

            # Log metrics
            logs.update({f"pc_{stage}_r2": pc_r2, f"pc_{stage}_loss": loss_point})
        
        # Image stream
        if pixel_preds != None:
            # Apply mask to predictions and labels
            valid_pixel_preds, valid_pixel_true = apply_mask(pixel_preds, pixel_labels, img_masks, multi_class=True)

            # Compute pixel-level loss
            if stage == "train":
                if self.cfg["weighted_loss"]:
                    self.weights = self.weights.to(valid_pixel_preds.device)
                else:
                    self.weight = torch.ones(9, dtype=float).to(valid_pixel_preds.device)
                loss_pixel = calc_masked_loss(self.loss_func, valid_pixel_preds, valid_pixel_true, self.weights)
            else:
                loss_pixel = self.criterion(valid_pixel_preds, valid_pixel_true)

            # Compute R² metric
            valid_pixel_preds_rounded = torch.round(valid_pixel_preds, decimals=2)
            pixel_r2 = r2_metric(valid_pixel_preds_rounded.view(-1), valid_pixel_true.view(-1))

            # Log metrics
            logs.update({f"pixel_{stage}_r2": pixel_r2, f"pixel_{stage}_loss": loss_pixel})
        
        # Fusion stream
        # Compute fusion loss
        if stage == "train":
            if self.cfg["weighted_loss"]:
                self.weights = self.weights.to(fuse_preds.device)
            else:
                self.weight = torch.ones(9, dtype=float).to(fuse_preds.device)
            loss_fuse = calc_masked_loss(self.loss_func, fuse_preds, labels, self.weights)
        else:
            loss_fuse = self.criterion(fuse_preds, labels)
        
        # Compute R² metric
        fuse_preds_rounded = torch.round(fuse_preds, decimals=2)
        fuse_r2 = r2_metric(fuse_preds_rounded.view(-1), labels.view(-1))
        
        logs.update({
            f"fuse_{stage}_r2": fuse_r2,
            f"fuse_{stage}_loss": loss_fuse})
        
        if self.cfg["head"]=="fuse_head":
            total_loss = loss_fuse
        else:
            if self.cfg["multitasks_uncertain_loss"] and stage == "train":
                if self.cfg["head"]=="all_head":
                    total_loss = self.awl(loss_pixel, loss_point, loss_fuse)
                elif self.cfg["head"]=="no_img_head":
                    total_loss = self.awl(loss_point, loss_fuse)
                elif self.cfg["head"]=="no_pc_head":
                    total_loss = self.awl(loss_pixel, loss_fuse)
            else:
                total_loss = loss_fuse*self.fuse_loss_weight + (loss_point*self.pc_loss_weight if loss_point != None else 0) +  (loss_pixel*self.img_loss_weight if loss_pixel != None else 0)
                
        if stage == "val":
            self.validation_step_outputs.append(
                {"val_target": labels, "val_pred": fuse_preds}
            )

        # Compute RMSE
        rmse = torch.sqrt(total_loss)
        logs.update(
            {
                f"{stage}_loss": total_loss,
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
            return labels, fuse_preds, total_loss
        else:
            return total_loss

    def training_step(self, batch, batch_idx):
        images = batch["images"] if "images" in batch else None
        point_clouds = batch["point_cloud"] if "point_cloud" in batch else None
        labels = batch["label"]
        per_pixel_labels = (
            batch["per_pixel_labels"] if "per_pixel_labels" in batch else None
        )
        image_masks = batch["mask"] if "mask" in batch else None
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
        image_masks = batch["mask"] if "mask" in batch else None
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

            self.save_to_file(test_true, test_pred, self.cfg["class_names"])

        self.validation_step_outputs.clear()
        self.val_r2.reset()

    def test_step(self, batch, batch_idx):
        images = batch["images"] if "images" in batch else None
        point_clouds = batch["point_cloud"] if "point_cloud" in batch else None
        labels = batch["label"]
        per_pixel_labels = (
            batch["per_pixel_labels"] if "per_pixel_labels" in batch else None
        )
        image_masks = batch["mask"] if "mask" in batch else None
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

        self.save_to_file(labels, fuse_preds, self.cfg["class_names"])
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
            self.cfg["save_dir"],
            self.cfg["log_name"],
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
        if self.cfg["multitasks_uncertain_loss"]:
            params.append({"params": [self.awl.params], "lr": self.lr})
            
        # Include parameters from the image model
        if self.cfg["use_ms"]:
            mf_params = list(self.mf_module.parameters())
            params.append({"params": mf_params, "lr": self.lr})
        if any(p.requires_grad for p in self.s2_model.parameters()):
            image_params = list(self.s2_model.parameters())
            params.append({"params": image_params, "lr": self.lr})

        # Include parameters from the point cloud model
        if any(p.requires_grad for p in self.pc_model.parameters()):
            point_params = list(self.pc_model.parameters()) 
            params.append({"params": point_params, "lr": self.lr})

        # Include parameters from the fusion layers
        fusion_params = list(self.fuse_head.parameters())
        params.append({"params": fusion_params, "lr": self.lr})

        # Choose the optimizer based on input parameter
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                params,
                betas=(0.9, 0.999),
                eps=1e-08,
            )
        elif self.optimizer_type == "adamW":
            optimizer = torch.optim.AdamW(
                params, weight_decay=self.cfg["weight_decay"]
            )
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                params,
                momentum=self.cfg["momentum"],
                weight_decay=self.cfg["weight_decay"],
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

        # Configure the scheduler based on the input parameter
        if self.scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=self.cfg["patience"], factor=0.5
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
                optimizer, step_size=self.cfg["step_size"], gamma=0.1
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=10,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.scheduler_type == "cosinewarmup":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer