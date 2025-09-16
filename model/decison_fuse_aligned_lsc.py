import os
import pandas as pd
import numpy as np

import torch
import pytorch_lightning as pl
import torch.nn as nn

from .pointnext import PointNextModel
from .decoder import DecisionLevelFusion

from torchmetrics.classification import Accuracy, ConfusionMatrix, F1Score
from .loss import apply_mask_per_batch


class FusionModel(pl.LightningModule):
    def __init__(self, config, n_classes):
        super().__init__()
        self.save_hyperparameters(config)

        self.cfg = config
        self.lr = self.cfg["lr"]
        self.ms_fusion = self.cfg["use_ms"]

        if self.ms_fusion:
            from .seasonal_fusion import FusionBlock
            self.mf_module = FusionBlock(n_inputs=len(self.cfg[f"{self.cfg['dataset']}_season_map"]), in_ch=self.cfg["n_bands"], n_filters=64)
            total_input_channels = 64
        else:
            total_input_channels = self.cfg["n_bands"] * len(self.cfg[f"{self.cfg['dataset']}_season_map"])

        # Image stream
        if self.cfg["network"] == "Unet":
            from .unet import UNet
            self.s2_model = UNet(n_channels=total_input_channels, n_classes=n_classes, return_type='logsoftmax', aligned=True)
        elif self.cfg["network"] == "ResUnet":
            from .ResUnet import ResUnet
            self.s2_model = ResUnet(n_channels=total_input_channels, n_classes=n_classes, return_type='logsoftmax', aligned=True)
        elif self.cfg["network"] == "FCNResNet":
            from .resnet_fcn import FCNResNet50
            self.s2_model = FCNResNet50(n_channels=total_input_channels, n_classes=n_classes, return_type='logsoftmax', aligned=True)
        elif self.cfg["network"] == "ResNet":
            from .ResNet import Resnet
            self.s2_model = Resnet(n_channels=total_input_channels, num_classes=n_classes, return_type='logsoftmax', aligned=True)
        elif self.cfg["network"] == "Vit":
            from .VitCls import S2Transformer
            self.s2_model = S2Transformer(num_classes=n_classes, usehead=True, return_type='logsoftmax', aligned=True)

        # Point cloud stream
        self.pc_model = PointNextModel(self.cfg, in_dim=3, n_classes=n_classes, return_type='logsoftmax', aligned=False)

        # Decision-level fusion module
        self.fuse_head = DecisionLevelFusion(
            n_classes=n_classes,
            method=self.cfg["decision_fuse_type"],
            weight_img=self.cfg.get("decision_weight_img", 0.7),
            weight_pc=self.cfg.get("decision_weight_pc", 0.3)
        )

        self.loss_func = nn.NLLLoss() #SmoothClsLoss(smoothing_ratio=0.1)
        self.metric = Accuracy(task="multiclass", num_classes=n_classes)
        self.f1_metric = F1Score(task="multiclass", num_classes=n_classes, average='macro')
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=n_classes)

        # Optimizer and scheduler settings
        self.optimizer_type = self.cfg["optimizer"]
        self.scheduler_type = self.cfg["scheduler"]
        
        self.validation_step_outputs = []

    def forward(self, images, pc_feat, xyz):
        if self.ms_fusion:
            stacked_features = self.mf_module(images)
        else:
            B, _, _, H, W = images.shape
            stacked_features = images.view(B, -1, H, W)

        img_preds, _ = self.s2_model(stacked_features)
        pc_preds, _ = self.pc_model(pc_feat, xyz)

        return img_preds, pc_preds

    def forward_and_metrics(self, images, img_masks, pc_feat, point_clouds, labels, pixel_labels, stage):
        pc_feat = pc_feat.permute(0, 2, 1) if pc_feat is not None else None
        point_clouds = point_clouds.permute(0, 2, 1) if point_clouds is not None else None
        true_cls = labels.argmax(dim=1)
        
        image_preds, pc_preds = self.forward(images, pc_feat, point_clouds)
        if self.cfg["network"] !="ResNet":
            img_logits_list = apply_mask_per_batch(image_preds, img_masks, multi_class=True)
            image_preds = torch.stack([p.mean(dim=0) if p.numel() > 0 else torch.zeros(image_preds.shape[1], device=image_preds.device) for p in img_logits_list], dim=0)
            
        fuse_preds = self.fuse_head(image_preds, pc_preds)
        pred_cls = fuse_preds.argmax(dim=1)

        acc = self.metric(pred_cls, true_cls)
        f1 = self.f1_metric(pred_cls, true_cls)
        
        # classification loss
        loss = self.criterion(pred_cls, true_cls)

        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_f1", f1, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if stage == "val":
            self.validation_step_outputs.append({"val_target": labels, "val_pred": fuse_preds})
        if stage == "test":
            return labels, fuse_preds, loss
        else:
            return loss

    def training_step(self, batch, batch_idx):
        return self.forward_and_metrics(batch["images"], batch.get("mask"), batch["pc_feat"], batch["point_cloud"], batch["label"], batch.get("per_pixel_labels"), "train")

    def validation_step(self, batch, batch_idx):
        return self.forward_and_metrics(batch["images"], batch.get("mask"), batch["pc_feat"], batch["point_cloud"], batch["label"], batch.get("per_pixel_labels"), "val")

    def test_step(self, batch, batch_idx):
        labels, fuse_preds, loss = self.forward_and_metrics(batch["images"], batch.get("mask"), batch["pc_feat"], batch["point_cloud"], batch["label"], batch.get("per_pixel_labels"), "test")
        self.save_to_file(labels, fuse_preds, self.cfg["class_names"])
        return loss

    def on_validation_epoch_end(self):
        sys_r2 = self.val_r2.compute()
        test_true = torch.cat([output["val_target"] for output in self.validation_step_outputs], dim=0)
        test_pred = torch.cat([output["val_pred"] for output in self.validation_step_outputs], dim=0)

        pred_classes = torch.argmax(test_pred, dim=1)
        true_classes = torch.argmax(test_true, dim=1)
        cm = self.confmat(pred_classes, true_classes)

        acc = self.metric(pred_classes, true_classes)
        f1 = self.f1_metric(test_pred, true_classes)
        self.log("val_acc_epoch", acc, sync_dist=True)
        self.log("val_f1_epoch", f1, sync_dist=True)
        print(f"Validation Accuracy: {acc}, F1 Score: {f1}")
        print("Confusion Matrix:")
        print(cm)

        #self.save_to_file(true_classes, test_pred, self.cfg["class_names"])

        self.validation_step_outputs.clear()

    def save_to_file(self, labels, outputs, classes):
        labels = labels.cpu().numpy()
        outputs = outputs.cpu().numpy()
        data = {
            "SampleID": np.arange(len(labels)),
            "True_class": labels,
            "Pred_class": np.argmax(outputs, axis=1),
        }
        df = pd.DataFrame(data)
        output_dir = os.path.join(self.cfg["save_dir"], self.cfg["log_name"], "outputs")
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "test_outputs.csv"), mode="a")

    def configure_optimizers(self):
        params = list(self.s2_model.parameters()) + list(self.pc_model.parameters()) + list(self.fuse_head.parameters())
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
                T_max=self.cfg["max_epochs"]
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.scheduler_type == "cosinewarmup":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.scheduler_type == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=0.1, epochs=self.cfg["max_epochs"], steps_per_epoch=len(self.train_dataloader()) / self.trainer.accumulate_grad_batches
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.scheduler_type == "CosLR":
            warmup_epochs = max(1, int(0.05 * self.cfg["max_epochs"]))  # ~5% warmup
            total_epochs = self.cfg["max_epochs"]

            warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)

            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                }}
        else:
            return optimizer