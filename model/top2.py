import os
import pandas as pd
import numpy as np

import torch
import pytorch_lightning as pl

from .decoder import MambaFusionDecoder
from .pointnext import PointNextModel

from torchmetrics.classification import (
    MultilabelPrecision, MultilabelRecall, MultilabelF1Score, MultilabelAccuracy
)

from .loss import MultiLabelFocalLoss


class FusionModel(pl.LightningModule):
    def __init__(self, config, n_classes):
        super().__init__()
        
        self.save_hyperparameters(config)
        
        self.cfg = config
        self.lr = self.cfg["lr"]
        
        # seasonal s2 data fusion block
        
        total_input_channels = (
            self.cfg["n_bands"] * 4
        )  # Concatenating all seasonal data directly

        # Image stream backbone
        if self.cfg["network"] == "Unet":
            from .unet import UNet

            self.s2_model = UNet(
                n_channels=total_input_channels,
                n_classes=n_classes,
                decoder=False,
                return_type='logits',
            )
        elif self.cfg["network"] == "ResUnet":
            from .ResUnet import ResUnet

            self.s2_model = ResUnet(
                n_channels=total_input_channels,
                n_classes=n_classes,
                decoder=False,
                return_type='logits',
            )
        elif self.cfg["network"] == "ResNet":
            from .resnet_fcn import FCNResNet50

            self.s2_model = FCNResNet50(
                n_channels=total_input_channels,
                n_classes=n_classes,
                return_type='logits',
                upsample_method='bilinear',
                pretrained=True,
                decoder=False
            )

        # PC stream backbone
        self.pc_model = PointNextModel(self.cfg, 
                                    in_dim=3, #if self.cfg["dataset"] in ["rmf", "ovf"] else 6, 
                                    n_classes=n_classes, 
                                    decoder=False
                                )

        # Late Fusion and classification layers with additional MLPs
        if self.cfg["network"] == "ResNet":
            img_chs=2048
        elif self.cfg["network"] == "ResUnet":
            img_chs=1024
        elif self.cfg["network"] == "Vit":
            img_chs=768
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
            return_type='logits',
            return_feature=False
        )
        
        # Metrics
        #self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = MultiLabelFocalLoss()

        # TorchMetrics for multi-label (top-2)
        self.val_acc = MultilabelAccuracy(num_labels=n_classes, threshold=0.3, average="macro")
        self.val_f1 = MultilabelF1Score(num_labels=n_classes, threshold=0.3, average="macro")
        self.val_precision = MultilabelPrecision(num_labels=n_classes, threshold=0.3, average="macro")
        self.val_recall = MultilabelRecall(num_labels=n_classes, threshold=0.3, average="macro")

        # Optimizer and scheduler settings
        self.optimizer_type = self.cfg["optimizer"]
        self.scheduler_type = self.cfg["scheduler"]

        self.validation_step_outputs = []

    
    def forward(self, images, pc_feat, xyz):
        img_emb = None
        pc_emb = None
        class_output = None

        # Process images
        if self.cfg["dataset"] in ['rmf', 'ovf']:
            stacked_features = torch.cat(images, dim=1)
        else:
            B, _, _, H, W = images.shape
            stacked_features = images.view(B, -1, H, W)
        img_emb = self.s2_model(stacked_features)  
        pc_emb = self.pc_model(pc_feat, xyz)  # torch.Size([bs, 768, 28])

        # Fusion and classification
        class_output = self.fuse_head(img_emb, pc_emb) # torch.Size([8, 1794, 8, 8])
        
        return class_output

    def forward_and_metrics(
        self, images, pc_feat, point_clouds, labels, stage
    ):
        """
        Forward operations, computes the masked loss, R² score, and logs the metrics.

        Args:
        - images: Image data
        - pc_feat: Point cloud features
        - point_clouds: Point cloud coordinates
        - labels: Ground truth labels for classification
        - stage: One of 'train', 'val', or 'test', used to select appropriate metrics and logging

        Returns:
        - loss: The computed loss
        """
        # Permute point cloud data if available
        pc_feat = pc_feat.permute(0, 2, 1) if pc_feat is not None else None
        point_clouds = point_clouds.permute(0, 2, 1) if point_clouds is not None else None
        
        # Top-2 2-hot ground truth
        label_2hot = torch.zeros_like(labels)
        top2_labels = torch.topk(labels, k=2, dim=1).indices  # shape: (B, 2)
        label_2hot.scatter_(1, top2_labels, 1.0)

        # Forward pass
        fuse_preds = self.forward(images, pc_feat, point_clouds)
        logs = {}
        
        # Compute fusion loss
        loss_fuse = self.criterion(fuse_preds, label_2hot)
        
        probs = torch.sigmoid(fuse_preds)
        self.val_acc.update(probs, label_2hot.int())
        self.val_f1.update(probs, label_2hot.int())
        self.val_precision.update(probs, label_2hot.int())
        self.val_recall.update(probs, label_2hot.int())

        
        logs.update({
            f"{stage}_loss": loss_fuse,
            f"{stage}_acc": self.val_acc.compute(),
            f"{stage}_f1": self.val_f1.compute(),
            f"{stage}_precision": self.val_precision.compute(),
            f"{stage}_recall": self.val_recall.compute(),
        })

                
        if stage == "val":
            self.validation_step_outputs.append(
                {"val_target": labels, "val_pred": fuse_preds}
            )


        # Log all metrics
        for key, value in logs.items():
            self.log(
                key,
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        
        if stage == "test":
            return labels, fuse_preds, loss_fuse
        else:
            return loss_fuse

    def training_step(self, batch, batch_idx):
        images = batch["images"] if "images" in batch else None
        point_clouds = batch["point_cloud"] if "point_cloud" in batch else None
        labels = batch["label"]
        pc_feat = batch["pc_feat"] if "pc_feat" in batch else None

        loss = self.forward_and_metrics(
            images,
            pc_feat,
            point_clouds,
            labels,
            stage="train",
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["images"] if "images" in batch else None
        point_clouds = batch["point_cloud"] if "point_cloud" in batch else None
        labels = batch["label"]
        pc_feat = batch["pc_feat"] if "pc_feat" in batch else None

        loss = self.forward_and_metrics(
            images,
            pc_feat,
            point_clouds,
            labels,
            stage="val",
        )
        return loss

    def on_validation_epoch_end(self):
        test_true = torch.cat(
            [output["val_target"] for output in self.validation_step_outputs], dim=0
        )
        test_pred = torch.cat(
            [output["val_pred"] for output in self.validation_step_outputs], dim=0
        )
        top2_preds = torch.topk(torch.sigmoid(test_pred), k=2, dim=1).indices
        top2_targets = torch.topk(test_true, k=2, dim=1).indices
        correct = [
            len(set(pred.tolist()) & set(target.tolist())) > 0
            for pred, target in zip(top2_preds, top2_targets)
        ]
        top2_accuracy = sum(correct) / len(correct)
        self.log("top2_acc", top2_accuracy, sync_dist=True)
        

        self.validation_step_outputs.clear()
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def test_step(self, batch, batch_idx):
        images = batch["images"] if "images" in batch else None
        point_clouds = batch["point_cloud"] if "point_cloud" in batch else None
        labels = batch["label"]
        pc_feat = batch["pc_feat"] if "pc_feat" in batch else None

        labels, fuse_preds, loss = self.forward_and_metrics(
            images,
            pc_feat,
            point_clouds,
            labels,
            stage="test"
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
        
        if any(p.requires_grad for p in self.s2_model.parameters()):
            image_params = list(self.s2_model.parameters())
            params.append({"params": image_params, "lr": 1e-4})

        # Include parameters from the point cloud model
        if any(p.requires_grad for p in self.pc_model.parameters()):
            point_params = list(self.pc_model.parameters()) 
            params.append({"params": point_params, "lr": self.lr})

        # Include parameters from the fusion layers
        if any(p.requires_grad for p in self.fuse_head.parameters()):
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
                T_max=10
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.scheduler_type == "cosinewarmup":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer