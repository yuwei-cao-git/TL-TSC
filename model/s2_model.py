import torch
import torch.nn as nn
import pytorch_lightning as pl


from torchmetrics.regression import R2Score
from torchmetrics.classification import (
    ConfusionMatrix,
)
from .loss import apply_mask, calc_masked_loss, get_class_grw_weight

class S2Model(pl.LightningModule):
    def __init__(self, config, n_classes):
        super().__init__()

        self.save_hyperparameters(config)

        self.cfg = config
        self.img_lr = self.img_lr = self.cfg.get("img_lr", 5e-4)

        # seasonal s2 data fusion block
        self.ms_fusion = self.cfg["use_ms"]
        if self.ms_fusion:
            from .seasonal_fusion import FusionBlock
            self.mf_module = FusionBlock(n_inputs=len(self.cfg[f"{self.cfg['dataset']}_season_map"]), in_ch=self.cfg["n_bands"], n_filters=64)
            total_input_channels = 64
        else:
            total_input_channels = self.cfg["n_bands"] * len(self.cfg[f"{self.cfg['dataset']}_season_map"])

        # Image stream backbone
        if self.cfg["network"] == "Unet":
            from .unet import UNet

            self.s2_model = UNet(
                n_channels=total_input_channels,
                n_classes=n_classes,
                decoder=True,
                aligned=(True if self.cfg['align_header'] in ['pc', 'both'] else False)
            )
        elif self.cfg["network"] == "ResUnet":
            from .ResUnet import ResUnet

            self.s2_model = ResUnet(
                n_channels=total_input_channels,
                n_classes=n_classes,
                decoder=True,
                aligned=(True if self.cfg['align_header'] in ['pc', 'both'] else False)
            )
        elif self.cfg["network"] == "ResNet":
            from .resnet_fcn import FCNResNet50

            self.s2_model = FCNResNet50(
                n_channels=total_input_channels,
                n_classes=n_classes,
                upsample_method='bilinear',
                pretrained=True,
                decoder=True
            )

        self.loss_func=self.cfg["loss_func"]
        # Define loss functions
        if self.loss_func in ["wmse", "wrmse", "wkl", "ewmse"]:
            self.weights = self.cfg[f"{self.cfg['dataset']}_class_weights"]
            if self.loss_func == "ewmse":
                self.weights = get_class_grw_weight(self.weights, n_classes, exp_scale=0.2)
        else:
            self.weights = None

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

    def forward(self, images):
        image_outputs = None

        # Process images
        if self.ms_fusion:  # Apply the MF module first to extract features from input
            stacked_features = self.mf_module(images)
        else:
            if self.cfg["dataset"] in ['rmf', 'ovf']:
                stacked_features = torch.cat(images, dim=1)
            else:
                B, _, _, H, W = images.shape
                stacked_features = images.view(B, -1, H, W)
        image_outputs, _ = self.s2_model(stacked_features) # torch.Size([bs, 9, 64, 64]), torch.Size([bs, 512, tile_size/16, tile_size/16])

        return image_outputs

    def forward_and_metrics(
        self, images, img_masks, labels, pixel_labels, stage
    ):
        """
        Forward operations, computes the masked loss, R² score, and logs the metrics.

        Args:
        - images: Image data
        - img_masks: Masks for images
        - labels: Ground truth labels for classification
        - pixel_labels: Ground truth labels for per-pixel predictions
        - stage: One of 'train', 'val', or 'test', used to select appropriate metrics and logging

        Returns:
        - loss: The computed loss
        """
        # Forward pass
        pixel_preds = self.forward(images)
        logs = {}
        loss = torch.tensor(0.0, device=pixel_preds.device)

        # Select appropriate metric instances based on the stage
        if stage == "train":
            r2_metric = self.train_r2
        elif stage == "val":
            r2_metric = self.val_r2
        else:  # stage == "test"
            r2_metric = self.test_r2
        weights = self.weights.to(pixel_preds.device) if self.weights is not None else None

        # Image stream
        if pixel_preds is not None:
            # Apply mask to predictions and labels
            valid_pixel_preds, valid_pixel_true = apply_mask(pixel_preds, pixel_labels, img_masks, multi_class=True)

            # Compute pixel-level loss
            if stage == "train":
                loss = calc_masked_loss(self.cfg["loss_func"], valid_pixel_preds, valid_pixel_true, weights)
            else:
                loss = self.criterion(valid_pixel_preds, valid_pixel_true)

            # Compute R² metric
            valid_pixel_preds_rounded = torch.round(valid_pixel_preds, decimals=2)
            pixel_r2 = r2_metric(valid_pixel_preds_rounded.view(-1), valid_pixel_true.view(-1))

            logs.update(
            {
                f"{stage}_r2": pixel_r2,
                f"{stage}_loss": loss
            }
            )
            if stage == "test":
                rmse = torch.sqrt(loss)
                logs.update(
                    {
                        f"{stage}_rmse": rmse,
                    }
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

        return loss

    def training_step(self, batch, batch_idx):
        images = batch["images"] if "images" in batch else None
        labels = batch["label"]
        per_pixel_labels = batch.get("per_pixel_labels", None)
        image_masks = batch.get("mask", None)

        loss = self.forward_and_metrics(
            images,
            image_masks,
            labels,
            per_pixel_labels,
            stage="train",
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["images"] if "images" in batch else None
        labels = batch["label"]
        per_pixel_labels = (
            batch["per_pixel_labels"] if "per_pixel_labels" in batch else None
        )
        image_masks = batch["mask"] if "mask" in batch else None

        loss = self.forward_and_metrics(
            images,
            image_masks,
            labels,
            per_pixel_labels,
            stage="val",
        )
        return loss

    def test_step(self, batch, batch_idx):
        images = batch["images"] if "images" in batch else None
        labels = batch["label"]
        per_pixel_labels = (
            batch["per_pixel_labels"] if "per_pixel_labels" in batch else None
        )
        image_masks = batch["mask"] if "mask" in batch else None

        loss = self.forward_and_metrics(
            images,
            image_masks,
            labels,
            per_pixel_labels,
            stage="test"
        )

        return loss

    def configure_optimizers(self):
        params = []

        # Include parameters from the image model
        if self.cfg["use_ms"]:
            mf_params = list(self.mf_module.parameters())
            params.append({"params": mf_params, "lr": self.img_lr})

        if any(p.requires_grad for p in self.s2_model.parameters()):
            image_params = list(self.s2_model.parameters())
            params.append({"params": image_params, "lr": self.img_lr})

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
        elif self.scheduler_type == "CosLR":
            warmup_epochs = max(1, int(0.05 * self.cfg["max_epochs"]))  # ~5% warmup
            total_epochs = self.cfg["max_epochs"]

            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
            )

            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        elif self.scheduler_type == "StepLRWarmup":
            # Determine the warmup phase length (e.g., 2 to 5 epochs is common)
            warmup_epochs = self.cfg.get("warmup_epochs", 3)

            # 1. Warmup: Bring LR from tiny value (1e-6) up to the full initial LR
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,  # Ensure start is close to zero
                end_factor=1.0,
                total_iters=warmup_epochs,
            )

            # 2. Main Policy: StepLR (takes over immediately after warmup)
            steplr = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.cfg["step_size"], gamma=0.1
            )

            # Combine the schedulers: switch from warmup to steplr after the milestone
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, steplr], milestones=[warmup_epochs]
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return optimizer
