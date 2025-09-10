# finetune.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from timm.scheduler import CosineLRScheduler
import os
import pandas as pd
import numpy as np

from .pointnext import PointNextModel
from .decoder import DecisionLevelFusion

from torchmetrics.regression import R2Score
from torchmetrics.functional import r2_score
from torchmetrics.classification import ConfusionMatrix
from .loss import apply_mask_per_batch, calc_masked_loss, get_class_grw_weight

# ---------- small utils ----------

def _name_starts_with_any(n: str, prefixes):
    return any(n.startswith(p) for p in prefixes)

def _is_norm_or_bias(n: str, p: torch.nn.Parameter):
    is_bn = (".bn" in n) or (".norm" in n) or isinstance(p, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm))
    is_bias_or_scale = (p.ndim == 1) or n.endswith(".bias")
    return is_bn or is_bias_or_scale

def reset_bn_running_stats(module: nn.Module):
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
        module.reset_running_stats()

def set_requires_grad_by_prefixes(model: nn.Module, train_prefixes):
    for n, p in model.named_parameters():
        p.requires_grad = _name_starts_with_any(n, train_prefixes)

def freeze_backbone_keep_heads(model: nn.Module, head_prefixes):
    for n, p in model.named_parameters():
        p.requires_grad = _name_starts_with_any(n, head_prefixes)

# ---------- heads / param groups / L2-SP ----------

def reinit_classifier_heads(model: nn.Module):
    """Fresh init final heads for dataset B (keeps shapes you already set)."""
    # S2 head
    if hasattr(model, "s2_model") and hasattr(model.s2_model, "classifier"):
        head = model.s2_model.classifier
        if hasattr(head, "conv_seg") and isinstance(head.conv_seg, nn.Conv2d):
            nn.init.kaiming_normal_(head.conv_seg.weight)
            if head.conv_seg.bias is not None:
                nn.init.zeros_(head.conv_seg.bias)

    # PointNext head
    if hasattr(model, "pc_model") and hasattr(model.pc_model, "decoder") and hasattr(model.pc_model.decoder, "cls_head"):
        last = model.pc_model.decoder.cls_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.kaiming_normal_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    # Fusion head
    if hasattr(model, "fuse_head") and hasattr(model.fuse_head, "reset_parameters"):
        model.fuse_head.reset_parameters()

def build_param_groups(model: nn.Module,
                       lr_early=5e-4, lr_mid=8e-4, lr_late=1e-3, lr_heads=5e-3,
                       wd=5e-3):
    """Create discriminative-LR param groups covering ALL params."""
    groups = []

    early = [
        "mf_module"
        "s2_model.encoder.stem",
        "s2_model.encoder.residual_block1",
        "pc_model.encoder.encoder.stem",
        "pc_model.encoder.encoder.encoder.0",
    ]
    mid = [
        "s2_model.encoder.residual_block2",
        "s2_model.encoder.residual_block3",
        "pc_model.encoder.encoder.encoder.1",
        "pc_model.encoder.encoder.encoder.2",
    ]
    late = [
        "s2_model.encoder.residual_block4",
        "pc_model.encoder.encoder.encoder.3",
        "pc_model.encoder.backbone.head",
        "s2_model.decoder",
    ]
    heads = [
        "s2_model.classifier",
        "pc_model.decoder.cls_head",
        "fuse_head",
    ]

    def add(prefixes, lr):
        wd_params, no_wd_params = [], []
        for n, p in model.named_parameters():
            if _name_starts_with_any(n, prefixes):
                (no_wd_params if _is_norm_or_bias(n, p) else wd_params).append(p)
        if wd_params:     groups.append({"params": wd_params,     "lr": lr, "weight_decay": wd})
        if no_wd_params:  groups.append({"params": no_wd_params,  "lr": lr, "weight_decay": 0.0})

    add(early, lr_early)
    add(mid,   lr_mid)
    add(late,  lr_late)
    add(heads, lr_heads)
    return groups

def snapshot_source_state(pl_module):
    """Call once right after loading A-weights, before fine-tuning.
    Store ONLY a CPU snapshot; we'll move per-param on the fly."""
    # Keep full state (params+buffers) to be safe, but we'll only use params.
    pl_module._src_state = {k: v.detach().clone().to("cpu") for k, v in pl_module.state_dict().items()}
    # Cache the param keys for quick membership checks
    pl_module._src_param_keys = set(dict(pl_module.named_parameters()).keys()) & set(pl_module._src_state.keys())

def l2sp_loss(pl_module, alpha: float = 5e-5):
    """Safe L2-SP: handles 'no matches', device/dtype/shape diffs."""
    # Ensure we always return a Tensor on the right device/dtype
    first_param = next(pl_module.parameters())
    device = first_param.device
    dtype = first_param.dtype
    # coerce alpha -> float safely
    try:
        alpha_f = float(alpha)
    except Exception:
        alpha_f = 5e-5  # fallback
    reg = torch.zeros((), device=device, dtype=dtype)

    if not hasattr(pl_module, "_src_state") or not hasattr(pl_module, "_src_param_keys"):
        return reg  # snapshot not made yet

    for k, p in pl_module.named_parameters():
        if not p.requires_grad:
            continue
        if k not in pl_module._src_param_keys:
            continue

        # bring source tensor to the same device/dtype lazily
        src = pl_module._src_state[k]
        if src.shape != p.shape:
            continue  # e.g., heads re-inited with different sizes—skip
        src = src.to(device=device, dtype=p.dtype)

        reg = reg + (p - src).pow(2).sum()

    # multiply by a tensor to keep dtype/device consistent
    return reg * torch.as_tensor(alpha_f, device=device, dtype=dtype)


# ---------- Gradual unfreeze callback ----------

class GradualUnfreeze(pl.Callback):
    """
    Stage 0: heads-only (0 .. e1-1)
    Stage 1: + late blocks (e1 .. e2-1)
    Stage 2: + some mid blocks (e2 .. end)
    """
    def __init__(self, e1=3, e2=8, warmup_epochs=0):
        super().__init__()
        self.e1, self.e2 = e1, e2
        self.warmup_epochs = warmup_epochs
        self._stage = -1

        self.head_prefixes = [
            "s2_model.classifier", "pc_model.decoder.cls_head", "fuse_head"
        ]
        self.stage2_prefixes = self.head_prefixes + [
            "s2_model.decoder",
            "s2_model.encoder.residual_block4",
            "pc_model.encoder.encoder.encoder.3",
            "pc_model.encoder.backbone.head",
        ]
        self.stage3_prefixes = self.stage2_prefixes + [
            "s2_model.encoder.residual_block3",
            "pc_model.encoder.encoder.encoder.2",
        ]

    def on_fit_start(self, trainer, pl_module):
        # Reset BN running stats and snapshot for L2-SP
        pl_module.apply(reset_bn_running_stats)
        snapshot_source_state(pl_module)

        # Start with heads-only
        freeze_backbone_keep_heads(pl_module, self.head_prefixes)
        self._stage = 0

    def on_train_epoch_start(self, trainer, pl_module):
        ep = trainer.current_epoch
        stage = 0 if ep < self.e1 else (1 if ep < self.e2 else 2)
        if stage == self._stage:
            return
        self._stage = stage

        if stage == 0:
            freeze_backbone_keep_heads(pl_module, self.head_prefixes)
        elif stage == 1:
            set_requires_grad_by_prefixes(pl_module, self.stage2_prefixes)
        else:
            set_requires_grad_by_prefixes(pl_module, self.stage3_prefixes)

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
            self.s2_model = UNet(n_channels=total_input_channels, n_classes=n_classes, aligned=True)
        elif self.cfg["network"] == "ResUnet":
            from .ResUnet import ResUnet
            self.s2_model = ResUnet(n_channels=total_input_channels, n_classes=n_classes, aligned=True)

        # Point cloud stream
        self.pc_model = PointNextModel(self.cfg, in_dim=3, n_classes=n_classes, aligned=False)

        # Decision-level fusion module
        self.fuse_head = DecisionLevelFusion(
            n_classes=n_classes,
            method=self.cfg["decision_fuse_type"],
            weight_img=self.cfg.get("decision_weight_img", 0.7),
            weight_pc=self.cfg.get("decision_weight_pc", 0.3)
        )

        self.loss_func = self.cfg["loss_func"]
        #self.criterion = nn.MSELoss()
        if self.loss_func in ["wmse", "wrmse", "wkl", "ewmse"]:
            self.weights = self.cfg[f"{self.cfg['dataset']}_class_weights"]
            if self.loss_func == "ewmse":
                self.weights = get_class_grw_weight(self.weights, n_classes, exp_scale=0.2)
        else:
            self.weights = None

        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=n_classes)

        # Optimizer and scheduler settings
        self.optimizer_type = self.cfg["optimizer"]
        self.scheduler_type = self.cfg["scheduler"]
        
        self.best_test_r2 = 0.0
        self.best_test_outputs = None
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

        image_preds, pc_preds = self.forward(images, pc_feat, point_clouds)
        if self.cfg["network"] !="ResNet":
            img_logits_list = apply_mask_per_batch(image_preds, img_masks, multi_class=True)
            #valid_pixel_preds, _ = apply_mask(image_preds, pixel_labels, img_masks, multi_class=True)
            image_preds = torch.stack([F.normalize(p.mean(dim=0), p=1, dim=0) if p.numel() > 0 else torch.zeros(image_preds.shape[1], device=image_preds.device) for p in img_logits_list], dim=0)
        fuse_preds = self.fuse_head(image_preds, pc_preds)

        r2_metric = getattr(self, f"{stage}_r2")
        weights = self.weights.to(fuse_preds.device) if self.weights is not None else None
        
        # classification loss
        loss = calc_masked_loss(self.loss_func, fuse_preds, labels, weights)
        
        r2 = r2_metric(torch.round(fuse_preds, decimals=2).view(-1), labels.view(-1))
        loss = loss + l2sp_loss(self, alpha=self.cfg.get("l2sp_alpha", 5e-5))

        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_r2", r2, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if stage == "val":
            self.validation_step_outputs.append({"val_target": labels, "val_pred": fuse_preds})
        if stage == "test":
            return labels, fuse_preds, loss
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

        last_epoch_val_r2 = r2_score(torch.round(test_pred.flatten(), decimals=1), test_true.flatten())
        self.log("ave_val_r2", last_epoch_val_r2, sync_dist=True)
        self.log("sys_r2", sys_r2, sync_dist=True)

        if last_epoch_val_r2 > self.best_test_r2:
            self.best_test_r2 = last_epoch_val_r2
            self.best_test_outputs = {"preds_all": test_pred, "true_labels_all": test_true}

            cm = self.confmat(torch.argmax(test_pred, dim=1), torch.argmax(test_true, dim=1))
            print("Confusion Matrix at best R²:", cm)
            print(f"R2 Score: {sys_r2}")
            print(f"Class R²: {r2_score(torch.round(test_pred, decimals=1), test_true, multioutput='raw_values')}")

            self.save_to_file(test_true, test_pred, self.cfg["class_names"])

        self.validation_step_outputs.clear()
        self.val_r2.reset()

    def save_to_file(self, labels, outputs, classes):
        labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
        outputs = outputs.cpu().numpy() if isinstance(outputs, torch.Tensor) else outputs
        data = {"SampleID": np.arange(labels.shape[0])}
        for i, class_name in enumerate(classes):
            data[f"True_{class_name}"] = labels[:, i]
            data[f"Pred_{class_name}"] = outputs[:, i]
        df = pd.DataFrame(data)
        output_dir = os.path.join(self.cfg["save_dir"], self.cfg["log_name"], "outputs")
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "test_outputs.csv"), mode="a")

    def configure_optimizers(self):
        param_groups = build_param_groups(self)
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-08)
        elif self.optimizer_type == "adamW":
            optimizer = torch.optim.AdamW(param_groups)  # per-group WD already set
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(param_groups, momentum=self.cfg["momentum"])
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

        # Configure the scheduler based on the input parameter
        if self.scheduler_type in ["cosine", "CosLR"]:  # treat CosLR same as cosine here
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
                },
            }
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
        else:
            return optimizer
