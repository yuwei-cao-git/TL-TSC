import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from torchmetrics.regression import R2Score
from torchmetrics.functional import r2_score
from .MambaFusionV2 import MambaFusionV2

# --- small helpers ---

def resize_mask_to_feat(mask, feat_h, feat_w):
    # mask: [B,H,W] or [B,1,H,W] -> [B,1,feat_h,feat_w] in [0,1]
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    return F.interpolate(mask, size=(feat_h, feat_w), mode="area")

def masked_avg(feat_map, mask_aligned, eps=1e-6):
    """
    feat_map:     [B, C, H, W]
    mask_aligned: [B, 1, H, W] (float in [0,1])  -- already resized to feat_map size
    returns:      [B, C]
    """
    if mask_aligned.dim() == 3:
        mask_aligned = mask_aligned.unsqueeze(1)  # [B,1,H,W]
    mask = mask_aligned.to(feat_map.dtype)

    num = (feat_map * mask).sum(dim=(2, 3))             # [B, C]
    den = mask.sum(dim=(2, 3)).clamp_min(eps)           # [B, 1]  <-- keep channel dim!
    return num / den                                    # [B, C] (broadcast on channel)


def kl_loss_from_logits(logits, target_props, reduction='batchmean', eps=1e-8):
    """
    logits: [B, C]
    target_props: [B, C], assumed >=0 and summing to 1 per row
    """
    log_pred = F.log_softmax(logits, dim=1)                 # distribution over classes
    target = target_props  # be safe
    return F.kl_div(log_pred, target, reduction=reduction)

class RegionHead(nn.Module):
    def __init__(self, feat_dim, n_classes):
        super().__init__()
        # Simple linear head works well for domain-specific calibration
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, n_classes)
        )

    def forward(self, feat):
        return self.head(feat)  # logits

# --- main module ---
class MultiHeadFusionModel(pl.LightningModule):
    """
    Shared backbones (S2 + PointNext) -> fuse to shared features -> region-specific heads.
    Loss: KLDiv on proportions per region.
    """
    def __init__(self, config, region_class_map, feat_dim=256):
        """
        region_class_map: dict like {'A': 9, 'B': 6}
        feat_dim: dimensionality of shared fused features
        """
        super().__init__()
        self.save_hyperparameters(config)
        self.cfg = config
        self.lr = self.cfg["lr"]
        self.ms_fusion = self.cfg.get("use_ms", False)
        self.fusion_dim = self.cfg.get("fusion_dim", 256)

        # -------- Seasonal fusion (if you use it) ----------
        if self.ms_fusion:
            from .seasonal_fusion import FusionBlock
            self.mf_module = FusionBlock(
                n_inputs=len(self.cfg[f"{self.cfg['dataset']}_season_map"]),
                in_ch=self.cfg["n_bands"],
                n_filters=64
            )
            img_in_ch = 64
        else:
            img_in_ch = self.cfg["n_bands"] * len(self.cfg[f"{self.cfg['dataset']}_season_map"])

        # -------- Image backbone (should return features) ----------
        # Modify your models to return features of size feat_dim
        if self.cfg["network"] == "Unet":
            from .unet import UNet
            self.s2_model = UNet(n_channels=img_in_ch, n_classes=None, decoder=False)  # <-- adapt UNet to output features
        elif self.cfg["network"] == "ResUnet":
            from .ResUnet import ResUnet
            self.s2_model = ResUnet(n_channels=img_in_ch, n_classes=None, decoder=False)
        else:
            raise ValueError("Unknown image network")

        # -------- Point cloud backbone (should return features) ----------
        from .pointnext import PointNextModel
        self.pc_model = PointNextModel(self.cfg, in_dim=3, n_classes=None, decoder=False)

        # -------- Feature-level fusion ----------
        # If your existing DecisionLevelFusion supports features, use it.
        # Otherwise, this simple weighted adder works.
        self.fuse_feat = MambaFusionV2(in_img_chs=1024,
            in_pc_chs=(self.cfg["emb_dims"]),
            out_ch=self.cfg.get("fusion_dim", 256),
            last_feat_size=8,
            d_state=16, d_conv=4, expand=2, width=256, use_film=True
            )

        # -------- Region-specific heads ----------
        self.region_heads = nn.ModuleDict()
        for region_key, ncls in region_class_map.items():
            self.region_heads[region_key] = RegionHead(self.fusion_dim, ncls)
        
        # -------- Weighted loss from different regions ----------
        from .loss import AutomaticWeightedLoss
        self.awl = AutomaticWeightedLoss(2)

        # -------- Metrics ----------
        # Maintain per-region R2 (on proportions flattened)
        self.train_r2 = {k: R2Score().to(self.device) for k in region_class_map.keys()}
        self.val_r2   = {k: R2Score().to(self.device) for k in region_class_map.keys()}
        self.test_r2  = {k: R2Score().to(self.device) for k in region_class_map.keys()}
        self.validation_step_outputs = []

        # Optimizer / scheduler
        self.optimizer_type = self.cfg["optimizer"]
        self.scheduler_type = self.cfg["scheduler"]

        # Optional class weights per region (if you want weighted KL or per-class scaling)
        self.region_weights = None  # hook point if needed

    # ----- forward -----
    def forward_backbones(self, images, pc_feat, xyz):
        if self.ms_fusion:
            stacked = self.mf_module(images)
        else:
            B, _, _, H, W = images.shape
            stacked = images.view(B, -1, H, W)

        # Expect your backbones to return (feature, aux/None)
        img_feat = self.s2_model(stacked)                 # [B, C_img, H, W]
        pc_feat = pc_feat.permute(0, 2, 1) if pc_feat is not None else None
        xyz = xyz.permute(0, 2, 1) if xyz is not None else None
        pc_token_raw = self.pc_model(pc_feat, xyz)             # [B, D]
        if pc_token_raw.dim() == 3:
            # quick attention or mean/max pool
            pc_emb = pc_token_raw.max(dim=1)[0]
        else:
            pc_emb = pc_token_raw

        fused_map = self.fuse_feat(img_feat, pc_emb)           # [B, C_out, H, W]
        return fused_map

    def forward(self, images, img_masks, pc_feat, point_clouds, region_key):
        fused_map = self.forward_backbones(images, pc_feat, point_clouds)
        #fused_feat = fused_map.mean(dim=(2,3))                  # GAP -> [B, C_out]
        mask8 = resize_mask_to_feat(img_masks, 8, 8)   # [B,1,8,8], float in [0,1]
        fused_feat = masked_avg(fused_map, mask8)        # [B,C]
        logits = self.region_heads[region_key](fused_feat)       # [B, C_region]
        return logits, mask8

    def on_fit_start(self):
        for d in [self.train_r2, self.val_r2, self.test_r2]:
            for m in d.values():
                m.to(self.device)

    # ----- one step (train/val/test) -----
    def step_impl(self, batch, stage):
        region_key = batch["region"]  # e.g. 'A' or 'B' (string)
        images = batch["images"]
        img_masks = batch.get("mask")
        pc_feat = batch["pc_feat"]
        point_clouds = batch["point_cloud"]
        labels = batch["label"]       # [B, C_region] proportions for this region
        
        if region_key == 'B':
            with torch.no_grad():
                sums = labels.sum(dim=1)
                self.log(f"{stage}_B/label_sum_mean", sums.mean(), prog_bar=False, batch_size=labels.size(0))
                self.log(f"{stage}_B/label_sum_abs_err", (sums-1).abs().mean(), prog_bar=False, batch_size=labels.size(0))
                self.log(f"{stage}_B/label_nan_frac", torch.isnan(labels).float().mean(), prog_bar=False, batch_size=labels.size(0))
                self.log(f"{stage}_B/label_var_mean", labels.var(dim=0).mean(), prog_bar=False, batch_size=labels.size(0))

        logits, mask8 = self.forward(images, img_masks, pc_feat, point_clouds, region_key)
        pred_props = F.softmax(logits, dim=1)
        
        # KLDiv on distributions
        loss_core = kl_loss_from_logits(logits, labels)
        
        # after computing mask8
        with torch.no_grad():
            cov = mask8.mean(dim=(2,3)).mean()  # [B,1,H,W] -> scalar
            self.log(f"{stage}_mask_cov/{region_key}", cov, on_step=True, prog_bar=False, batch_size=labels.size(0))
            if cov < 0.2:
                loss_core = (loss * mask8.mean(dim=(2,3)).squeeze(1)).mean()
        
        # entropy regularizer (prevent overconfident wrong preds)
        entropy = -(pred_props * pred_props.clamp_min(1e-8).log()).sum(dim=1).mean()
        loss = loss_core - self.cfg.get("entropy_w", 0.01) * entropy

        # For R2 we can compare softmaxed preds vs labels per region
        with torch.no_grad():
            r2_metric = getattr(self, f"{stage}_r2")[region_key]
            r2_val = r2_metric(pred_props.view(-1), labels.view(-1))
            self.log(f"{stage}_B/pred_sum", pred_props.sum(dim=1).mean(), prog_bar=False, batch_size=labels.size(0))
            self.log(f"{stage}_B/label_sum", labels.sum(dim=1).mean(), prog_bar=False, batch_size=labels.size(0))

        bs = labels.shape[0]

        self.log(f"{stage}_loss/{region_key}", loss,
                on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
                batch_size=bs)
        self.log(f"{stage}_r2/{region_key}", r2_val,
                on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
                batch_size=bs)

        if stage == "val":
            self.validation_step_outputs.append({
                "region": region_key, "target": labels.detach(), "pred": pred_props.detach()
            })

        if stage == "test":
            return loss
        return loss

    # ----- lightning hooks -----
    def training_step(self, batch, batch_idx):
        loss_A = self.step_impl(batch["A"], "train")
        loss_B = self.step_impl(batch["B"], "train")
        loss = self.awl(loss_A, loss_B)
        return loss

    def validation_step(self, batch, batch_idx):
        loss_A = self.step_impl(batch["A"], "val")
        loss_B = self.step_impl(batch["B"], "val")
        return 0.5 * (loss_A + loss_B)

    def on_validation_epoch_end(self):
        by_region = {}
        for out in self.validation_step_outputs:
            k = out["region"]
            by_region.setdefault(k, {"t": [], "p": []})
            by_region[k]["t"].append(out["target"])
            by_region[k]["p"].append(out["pred"])

        r2_values = []
        for k, d in by_region.items():
            t = torch.cat(d["t"], dim=0)
            p = torch.cat(d["p"], dim=0)
            r2_epoch = r2_score(p.flatten(), t.flatten())
            self.log(f"val_epoch_r2/{k}", r2_epoch, prog_bar=True, sync_dist=True)
            r2_values.append(r2_epoch)

        if r2_values:
            r2_mean = torch.stack(r2_values).mean()
            self.log("val_epoch_r2/mean", r2_mean, prog_bar=True, sync_dist=True)

        self.validation_step_outputs.clear()
        for m in self.val_r2.values():
            m.reset()


    def test_step(self, batch, batch_idx):
        loss_A = self.step_impl(batch["A"], "test")
        loss_B = self.step_impl(batch["B"], "test")
        return 0.5 * (loss_A + loss_B)


    # ----- save outputs (optional) -----
    def save_to_file(self, labels, outputs, classes, save_path):
        labels = labels.cpu().numpy()
        outputs = outputs.cpu().numpy()
        data = {"SampleID": np.arange(labels.shape[0])}
        for i, name in enumerate(classes):
            data[f"True_{name}"] = labels[:, i]
            data[f"Pred_{name}"] = outputs[:, i]
        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

    # ----- optimizers / schedulers -----
    def configure_optimizers(self):
        params = list(self.s2_model.parameters()) + \
                list(self.pc_model.parameters()) + \
                list(self.fuse_feat.parameters()) + \
                list(self.region_heads.parameters()) + \
                list(self.awl.parameters())

        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg["step_size"], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}