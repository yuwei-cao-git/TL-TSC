import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .decoder import MambaFusionDecoder
from .pointnext import PointNextModel
from torchmetrics import Accuracy, ConfusionMatrix, F1Score
from .loss import apply_mask


class FusionModel(pl.LightningModule):
    def __init__(self, config, n_classes):
        super().__init__()
        self.save_hyperparameters(config)

        self.cfg = config
        self.pc_lr = self.cfg["pc_lr"]
        self.img_lr = self.cfg["img_lr"]
        self.fuse_lr = self.cfg["fuse_lr"]
        self.ms_fusion = self.cfg["use_ms"]

        if self.ms_fusion:
            from .seasonal_fusion import FusionBlock
            self.mf_module = FusionBlock(n_inputs=4, in_ch=self.cfg["n_bands"], n_filters=64)
            total_input_channels = 64
        else:
            total_input_channels = self.cfg["n_bands"] * 4

        if self.cfg["network"] == "Unet":
            from .unet import UNet
            self.s2_model = UNet(n_channels=total_input_channels, n_classes=n_classes, decoder=self.cfg["head"] in ["no_pc_head", "all_head"], return_type='logsoftmax')
        elif self.cfg["network"] == "ResUnet":
            from .ResUnet import ResUnet
            self.s2_model = ResUnet(n_channels=total_input_channels, n_classes=n_classes, decoder=self.cfg["head"] in ["no_pc_head", "all_head"], return_type='logsoftmax')
        elif self.cfg["network"] == "ResNet":
            from .resnet_fcn import FCNResNet50
            self.s2_model = FCNResNet50(n_channels=total_input_channels, n_classes=n_classes, upsample_method='bilinear', pretrained=True, decoder=self.cfg["head"] in ["no_pc_head", "all_head"], return_type='logsoftmax')

        self.pc_model = PointNextModel(self.cfg, in_dim=3, n_classes=n_classes, decoder=self.cfg["head"] in ["no_img_head", "all_head"], return_type='logsoftmax')

        if self.cfg["network"] == "ResNet":
            img_chs = 2048
        elif self.cfg["network"] == "ResUnet":
            img_chs = 1024
        elif self.cfg["network"] == "Vit":
            img_chs = 768
        else:
            img_chs = 512

        self.fuse_head = MambaFusionDecoder(
            in_img_chs=img_chs,
            in_pc_chs=self.cfg["emb_dims"],
            dim=self.cfg["fusion_dim"],
            hidden_ch=self.cfg["linear_layers_dims"],
            num_classes=n_classes,
            drop=self.cfg["dp_fuse"],
            last_feat_size=(self.cfg["tile_size"] // 8) if self.cfg["network"] == "ResNet" else (self.cfg["tile_size"] // 16),
            return_type='logsoftmax'
        )

        self.criterion = nn.NLLLoss() #SmoothClsLoss(smoothing_ratio=0.1)
        self.metric = Accuracy(task="multiclass", num_classes=n_classes)
        self.f1_metric = F1Score(task="multiclass", num_classes=n_classes, average='macro')
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=n_classes)
        self.validation_step_outputs = []

    def forward(self, images, pc_feat, xyz):
        pc_feat = pc_feat.permute(0, 2, 1) if pc_feat is not None else None
        xyz = xyz.permute(0, 2, 1) if xyz is not None else None

        if self.ms_fusion:
            stacked_features = self.mf_module(images)
        else:
            if self.cfg["dataset"] in ['rmf', 'ovf']:
                stacked_features = torch.cat(images, dim=1)
            else:
                B, _, _, H, W = images.shape
                stacked_features = images.view(B, -1, H, W)

        image_outputs, img_emb = None, None
        if self.cfg["head"] in ["no_pc_head", "all_head"]:
            image_outputs, img_emb = self.s2_model(stacked_features)
        else:
            img_emb = self.s2_model(stacked_features)

        point_outputs, pc_emb = None, None
        if self.cfg["head"] in ["no_img_head", "all_head"]:
            point_outputs, pc_emb = self.pc_model(pc_feat, xyz)
        else:
            pc_emb = self.pc_model(pc_feat, xyz)

        class_output = self.fuse_head(img_emb, pc_emb)

        if self.cfg["head"] == "no_pc_head":
            return image_outputs, None, class_output
        elif self.cfg["head"] == "fuse_head":
            return None, None, class_output
        elif self.cfg["head"] == "no_img_head":
            return None, point_outputs, class_output
        else:
            return image_outputs, point_outputs, class_output

    def forward_and_metrics(self, images, img_masks, pc_feat, point_clouds, labels, pixel_labels, stage):
        pixel_preds, pc_preds, fuse_preds = self.forward(images, pc_feat, point_clouds)
        logs = {}

        true_cls = labels.argmax(dim=1)

        if pc_preds is not None:
            loss_point = self.criterion(pc_preds, true_cls)
            pred_pc_cls = pc_preds.argmax(dim=1)
            acc_pc = self.metric(pred_pc_cls, true_cls)
            f1_pc = self.f1_metric(pc_preds, true_cls)
            logs.update({f"pc_{stage}_acc": acc_pc, f"pc_{stage}_f1": f1_pc, f"pc_{stage}_loss": loss_point})
        else:
            loss_point = 0

        if pixel_preds is not None and pixel_labels is not None:
            valid_pixel_preds, valid_pixel_true = apply_mask(pixel_preds, pixel_labels, img_masks, multi_class=True)
            if valid_pixel_preds.numel() > 0:
                true_pix_cls = valid_pixel_true.argmax(dim=1)
                loss_pixel = self.criterion(valid_pixel_preds, true_pix_cls)
                pred_pix_cls = valid_pixel_preds.argmax(dim=1)
                acc_pix = self.metric(pred_pix_cls, true_pix_cls)
                f1_pix = self.f1_metric(valid_pixel_preds, true_pix_cls)
                logs.update({f"pixel_{stage}_acc": acc_pix, f"pixel_{stage}_f1": f1_pix, f"pixel_{stage}_loss": loss_pixel})

                avg_preds = valid_pixel_preds.mean(dim=0, keepdim=True)
                consistency_loss = nn.KLDivLoss(reduction='batchmean')(valid_pixel_preds.log_softmax(dim=1), avg_preds.softmax(dim=1).expand_as(valid_pixel_preds))
                logs.update({f"pixel_{stage}_consistency": consistency_loss})
                loss_pixel += 0.1 * consistency_loss
            else:
                loss_pixel = 0
        else:
            loss_pixel = 0

        loss_fuse = self.criterion(fuse_preds, true_cls)
        pred_fuse_cls = fuse_preds.argmax(dim=1)
        acc_fuse = self.metric(pred_fuse_cls, true_cls)
        f1_fuse = self.f1_metric(fuse_preds, true_cls)
        logs.update({f"{stage}_acc": acc_fuse, f"{stage}_f1": f1_fuse, f"{stage}_loss": loss_fuse})

        total_loss = loss_fuse + loss_point + loss_pixel
        logs.update({f"{stage}_loss": total_loss})

        for key, value in logs.items():
            self.log(key, value, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if stage == "val":
            self.validation_step_outputs.append({"val_target": labels, "val_pred": fuse_preds})

        if stage == "test":
            return labels, fuse_preds, total_loss
        else:
            return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="test")

    def _shared_step(self, batch, stage):
        return self.forward_and_metrics(
            batch.get("images"),
            batch.get("mask"),
            batch.get("pc_feat"),
            batch.get("point_cloud"),
            batch.get("label"),
            batch.get("per_pixel_labels"),
            stage
        )

    def on_validation_epoch_end(self):
        test_true = torch.cat([x["val_target"] for x in self.validation_step_outputs], dim=0)
        test_pred = torch.cat([x["val_pred"] for x in self.validation_step_outputs], dim=0)
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

        self.save_to_file(true_classes, test_pred, self.cfg["class_names"])
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
        params = []

        # Include parameters from the image model
        if self.ms_fusion:
            mf_params = list(self.mf_module.parameters())
            params.append({"params": mf_params, "lr": self.img_lr})

        if any(p.requires_grad for p in self.s2_model.parameters()):
            image_params = list(self.s2_model.parameters())
            params.append({"params": image_params, "lr": self.img_lr})

        # Include parameters from the point cloud model
        if any(p.requires_grad for p in self.pc_model.parameters()):
            point_params = list(self.pc_model.parameters())
            params.append({"params": point_params, "lr": self.pc_lr})

        # Include parameters from the fusion layers
        if any(p.requires_grad for p in self.fuse_head.parameters()):
            fusion_params = list(self.fuse_head.parameters())
            params.append({"params": fusion_params, "lr": self.fuse_lr})

        if self.cfg["use_ms"]:
            mf_params = list(self.mf_module.parameters())
            params.append({"params": mf_params, "lr": self.img_lr})
        optimizer = torch.optim.AdamW(params, weight_decay=self.cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
