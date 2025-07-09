import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR


from torchmetrics import Accuracy, F1Score, Precision, Recall


from .loss import MultiLabelFocalLoss


class PCModel(pl.LightningModule):
    def __init__(self, params, n_classes):
        super(PCModel, self).__init__()
        self.params = params
        if self.params["network"] == "pointnext":
            from .pointnext import PointNextModel
            self.model = PointNextModel(self.params, 
                                        in_dim=3, #if self.cfg["dataset"] in ["rmf", "ovf"] else 6, 
                                        n_classes=n_classes, 
                                        decoder=True,
                                        return_type='logsoftmax'
                                    )
        elif self.params["network"] == "repsurf":
            from .repsurf_ssg_umb import RepsurfaceModel
            self.model = RepsurfaceModel(n_classes=n_classes, return_type='logsoftmax', decoder=True)
        elif self.params["network"] == "repsurf2x":
            from .repsurf_ssg_umb_2x import RepsurfaceModel
            self.model = RepsurfaceModel(n_classes=n_classes, return_type='logsoftmax', decoder=True)

        # Metrics
        self.criterion = nn.NLLLoss() # or nn.CrossEntropyLoss()  if using 'logits' as output
        self.val_acc = Accuracy(task="multiclass", num_classes=n_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=n_classes, average="macro")
        self.val_precision = Precision(task="multiclass", num_classes=n_classes, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=n_classes, average="macro")


        # Optimizer and scheduler settings
        self.optimizer_type = self.params["optimizer"]
        self.scheduler_type = self.params["scheduler"]

        self.validation_step_outputs = []


    def forward(self, point_cloud, feats):
        """
        Args:
            point_cloud: Input point cloud tensor (B, N, 3), where:
            B = Batch size, N = Number of points, 3 = (x, y, z) coordinates
            feats: The normals of points
            category: Optional category tensor if categories are used
        Returns:
            logits: Class logits for each point (B, N, num_classes)
        """
        if self.params["network"] == "pointnext":
            logits, _ = self.model(point_cloud, feats)
        else:
            logits, _ = self.model(point_cloud)
        return logits

    def foward_compute_loss_and_metrics(self, xyz, feats, targets, stage="val"):
        logs = {}
        xyz = xyz.permute(0, 2, 1)
        feats = feats.permute(0, 2, 1)
        logits = self.forward(xyz, feats)  # (B, n_classes)

        # Prepare single-label target: index of max proportion per sample
        leading_label = targets.argmax(dim=1)  # shape: (B,)

        if logits.shape == leading_label.shape:  # this is not expected, check!
            raise RuntimeError("logits should be (B, n_classes), target should be (B,)")

        # If output is logits, use CrossEntropyLoss. If log_softmax, use NLLLoss.
        loss = self.criterion(logits, leading_label)

        preds = logits.argmax(dim=1)  # predicted class indices

        self.val_acc.update(preds, leading_label)
        self.val_f1.update(preds, leading_label)
        self.val_precision.update(preds, leading_label)
        self.val_recall.update(preds, leading_label)

        logs.update({
            f"{stage}_loss": loss,
            f"{stage}_acc": self.val_acc.compute(),
            f"{stage}_f1": self.val_f1.compute(),
            f"{stage}_precision": self.val_precision.compute(),
            f"{stage}_recall": self.val_recall.compute(),
        })

        # Log all metrics
        for key, value in logs.items():
            self.log(key, value, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss


    def training_step(self, batch, batch_idx):
        point_cloud = batch["point_cloud"] if "point_cloud" in batch else None
        labels = batch["label"]
        pc_feat = batch["pc_feat"] if "pc_feat" in batch else None

        return self.foward_compute_loss_and_metrics(point_cloud, pc_feat, labels, "train")

    def validation_step(self, batch, batch_idx):
        point_cloud = batch["point_cloud"] if "point_cloud" in batch else None
        labels = batch["label"]
        pc_feat = batch["pc_feat"] if "pc_feat" in batch else None

        return self.foward_compute_loss_and_metrics(point_cloud, pc_feat, labels, "val")

    def test_step(self, batch, batch_idx):
        point_cloud = batch["point_cloud"] if "point_cloud" in batch else None
        labels = batch["label"]
        pc_feat = batch["pc_feat"] if "pc_feat" in batch else None

        return self.foward_compute_loss_and_metrics(point_cloud, pc_feat, labels, "test")

    def configure_optimizers(self):
        if self.params["optimizer"] == "Adam":
            optimizer = Adam(
                self.parameters(),
                lr=self.params["lr"],
                betas=(0.9, 0.999),
                eps=1e-08,
            )
        if self.params["optimizer"] == "AdamW":
            optimizer = AdamW(
                self.parameters(), lr=self.params["lr"], weight_decay=0.05
            )
        else:
            optimizer = SGD(
                params=self.parameters(),
                lr=self.params["lr"],
                momentum=0.9,
                weight_decay=1e-4,
            )

        # Configure the scheduler based on the input parameter
        if self.params["scheduler"] == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer, patience=self.params["patience"], factor=0.5
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",  # Reduce learning rate when 'val_loss' plateaus
                },
            }
        elif self.params["scheduler"] == "steplr":
            scheduler = StepLR(optimizer, step_size=self.params["step_size"])
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.params["scheduler"] == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=200,
                eta_min=0,
                last_epoch=-1,
                verbose=False,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer
