import torch.nn as nn
import torch.nn.functional as F
from pointnext import pointnext_s, PointNext, pointnext_b, pointnext_l, pointnext_xl


class PointNextModel(nn.Module):
    def __init__(self, config, in_dim, decoder=True):
        super(PointNextModel, self).__init__()
        self.config = config
        self.n_classes = config["test_n_classes"]
        self.task = config["task"]
        self.use_decoder = decoder

        # Initialize the PointNext encoder and decoder
        if config["encoder"] == "s":
            self.encoder = pointnext_s(
                in_dim=in_dim
            )  # Load the pointnext_s() as the encoder
        elif config["encoder"] == "b":
            self.encoder = pointnext_b(
                in_dim=in_dim
            )  # Load the pointnext_s() as the encoder
        elif config["encoder"] == "l":
            self.encoder = pointnext_l(
                in_dim=in_dim
            )  # Load the pointnext_s() as the encoder
        else:
            self.encoder = pointnext_xl(
                in_dim=in_dim
            )  # Load the pointnext_s() as the encoder

        self.backbone = PointNext(self.config["emb_dims"], encoder=self.encoder)

        self.norm = nn.BatchNorm1d(self.config["emb_dims"])
        if self.use_decoder:
            self.act = nn.ReLU()
            self.cls_head = nn.Sequential(
                nn.Linear(self.config["emb_dims"], 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(self.config["dp_pc"]),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(self.config["dp_pc"]),
                nn.Linear(256, self.n_classes),
            )

    def forward(self, pc_feat, xyz):
        pc_feats = self.norm(self.backbone(pc_feat, xyz))
        
        if self.use_decoder:
            out = pc_feats.mean(dim=-1)  # (bs, emb_dim)
            out = self.act(out)
            logits = self.cls_head(out)
            if self.task == "classify":
                return F.log_softmax(logits, dim=1), pc_feats
                # Lower T -> sharper, Higher T -> flatter
                # return F.softmax(logits / 0.5, dim=1), pc_feats
            else:
                preds = F.softmax(logits, dim=1)
                return preds, pc_feats
        else:
            return pc_feats
