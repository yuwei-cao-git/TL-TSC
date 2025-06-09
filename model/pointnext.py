import torch.nn as nn
import torch.nn.functional as F
from pointnext import pointnext_s, PointNext, pointnext_b, pointnext_l, pointnext_xl


class PointNextEncoder(nn.Module):
    def __init__(self, config, in_dim):
        super(PointNextEncoder, self).__init__()
        self.config = config

        # Choose encoder type
        if config["encoder"] == "s":
            self.encoder = pointnext_s(in_dim=in_dim)
        elif config["encoder"] == "b":
            self.encoder = pointnext_b(in_dim=in_dim)
        elif config["encoder"] == "l":
            self.encoder = pointnext_l(in_dim=in_dim)
        else:
            self.encoder = pointnext_xl(in_dim=in_dim)

        self.backbone = PointNext(config["emb_dims"], encoder=self.encoder)
        self.norm = nn.BatchNorm1d(config["emb_dims"])

    def forward(self, pc_feat, xyz):
        features = self.backbone(pc_feat, xyz)  # (B, C, N)
        features = self.norm(features)
        return features

class PointNextClassifier(nn.Module):
    def __init__(self, config, n_classes):
        super(PointNextClassifier, self).__init__()
        self.config = config
        self.n_classes = n_classes
        self.task = config["task"]

        self.act = nn.ReLU()
        self.cls_head = nn.Sequential(
            nn.Linear(config["emb_dims"], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(config["dp_pc"]),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(config["dp_pc"]),
            nn.Linear(256, self.n_classes),
        )

    def forward(self, pc_feats):
        out = pc_feats.mean(dim=-1)  # Global feature: (B, C)
        out = self.act(out)
        logits = self.cls_head(out)

        preds = F.softmax(logits, dim=1)
        return preds
        
class PointNextModel(nn.Module):
    def __init__(self, config, in_dim, n_classes, decoder=True):
        super(PointNextModel, self).__init__()
        self.use_decoder = decoder
        self.encoder = PointNextEncoder(config, in_dim)
        self.decoder = PointNextClassifier(config, n_classes) if decoder else None

    def forward(self, pc_feat, xyz):
        pc_feats = self.encoder(pc_feat, xyz)
        if self.use_decoder:
            return self.decoder(pc_feats), pc_feats
        else:
            return pc_feats