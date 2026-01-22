import torch.nn as nn
import torch.nn.functional as F
from pointnext import pointnext_s, PointNext, pointnext_b, pointnext_l, pointnext_xl
from .decoder import PCCAHead

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
        self.act = nn.ReLU()

    def forward(self, pc_feat, xyz):
        features = self.backbone(pc_feat, xyz)  # (B, C, N)
        features = self.norm(features)
        out = features.mean(dim=-1)  # Global feature: (B, C)
        out = self.act(out)
        return out

class PointNextClassifier(nn.Module):
    def __init__(self, config, n_classes, return_type, aligned):
        super(PointNextClassifier, self).__init__()
        self.config = config
        self.n_classes = n_classes
        self.return_type = return_type
        self.aligned = aligned
        
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
        if aligned:
            self.disalign_head = PCCAHead(self.n_classes, self.n_classes)

    def forward(self, pc_feats):
        logits = self.cls_head(pc_feats)
        if self.aligned:
            logits = self.disalign_head(logits)
        if self.return_type == 'logsoftmax':
            return F.log_softmax(logits, dim=1)
        elif self.return_type == 'logits':
            return logits
        elif self.return_type == 'softmax':
            preds = F.softmax(logits, dim=1)
            return preds
        else:
            return logits
        
class PointNextModel(nn.Module):
    def __init__(self, config, in_dim, n_classes, decoder=True, return_type='softmax', aligned=False):
        super(PointNextModel, self).__init__()
        self.use_decoder = decoder
        self.align_header = aligned
        self.encoder = PointNextEncoder(config, in_dim)
        if self.use_decoder:
            self.decoder = PointNextClassifier(config, n_classes, return_type=return_type, aligned=aligned)

    def forward(self, pc_feat, xyz):
        pc_feats = self.encoder(pc_feat, xyz)
        if self.use_decoder:
            return self.decoder(pc_feats), pc_feats
        else:
            return pc_feats