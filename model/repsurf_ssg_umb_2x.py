"""
Author: Haoxi Ran
Date: 05/10/2022
"""

import torch.nn as nn
import torch.nn.functional as F
from modules.repsurface_utils import SurfaceAbstractionCD, UmbrellaSurfaceConstructor


class RepsurfaceModel(nn.Module):
    def __init__(self, n_classes, return_type='softmax', decoder=True):
        super(RepsurfaceModel, self).__init__()
        center_channel = 3
        repsurf_channel = 10

        self.init_nsample = 1024
        self.return_dist = False
        
        self.return_type = return_type
        self.decoder = decoder
        self.surface_constructor = UmbrellaSurfaceConstructor(8 + 1, repsurf_channel,
                                        return_dist=False, aggr_type=sum,
                                        cuda=True)
        self.sa1 = SurfaceAbstractionCD(npoint=512, radius=0.1, nsample=24, feat_channel=repsurf_channel,
                                        pos_channel=center_channel, mlp=[128, 128, 256], group_all=False,
                                        return_polar=False, cuda=True)
        self.sa2 = SurfaceAbstractionCD(npoint=128, radius=0.2, nsample=24, feat_channel=256 + repsurf_channel,
                                        pos_channel=center_channel, mlp=[256, 256, 512], group_all=False,
                                        return_polar=False, cuda=True)
        self.sa3 = SurfaceAbstractionCD(npoint=32, radius=0.4, nsample=24, feat_channel=512 + repsurf_channel,
                                        pos_channel=center_channel, mlp=[512, 512, 1024], group_all=False,
                                        return_polar=False, cuda=True)
        self.sa4 = SurfaceAbstractionCD(npoint=None, radius=None, nsample=None, feat_channel=1024 + repsurf_channel,
                                        pos_channel=center_channel, mlp=[1024, 1024, 2048], group_all=True,
                                        return_polar=False, cuda=True)
        if self.decoder:
            self.classfier = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(True),
                nn.Dropout(0.4),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.Dropout(0.4),
                nn.Linear(256, n_classes))

    def forward(self, points):
        # init
        center = points[:, :3, :]

        normal = self.surface_constructor(center)

        center, normal, feature = self.sa1(center, normal, None)
        center, normal, feature = self.sa2(center, normal, feature)
        center, normal, feature = self.sa3(center, normal, feature)
        _, _, feature = self.sa4(center, normal, feature)

        feature = feature.view(-1, 2048)
        if self.decoder:
            logits = self.classfier(feature)
            if self.return_type == 'logsoftmax':
                return F.log_softmax(logits, -1), feature
            elif self.return_type == 'logits':
                return logits, feature
            else:
                return F.softmax(feature, dim=1), feature
        else:
            return feature
