import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------------
# Parts of the fusion module - s2 seasonal data early fusion
# -----------------------------------------------------------------------------------
class ConvBatchNormAct_x2(nn.Module):
    def __init__(
        self,
        in_ch,
        n_filters,
        activation=F.leaky_relu,
        spatial_dropout=0,
        inference_dropout=False,
    ):
        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(in_ch, n_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.dropout = (
            nn.Dropout2d(spatial_dropout) if spatial_dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x


class FusionBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        in_ch,
        n_filters,
        activation=nn.LeakyReLU(),
        spatial_dropout=0,
        inference_dropout=False,
    ):
        super().__init__()
        self.n_filters = n_filters
        self.activation = activation
        self.spatial_dropout = spatial_dropout
        self.inference_dropout = inference_dropout

        # Convolutional branches for each input (assuming same channels)
        self.conv_branches = nn.ModuleList(
            [
                ConvBatchNormAct_x2(
                    in_ch,
                    n_filters,
                    activation=self.activation,
                    spatial_dropout=self.spatial_dropout,
                )
                for _ in range(n_inputs)
            ]
        )

        # Excitation blocks for each branch
        self.excitation_pools = nn.ModuleList(
            [nn.AdaptiveAvgPool2d(1) for _ in range(n_inputs)]
        )
        self.excitation_denses1 = nn.ModuleList(
            [nn.Linear(n_filters, n_filters) for _ in range(n_inputs)]
        )
        self.excitation_denses2 = nn.ModuleList(
            [nn.Linear(n_filters, n_filters) for _ in range(n_inputs)]
        )

    def forward(self, xs):
        outputs = []
        for x, conv, pool, dense1, dense2 in zip(
            xs,
            self.conv_branches,
            self.excitation_pools,
            self.excitation_denses1,
            self.excitation_denses2,
        ):
            _conv = conv(x)
            _pool = pool(_conv).view(_conv.shape[0], -1)
            _dense1 = self.activation(dense1(_pool))
            _dense2 = torch.sigmoid(dense2(_dense1)).view(
                _conv.shape[0], _conv.shape[1], 1, 1
            )
            outputs.append(_conv * _dense2)
        y = torch.stack(outputs).sum(dim=0)
        return y
