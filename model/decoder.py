import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from mamba_ssm import Mamba
from einops import rearrange
from timm.layers import trunc_normal_

# -----------------------------------------------------------------------------------
# Fusion block & Classify head - late feature fusion
# -----------------------------------------------------------------------------------

class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        bias=False,
    ):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
            ),
            norm_layer(out_channels),
            nn.ReLU6(),
        )

class MambaFusion(nn.Module):
    def __init__(
        self,
        in_img_chs,  # Input channels for image
        in_pc_chs,  # Input channels for point cloud
        dim=128,
        d_state=16,
        d_conv=4,
        expand=2,
        last_feat_size=16
    ):
        super().__init__()
        # Sample the grids in 2D space
        xx = np.linspace(-0.3, 0.3, last_feat_size, dtype=np.float32)
        yy = np.linspace(-0.3, 0.3, last_feat_size, dtype=np.float32)
        self.grid = np.array(np.meshgrid(xx, yy))  # (2, 8, 8)

        # reshape
        self.grid = torch.Tensor(self.grid).view(2, -1)  # (2, 8, 8) -> (2, 8 * 8)

        self.m = self.grid.shape[1]
        
        # Calculate the combined input channels
        combined_in_chs = in_img_chs + in_pc_chs + 2
        assert isinstance(combined_in_chs, int) and isinstance(
            combined_in_chs, int
        ), "in_channels and out_channels must be integers"
        
        # Pooling scales for the pooling layers
        pool_scales = self.generate_arithmetic_sequence(1, last_feat_size, last_feat_size // 4)
        self.pool_len = len(pool_scales)
        # Initialize pooling layers
        self.pool_layers = nn.ModuleList()

        # First pooling layer with 1x1 convolution and adaptive average pool
        self.pool_layers.append(
            nn.Sequential(
                ConvBNReLU(combined_in_chs, dim, kernel_size=1), nn.AdaptiveAvgPool2d(1)
            )
        )

        # Add the rest of the pooling layers based on the pooling scales
        for pool_scale in pool_scales[1:]:
            self.pool_layers.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvBNReLU(combined_in_chs, dim, kernel_size=1),
                )
            )

        # Mamba module
        self.mamba = Mamba(
            d_model=dim * self.pool_len
            + combined_in_chs,  # Model dimension, to be set dynamically in forward
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x, pc_emb):
        B, _, H, W = x.shape
        # repeat grid for batch operation
        grid = self.grid.to(x.device)  # (2, 8 * 8)
        grid = grid.unsqueeze(0).repeat(B, 1, 1)  # (B, 2, 88 * 45)
    
        # Pool over points (max pooling over point cloud features)
        #pc_emb = torch.max(pc_emb, dim=2)[0]  # Shape: (batch_size, feature_dim)
        
        # Expand point cloud features to (B, C_point, H, W)
        point_cloud_expanded = pc_emb.unsqueeze(2).repeat(
            1, 1, self.m
        )  # (BS, feature_dim, 8 * 8)
        point_cloud_expanded = point_cloud_expanded.view(B, -1, H, W)
    
        grid = grid.view(B, -1, H, W)

        # Concatenate image and point cloud features
        combined_features = torch.cat([x, grid, point_cloud_expanded], dim=1)

        # Pooling and Mamba layers
        res = combined_features

        ppm_out = [res]
        for p in self.pool_layers:
            pool_out = p(combined_features)
            pool_out = F.interpolate(
                pool_out, (H, W), mode="bilinear", align_corners=False
            )
            ppm_out.append(pool_out)
        x = torch.cat(ppm_out, dim=1)
        _, chs, _, _ = x.shape
        x = rearrange(x, "b c h w -> b (h w) c", b=B, c=chs, h=H, w=W)
        x = self.mamba(x)
        x = x.transpose(2, 1).view(B, chs, H, W)
        return x

    def generate_arithmetic_sequence(self, start, stop, step):
        sequence = []
        for i in range(start, stop, step):
            sequence.append(i)
        return sequence  # 1, 3, 5, 7

class MLP(nn.Module):
    def __init__(
        self,
        in_ch=1024,
        hidden_ch=[128, 128],
        num_classes=9,
        dropout_prob=0.1,
        return_type='logits',
    ):
        super(MLP, self).__init__()
        self.return_type = return_type
        self.conv = ConvBNReLU(in_ch, in_ch, kernel_size=3)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_ch, hidden_ch[0])
        self.bn1 = nn.BatchNorm1d(hidden_ch[0])
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_ch[0], hidden_ch[1])
        self.bn2 = nn.BatchNorm1d(hidden_ch[1])
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(hidden_ch[1], num_classes)  # Output layer

    def forward(self, x):
        x = self.conv(x)
        x = self.pooling(x)  # Global pooling to (B, in_ch)
        x = x.view(x.size(0), -1)     # (B, C)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        logits = self.fc3(x)  # [batch_size, num_classes]
        if self.return_type == 'logsoftmax':
            return F.log_softmax(logits, dim=1)
        elif self.return_type == 'logits':
            return logits
        else:
            # Lower T -> sharper, Higher T -> flatter
            # return F.log_softmax(logits / 0.5, dim=1)
            class_output = F.softmax(logits, dim=1)
            return class_output

class MambaFusionDecoder(nn.Module):
    def __init__(
        self,
        in_img_chs,  # Input channels for image
        in_pc_chs,  # Input channels for point cloud
        dim=128,
        hidden_ch=[128, 128],
        num_classes=9,
        drop=0.1,
        d_state=8,
        d_conv=4,
        expand=2,
        last_feat_size=8,
        return_type='softmax',
        return_feature=False
    ):
        super(MambaFusionDecoder, self).__init__()
        self.mamba = MambaFusion(
            in_img_chs,  # Input channels for image
            in_pc_chs,
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            last_feat_size=last_feat_size
        )
        # Initialize MLPBlock (now it takes output channels as num_classes)
        self.mlp_block = MLP(
            in_ch=dim * self.mamba.pool_len
            + in_img_chs
            + in_pc_chs
            + 2,  # Adjusted input channels after fusion
            hidden_ch=hidden_ch,
            num_classes=num_classes,
            dropout_prob=drop,
            return_type=return_type,
        )
        self.return_feature=return_feature

    def forward(self, img_emb, pc_emb):
        x = self.mamba(img_emb, pc_emb) # torch.Size([8, 2306, 8, 8])
        class_output = self.mlp_block(x)  # Class output of shape (B, num_classes)
        if self.return_feature:
            return class_output, x
        else:
            return class_output
    
# -----------------------------------------------------------------------------------
# Use mamba & upsamping as decoder: MambaDecoder 
# -----------------------------------------------------------------------------------
class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )
            
class ConvFFN(nn.Module):
    def __init__(self, in_ch=128, hidden_ch=512, out_ch=128, drop=0.):
        super(ConvFFN, self).__init__()
        self.conv = ConvBNReLU(in_ch, in_ch, kernel_size=3)
        self.fc1 = Conv(in_ch, hidden_ch, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = Conv(hidden_ch, out_ch, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class MambaBlock(nn.Module):
    def __init__(self, in_chs=512, dim=128, hidden_ch=512, out_ch=128, drop=0.1, d_state=16, d_conv=4, expand=2, last_feat_size=16):
        super(MambaBlock, self).__init__()
        self.mamba = MambaFusion(in_img_chs=in_chs, in_pc_chs=0, dim=dim, d_state=d_state, d_conv=d_conv, expand=expand, last_feat_size=last_feat_size, fusion=False)
        self.conv_ffn = ConvFFN(in_ch=dim*self.mamba.pool_len+in_chs, hidden_ch=hidden_ch, out_ch=out_ch, drop=drop)

    def forward(self, x):
        x = self.mamba(x)
        x = self.conv_ffn(x)

        return x

class MambaDecoder(nn.Module):
    def __init__(self, encoder_channels=(64, 128, 256, 512), decoder_channels=128, num_classes=9, last_feat_size=16):
        super().__init__()
        self.b3 = MambaBlock(in_chs=encoder_channels[-1], dim=decoder_channels, last_feat_size=last_feat_size)
        self.up_conv = nn.Sequential(ConvBNReLU(decoder_channels, decoder_channels),
                            nn.Upsample(scale_factor=2),
                            ConvBNReLU(decoder_channels, decoder_channels),
                            nn.Upsample(scale_factor=2),
                            ConvBNReLU(decoder_channels, decoder_channels),
                            nn.Upsample(scale_factor=2),
                            )
        self.pre_conv = ConvBNReLU(encoder_channels[0], decoder_channels)
        self.head = nn.Sequential(ConvBNReLU(decoder_channels, decoder_channels // 2),
                        nn.Upsample(scale_factor=2, mode='bilinear'),
                        ConvBNReLU(decoder_channels // 2, decoder_channels // 2),
                        nn.Upsample(scale_factor=2, mode='bilinear'),
                        Conv(decoder_channels // 2, num_classes, kernel_size=1))
        self.apply(self._init_weights)

    def forward(self, x0, x3):
        x3 = self.b3(x3)
        x4 = self.up_conv(x3)
        x = x4 + self.pre_conv(x0) #shortcut
        x = self.head(x)
        return x, x3

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
# -----------------------------------------------------------------------------------
# Use just upsamping as decoder: SimpleDecoder 
# -----------------------------------------------------------------------------------
            
class SimpleUpDecoder(nn.Module):
    def __init__(self, encoder_channel=512, decoder_channels=128, num_classes=9):
        super().__init__()
        self.up_conv = nn.Sequential(ConvBNReLU(encoder_channel, decoder_channels),
                            nn.Upsample(scale_factor=2),
                            ConvBNReLU(decoder_channels, decoder_channels),
                            nn.Upsample(scale_factor=2),
                            ConvBNReLU(decoder_channels, decoder_channels),
                            nn.Upsample(scale_factor=2),
                            )
        self.head = nn.Sequential(ConvBNReLU(decoder_channels, decoder_channels // 2),
                        ConvBNReLU(decoder_channels // 2, decoder_channels // 2),
                        Conv(decoder_channels // 2, num_classes, kernel_size=1))
        self.apply(self._init_weights)

    def forward(self, x):
        x1 = self.up_conv(x)
        x = self.head(x1)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
            
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionLevelFusion(nn.Module):
    def __init__(self, n_classes, method="average", weight_img=0.5, weight_pc=0.5):
        super().__init__()
        self.method = method
        self.weight_img = weight_img
        self.weight_pc = weight_pc

        if method == "mlp":
            self.fuse_mlp = nn.Sequential(
                nn.Linear(2 * n_classes, 128),
                nn.ReLU(),
                nn.Linear(128, n_classes)
            )

    def forward(self, img_logits, pc_logits):
        # Only one modality available
        if img_logits is None:
            return pc_logits
        if pc_logits is None:
            return img_logits

        if self.method == "average":
            return (img_logits + pc_logits) / 2

        elif self.method == "weighted":
            return self.weight_img * img_logits + self.weight_pc * pc_logits

        elif self.method == "mlp":
            # Ensure both logits have the same shape
            if img_logits.shape != pc_logits.shape:
                raise ValueError("Shape mismatch between img_logits and pc_logits")
            fused_input = torch.cat([img_logits, pc_logits], dim=1)
            return self.fuse_mlp(fused_input)

        else:
            raise ValueError(f"Unknown fusion method: {self.method}")