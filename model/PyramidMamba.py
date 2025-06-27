# https://github.com/WangLibo1995/GeoSeg/blob/main/geoseg/models/PyramidMamba.py
import torch.nn as nn

import timm
from .decoder import MambaDecoder
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )

class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

class EfficientPyramidMamba(nn.Module):
    def __init__(self,
                backbone_name='swsl_resnet18',
                pretrained=True,
                num_classes=6,
                decoder_channels=128,
                last_feat_size=16  # last_feat_size=input_img_size // 32
                ):
        super().__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                            out_indices=(1, 4), pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()
        self.decoder = MambaDecoder(encoder_channels=encoder_channels, decoder_channels=decoder_channels, num_classes=num_classes, last_feat_size=last_feat_size)

    def forward(self, x):
        x0, x3 = self.backbone(x)
        logits, feat = self.decoder(x0, x3)
        preds = F.softmax(logits, dim=1)
        return preds, feat

class PyramidMamba(nn.Module):
    def __init__(self,
                backbone_name='swin_base_patch4_window12_384.ms_in22k_ft_in1k',
                pretrained=True,
                num_classes=6,
                decoder_channels=128,
                last_feat_size=32,
                img_size=1024
                ):
        super().__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32, img_size=img_size,
                            out_indices=(-4, -1), pretrained=pretrained)

        encoder_channels = self.backbone.feature_info.channels()
        self.decoder = MambaDecoder(encoder_channels=encoder_channels, decoder_channels=decoder_channels, num_classes=num_classes, last_feat_size=last_feat_size)

    def forward(self, x):
        x0, x3 = self.backbone(x)
        x0 = x0.permute(0, 3, 1, 2)
        x3 = x3.permute(0, 3, 1, 2)
        x, feat = self.decoder(x0, x3)

        return x, feat