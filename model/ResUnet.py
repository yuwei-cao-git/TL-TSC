import torch
import torch.nn as nn
import torch.nn.functional as F

# ResidualUNet model
class ResUnetEncoder(nn.Module):
    """
    Encoder for ResUnet: includes stem, residual blocks, and bridge.
    """
    def __init__(self, n_channels=52, block_expansion=1):
        super().__init__()
        f = [
            32 * block_expansion,
            64 * block_expansion,
            128 * block_expansion,
            256 * block_expansion,
            512 * block_expansion,
        ]

        self.stem = Stem(n_channels, f[0])
        self.residual_block1 = ResidualBlock(f[0], f[1], stride=2)
        self.residual_block2 = ResidualBlock(f[1], f[2], stride=2)
        self.residual_block3 = ResidualBlock(f[2], f[3], stride=2)
        self.residual_block4 = ResidualBlock(f[3], f[4], stride=2)

        self.conv_block1 = ConvBlock(f[4], f[4])
        self.conv_block2 = ConvBlock(f[4], f[4])

    def forward(self, x):
        e1 = self.stem(x)
        e2 = self.residual_block1(e1)
        e3 = self.residual_block2(e2)
        e4 = self.residual_block3(e3)
        e5 = self.residual_block4(e4)

        b0 = self.conv_block1(e5)
        b1 = self.conv_block2(b0)

        return b1, [e1, e2, e3, e4]  # b1 is bridge output, rest are skip connections

class ResUnetDecoder(nn.Module):
    """
    Decoder for ResUnet using residual upsampling blocks and final prediction layer.
    """
    def __init__(self, block_expansion=1):
        super().__init__()
        f = [
            32 * block_expansion,
            64 * block_expansion,
            128 * block_expansion,
            256 * block_expansion,
            512 * block_expansion,
        ]

        self.upsample_concat = UpSampleConcat()
        self.residual_block_d1 = ResidualBlock(f[4] + f[3], f[3])
        self.residual_block_d2 = ResidualBlock(f[3] + f[2], f[2])
        self.residual_block_d3 = ResidualBlock(f[2] + f[1], f[1])
        self.residual_block_d4 = ResidualBlock(f[1] + f[0], f[0])

    def forward(self, b1, skip_connections):
        e1, e2, e3, e4 = skip_connections

        u1 = self.upsample_concat(b1, e4)
        d1 = self.residual_block_d1(u1)

        u2 = self.upsample_concat(d1, e3)
        d2 = self.residual_block_d2(u2)

        u3 = self.upsample_concat(d2, e2)
        d3 = self.residual_block_d3(u3)

        u4 = self.upsample_concat(d3, e1)
        d4 = self.residual_block_d4(u4)
        
        return d4


class ResUnetClassifier(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.out_conv = nn.Conv2d(in_channels, n_classes, kernel_size=1)

    def forward(self, x):
        logits = self.out_conv(x)
        preds = F.softmax(logits, dim=1)
        return preds

class ResUnet(nn.Module):
    def __init__(self, n_channels=52, n_classes=9, block_expansion=2, decoder=True):
        super().__init__()
        self.use_decoder = decoder
        self.encoder = ResUnetEncoder(n_channels, block_expansion)
        if self.use_decoder:
            self.decoder = ResUnetDecoder(block_expansion)
            self.classifier = ResUnetClassifier(32 * block_expansion, n_classes)

    def forward(self, x):
        b1, skip_connections = self.encoder(x)
        if self.use_decoder:
            decoded = self.decoder(b1, skip_connections)
            preds = self.classifier(decoded)
            return preds, b1
        else:
            return b1


# -----------------------------------------------------------------------------------
# Parts of the ResU-Net model
# -----------------------------------------------------------------------------------
class BNAct(nn.Module):
    """Batch Normalization followed by an optional ReLU activation."""

    def __init__(self, num_features, act=True):
        super(BNAct, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.act = act
        if self.act:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        if self.act:
            x = self.activation(x)
        return x


# ConvBlock module
class ConvBlock(nn.Module):
    """Convolution Block with BN and Activation."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ConvBlock, self).__init__()
        self.bn_act = BNAct(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.bn_act(x)
        x = self.conv(x)
        return x


# Stem module
class Stem(nn.Module):
    """Initial convolution block with residual connection."""

    def __init__(self, in_channels, out_channels):
        super(Stem, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_block = ConvBlock(out_channels, out_channels)
        self.shortcut_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding=0
        )
        self.bn_act = BNAct(out_channels, act=False)

    def forward(self, x):
        conv = self.conv1(x)
        conv = self.conv_block(conv)
        shortcut = self.shortcut_conv(x)
        shortcut = self.bn_act(shortcut)
        output = conv + shortcut
        return output


# ResidualBlock module
class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers and a shortcut connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv_block1 = ConvBlock(in_channels, out_channels, stride=stride)
        self.conv_block2 = ConvBlock(out_channels, out_channels)
        self.shortcut_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=stride
        )
        self.bn_act = BNAct(out_channels, act=False)

    def forward(self, x):
        res = self.conv_block1(x)
        res = self.conv_block2(res)
        shortcut = self.shortcut_conv(x)
        shortcut = self.bn_act(shortcut)
        output = res + shortcut
        return output


# UpSampleConcat module
class UpSampleConcat(nn.Module):
    """Upsamples the input and concatenates with the skip connection."""

    def __init__(self):
        super(UpSampleConcat, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # else:
        # self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x, xskip):
        # x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.up(x)
        x = torch.cat([x, xskip], dim=1)
        return x
    