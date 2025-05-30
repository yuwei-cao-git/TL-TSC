import torch.nn as nn
import torch.nn.functional as F
from .blocks import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, return_logits=False, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.return_logits = return_logits

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)  # Initial convolution
        x2 = self.down1(x1)  # Down 1
        x3 = self.down2(x2)  # Down 2
        x4 = self.down3(x3)  # Down 3
        x5 = self.down4(x4)  # Down 4
        x = self.up1(x5, x4)  # Up 1
        x = self.up2(x, x3)  # Up 2
        x = self.up3(x, x2)  # Up 3
        x = self.up4(x, x1)  # Up 4
        logits = self.outc(x)  # Output layer
        if self.return_logits:
            return F.log_softmax(logits, dim=1), x5
            # Lower T -> sharper, Higher T -> flatter
            # return F.softmax(logits / 0.5, dim=1), x5
        else:
            preds = F.softmax(logits, dim=1)
            return preds, x5
