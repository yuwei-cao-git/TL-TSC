import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50
import torch.nn.functional as F
from .decoder import SimpleDecoder

class FCNResNet50(nn.Module):
    """
    FCN with ResNet-50 backbone for semantic segmentation.
    Adjusts first conv to accept custom number of input channels.
    """
    def __init__(self, n_channels, n_classes, pretrained=False):
        super().__init__()
        # Load FCN-ResNet50 with optional pretrained weights
        fcn = fcn_resnet50(
            pretrained=pretrained,
            progress=True,
            num_classes=n_classes,
            aux_loss=None
        )
        
        # Replace first conv if input channels differ from 3
        if n_channels != 3:
            fcn.backbone.conv1 = nn.Conv2d(
                n_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )
            
        # pull apart the pieces
        self.backbone = fcn.backbone      # IntermediateLayerGetter
        self.classifier = fcn.classifier    # FCNHead
        # self.decoder = SimpleDecoder(encoder_channel=2048, decoder_channels=128, num_classes=n_classes)

    def forward(self, x):
        feat = self.backbone(x)['out'] #(B, 2048, H/8, W/8)
        # FCN returns a dict with 'out' key
        logits = self.classifier(feat) # torch.Size([8, 9, 16, 16])
        # 3) upsample logits back to input H×W
        logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        # logits = self.decoder(feat)
        preds = F.softmax(logits, dim=1)
        return preds, feat