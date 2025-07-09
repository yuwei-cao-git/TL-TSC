import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50
import torch.nn.functional as F

class FCNResNet50Encoder(nn.Module):
    """
    FCN-ResNet50 encoder that outputs high-level feature maps.
    Replaces the first conv layer if input channels ≠ 3.
    """
    def __init__(self, n_channels, pretrained=False):
        super().__init__()
        # Load FCN-ResNet50 with optional pretrained weights
        fcn = fcn_resnet50(
            pretrained=pretrained,
            progress=True,
            num_classes=21,  # placeholder, won't be used
            aux_loss=None
        )

        # Replace first conv layer if input channels ≠ 3
        if n_channels != 3:
            fcn.backbone.conv1 = nn.Conv2d(
                n_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )

        self.backbone = fcn.backbone  # IntermediateLayerGetter

    def forward(self, x):
        return self.backbone(x)['out']  # Output: (B, 2048, H/8, W/8)
    
class FCNResNet50Decoder(nn.Module):
    """
    FCN classifier head and upsampling for segmentation output.
    """
    def __init__(self, return_type, encoder_channels=2048, n_classes=9, upsample='bilinear'):
        super().__init__()
        self.decoder_upsample = upsample
        self.return_type = return_type
        if self.decoder_upsample == 'bilinear':
            from torchvision.models.segmentation.fcn import FCNHead
            self.classifier = FCNHead(encoder_channels, n_classes)
        else:
            from .decoder import SimpleUpDecoder
            self.classifier = SimpleUpDecoder(encoder_channel=2048, decoder_channels=512, num_classes=n_classes)

    def forward(self, features, input_shape):
        logits = self.classifier(features)  # (B, n_classes, H/8, W/8)
        if self.decoder_upsample == 'bilinear':
            logits = F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=False)
        if self.return_type == 'logsoftmax':
            return F.log_softmax(logits, dim=1)
        elif self.return_type == 'logits':
            return logits
        else:
            return F.softmax(logits, dim=1)
    
    
class FCNResNet50(nn.Module):
    def __init__(self, n_channels, n_classes, return_type, upsample_method='bilinear', pretrained=False, decoder=True):
        super().__init__()
        self.use_decoder = decoder
        self.encoder = FCNResNet50Encoder(n_channels, pretrained=pretrained)
        if self.use_decoder:
            self.decoder = FCNResNet50Decoder(return_type=return_type, encoder_channels=2048, n_classes=n_classes, upsample=upsample_method)

    def forward(self, x):
        features = self.encoder(x)
        if self.use_decoder:
            preds = self.decoder(features, input_shape=x.shape[2:])
            return preds, features
        else:
            return features
