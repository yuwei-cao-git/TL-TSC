import torch.nn as nn
import torch.nn.functional as F
import timm
from .decoder import SimpleDecoder

class ViTSeg(nn.Module):
    """
    Simple ViT-based segmentation model.
    Uses a timm Vision Transformer as backbone with a lightweight decoder.
    """
    def __init__(self,
                n_channels,
                n_classes,
                model_name='vit_base_patch16_224',
                pretrained=True):
        super().__init__()
        # Feature extractor backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=n_channels,
            img_size=128
        )
        # Number of channels in the last feature map
        in_chs = self.backbone.feature_info[-1]['num_chs']
        self.decoder = SimpleDecoder(encoder_channel=in_chs, decoder_channels=128, num_classes=n_classes)


    def forward(self, x):
        # Extract feature maps
        features = self.backbone(x)
        x = features[-1]
        # Decode to class logits
        logits = self.decoder(x)
        preds = F.softmax(logits, dim=1)
        return preds, x