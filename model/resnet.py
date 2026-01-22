import timm
from torchgeo.models import ResNet50_Weights, resnet50
import torch.nn as nn
import torch.nn.functional as F
    
class Resnet(nn.Module):
    """
    ResNet for S2. 
    """
    def __init__(self, n_channels, num_classes=9, aligned=False):
        super().__init__()
        self.aligned = aligned
        weights = ResNet50_Weights.SENTINEL2_MI_MS_SATLAS
        res = resnet50(weights)
        
        if n_channels != 9:
            conv1 = nn.Conv2d(
                n_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )
        # Initialize the new conv1 layer's weights by averaging the pretrained weights across the channel dimension
        original_weights = res.conv1.weight.data.mean(dim=1, keepdim=True)
        # Expand the averaged weights to the number of input channels of the new dataset
        res.conv1.weight.data = original_weights.repeat(1, n_channels, 1, 1)
        res.conv1 = conv1
        fc = nn.Linear(res.fc.in_features, num_classes)
        res.fc = fc
        self.backbone = res
        if self.aligned:
            from .decoder import PCCAHead
            self.disalign_head = PCCAHead(num_classes, num_classes)
        
            
    def forward(self, x):
        out = self.backbone(x)
        if self.aligned:
            out = self.disalign_head(out)
            
        probs = F.softmax(out, dim=1)
        
        return probs, out