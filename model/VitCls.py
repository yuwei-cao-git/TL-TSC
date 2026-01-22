import timm
from torchgeo.models import ViTBase16_Weights, vit_base_patch16_224
import torch.nn as nn
import torch.nn.functional as F
    
class S2Transformer(nn.Module):
    """
    Vit for S2. 
    """
    def __init__(self, n_channels, num_classes=9, aligned=False):
        super().__init__()
        self.aligned = aligned
        weights = ViTBase16_Weights.SENTINEL2_MI_MS_SATLAS
        res = vit_base_patch16_224(weights, in_chans=n_channels)
        
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