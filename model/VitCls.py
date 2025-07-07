from timm.models.vision_transformer import VisionTransformer
import torch.nn as nn

class HeadlessVIT(VisionTransformer):
    """Vision transformer without the classification head module. Just acts as
    a feature extractor.
    """
    
    def __init__(self,
                 **kwargs):
        
        super().__init__(**kwargs)

        del self.head
    
    def forward(self, x):
        x = self.forward_features(x)
        return x

    
class S2Transformer(nn.Module):
    """Transformer for S2. 
    
    Note
    ----
    Should use the S2Transformer dataloader.
    """
    def __init__(self, 
                    num_classes,
                    p_dropout = 0.3,
                    # args for VIT
                    n_bands_vit=9,
                    img_size=128, 
                    patch_size=6, 
                    embed_dim=768, 
                    depth=12,             
                    num_heads=12, 
                    mlp_ratio=4.,             
                    qkv_bias=True,
                    usehead=False
                ):
        super().__init__()
        self.usehead= usehead
        
        self.vit = HeadlessVIT(img_size=img_size, patch_size=patch_size, 
                                in_chans=n_bands_vit, num_classes=num_classes,
                                embed_dim=embed_dim, depth=depth, 
                                num_heads=num_heads, mlp_ratio=mlp_ratio, 
                                qkv_bias=qkv_bias)
        if usehead:
            self.head = nn.Sequential(
                                        nn.Linear(512+embed_dim, 2048),
                                        nn.ReLU(),
                                        nn.Dropout(p = p_dropout),
                                        nn.Linear(2048, num_classes)
                                    )
            
    def forward(self, x):
        x1 = self.vit(x) # [batch, embed_dim]
        
        x1 = x1.squeeze() # [batch, embed_dim]
        
        if self.usehead:
            x = self.head(x1) # [batch, n_class]
            return x, x1
        else:
            return x1