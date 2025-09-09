import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from math import gcd

def gn(num_channels: int) -> nn.GroupNorm:
    # choose a divisor of num_channels for GroupNorm
    # prefer 32, otherwise fallback to a gcd
    g = 32 if num_channels % 32 == 0 else gcd(num_channels, 32)
    g = max(1, min(g, num_channels))
    return nn.GroupNorm(g, num_channels)

def get_norm(norm_type: str, num_channels: int) -> nn.Module:
    if norm_type == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm_type == "gn":
        return gn(num_channels)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")

class DualBN2d(nn.Module):
    def __init__(self, num_feats):
        super().__init__()
        self.bnA = nn.BatchNorm2d(num_feats)
        self.bnB = nn.BatchNorm2d(num_feats)
    def forward(self, x, region: str):
        return self.bnA(x) if region == 'A' else self.bnB(x)

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, act='silu', norm_type='gn', bias=False, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias, groups=groups)
        self.norm = get_norm(norm_type, out_ch)
        self.act  = nn.SiLU(inplace=True) if act == 'silu' else nn.GELU()
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, norm_type='gn'):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.norm = get_norm(norm_type, out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return self.act(self.norm(x))

class MambaFusionV2(nn.Module):
    """
    Plug-and-play fusion before/around Mamba.
    Inputs:
        - x:      [B, C_img, H, W]  (image feature map)
        pc_emb: [B, C_pc]         (stand-level point cloud embedding)
    Returns:
        - fused feature map [B, out_ch, H, W]
    """
    def __init__(
        self,
        in_img_chs: int,
        in_pc_chs: int,
        out_ch: int = 128,        # final channels after fusion
        last_feat_size: int = 16, # H=W expected at this stage
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        width: int = 256,         # shared width D for aligned streams
        use_film: bool = True
    ):
        super().__init__()
        self.HW = last_feat_size
        self.m = last_feat_size * last_feat_size

        # --- learned 2D positional embedding (from normalized coords)
        self.pos_proj = nn.Conv2d(2, width, 1, bias=False)
        nn.init.zeros_(self.pos_proj.weight)  # start as identity-ish

        # --- align channels
        self.img_align = ConvBNAct(in_img_chs, width, k=1, norm_type='gn')
        self.pc_align  = nn.Sequential(nn.Linear(in_pc_chs, width, bias=False),
                                        nn.LayerNorm(width))

        # --- optional FiLM conditioning of image by pc
        self.use_film = use_film
        if use_film:
            self.film_scale = nn.Linear(width, width)
            self.film_shift = nn.Linear(width, width)
            self.global_gate = nn.Sequential(
                nn.Linear(2*width, width),
                nn.SiLU(inplace=True),
                nn.Linear(width, 1),
                nn.Sigmoid()
            )

        # --- fused building: img + pos + pc_map -> bottleneck
        # concat design, but with aligned widths
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(width*3, width, norm_type='gn'),
            DepthwiseSeparableConv(width,   width, norm_type='gn'),
        )

        # --- PPM
        pools = self._arith_seq(1, last_feat_size, max(1, last_feat_size // 4))
        self.pool_layers = nn.ModuleList()
        for i, psize in enumerate(pools):
            if i == 0:
                # match your "pool to 1 then 1x1"
                self.pool_layers.append(nn.Sequential(
                    ConvBNAct(width*3, width, k=1, norm_type='gn'),
                    nn.AdaptiveAvgPool2d(1)
                ))
            else:
                self.pool_layers.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(psize),
                    ConvBNAct(width*3, width, k=1, norm_type='gn')
                ))
        self.pool_len = len(self.pool_layers)

        # --- Mamba expects [B, L, C]
        from mamba_ssm import Mamba
        self.pre_norm  = nn.LayerNorm(width * (self.pool_len + 1))
        self.mamba     = Mamba(
            d_model=width * (self.pool_len + 1),  # token dim after concat PPM maps
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.post_norm = nn.LayerNorm(width * (self.pool_len + 1))

        # --- project back to desired out_ch
        self.out_proj = ConvBNAct(width * (self.pool_len + 1), out_ch, k=1, norm_type='gn')

    @torch.no_grad()
    def _grid(self, B, device):
        ys = torch.linspace(-1, 1, steps=self.HW, device=device)
        xs = torch.linspace(-1, 1, steps=self.HW, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        pe = torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B,2,H,W]
        return pe

    def _arith_seq(self, start, stop, step):
        # like your generate_arithmetic_sequence
        return list(range(start, stop, step))

    def forward(self, x, pc_emb):
        """
        x: [B, C_img, H, W], pc_emb: [B, C_pc]
        """
        B, _, H, W = x.shape
        assert H == self.HW and W == self.HW, f"Expected {self.HW}x{self.HW}, got {H}x{W}"

        # align
        x = self.img_align(x)              # [B, D, H, W]
        p = self.pc_align(pc_emb)          # [B, D]

        # learned pos from coords
        pos = self.pos_proj(self._grid(B, x.device))  # [B, D, H, W]

        # FiLM conditioning + global gate (stable default)
        if self.use_film:
            scale = self.film_scale(p).unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1]
            shift = self.film_shift(p).unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1]
            x_cond = (1.0 + scale) * x + shift
            g = self.global_gate(torch.cat([p, x.mean(dim=(2,3))], dim=1)).view(B,1,1,1)
            x = g * x_cond + (1 - g) * x

        # broadcast pc to spatial (no heavy repeat)
        pc_map = p.unsqueeze(-1).unsqueeze(-1).expand(B, p.size(1), H, W)  # [B, D, H, W]

        # concat (img + pos + pc_map), then a bottleneck
        fused = torch.cat([x, pos, pc_map], dim=1)        # [B, 3D, H, W]
        fused = self.bottleneck(fused)                    # [B, D, H, W]

        # multi-scale pyramid, upsample to (H,W), concat with original fused
        ppm = [fused]
        for layer in self.pool_layers:
            o = layer(torch.cat([x, pos, pc_map], dim=1)) # each branch uses same inputs, 1x1 after pooling
            o = F.interpolate(o, size=(H, W), mode="bilinear", align_corners=False)
            ppm.append(o)
        cat = torch.cat(ppm, dim=1)                      # [B, D*(pool_len+1), H, W]

        # to tokens -> Mamba (B, L, C)
        B, Cc, H, W = cat.shape
        tokens = rearrange(cat, "b c h w -> b (h w) c")  # [B, L=H*W, Cc]
        tokens = self.pre_norm(tokens)
        tokens = self.mamba(tokens)
        tokens = self.post_norm(tokens)
        cat_out = rearrange(tokens, "b (h w) c -> b c h w", h=H, w=W)  # [B, Cc, H, W]

        # final projection
        return self.out_proj(cat_out)                     # [B, out_ch, H, W]
