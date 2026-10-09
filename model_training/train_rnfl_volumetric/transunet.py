"""
transunet.py
============
TransUNet hybrid CNN-Transformer architecture adapted for Volumetric 2.5D RNFL OCT segmentation.

Architectural Innovations & RNFL Specializations:
- Anisotropic Axial-Preserving Tokenization:
  Decoupled depth vs. lateral pooling retains 96-192 axial tokens (bin size ~12.5 µm vs. 50 µm in naive ViT),
  preserving sub-millimeter axial gradients essential for the thin retinal nerve fiber layer.
- Multi-Scale Boundary Injection:
  The 1D continuous boundary regression heads directly tap high-frequency CNN stem features (1/1 scale)
  coupled to transformer decoder context, giving sharp physical voxel edge alignment.
- Constrained Spatial Transformer (STN) Support:
  Canonicalizes variable patient head tilt (theta, t_y) to horizontal frame prior to tokenization,
  preventing diagonal self-attention interpolation artifacts.
- Bi-Planar Orthogonal Consensus Fusion:
  Supports simultaneous horizontal (X-Z) and vertical (Y-Z) passes for 3D consensus segmentation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from canonicalizer import ConstrainedSpatialTransformer, invert_dense_mask
except ImportError:
    try:
        from .canonicalizer import ConstrainedSpatialTransformer, invert_dense_mask
    except ImportError:
        ConstrainedSpatialTransformer = None
        invert_dense_mask = None


class ConvBlock(nn.Module):
    """Dual 3x3 Conv-BN-ReLU block used in CNN feature stages and decoder."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DecoderCupBlock(nn.Module):
    """
    Cascaded Upsampling Block (CUP) with dynamic shape alignment and skip concatenation.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        if skip is not None:
            # Interpolate coarse features up to exact target skip shape
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return self.conv(x)


class MultiScaleBoundaryHead(nn.Module):
    """
    Multi-Scale 1D Boundary Regression Head branching from both:
    1. Coarse Transformer Decoder features (global peripapillary context)
    2. High-resolution CNN Stem features (sharp sub-millimeter axial voxel edges)
    """
    def __init__(self, in_channels_dec: int, in_channels_stem: int = 32, width: int = 320):
        super().__init__()
        self.vertical_pool_dec = nn.AdaptiveAvgPool2d((1, width))
        self.vertical_pool_stem = nn.AdaptiveAvgPool2d((1, width))

        combined_channels = in_channels_dec + in_channels_stem
        self.conv1d_ilm = nn.Sequential(
            nn.Conv1d(combined_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
        )
        self.conv1d_nfl = nn.Sequential(
            nn.Conv1d(combined_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
        )
        self.conv1d_cup = nn.Sequential(
            nn.Conv1d(combined_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, dec_feat: torch.Tensor, stem_feat: torch.Tensor = None):
        p_dec = self.vertical_pool_dec(dec_feat).squeeze(2)  # (B, C_dec, W)
        if stem_feat is not None:
            p_stem = self.vertical_pool_stem(stem_feat).squeeze(2)  # (B, C_stem, W)
            combined = torch.cat([p_dec, p_stem], dim=1)
        else:
            combined = p_dec

        ilm = self.conv1d_ilm(combined).squeeze(1)
        nfl = self.conv1d_nfl(combined).squeeze(1)
        cup = self.conv1d_cup(combined).squeeze(1)
        return ilm, nfl, cup


class AnisotropicTransUNetEncoder(nn.Module):
    """
    Anisotropic Hybrid CNN + Vision Transformer Encoder.
    Decoupled axial vs. lateral downsampling preserves 96-192 depth rows (bin size ~12.5 µm),
    preventing axial boundary blurring.
    """
    def __init__(
        self,
        in_channels: int = 5,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        axial_height: int = 768,
        width: int = 320,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.axial_height = axial_height
        self.width = width

        # Stage 1: (H, W) -> (H/2, W/2) = (384, 160)
        self.stage1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d((2, 2))

        # Stage 2: (H/2, W/2) -> (H/4, W/4) = (192, 80)
        self.stage2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d((2, 2))

        # Stage 3: (192, 80) -> Lateral-only downsampling (1, 2) = (192, 40)
        # Keeps axial depth at 192 rows!
        self.stage3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d((1, 2))

        # Stage 4: (192, 40) -> Conv stride (2, 2) = (96, 20) token grid
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, hidden_size, kernel_size=3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
        )

        self.grid_size = (96, 20)
        self.num_tokens = self.grid_size[0] * self.grid_size[1]  # 1920 tokens

        # Learnable 2D positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor):
        # x: (B, in_channels, H, W)
        f1 = self.stage1(x)         # (B, 32, 768, 320) -> Native stem skip
        p1 = self.pool1(f1)        # (B, 32, 384, 160)

        f2 = self.stage2(p1)        # (B, 64, 384, 160)
        p2 = self.pool2(f2)        # (B, 64, 192, 80)

        f3 = self.stage3(p2)        # (B, 128, 192, 80)
        p3 = self.pool3(f3)        # (B, 128, 192, 40) -> Preserved axial depth

        f4 = self.stage4(p3)        # (B, hidden_size, 96, 20)
        B, C, H_tok, W_tok = f4.shape

        tokens = f4.flatten(2).transpose(1, 2)
        if tokens.shape[1] == self.pos_embed.shape[1]:
            tokens = tokens + self.pos_embed
        else:
            orig_h, orig_w = self.grid_size
            pos = self.pos_embed.reshape(1, orig_h, orig_w, C).permute(0, 3, 1, 2)
            pos = F.interpolate(pos, size=(H_tok, W_tok), mode="bilinear", align_corners=False)
            pos = pos.flatten(2).transpose(1, 2)
            tokens = tokens + pos

        tokens = self.pos_drop(tokens)
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        bottleneck = tokens.transpose(1, 2).reshape(B, C, H_tok, W_tok)
        return bottleneck, f1, p1, p2, p3


class TransUNetRNFLNet(nn.Module):
    """
    Advanced TransUNet architecture tailored for Volumetric RNFL OCT segmentation.
    Features:
    - 2.5D multi-slice context
    - Anisotropic axial-preserving Vision Transformer encoder
    - Multi-scale Cascaded Upsampler (CUP) with dynamic shape interpolation
    - Multi-Scale 1D Boundary Regression Head tapping stem features
    - Optional Constrained Spatial Transformer (STN) canonicalizer
    """
    def __init__(
        self,
        in_channels: int = 5,
        base_channels: int = 16,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 512,
        axial_height: int = 768,
        width: int = 320,
        use_laterality_embedding: bool = False,
        use_stn: bool = False,
        max_angle_deg: float = 25.0,
    ):
        super().__init__()
        self.axial_height = axial_height
        self.width = width
        self.use_laterality_embedding = use_laterality_embedding
        self.use_stn = use_stn

        if use_stn and ConstrainedSpatialTransformer is not None:
            self.stn = ConstrainedSpatialTransformer(
                in_channels=in_channels,
                max_angle_deg=max_angle_deg,
                max_shift_y_ratio=0.20,
                axial_height=axial_height,
                width=width,
            )
        else:
            self.stn = None

        self.encoder = AnisotropicTransUNetEncoder(
            in_channels=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            axial_height=axial_height,
            width=width,
        )

        # Cascaded Upsampler Decoder (CUP)
        # d1: (96, 20) -> (192, 40) with skip p3 (128 channels)
        self.cup1 = DecoderCupBlock(in_channels=hidden_size, skip_channels=128, out_channels=128)
        # d2: (192, 40) -> (192, 80) with skip p2 (64 channels)
        self.cup2 = DecoderCupBlock(in_channels=128, skip_channels=64, out_channels=64)
        # d3: (192, 80) -> (384, 160) with skip p1 (32 channels)
        self.cup3 = DecoderCupBlock(in_channels=64, skip_channels=32, out_channels=32)
        # d4: (384, 160) -> (768, 320) with skip f1 (32 channels)
        self.cup4 = DecoderCupBlock(in_channels=32, skip_channels=32, out_channels=base_channels)

        if use_laterality_embedding:
            self.eye_emb = nn.Embedding(2, base_channels)

        self.mask_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.boundary_head = MultiScaleBoundaryHead(
            in_channels_dec=base_channels,
            in_channels_stem=32,
            width=width,
        )

    def forward(self, x: torch.Tensor, eye_idx: torch.Tensor = None, override_theta_deg=None):
        """
        x: (B, in_channels, H, W)
        eye_idx: (B,) long tensor (0=OD, 1=OS), optional
        """
        mat_inv = None
        mat_fwd = None
        theta_deg = None

        if self.stn is not None:
            x, mat_fwd, mat_inv, theta_deg = self.stn(x, override_theta_deg=override_theta_deg)

        bottleneck, f1, p1, p2, p3 = self.encoder(x)

        d1 = self.cup1(bottleneck, p3)  # (B, 128, 192, 40)
        d2 = self.cup2(d1, p2)          # (B, 64, 192, 80)
        d3 = self.cup3(d2, p1)          # (B, 32, 384, 160)
        feats = self.cup4(d3, f1)       # (B, base_channels, 768, 320)

        if self.use_laterality_embedding and eye_idx is not None:
            emb = self.eye_emb(eye_idx).unsqueeze(-1).unsqueeze(-1)
            feats = feats + emb

        mask_logits = self.mask_head(feats)  # (B, 1, H, W)

        # Multi-Scale 1D Continuous Boundary Regression
        ilm_raw, nfl_raw, cup_logits = self.boundary_head(feats, f1)
        ilm_pred = torch.sigmoid(ilm_raw) * float(self.axial_height)
        nfl_pred = torch.sigmoid(nfl_raw) * float(self.axial_height)

        # If STN was applied, invert dense mask back to native coordinates
        if mat_inv is not None and invert_dense_mask is not None:
            native_mask_logits = invert_dense_mask(mask_logits, mat_inv)
            canonical_mask_logits = mask_logits
            mask_logits = native_mask_logits
        else:
            native_mask_logits = mask_logits
            canonical_mask_logits = mask_logits

        probs = torch.sigmoid(mask_logits).squeeze(1)
        column_thickness = probs.sum(dim=1)

        out = {
            'mask_logits': mask_logits,
            'native_mask_logits': native_mask_logits,
            'canonical_mask_logits': canonical_mask_logits,
            'ilm_pred': ilm_pred,
            'nfl_pred': nfl_pred,
            'cup_logits': cup_logits,
            'column_thickness': column_thickness,
        }
        if theta_deg is not None:
            out['predicted_tilt_deg'] = theta_deg
            out['mat_fwd'] = mat_fwd
            out['mat_inv'] = mat_inv

        return out
