"""
transunet.py
============
TransUNet hybrid CNN-Transformer architecture adapted for Volumetric 2.5D RNFL OCT segmentation.

Reference:
- Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation" (arXiv:2102.04306)

Key adaptations for Retinal OCT RNFL:
- Native 2.5D multi-slice input (in_channels=5 for z-2..z+2 axial slices).
- Anisotropic aspect ratio support: (H, W) = (768, 320).
- Hybrid ResNet/CNN stem + Vision Transformer (ViT) bottleneck + Cascaded Upsampler (CUP) with CNN skip connections.
- Seamless compatibility with VolumetricRNFLNet multi-task interface:
  1. Dense Voxel Mask Logits: (B, 1, H, W)
  2. 1D Boundary Regression Head: (B, W) for ILM and NFL surface depths
  3. 1D Optic Cup Absence Classifier: (B, W) detecting absence over the cup cavity
  4. Differentiable column thickness: (B, W)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model import BoundaryRegressionHead
except ImportError:
    from .model import BoundaryRegressionHead


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
    Cascaded Upsampling Block (CUP) with optional CNN skip connections.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        # Upsample by factor of 2 using bilinear interpolation
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TransUNetEncoder(nn.Module):
    """
    Hybrid CNN + Transformer Encoder.
    - CNN stem downsamples the input and extracts multi-scale feature skips (1/2, 1/4, 1/8).
    - Bottleneck projection downsamples to 1/16 (e.g., 48 x 20 for 768 x 320 input).
    - TransformerEncoder models global spatial contextual relationships across the retina.
    """
    def __init__(
        self,
        in_channels: int = 5,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        grid_size: tuple = (48, 20),
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_tokens = grid_size[0] * grid_size[1]
        self.hidden_size = hidden_size

        # CNN Feature Stages (1/2, 1/4, 1/8)
        # Stage 1: (H, W) -> (H/2, W/2)
        self.stage1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Stage 2: (H/2, W/2) -> (H/4, W/4)
        self.stage2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Stage 3: (H/4, W/4) -> (H/8, W/8)
        self.stage3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bottleneck projection to hidden_size at 1/16: (H/8, W/8) -> (H/16, W/16)
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, hidden_size, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
        )

        # Learnable 2D positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=dropout)

        # Standard Transformer Encoder
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
        f1 = self.stage1(x)         # (B, 32, H, W)
        p1 = self.pool1(f1)        # (B, 32, H/2, W/2)

        f2 = self.stage2(p1)        # (B, 64, H/2, W/2)
        p2 = self.pool2(f2)        # (B, 64, H/4, W/4)

        f3 = self.stage3(p2)        # (B, 128, H/4, W/4)
        p3 = self.pool3(f3)        # (B, 128, H/8, W/8)

        f4 = self.stage4(p3)        # (B, hidden_size, H/16, W/16)
        B, C, H_tok, W_tok = f4.shape

        # Flatten into token sequence (B, N, C)
        tokens = f4.flatten(2).transpose(1, 2)

        # Add positional embedding (with dynamic interpolation if dimensions differ)
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

        # Reshape back to 2D feature map
        bottleneck = tokens.transpose(1, 2).reshape(B, C, H_tok, W_tok)

        # Skips: f3 is at (H/4, W/4); p3 is at (H/8, W/8).
        # We supply skips at (H/8, W/8), (H/4, W/4), (H/2, W/2)
        return bottleneck, p3, p2, p1


class TransUNetRNFLNet(nn.Module):
    """
    TransUNet architecture tailored for Volumetric RNFL OCT segmentation.
    Integrates:
    - 2.5D context slicing
    - Hybrid CNN + Transformer encoder
    - Cascaded Upsampler (CUP) decoder
    - Multi-task dense mask, 1D boundary regression, and optic cup absence heads
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
    ):
        super().__init__()
        self.axial_height = axial_height
        self.width = width
        self.use_laterality_embedding = use_laterality_embedding

        grid_size = (axial_height // 16, width // 16)
        self.encoder = TransUNetEncoder(
            in_channels=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            grid_size=grid_size,
        )

        # Cascaded Upsampler Decoder (CUP)
        # d1: 1/16 -> 1/8 with skip f3 (128 channels)
        self.cup1 = DecoderCupBlock(in_channels=hidden_size, skip_channels=128, out_channels=128)
        # d2: 1/8 -> 1/4 with skip f2 (64 channels)
        self.cup2 = DecoderCupBlock(in_channels=128, skip_channels=64, out_channels=64)
        # d3: 1/4 -> 1/2 with skip f1 (32 channels)
        self.cup3 = DecoderCupBlock(in_channels=64, skip_channels=32, out_channels=32)
        # d4: 1/2 -> 1/1
        self.cup4 = DecoderCupBlock(in_channels=32, skip_channels=0, out_channels=base_channels)

        # Optional Laterality Conditioning
        if use_laterality_embedding:
            self.eye_emb = nn.Embedding(2, base_channels)

        # Dense Mask Output Head
        self.mask_head = nn.Conv2d(base_channels, 1, kernel_size=1)

        # 1D Boundary Regression Head (fed with full-resolution features)
        self.boundary_head = BoundaryRegressionHead(in_channels=base_channels, width=width)

    def forward(self, x: torch.Tensor, eye_idx: torch.Tensor = None):
        """
        x: (B, in_channels, H, W)
        eye_idx: (B,) long tensor (0=OD, 1=OS), optional
        """
        bottleneck, f3, f2, f1 = self.encoder(x)

        d1 = self.cup1(bottleneck, f3) # (B, 128, H/8, W/8)
        d2 = self.cup2(d1, f2)         # (B, 64, H/4, W/4)
        d3 = self.cup3(d2, f1)         # (B, 32, H/2, W/2)
        feats = self.cup4(d3)          # (B, base_channels, H, W)

        if self.use_laterality_embedding and eye_idx is not None:
            emb = self.eye_emb(eye_idx).unsqueeze(-1).unsqueeze(-1)
            feats = feats + emb

        mask_logits = self.mask_head(feats)  # (B, 1, H, W)

        ilm_raw, nfl_raw, cup_logits = self.boundary_head(feats)
        ilm_pred = torch.sigmoid(ilm_raw) * float(self.axial_height)
        nfl_pred = torch.sigmoid(nfl_raw) * float(self.axial_height)

        probs = torch.sigmoid(mask_logits).squeeze(1)
        column_thickness = probs.sum(dim=1)

        return {
            'mask_logits': mask_logits,
            'ilm_pred': ilm_pred,
            'nfl_pred': nfl_pred,
            'cup_logits': cup_logits,
            'column_thickness': column_thickness,
        }
