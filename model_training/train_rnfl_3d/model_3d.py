"""Dense anisotropic 3D U-Net selected by the Phase 0 A100 gate."""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class AnisotropicUNetConfig:
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 32
    patch_shape: Tuple[int, int, int] = (64, 768, 64)
    strides: Tuple[Tuple[int, int, int], ...] = (
        (1, 2, 1),
        (1, 2, 1),
        (2, 2, 2),
        (2, 2, 2),
    )


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = min(8, out_channels)
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DownBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: Tuple[int, int, int]):
        super().__init__()
        self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=stride, stride=stride)
        self.features = ConvBlock3D(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(self.downsample(x))


class UpBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.reduce = nn.Conv3d(in_channels, out_channels, 1)
        self.features = ConvBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.features(torch.cat((self.reduce(x), skip), dim=1))


class BoundaryRegressionHead3D(nn.Module):
    """Transverse 2D Convolutional Head branching from decoded features.

    Pools axially along depth H (axial_y=768) to obtain a transverse surface feature
    map (B, C, slow_z, fast_x), then uses 2D convolutions to regress:
    - ilm_pred: (B, slow_z, fast_x) continuous ILM surface depth
    - nfl_pred: (B, slow_z, fast_x) continuous NFL surface depth
    - cup_logits: (B, slow_z, fast_x) optic cup / absence logits
    """

    def __init__(self, in_channels: int):
        super().__init__()
        groups = min(8, in_channels)
        self.conv2d_ilm = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(min(8, 64), 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(min(8, 32), 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )
        self.conv2d_nfl = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(min(8, 64), 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(min(8, 32), 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )
        self.conv2d_cup = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(min(8, 32), 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # feat: (B, C, slow_z, axial_y, fast_x)
        # Pool vertically across axial dimension (dim 3)
        transverse = feat.mean(dim=3)  # (B, C, slow_z, fast_x)
        ilm = self.conv2d_ilm(transverse).squeeze(1)  # (B, slow_z, fast_x)
        nfl = self.conv2d_nfl(transverse).squeeze(1)  # (B, slow_z, fast_x)
        cup = self.conv2d_cup(transverse).squeeze(1)  # (B, slow_z, fast_x)
        return ilm, nfl, cup


class AnisotropicRNFLUNet3D(nn.Module):
    """Dense anisotropic 3D RNFL successor with multi-task boundary and absence heads.

    Predicts:
    1. mask_logits: (B, 1, slow_z, axial_y, fast_x) dense volumetric RNFL mask logits
    2. ilm_pred: (B, slow_z, fast_x) continuous ILM surface depth in pixel coordinates
    3. nfl_pred: (B, slow_z, fast_x) continuous NFL surface depth in pixel coordinates
    4. cup_logits: (B, slow_z, fast_x) optic cup / absence logits
    5. column_thickness: (B, slow_z, fast_x) differentiable column thickness from voxel probabilities
    """

    def __init__(self, config: AnisotropicUNetConfig = AnisotropicUNetConfig(), axial_height: float = 768.0):
        super().__init__()
        self.config = config
        self.axial_height = float(axial_height)
        base = config.base_channels
        channels = (base, base * 2, base * 4, base * 8, base * 12)
        self.encoder0 = ConvBlock3D(config.in_channels, channels[0])
        self.encoder1 = DownBlock3D(channels[0], channels[1], config.strides[0])
        self.encoder2 = DownBlock3D(channels[1], channels[2], config.strides[1])
        self.encoder3 = DownBlock3D(channels[2], channels[3], config.strides[2])
        self.bottleneck = DownBlock3D(channels[3], channels[4], config.strides[3])
        self.decoder3 = UpBlock3D(channels[4], channels[3], channels[3])
        self.decoder2 = UpBlock3D(channels[3], channels[2], channels[2])
        self.decoder1 = UpBlock3D(channels[2], channels[1], channels[1])
        self.decoder0 = UpBlock3D(channels[1], channels[0], channels[0])
        self.mask_head = nn.Conv3d(channels[0], config.out_channels, 1)
        self.boundary_head = BoundaryRegressionHead3D(in_channels=channels[0])

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        if image.ndim != 5:
            raise ValueError("image must have shape (B, C, slow_z, axial_y, fast_x)")
        x0 = self.encoder0(image)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.bottleneck(x3)
        decoded = self.decoder0(
            self.decoder1(self.decoder2(self.decoder3(x4, x3), x2), x1),
            x0,
        )
        mask_logits = self.mask_head(decoded)

        # Multi-task 1D/2D continuous surface regression & cup absence
        ilm_raw, nfl_raw, cup_logits = self.boundary_head(decoded)
        ilm_pred = torch.sigmoid(ilm_raw) * self.axial_height
        nfl_pred = torch.sigmoid(nfl_raw) * self.axial_height

        # Differentiable column thickness along axial depth (dim 3)
        probs = torch.sigmoid(mask_logits).squeeze(1)  # (B, slow_z, axial_y, fast_x)
        column_thickness = probs.sum(dim=2)  # (B, slow_z, fast_x)

        return {
            "mask_logits": mask_logits,
            "ilm_pred": ilm_pred,
            "nfl_pred": nfl_pred,
            "cup_logits": cup_logits,
            "column_thickness": column_thickness,
        }

