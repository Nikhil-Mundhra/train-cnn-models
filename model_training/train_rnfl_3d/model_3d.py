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


class AnisotropicRNFLUNet3D(nn.Module):
    """Dense-mask-only first ablation of the 3D RNFL successor.

    Auxiliary NFL-absence and surface heads are intentionally excluded until
    the dense baseline has a frozen benchmark result.
    """

    def __init__(self, config: AnisotropicUNetConfig = AnisotropicUNetConfig()):
        super().__init__()
        self.config = config
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
        return {"mask_logits": self.mask_head(decoded)}

