"""
model.py
========
Deep learning architecture for Volumetric RNFL segmentation and surface regression.
Features:
- 2.5D Multi-Slice Input (5 channels: z-2, z-1, z, z+1, z+2)
- High-resolution U-Net backbone with residual units (MONAI)
- Multi-task heads:
  1. Dense Voxel Mask Logits: (B, 1, H, W) for 3D Slicer volumetric labelmap
  2. 1D Boundary Regression Head: (B, W) for ILM and NFL surface depths
  3. 1D Optic Cup Absence Classifier: (B, W) detecting absence over the cup cavity
- Differentiable soft-boundary extraction layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet


class BoundaryRegressionHead(nn.Module):
    """
    1D Convolutional Head branching from bottleneck features to directly
    regress ILM depth, NFL depth, and cup absence probability per A-scan column.
    """
    def __init__(self, in_channels, width=320):
        super().__init__()
        # Pool vertically across depth H (axial depth), preserving horizontal fast axis W
        self.vertical_pool = nn.AdaptiveAvgPool2d((1, width))
        self.conv1d_ilm = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=3, padding=1)
        )
        self.conv1d_nfl = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=3, padding=1)
        )
        self.conv1d_cup = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, feat):
        # feat: (B, C, H, W)
        pooled = self.vertical_pool(feat).squeeze(2)  # (B, C, W)
        ilm = self.conv1d_ilm(pooled).squeeze(1)      # (B, W)
        nfl = self.conv1d_nfl(pooled).squeeze(1)      # (B, W)
        cup = self.conv1d_cup(pooled).squeeze(1)      # (B, W)
        return ilm, nfl, cup


class VolumetricRNFLNet(nn.Module):
    """
    Multi-Task 2.5D Volumetric Network for RNFL Segmentation and Surface Extraction.
    """
    def __init__(
        self,
        in_channels=5,
        base_channels=16,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        axial_height=768,
        width=320
    ):
        super().__init__()
        self.axial_height = axial_height
        self.width = width

        # U-Net Backbone
        self.backbone = UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=base_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units
        )

        # Dense Mask Output Head
        self.mask_head = nn.Conv2d(base_channels, 1, kernel_size=1)

        # Boundary Regression & Cup Absence Head
        self.boundary_head = BoundaryRegressionHead(in_channels=base_channels, width=width)

    def forward(self, x):
        """
        x: (B, context_slices, H, W)
        Returns:
            mask_logits: (B, 1, H, W)
            ilm_pred:    (B, W) in pixel row units
            nfl_pred:    (B, W) in pixel row units
            cup_logits:  (B, W) logits for cup absence
            soft_surfaces: dict with differentiable surfaces derived from mask_logits
        """
        feats = self.backbone(x)  # (B, base_channels, H, W)
        mask_logits = self.mask_head(feats)  # (B, 1, H, W)

        # Explicit 1D regression heads (scaled to height)
        ilm_raw, nfl_raw, cup_logits = self.boundary_head(feats)
        # Apply sigmoid and scale to [0, axial_height]
        ilm_pred = torch.sigmoid(ilm_raw) * float(self.axial_height)
        nfl_pred = torch.sigmoid(nfl_raw) * float(self.axial_height)

        # Differentiable column thickness from dense probabilities
        probs = torch.sigmoid(mask_logits).squeeze(1)  # (B, H, W)
        column_thickness = probs.sum(dim=1)            # (B, W)

        return {
            'mask_logits': mask_logits,
            'ilm_pred': ilm_pred,
            'nfl_pred': nfl_pred,
            'cup_logits': cup_logits,
            'column_thickness': column_thickness
        }
