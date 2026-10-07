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

try:
    from canonicalizer import ConstrainedSpatialTransformer, invert_dense_mask
except ImportError:
    from .canonicalizer import ConstrainedSpatialTransformer, invert_dense_mask




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
        width=320,
        use_laterality_embedding=False
    ):
        super().__init__()
        self.axial_height = axial_height
        self.width = width
        self.use_laterality_embedding = use_laterality_embedding

        # U-Net Backbone
        self.backbone = UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=base_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units
        )

        # Optional Laterality Conditioning Embedding
        if use_laterality_embedding:
            self.eye_emb = nn.Embedding(2, base_channels)

        # Dense Mask Output Head
        self.mask_head = nn.Conv2d(base_channels, 1, kernel_size=1)

        # Boundary Regression & Cup Absence Head
        self.boundary_head = BoundaryRegressionHead(in_channels=base_channels, width=width)

    def forward(self, x, eye_idx=None):
        """
        x: (B, context_slices, H, W)
        eye_idx: (B,) long tensor (0=OD, 1=OS), optional
        Returns:
            mask_logits: (B, 1, H, W)
            ilm_pred:    (B, W) in pixel row units
            nfl_pred:    (B, W) in pixel row units
            cup_logits:  (B, W) logits for cup absence
            column_thickness: (B, W) differentiable column thickness
        """
        feats = self.backbone(x)  # (B, base_channels, H, W)
        if self.use_laterality_embedding and eye_idx is not None:
            emb = self.eye_emb(eye_idx).unsqueeze(-1).unsqueeze(-1)
            feats = feats + emb

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


class CanonicalVolumetricRNFLNet(nn.Module):
    """
    Volumetric RNFL Network equipped with a Constrained Spatial Transformer (STN).
    
    1. Canonicalizes the input slice to a flat horizontal coordinate frame via
       the ConstrainedSpatialTransformer.
    2. Executes the multi-task U-Net backbone and continuous 1D boundary regression
       in the canonical coordinate frame.
    3. Inverts the spatial transformation for dense voxel masks back to native scan space.
    """
    def __init__(
        self,
        in_channels=5,
        base_channels=16,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        axial_height=768,
        width=320,
        use_laterality_embedding=False,
        max_angle_deg=25.0,
    ):
        super().__init__()
        self.axial_height = axial_height
        self.width = width
        self.use_laterality_embedding = use_laterality_embedding
        self.stn = ConstrainedSpatialTransformer(
            in_channels=in_channels,
            max_angle_deg=max_angle_deg,
            max_shift_y_ratio=0.20,
            axial_height=axial_height,
            width=width
        )
        self.backbone_net = VolumetricRNFLNet(
            in_channels=in_channels,
            base_channels=base_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            axial_height=axial_height,
            width=width,
            use_laterality_embedding=use_laterality_embedding
        )

    def forward(self, x, eye_idx=None, override_theta_deg=None):
        """
        x: (B, context_slices, H, W)
        eye_idx: (B,) long tensor (0=OD, 1=OS), optional
        override_theta_deg: Optional manual rotation angle override for ablation/stress testing
        """
        # 1. Canonicalize input orientation
        x_canonical, mat_fwd, mat_inv, theta_deg = self.stn(x, override_theta_deg=override_theta_deg)

        # 2. Forward pass through backbone in canonical orientation
        preds = self.backbone_net(x_canonical, eye_idx=eye_idx)

        # 3. Inverse warp dense mask logits back to native coordinates
        native_mask_logits = invert_dense_mask(preds['mask_logits'], mat_inv)

        preds['native_mask_logits'] = native_mask_logits
        preds['canonical_mask_logits'] = preds['mask_logits']
        # Set primary mask_logits to native for standard loss computation against native ground truth
        preds['mask_logits'] = native_mask_logits
        preds['predicted_tilt_deg'] = theta_deg
        preds['mat_fwd'] = mat_fwd
        preds['mat_inv'] = mat_inv

        return preds


try:
    from transunet import TransUNetRNFLNet
except ImportError:
    try:
        from .transunet import TransUNetRNFLNet
    except ImportError:
        pass

