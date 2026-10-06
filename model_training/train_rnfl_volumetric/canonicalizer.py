"""
canonicalizer.py
================
Spatial Transformer Network (STN) for orientation canonicalization in OCT scans.

Addresses ocular tilt, scanner alignment variations, and patient head rotation by:
1. Predicting rigid in-plane tilt angle (theta) and vertical translation (t_y)
   via a lightweight, downsampled localization sub-network.
2. Applying aspect-ratio-corrected affine warping into a canonical horizontal plane
   where retinal layers are flat, enabling optimal 1D boundary regression and 2D U-Net feature extraction.
3. Inverting the transformation to map predictions back into native patient coordinates.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstrainedSpatialTransformer(nn.Module):
    """
    Constrained 2D Spatial Transformer Network.
    
    Restricted to rotation (theta) and vertical shift (t_y) to strictly prevent
    unconstrained affine shearing or axial scaling that would artificially alter
    retinal layer thickness (a critical clinical biomarker).
    """

    def __init__(
        self,
        in_channels: int = 5,
        max_angle_deg: float = 25.0,
        max_shift_y_ratio: float = 0.20,
        axial_height: int = 768,
        width: int = 320,
    ):
        super().__init__()
        self.axial_height = float(axial_height)
        self.width = float(width)
        self.aspect_ratio = self.axial_height / self.width  # e.g., 768 / 320 = 2.40
        self.max_angle_rad = float(max_angle_deg * (torch.pi / 180.0))
        self.max_shift_y_ratio = float(max_shift_y_ratio)

        # Multi-scale downsampling localization backbone
        # Operates on downsampled features to be invariant to fine speckle noise
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),  # (B, 16, H/2, W/2)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                              # (B, 16, H/4, W/4)

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),           # (B, 32, H/8, W/8)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                              # (B, 32, H/16, W/16)

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),          # (B, 64, H/32, W/32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),                                     # (B, 64, 1, 1)
            nn.Flatten(),
        )

        # Parameter regression head: [theta, t_y]
        self.fc_loc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )

        # Identity initialization: Start with exactly 0.0 tilt and 0.0 shift
        # Guarantees that at epoch 0, the STN behaves identically to the baseline unwarped network
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.zero_()

    def build_affine_matrix(
        self,
        theta: torch.Tensor,
        ty: torch.Tensor,
        inverse: bool = False,
    ) -> torch.Tensor:
        """
        Builds a (B, 2, 3) affine transformation matrix for F.affine_grid.

        Because PyTorch's affine_grid operates on normalized coordinates [-1, 1] x [-1, 1],
        a naive 2D rotation matrix produces elliptical shear on non-square images (H != W).
        We correct for this aspect ratio so that rotation in pixel space is physically isometric:
            x_norm = x_px / (W / 2)
            y_norm = y_px / (H / 2)
        """
        batch_size = theta.shape[0]
        device = theta.device

        angle = -theta if inverse else theta
        shift_y = -ty if inverse else ty

        cos_t = torch.cos(angle)
        sin_t = torch.sin(angle)
        s = self.aspect_ratio  # H / W

        # Corrected 2D rotation components in normalized coordinate space
        r00 = cos_t
        r01 = -sin_t * s
        r10 = sin_t / s
        r11 = cos_t
        zeros = torch.zeros_like(theta)

        row0 = torch.stack([r00, r01, zeros], dim=-1)
        row1 = torch.stack([r10, r11, shift_y], dim=-1)
        theta_mat = torch.stack([row0, row1], dim=1)  # (B, 2, 3)

        return theta_mat


    def forward(
        self,
        x: torch.Tensor,
        override_theta_deg: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transforms input slices into canonical horizontal coordinates.

        Args:
            x: Input tensor of shape (B, C, H, W)
            override_theta_deg: Optional manually specified rotation angle in degrees for testing/ablation
        Returns:
            x_canonical: (B, C, H, W) resampled tensor
            mat_fwd: (B, 2, 3) forward affine transform matrix
            mat_inv: (B, 2, 3) inverse affine transform matrix
            theta_deg: (B,) predicted orientation tilt in degrees
        """
        batch_size = x.size(0)

        if override_theta_deg is not None:
            theta = override_theta_deg * (torch.pi / 180.0)
            ty = torch.zeros((batch_size,), device=x.device, dtype=x.dtype)
        else:
            feats = self.localization(x)
            raw_params = self.fc_loc(feats)
            # Bound rotation to [-max_angle, +max_angle] and vertical shift to [-max_shift_y, +max_shift_y]
            theta = torch.tanh(raw_params[:, 0]) * self.max_angle_rad
            ty = torch.tanh(raw_params[:, 1]) * self.max_shift_y_ratio

        mat_fwd = self.build_affine_matrix(theta, ty, inverse=False)
        mat_inv = self.build_affine_matrix(theta, ty, inverse=True)

        grid_fwd = F.affine_grid(mat_fwd, x.size(), align_corners=True)
        x_canonical = F.grid_sample(
            x,
            grid_fwd,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        theta_deg = theta * (180.0 / torch.pi)
        return x_canonical, mat_fwd, mat_inv, theta_deg


def invert_dense_mask(
    canonical_mask_logits: torch.Tensor,
    mat_inv: torch.Tensor,
) -> torch.Tensor:
    """
    Inverts canonical dense mask logits back to native patient scan coordinates.
    """
    grid_inv = F.affine_grid(mat_inv, canonical_mask_logits.size(), align_corners=True)
    native_mask_logits = F.grid_sample(
        canonical_mask_logits,
        grid_inv,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return native_mask_logits


def invert_1d_boundary_points(
    x_coords: torch.Tensor,
    y_coords: torch.Tensor,
    theta_deg: torch.Tensor,
    ty_ratio: torch.Tensor,
    height: float = 768.0,
    width: float = 320.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Analytically projects canonical 1D continuous boundary coordinates back to native space.
    
    Args:
        x_coords: Tensor of horizontal positions (e.g., 0 to W-1)
        y_coords: Regressed vertical depth row values [0, H]
        theta_deg: Rotation angle in degrees
        ty_ratio: Normalized vertical shift ratio [-0.2, 0.2]
    Returns:
        (x_native, y_native): Projected coordinates in native pixel space
    """
    theta = theta_deg * (torch.pi / 180.0)
    
    # 1. Shift to center
    x_c = x_coords - (width / 2.0)
    y_c = y_coords - (height / 2.0)

    # 2. Inverse rotation by -theta
    cos_t = torch.cos(-theta)
    sin_t = torch.sin(-theta)
    
    x_native = cos_t * x_c - sin_t * y_c + (width / 2.0)
    y_native = (sin_t * x_c + cos_t * y_c) - (ty_ratio * height / 2.0) + (height / 2.0)

    return x_native, y_native
