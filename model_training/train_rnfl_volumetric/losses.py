"""
losses.py
=========
Combined multi-task loss for Volumetric RNFL segmentation:
- Dice + BCE loss on the dense voxel mask
- Smooth L1 (Huber) loss on boundary depths in microns
- Binary Cross Entropy loss on optic cup absence detection
- Topological ordering constraint: penalizes ILM crossing below NFL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, TverskyLoss

AXIAL_UM = 3.12367


class VolumetricRNFLLoss(nn.Module):
    def __init__(
        self,
        weight_dice=1.0,
        weight_bce=1.0,
        weight_boundary=0.4,
        weight_cup=0.5,
        weight_topo=0.1,
        weight_edge=0.2,
        weight_thickness=0.3,
        peripapillary_weight=1.5,
        use_tversky=True,
        tversky_alpha=0.3,
        tversky_beta=0.7
    ):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.weight_boundary = weight_boundary
        self.weight_cup = weight_cup
        self.weight_topo = weight_topo
        self.weight_edge = weight_edge
        self.weight_thickness = weight_thickness
        self.peripapillary_weight = peripapillary_weight
        self.use_tversky = use_tversky

        if use_tversky:
            # Tversky loss with beta=0.7 penalizes False Negatives (missing thin RNFL) heavily
            self.seg_overlap_loss = TverskyLoss(sigmoid=True, alpha=tversky_alpha, beta=tversky_beta)
        else:
            self.seg_overlap_loss = DiceLoss(sigmoid=True, smooth_nr=1e-5, smooth_dr=1e-5)

        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, preds, targets):
        """
        preds: dict containing:
            'mask_logits': (B, 1, H, W)
            'ilm_pred':    (B, W)
            'nfl_pred':    (B, W)
            'cup_logits':  (B, W)
        targets: dict containing:
            'mask':             (B, 1, H, W)
            'ilm_surface':      (B, W)
            'nfl_surface':      (B, W)
            'cup_absent':       (B, W)
            'image':            (B, C, H, W) optional for edge loss
            'is_peripapillary': (B,) optional for rim weighting
        """
        # 1. Dense Segmentation Losses
        target_mask = targets['mask']
        loss_overlap = self.seg_overlap_loss(preds['mask_logits'], target_mask)

        # Boundary-aware spatial weighting (ReLayNet-inspired):
        # 4x boost on layer transition boundaries, 2x boost on foreground voxels.
        # This prevents the 99.4% background voxels from zeroing out thin peripheral layers.
        boundary = torch.abs(target_mask[:, :, 1:, :] - target_mask[:, :, :-1, :])
        boundary_pad = F.pad(boundary, (0, 0, 1, 0))
        spatial_weights = 1.0 + 4.0 * boundary_pad + 2.0 * target_mask
        loss_bce = (F.binary_cross_entropy_with_logits(
            preds['mask_logits'], target_mask, reduction='none'
        ) * spatial_weights).mean()

        loss_seg = self.weight_dice * loss_overlap + self.weight_bce * loss_bce

        # 2. Surface Boundary Depth Losses (scaled to microns)
        ilm_err = self.smooth_l1(preds['ilm_pred'], targets['ilm_surface']) * AXIAL_UM
        loss_ilm = ilm_err.mean()

        # NFL loss only where tissue is present (cup_absent == 0)
        cup_absent = targets['cup_absent']  # (B, W)
        tissue_mask = (1.0 - cup_absent)

        # Peripapillary loss weighting: heavily penalize errors around the disc margin
        is_peri = targets.get('is_peripapillary', None)
        if is_peri is not None and is_peri.any():
            peri_weight = torch.ones_like(tissue_mask)
            peri_weight[is_peri] *= self.peripapillary_weight
        else:
            peri_weight = 1.0

        nfl_err = self.smooth_l1(preds['nfl_pred'], targets['nfl_surface']) * AXIAL_UM
        nfl_valid_err = nfl_err * tissue_mask * peri_weight
        loss_nfl = nfl_valid_err.sum() / ((tissue_mask * peri_weight).sum() + 1e-6)

        loss_boundary = loss_ilm + loss_nfl

        # 3. Optic Cup Absence Loss
        loss_cup = F.binary_cross_entropy_with_logits(preds['cup_logits'], cup_absent)

        # 4. Topological Ordering Constraint: ILM must be <= NFL (depth increases downwards)
        topo_violation = F.relu(preds['ilm_pred'] - preds['nfl_pred']) * tissue_mask
        loss_topo = topo_violation.mean()

        # 5. Optical Reflectance Edge Alignment Loss (snaps to physical bright-to-dark drop-off)
        loss_edge = torch.tensor(0.0, device=preds['nfl_pred'].device)
        if self.weight_edge > 0 and 'image' in targets:
            img = targets['image'][:, 2:3, :, :].float()  # central B-scan
            B, _, H_img, W_img = img.shape
            edge_raw = F.relu(img[:, :, :-1, :] - img[:, :, 1:, :])

            # Vertical Gaussian smoothing to provide a wide basin of attraction
            k_size = 15
            sigma = 3.0
            k = torch.exp(-torch.arange(-(k_size//2), k_size//2 + 1, dtype=torch.float32, device=img.device)**2 / (2 * sigma**2))
            k = (k / k.sum()).view(1, 1, k_size, 1)
            edge_smooth = F.conv2d(edge_raw, k, padding=(k_size//2, 0))

            x_coords = torch.linspace(-1, 1, W_img, dtype=torch.float32, device=img.device).unsqueeze(0).expand(B, -1)
            norm_y = 2.0 * preds['nfl_pred'] / (H_img - 2) - 1.0
            grid = torch.stack([x_coords, norm_y], dim=-1).unsqueeze(1)
            sampled_edge = F.grid_sample(edge_smooth, grid, mode='bilinear', align_corners=True).squeeze(1).squeeze(1)

            # Maximize gradient at boundary (minimize -sampled_edge) where tissue is present
            edge_masked = sampled_edge * tissue_mask * (peri_weight if is_peri is not None else 1.0)
            loss_edge = -edge_masked.sum() / ((tissue_mask * peri_weight).sum() + 1e-6)

        # 6. Differentiable Column-Thickness Consistency Loss
        # Enforces that the vertical integral of dense probabilities equals the 1D physical thickness
        loss_thick = torch.tensor(0.0, device=preds['nfl_pred'].device)
        if self.weight_thickness > 0 and 'column_thickness' in preds:
            gt_thickness = F.relu(targets['nfl_surface'] - targets['ilm_surface']) * tissue_mask
            pred_thickness = preds['column_thickness'] * tissue_mask
            thick_err = self.smooth_l1(pred_thickness, gt_thickness) * AXIAL_UM
            loss_thick = thick_err.sum() / (tissue_mask.sum() + 1e-6)

        # Total Weighted Loss
        total_loss = (
            loss_seg
            + self.weight_boundary * loss_boundary
            + self.weight_cup * loss_cup
            + self.weight_topo * loss_topo
            + self.weight_edge * loss_edge
            + self.weight_thickness * loss_thick
        )

        return {
            'loss': total_loss,
            'loss_seg': loss_seg.detach(),
            'loss_overlap': loss_overlap.detach(),
            'loss_boundary': loss_boundary.detach(),
            'loss_ilm': loss_ilm.detach(),
            'loss_nfl': loss_nfl.detach(),
            'loss_cup': loss_cup.detach(),
            'loss_topo': loss_topo.detach(),
            'loss_edge': loss_edge.detach(),
            'loss_thick': loss_thick.detach(),
        }
