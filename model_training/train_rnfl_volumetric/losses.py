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
from monai.losses import DiceLoss

AXIAL_UM = 3.12367


class VolumetricRNFLLoss(nn.Module):
    def __init__(
        self,
        weight_dice=1.0,
        weight_bce=1.0,
        weight_boundary=0.05,
        weight_cup=0.5,
        weight_topo=0.1
    ):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.weight_boundary = weight_boundary
        self.weight_cup = weight_cup
        self.weight_topo = weight_topo

        self.dice_loss = DiceLoss(sigmoid=True, smooth_nr=1e-5, smooth_dr=1e-5)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, preds, targets):
        """
        preds: dict containing:
            'mask_logits': (B, 1, H, W)
            'ilm_pred':    (B, W)
            'nfl_pred':    (B, W)
            'cup_logits':  (B, W)
        targets: dict containing:
            'mask':        (B, 1, H, W)
            'ilm_surface': (B, W)
            'nfl_surface': (B, W)
            'cup_absent':  (B, W)
        """
        # 1. Dense Segmentation Losses
        loss_dice = self.dice_loss(preds['mask_logits'], targets['mask'])
        loss_bce = self.bce_loss(preds['mask_logits'], targets['mask'])
        loss_seg = self.weight_dice * loss_dice + self.weight_bce * loss_bce

        # 2. Surface Boundary Depth Losses (scaled to microns)
        ilm_err = self.smooth_l1(preds['ilm_pred'], targets['ilm_surface']) * AXIAL_UM
        loss_ilm = ilm_err.mean()

        # NFL loss only where tissue is present (cup_absent == 0)
        cup_absent = targets['cup_absent']  # (B, W)
        tissue_mask = (1.0 - cup_absent)
        nfl_err = self.smooth_l1(preds['nfl_pred'], targets['nfl_surface']) * AXIAL_UM
        nfl_valid_err = nfl_err * tissue_mask
        loss_nfl = nfl_valid_err.sum() / (tissue_mask.sum() + 1e-6)

        loss_boundary = loss_ilm + loss_nfl

        # 3. Optic Cup Absence Loss
        loss_cup = self.bce_loss(preds['cup_logits'], cup_absent)

        # 4. Topological Ordering Constraint: ILM must be <= NFL (depth increases downwards)
        # Violations occur when ilm_pred > nfl_pred
        topo_violation = F.relu(preds['ilm_pred'] - preds['nfl_pred']) * tissue_mask
        loss_topo = topo_violation.mean()

        # Total Weighted Loss
        total_loss = (
            loss_seg
            + self.weight_boundary * loss_boundary
            + self.weight_cup * loss_cup
            + self.weight_topo * loss_topo
        )

        return {
            'loss': total_loss,
            'loss_seg': loss_seg.detach(),
            'loss_dice': loss_dice.detach(),
            'loss_boundary': loss_boundary.detach(),
            'loss_ilm': loss_ilm.detach(),
            'loss_nfl': loss_nfl.detach(),
            'loss_cup': loss_cup.detach(),
            'loss_topo': loss_topo.detach(),
        }
