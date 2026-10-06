"""
test_canonicalizer_and_augmentations.py
======================================
Unit tests for ConstrainedSpatialTransformer, CanonicalVolumetricRNFLNet,
and OCTRobustnessAugmenter.
"""

import sys
import os
import torch

# Ensure model directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from canonicalizer import (
    ConstrainedSpatialTransformer,
    invert_dense_mask,
    invert_1d_boundary_points,
)
from augmentations import (
    OCTRobustnessAugmenter,
    apply_speckle_noise,
    apply_vessel_shadows,
    apply_axial_roll_off,
    apply_focal_dropout,
)
from model import CanonicalVolumetricRNFLNet, VolumetricRNFLNet


def test_stn_identity_initialization():
    """Verify that newly initialized STN has zero tilt and reproduces identity warp."""
    stn = ConstrainedSpatialTransformer(in_channels=5, axial_height=768, width=320)
    x = torch.rand(2, 5, 768, 320)

    x_can, mat_fwd, mat_inv, theta_deg = stn(x)

    assert x_can.shape == (2, 5, 768, 320)
    assert mat_fwd.shape == (2, 2, 3)
    assert mat_inv.shape == (2, 2, 3)
    assert torch.allclose(theta_deg, torch.zeros_like(theta_deg), atol=1e-4)

    # In identity state, forward output should match input closely (minor bilinear border tolerance)
    diff = torch.abs(x_can[:, :, 10:-10, 10:-10] - x[:, :, 10:-10, 10:-10]).max()
    assert diff < 1e-3, f"Identity warp error too high: {diff}"


def test_stn_differentiability():
    """Verify that gradients flow through grid_sample back to the STN parameters."""
    stn = ConstrainedSpatialTransformer(in_channels=5, axial_height=768, width=320)
    x = torch.rand(2, 5, 768, 320, requires_grad=True)

    x_can, _, _, theta_deg = stn(x)
    loss = x_can.sum() + theta_deg.sum()
    loss.backward()

    # The final regression head receives non-zero gradients directly at identity init
    assert stn.fc_loc[-1].weight.grad is not None and stn.fc_loc[-1].weight.grad.abs().sum() > 0
    assert stn.fc_loc[-1].bias.grad is not None and stn.fc_loc[-1].bias.grad.abs().sum() > 0

    # With non-zero weights (after initial optimizer update), earlier layers also receive gradients
    stn.zero_grad()
    with torch.no_grad():
        stn.fc_loc[-1].weight.normal_(0.0, 0.05)
    x_can2, _, _, theta_deg2 = stn(x)
    loss2 = x_can2.sum() + theta_deg2.sum()
    loss2.backward()
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in stn.localization.parameters())
    assert has_grads, "No gradients reached the STN localization parameters after weight perturbation!"



def test_stn_override_rotation_and_inversion():
    """Verify that manual rotation and inverse mask warping function properly."""
    stn = ConstrainedSpatialTransformer(in_channels=1, axial_height=768, width=320)
    
    # Create an artificial horizontal bar mask
    mask = torch.zeros(1, 1, 768, 320)
    mask[:, :, 380:400, :] = 1.0

    # Rotate forward by +15 degrees
    override_deg = torch.tensor([15.0])
    can_mask, mat_fwd, mat_inv, _ = stn(mask, override_theta_deg=override_deg)

    # Invert back to native space
    restored_mask = invert_dense_mask(can_mask, mat_inv)

    assert can_mask.shape == (1, 1, 768, 320)
    assert restored_mask.shape == (1, 1, 768, 320)

    # Center of restored mask should overlap original bar
    overlap = (restored_mask[:, :, 380:400, 100:220] > 0.5).float().mean()
    assert overlap > 0.85, f"Restored mask overlap too low: {overlap}"


def test_robustness_augmentations():
    """Verify that all OCT physics noise transforms maintain shape and valid ranges."""
    x = torch.rand(4, 5, 768, 320)

    # Speckle noise
    speckled = apply_speckle_noise(x, prob=1.0, sigma_min=0.05, sigma_max=0.10)
    assert speckled.shape == x.shape
    assert speckled.min() >= 0.0 and speckled.max() <= 1.0

    # Vessel shadow bands
    shadowed = apply_vessel_shadows(x, prob=1.0, max_vessels=2)
    assert shadowed.shape == x.shape
    assert shadowed.min() >= 0.0 and shadowed.max() <= 1.0

    # Axial roll-off
    decayed = apply_axial_roll_off(x, prob=1.0)
    assert decayed.shape == x.shape
    assert decayed.min() >= 0.0 and decayed.max() <= 1.0

    # Focal dropout
    dropped = apply_focal_dropout(x, prob=1.0)
    assert dropped.shape == x.shape
    assert dropped.min() >= 0.0 and dropped.max() <= 1.0

    # Unified module
    augmenter = OCTRobustnessAugmenter()
    batch = {"image": x.clone(), "mask": torch.ones(4, 1, 768, 320)}
    out_batch = augmenter(batch)
    assert out_batch["image"].shape == x.shape
    assert torch.equal(out_batch["mask"], torch.ones(4, 1, 768, 320))


def test_canonical_volumetric_rnfl_net_end_to_end():
    """Verify that CanonicalVolumetricRNFLNet executes end-to-end forward and backward passes."""
    model = CanonicalVolumetricRNFLNet(
        in_channels=5,
        base_channels=8,
        channels=(8, 16, 32, 64, 128),
        num_res_units=1,
        axial_height=768,
        width=320,
    )
    model.train()

    x = torch.rand(2, 5, 768, 320, requires_grad=True)
    preds = model(x)

    assert "mask_logits" in preds
    assert "native_mask_logits" in preds
    assert "canonical_mask_logits" in preds
    assert "ilm_pred" in preds
    assert "nfl_pred" in preds
    assert "cup_logits" in preds
    assert "predicted_tilt_deg" in preds

    assert preds["mask_logits"].shape == (2, 1, 768, 320)
    assert preds["ilm_pred"].shape == (2, 320)
    assert preds["nfl_pred"].shape == (2, 320)
    assert preds["cup_logits"].shape == (2, 320)
    assert preds["predicted_tilt_deg"].shape == (2,)

    # Test backward loss
    loss = (
        preds["mask_logits"].sum()
        + preds["ilm_pred"].sum()
        + preds["nfl_pred"].sum()
        + preds["predicted_tilt_deg"].pow(2).sum()
    )
    loss.backward()
    assert x.grad is not None
