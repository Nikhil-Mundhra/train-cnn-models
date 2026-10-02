"""Unit tests for 3D Anisotropic RNFL training components."""

import unittest
import numpy as np
import torch
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model_3d import AnisotropicRNFLUNet3D, AnisotropicUNetConfig
from dataset_3d import build_ground_truth_mask, predict_whole_volume_mask
from train_3d import DiceBCELoss


class TestTrain3D(unittest.TestCase):
    def test_dice_bce_loss_perfect_match(self):
        criterion = DiceBCELoss(bce_weight=1.0, dice_weight=1.0)
        # Perfect match: logits high positive where mask is 1, large negative where mask is 0
        mask = torch.zeros((1, 1, 16, 64, 16), dtype=torch.float32)
        mask[:, :, 4:12, 20:40, 4:12] = 1.0
        logits = (mask * 20.0) - 10.0

        total_loss, bce, dice_loss, dice = criterion(logits, mask)
        self.assertAlmostEqual(dice.item(), 1.0, places=2)
        self.assertAlmostEqual(dice_loss.item(), 0.0, places=2)
        self.assertLess(total_loss.item(), 0.1)

    def test_dice_bce_loss_gradients(self):
        criterion = DiceBCELoss()
        logits = torch.randn((1, 1, 8, 32, 8), requires_grad=True)
        mask = (torch.rand((1, 1, 8, 32, 8)) > 0.5).float()
        total_loss, _, _, _ = criterion(logits, mask)
        total_loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_model_3d_small_forward(self):
        # Test model with smaller base channels for fast unit test
        config = AnisotropicUNetConfig(
            in_channels=1,
            out_channels=1,
            base_channels=8,
            patch_shape=(16, 128, 16),
            strides=((1, 2, 1), (1, 2, 1), (2, 2, 2), (2, 2, 2)),
        )
        model = AnisotropicRNFLUNet3D(config=config)
        dummy_input = torch.randn(1, 1, 16, 128, 16)
        out = model(dummy_input)
        self.assertIn("mask_logits", out)
        self.assertEqual(out["mask_logits"].shape, (1, 1, 16, 128, 16))

    def test_build_ground_truth_mask(self):
        ilm = np.full((32, 32), 10.0, dtype=np.float32)
        nfl = np.full((32, 32), 20.0, dtype=np.float32)
        # sentinel absent in corner
        nfl[:5, :5] = np.nan
        curves = {"ILM": ilm, "NFL": nfl}
        mask = build_ground_truth_mask(curves, shape=(32, 64, 32))
        self.assertEqual(mask.shape, (32, 64, 32))
        # Where NFL is valid and 10 <= y < 20, mask should be 1
        self.assertTrue((mask[10:20, 10:20, 10:20] == 1).all())
        # Where NFL is absent, mask should be 0
        self.assertTrue((mask[:5, :, :5] == 0).all())

    def test_predict_whole_volume_mask_synthetic(self):
        # Synthetic tiny volume
        class DummyModel(torch.nn.Module):
            def forward(self, x):
                return {"mask_logits": torch.full_like(x, 5.0)}

        dummy_model = DummyModel()
        volume = np.ones((64, 64, 64), dtype=np.uint16) * 1000
        pred = predict_whole_volume_mask(
            model=dummy_model,
            volume=volume,
            patch_shape=(32, 64, 32),
            stride=(32, 32),
            device="cpu",
            use_amp=False,
        )
        self.assertEqual(pred.shape, (64, 64, 64))
        self.assertEqual(pred.dtype, np.uint8)
        self.assertTrue((pred == 1).all())


if __name__ == "__main__":
    unittest.main()
