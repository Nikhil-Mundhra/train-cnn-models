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
from train_3d import VolumetricRNFL3DLoss


class TestTrain3D(unittest.TestCase):
    def test_volumetric_loss_perfect_match(self):
        criterion = VolumetricRNFL3DLoss()
        # Perfect match: logits high positive where mask is 1, large negative where mask is 0
        mask = torch.zeros((1, 1, 8, 64, 8), dtype=torch.float32)
        mask[:, :, 2:6, 20:40, 2:6] = 1.0
        logits = (mask * 20.0) - 10.0
        ilm_surface = torch.full((1, 8, 8), 20.0)
        nfl_surface = torch.full((1, 8, 8), 40.0)

        preds = {
            "mask_logits": logits,
            "ilm_pred": ilm_surface.clone(),
            "nfl_pred": nfl_surface.clone(),
            "cup_logits": torch.full((1, 8, 8), -10.0),
            "column_thickness": torch.full((1, 8, 8), 20.0),
        }
        batch = {
            "mask": mask,
            "ilm_surface": ilm_surface,
            "nfl_surface": nfl_surface,
            "ilm_valid": torch.ones((1, 8, 8), dtype=torch.bool),
            "nfl_valid": torch.ones((1, 8, 8), dtype=torch.bool),
            "nfl_absent": torch.zeros((1, 8, 8), dtype=torch.bool),
        }

        total_loss, metrics = criterion(preds, batch)
        self.assertAlmostEqual(metrics["dice"], 1.0, places=2)
        self.assertAlmostEqual(metrics["dice_loss"], 0.0, places=2)
        self.assertLess(metrics["loss_ilm"], 0.1)
        self.assertLess(metrics["loss_nfl"], 0.1)

    def test_volumetric_loss_gradients(self):
        criterion = VolumetricRNFL3DLoss()
        logits = torch.randn((1, 1, 8, 32, 8), requires_grad=True)
        ilm_pred = torch.randn((1, 8, 8), requires_grad=True)
        nfl_pred = torch.randn((1, 8, 8), requires_grad=True)
        cup_logits = torch.randn((1, 8, 8), requires_grad=True)
        column_thick = torch.randn((1, 8, 8), requires_grad=True)

        preds = {
            "mask_logits": logits,
            "ilm_pred": ilm_pred,
            "nfl_pred": nfl_pred,
            "cup_logits": cup_logits,
            "column_thickness": column_thick,
        }
        batch = {
            "mask": (torch.rand((1, 1, 8, 32, 8)) > 0.5).float(),
            "ilm_surface": torch.rand((1, 8, 8)) * 10,
            "nfl_surface": torch.rand((1, 8, 8)) * 20,
            "ilm_valid": torch.ones((1, 8, 8), dtype=torch.bool),
            "nfl_valid": torch.ones((1, 8, 8), dtype=torch.bool),
            "nfl_absent": torch.zeros((1, 8, 8), dtype=torch.bool),
        }
        total_loss, _ = criterion(preds, batch)
        total_loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(ilm_pred.grad)
        self.assertIsNotNone(nfl_pred.grad)
        self.assertIsNotNone(cup_logits.grad)
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
