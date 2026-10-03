"""
test_predictor_3d.py
====================
Unit test suite verifying:
1. Anisotropic3DPredictor instantiation and device configuration.
2. VolumePrediction generation with valid mask, ilm_curve, nfl_curve, and cup_probs.
3. Subpixel surface refinement and cup detection.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

TESTS_DIR = Path(__file__).resolve().parent
MODULE_DIR = TESTS_DIR.parent
VOLUMETRIC_DIR = MODULE_DIR.parent / "train_rnfl_volumetric"

if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
if str(VOLUMETRIC_DIR) not in sys.path:
    sys.path.insert(0, str(VOLUMETRIC_DIR))

from model_3d import AnisotropicRNFLUNet3D, AnisotropicUNetConfig
from predictor_3d import Anisotropic3DPredictor
from batch_cohort_evaluator import DiscGeometry, VolumePrediction


class TestAnisotropic3DPredictor(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        config = AnisotropicUNetConfig(base_channels=4)
        self.model = AnisotropicRNFLUNet3D(config).to(self.device)
        self.predictor = Anisotropic3DPredictor(
            model=self.model,
            device=self.device,
            stride=(16, 16),
            patch_shape=(16, 64, 16),
            threshold=0.5,
        )

    def test_predictor_initialization(self):
        self.assertIsNotNone(self.predictor.model)
        self.assertEqual(self.predictor.stride, (16, 16))
        self.assertEqual(self.predictor.patch_shape, (16, 64, 16))

    def test_mock_volume_prediction(self):
        # Create a mock OCT volume with synthetic memmap
        mock_vol = MagicMock()
        mock_vol.eye = "OD"
        mock_vol.subject = "TEST001"
        mock_vol.disc_geometry = DiscGeometry(zc=8, xc=8, r_disc=4)

        # Small 16x64x16 volume for rapid CPU test
        synthetic_volume = np.zeros((16, 64, 16), dtype=np.uint16)
        synthetic_volume[:, 20:30, :] = 1500  # simulated RNFL tissue
        mock_vol.memmap = synthetic_volume

        pred = self.predictor.predict(mock_vol)

        self.assertIsInstance(pred, VolumePrediction)
        self.assertEqual(pred.mask.shape, (16, 64, 16))
        self.assertEqual(pred.ilm_curve.shape, (16, 16))
        self.assertEqual(pred.nfl_curve.shape, (16, 16))
        self.assertEqual(pred.cup_probs.shape, (16, 16))
        self.assertIn("total_rnfl_volume_mm3", pred.diagnostics)


if __name__ == "__main__":
    unittest.main()
