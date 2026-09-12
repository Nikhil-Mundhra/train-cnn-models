"""
test_pipeline.py
================
Unit tests for the Volumetric RNFL training pipeline:
- Dataset loading and eye standardization
- Model forward pass
- Loss calculation and backward pass
- Optic disc geometry computation
"""

import os
import sys
import unittest
from pathlib import Path
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PKG_DIR = Path(__file__).resolve().parent.parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from dataset import SolixRNFLDataset
from model import VolumetricRNFLNet
from losses import VolumetricRNFLLoss
from evaluate_baseline import compute_disc_geometry, load_curves


class TestRNFLPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_root = "/Users/nikhilmundhra/Library/CloudStorage/Box-Box/deidentified"
        cls.sample_subject = "BEH0181"

    def test_curve_loading_and_disc_geometry(self):
        xml_path = os.path.join(
            self.dataset_root, "tsv", "good", self.sample_subject,
            "curve", f"{self.sample_subject}, {self.sample_subject} _OD_Disc Cube_175_390_6_1.xml"
        )
        self.assertTrue(os.path.exists(xml_path), f"File not found: {xml_path}")

        curves = load_curves(xml_path)
        self.assertIn('ILM', curves)
        self.assertIn('NFL', curves)
        self.assertEqual(curves['ILM'].shape, (320, 320))

        zc, xc, radius = compute_disc_geometry(curves['NFL'])
        self.assertGreater(zc, 100.0)
        self.assertLess(zc, 220.0)
        self.assertGreater(xc, 100.0)
        self.assertLess(xc, 220.0)
        self.assertGreater(radius, 20.0)
        self.assertLess(radius, 80.0)

    def test_dataset_item_shapes(self):
        ds = SolixRNFLDataset(
            dataset_root=self.dataset_root,
            subjects=[self.sample_subject],
            context_slices=5
        )
        self.assertGreater(len(ds), 0)

        item = ds[160]
        self.assertEqual(item['image'].shape, (5, 768, 320))
        self.assertEqual(item['mask'].shape, (1, 768, 320))
        self.assertEqual(item['ilm_surface'].shape, (320,))
        self.assertEqual(item['nfl_surface'].shape, (320,))
        self.assertEqual(item['cup_absent'].shape, (320,))

    def test_model_forward_and_loss(self):
        device = torch.device("cpu")
        model = VolumetricRNFLNet(in_channels=5, base_channels=8).to(device)
        criterion = VolumetricRNFLLoss().to(device)

        dummy_img = torch.randn(2, 5, 768, 320, device=device)
        dummy_targets = {
            'mask': torch.randint(0, 2, (2, 1, 768, 320), device=device).float(),
            'ilm_surface': torch.full((2, 320), 260.0, device=device),
            'nfl_surface': torch.full((2, 320), 285.0, device=device),
            'cup_absent': torch.zeros((2, 320), device=device)
        }

        preds = model(dummy_img)
        self.assertEqual(preds['mask_logits'].shape, (2, 1, 768, 320))
        self.assertEqual(preds['ilm_pred'].shape, (2, 320))
        self.assertEqual(preds['nfl_pred'].shape, (2, 320))
        self.assertEqual(preds['cup_logits'].shape, (2, 320))

        losses = criterion(preds, dummy_targets)
        self.assertTrue(torch.isfinite(losses['loss']))
        self.assertGreater(losses['loss'].item(), 0.0)

        losses['loss'].backward()
        # Verify gradients exist
        self.assertIsNotNone(model.mask_head.weight.grad)


if __name__ == "__main__":
    unittest.main()
