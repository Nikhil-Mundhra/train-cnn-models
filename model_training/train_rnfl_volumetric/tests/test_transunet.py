"""
test_transunet.py
=================
Unit tests for TransUNet RNFL architecture:
- Forward pass tensor dimensions
- Laterality conditioning embedding
- Loss computation & backward gradient flow
- Multi-task output heads integrity
"""

import sys
import unittest
from pathlib import Path
import torch

SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from transunet import TransUNetRNFLNet
from losses import VolumetricRNFLLoss


class TestTransUNet(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.batch_size = 2
        self.in_channels = 5
        self.H = 768
        self.W = 320
        self.model = TransUNetRNFLNet(
            in_channels=self.in_channels,
            base_channels=16,
            hidden_size=256,
            num_layers=4,
            num_heads=8,
            mlp_dim=512,
            axial_height=self.H,
            width=self.W,
            use_laterality_embedding=True,
        ).to(self.device)

    def test_forward_pass_dimensions(self):
        x = torch.randn(self.batch_size, self.in_channels, self.H, self.W, device=self.device)
        eye_idx = torch.tensor([0, 1], device=self.device)
        preds = self.model(x, eye_idx=eye_idx)

        self.assertIn("mask_logits", preds)
        self.assertIn("ilm_pred", preds)
        self.assertIn("nfl_pred", preds)
        self.assertIn("cup_logits", preds)
        self.assertIn("column_thickness", preds)

        self.assertEqual(preds["mask_logits"].shape, (self.batch_size, 1, self.H, self.W))
        self.assertEqual(preds["ilm_pred"].shape, (self.batch_size, self.W))
        self.assertEqual(preds["nfl_pred"].shape, (self.batch_size, self.W))
        self.assertEqual(preds["cup_logits"].shape, (self.batch_size, self.W))
        self.assertEqual(preds["column_thickness"].shape, (self.batch_size, self.W))

    def test_loss_backward_gradient(self):
        x = torch.randn(self.batch_size, self.in_channels, self.H, self.W, device=self.device, requires_grad=True)
        preds = self.model(x)

        targets = {
            "mask": torch.randint(0, 2, (self.batch_size, 1, self.H, self.W), dtype=torch.float32, device=self.device),
            "ilm_surface": torch.full((self.batch_size, self.W), 120.0, device=self.device),
            "nfl_surface": torch.full((self.batch_size, self.W), 160.0, device=self.device),
            "cup_absent": torch.zeros((self.batch_size, self.W), device=self.device),
            "is_peripapillary": torch.ones(self.batch_size, dtype=torch.bool, device=self.device),
            "image": x,
        }

        criterion = VolumetricRNFLLoss()
        loss_dict = criterion(preds, targets)
        loss = loss_dict["loss"]
        self.assertTrue(torch.isfinite(loss).item())

        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.norm().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
