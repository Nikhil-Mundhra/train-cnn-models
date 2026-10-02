import sys
import unittest
from pathlib import Path

import torch


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from profile_feasibility import AnisotropicUNetCapacityProbe, parse_shapes


class TestCapacityProbe(unittest.TestCase):
    def test_parse_shapes(self):
        self.assertEqual(
            parse_shapes("32x768x32,64x384x64"),
            [(32, 768, 32), (64, 384, 64)],
        )

    def test_probe_preserves_input_shape(self):
        model = AnisotropicUNetCapacityProbe(base_channels=4)
        image = torch.randn(1, 1, 8, 32, 8)
        self.assertEqual(model(image).shape, image.shape)


if __name__ == "__main__":
    unittest.main()

