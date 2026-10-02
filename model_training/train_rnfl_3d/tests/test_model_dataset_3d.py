import sys
import unittest
from pathlib import Path

import numpy as np
import torch


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from dataset_3d import SolixRNFL3DPatchDataset
from model_3d import AnisotropicRNFLUNet3D, AnisotropicUNetConfig


class TestDense3DModel(unittest.TestCase):
    def test_dense_head_preserves_shape(self):
        config = AnisotropicUNetConfig(base_channels=4, patch_shape=(8, 32, 8))
        model = AnisotropicRNFLUNet3D(config)
        image = torch.randn(1, 1, 8, 32, 8)
        output = model(image)
        self.assertEqual(output["mask_logits"].shape, image.shape)
        self.assertNotIn("cup_logits", output)


class TestPatchExtraction(unittest.TestCase):
    def _dataset(self, eye="OD"):
        dataset = SolixRNFL3DPatchDataset.__new__(SolixRNFL3DPatchDataset)
        dataset.patch_shape = (2, 8, 4)
        dataset.standardize_eye = True
        dataset.augment = False
        volume = np.arange(4 * 8 * 8, dtype=np.uint16).reshape(4, 8, 8)
        ilm = np.full((4, 8), 2.0, dtype=np.float32)
        nfl = np.full((4, 8), 5.0, dtype=np.float32)
        nfl[0, 0] = np.nan
        scan = {
            "subject": "TEST",
            "eye": eye,
            "volume": volume,
            "curves": {"ILM": ilm, "NFL": nfl},
        }
        return dataset, scan

    def test_targets_keep_ilm_valid_when_nfl_is_absent(self):
        dataset, scan = self._dataset("OD")
        patch = dataset._extract_patch(scan, 0, 0)
        self.assertEqual(patch["image"].shape, (1, 2, 8, 4))
        self.assertEqual(patch["mask"].shape, (1, 2, 8, 4))
        self.assertTrue(patch["ilm_valid"][0, 0])
        self.assertFalse(patch["nfl_valid"][0, 0])
        self.assertTrue(patch["nfl_absent"][0, 0])
        self.assertEqual(float(patch["mask"][0, 0, :, 0].sum()), 0.0)
        self.assertEqual(float(patch["mask"][0, 0, :, 1].sum()), 3.0)

    def test_os_patch_is_returned_in_canonical_x_orientation(self):
        dataset, scan = self._dataset("OS")
        patch = dataset._extract_patch(scan, 0, 0)
        expected = np.flip(scan["volume"][0:2, :, 4:8], axis=-1).copy() / 2560.0
        np.testing.assert_allclose(patch["image"].numpy()[0], expected)


if __name__ == "__main__":
    unittest.main()

