import math
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from evaluation_metrics import SOLIX_SPACING_UM, compute_volumetric_metrics


class TestVolumetricMetricContract(unittest.TestCase):
    def test_identical_single_voxel(self):
        mask = np.zeros((5, 7, 5), dtype=np.uint8)
        mask[2, 3, 2] = 1
        metrics = compute_volumetric_metrics(mask, mask)
        self.assertAlmostEqual(metrics.dice, 1.0)
        self.assertAlmostEqual(metrics.hd_um, 0.0)
        self.assertAlmostEqual(metrics.hd95_um, 0.0)
        self.assertAlmostEqual(metrics.asd_um, 0.0)
        self.assertAlmostEqual(metrics.volume_similarity, 1.0)

    def test_one_voxel_shift_respects_each_physical_axis(self):
        for axis, expected_um in enumerate(SOLIX_SPACING_UM):
            with self.subTest(axis=axis):
                gt = np.zeros((5, 7, 5), dtype=np.uint8)
                pred = np.zeros_like(gt)
                center = [2, 3, 2]
                gt[tuple(center)] = 1
                center[axis] += 1
                pred[tuple(center)] = 1
                metrics = compute_volumetric_metrics(gt, pred)
                self.assertAlmostEqual(metrics.hd_um, expected_um, places=5)
                self.assertAlmostEqual(metrics.hd95_um, expected_um, places=5)
                self.assertAlmostEqual(metrics.asd_um, expected_um, places=5)
                self.assertAlmostEqual(metrics.volume_similarity, 1.0)

    def test_empty_mask_policy(self):
        empty = np.zeros((3, 3, 3), dtype=np.uint8)
        nonempty = empty.copy()
        nonempty[1, 1, 1] = 1
        both_empty = compute_volumetric_metrics(empty, empty)
        one_empty = compute_volumetric_metrics(empty, nonempty)
        self.assertEqual(both_empty.dice, 1.0)
        self.assertEqual(both_empty.hd95_um, 0.0)
        self.assertEqual(one_empty.dice, 0.0)
        self.assertTrue(math.isnan(one_empty.hd95_um))
        self.assertEqual(one_empty.volume_similarity, 0.0)

    def test_volume_similarity_is_bounded_and_symmetric(self):
        gt = np.zeros((3, 3, 3), dtype=np.uint8)
        pred = np.zeros_like(gt)
        gt[1, 1, 1] = 1
        pred[1, 1, 1] = 1
        pred[1, 1, 2] = 1
        forward = compute_volumetric_metrics(gt, pred)
        reverse = compute_volumetric_metrics(pred, gt)
        self.assertAlmostEqual(forward.volume_similarity, 2.0 / 3.0)
        self.assertAlmostEqual(reverse.volume_similarity, 2.0 / 3.0)

    def test_rejects_non_binary_and_shape_mismatch(self):
        with self.assertRaises(ValueError):
            compute_volumetric_metrics(np.full((2, 2, 2), 2), np.zeros((2, 2, 2)))
        with self.assertRaises(ValueError):
            compute_volumetric_metrics(np.zeros((2, 2, 2)), np.zeros((3, 2, 2)))


if __name__ == "__main__":
    unittest.main()
