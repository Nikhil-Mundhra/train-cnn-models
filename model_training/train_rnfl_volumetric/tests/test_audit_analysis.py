"""Unit tests for edit-focused audit-correction metrics."""

import sys
import unittest
from pathlib import Path

import numpy as np


TESTS_DIR = Path(__file__).resolve().parent
MODULE_DIR = TESTS_DIR.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from audit_analysis import compute_audit_correction_metrics


class TestAuditCorrectionMetrics(unittest.TestCase):
    def test_separates_edited_and_unchanged_columns(self):
        human_reference = np.array([[10.0, 10.0, 10.0, np.nan, np.nan]])
        raw = np.array([[10.0, 12.0, 14.0, np.nan, np.nan]])
        predicted = np.array([[10.5, 10.5, 13.0, np.nan, np.nan]])
        cup_probability = np.array([[0.0, 0.0, 0.0, 1.0, 1.0]])

        metrics = compute_audit_correction_metrics(
            human_reference,
            raw,
            predicted,
            cup_probability,
            np.array([True]),
            axial_res_um=3.0,
        )

        self.assertEqual(metrics["audit_edited_columns"], 2)
        self.assertEqual(metrics["audit_unchanged_columns"], 1)
        self.assertAlmostEqual(metrics["raw_edit_mabe_um"], 9.0)
        self.assertAlmostEqual(metrics["unet_edit_mabe_um"], 5.25)
        self.assertAlmostEqual(metrics["delta_edit_mabe_um"], 3.75)
        self.assertAlmostEqual(metrics["audit_correction_gain"], 1.0 - 5.25 / 9.0)
        self.assertAlmostEqual(metrics["audit_ratio_gain"], 1.0 - 5.25 / 9.0)
        self.assertAlmostEqual(metrics["audit_symmetric_gain"], (9.0 - 5.25) / (9.0 + 5.25))
        self.assertAlmostEqual(metrics["audit_edit_recovery_rate"], 1.0)
        self.assertAlmostEqual(metrics["audit_unchanged_preservation_rate"], 1.0)

    def test_outlier_clamping_and_symmetric_gain(self):
        human_reference = np.array([[10.0]])
        raw = np.array([[11.0]])       # raw error 1 px = 3.0 um
        predicted = np.array([[19.0]]) # unet error 9 px = 27.0 um (huge regression)
        cup_probability = np.zeros_like(human_reference)

        metrics = compute_audit_correction_metrics(
            human_reference,
            raw,
            predicted,
            cup_probability,
            np.array([True]),
            axial_res_um=3.0,
        )

        self.assertAlmostEqual(metrics["raw_edit_mabe_um"], 3.0)
        self.assertAlmostEqual(metrics["unet_edit_mabe_um"], 27.0)
        self.assertAlmostEqual(metrics["delta_edit_mabe_um"], -24.0)
        # Unclamped ratio is 1 - 27/3 = -8.0 (-800%)
        self.assertAlmostEqual(metrics["audit_ratio_gain"], -8.0)
        # Clamped gain must be strictly bounded at -1.0 (-100%)
        self.assertEqual(metrics["audit_correction_gain"], -1.0)
        # Symmetric gain is (3 - 27) / (3 + 27) = -24 / 30 = -0.8 (-80%)
        self.assertAlmostEqual(metrics["audit_symmetric_gain"], -0.8)

    def test_cup_edge_localization_uses_fast_axis_resolution(self):
        human_reference = np.array([[10.0, np.nan, np.nan, 10.0, 10.0]])
        predicted = np.array([[10.0, np.nan, np.nan, 10.0, 10.0]])
        cup_probability = np.array([[0.0, 0.0, 1.0, 1.0, 0.0]])

        metrics = compute_audit_correction_metrics(
            human_reference,
            None,
            predicted,
            cup_probability,
            np.array([True]),
            axial_res_um=3.0,
            fast_axis_res_um=20.0,
        )

        self.assertEqual(metrics["unet_cup_edge_valid_slices"], 1)
        self.assertAlmostEqual(metrics["unet_cup_left_edge_mae_um"], 20.0)
        self.assertAlmostEqual(metrics["unet_cup_right_edge_mae_um"], 20.0)
        self.assertAlmostEqual(metrics["unet_cup_width_mae_um"], 0.0)
        self.assertAlmostEqual(metrics["unet_cup_presence_recall"], 1.0)

    def test_material_edit_threshold_is_inclusive(self):
        human_reference = np.array([[10.0, 10.0]])
        raw = np.array([[11.0, 10.9]])
        predicted = np.array([[10.0, 10.0]])
        cup_probability = np.zeros_like(human_reference)

        metrics = compute_audit_correction_metrics(
            human_reference,
            raw,
            predicted,
            cup_probability,
            np.array([True]),
            axial_res_um=3.0,
        )

        self.assertEqual(metrics["audit_edited_columns"], 1)
        self.assertEqual(metrics["audit_unchanged_columns"], 1)


if __name__ == "__main__":
    unittest.main()
