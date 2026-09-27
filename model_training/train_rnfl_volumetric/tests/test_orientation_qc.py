import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODULE_DIR = Path(__file__).resolve().parent.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from evaluation_manifest import is_evaluation_split, load_split_manifest
from orientation import horizontal_requires_flip, vertical_source_and_destination
from quality_control import assess_prediction


class TestOrientationMapping(unittest.TestCase):
    def test_corrected_os_vertical_prediction_returns_to_native_column(self):
        self.assertEqual(vertical_source_and_destination(0, 320, "OS", "corrected"), (319, 319))
        self.assertEqual(vertical_source_and_destination(319, 320, "OS", "corrected"), (0, 0))

    def test_legacy_mode_reproduces_misaligned_write(self):
        self.assertEqual(vertical_source_and_destination(0, 320, "OS", "legacy_vertical_mirror"), (319, 0))

    def test_od_and_native_modes_are_identity_mappings(self):
        self.assertEqual(vertical_source_and_destination(37, 320, "OD", "corrected"), (37, 37))
        self.assertEqual(vertical_source_and_destination(37, 320, "OS", "native"), (37, 37))
        self.assertTrue(horizontal_requires_flip("OS", "corrected"))
        self.assertFalse(horizontal_requires_flip("OS", "native"))


class TestEvaluationManifest(unittest.TestCase):
    def test_json_manifest_and_evaluation_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            path.write_text(json.dumps({"subjects": {"A": "train", "B": "external_test"}}))
            manifest = load_split_manifest(str(path))
        self.assertEqual(manifest, {"A": "train", "B": "external_test"})
        self.assertFalse(is_evaluation_split(manifest["A"]))
        self.assertTrue(is_evaluation_split(manifest["B"]))


class TestQualityControl(unittest.TestCase):
    def _prediction(self):
        ilm = np.full((4, 5), 10.0, dtype=np.float32)
        nfl = np.full((4, 5), 20.0, dtype=np.float32)
        mask = np.zeros((4, 32, 5), dtype=np.uint8)
        mask[:, 10:20, :] = 1
        cup = np.zeros((4, 5), dtype=np.float32)
        cup[:, 2] = 1.0
        return SimpleNamespace(mask=mask, ilm_curve=ilm, nfl_curve=nfl, cup_probs=cup, diagnostics={})

    def test_clean_prediction_passes_prediction_only_checks(self):
        assessment = assess_prediction(self._prediction())
        self.assertEqual(assessment.status, "pass")
        self.assertFalse(assessment.flags)

    def test_reference_failure_routes_to_manual_review(self):
        evaluation = SimpleNamespace(unet_dice=0.60, unet_mabe=100.0, unet_cup_iou=0.70)
        assessment = assess_prediction(self._prediction(), evaluation=evaluation)
        self.assertEqual(assessment.status, "manual_review")
        self.assertIn("reference_mabe_above_operational_limit", assessment.flags)

    def test_biplanar_disagreement_routes_to_manual_review(self):
        prediction = self._prediction()
        prediction.diagnostics = {"biplanar_nfl_disagreement_um": 40.0}
        assessment = assess_prediction(prediction)
        self.assertIn("biplanar_boundary_disagreement", assessment.flags)


if __name__ == "__main__":
    unittest.main()
