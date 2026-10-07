"""Checks the paired full-cube thickness comparison and shared rendering scale."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")

MODULE_DIR = Path(__file__).resolve().parent.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from thickness_maps import (
    compute_paired_thickness_errors,
    render_signed_error_map,
    shared_color_limit,
    summarize_signed_error,
)
from batch_cohort_evaluator import CohortEvaluatorPipeline, ScanEvaluationResult


class TestThicknessMaps(unittest.TestCase):
    def test_signed_errors_use_one_full_cube_mask(self):
        good = {
            "ILM": np.zeros((2, 3), dtype=np.float32),
            "NFL": np.array([[10, 10, np.nan], [10, 0, 10]], dtype=np.float32),
        }
        raw = {
            "ILM": np.zeros((2, 3), dtype=np.float32),
            "NFL": np.array([[11, 8, 8], [10, 3, np.nan]], dtype=np.float32),
        }
        pred = SimpleNamespace(
            ilm_curve=np.zeros((2, 3), dtype=np.float32),
            nfl_curve=np.array([[9, 12, 10], [10, 10, 10]], dtype=np.float32),
        )
        unet, commercial, valid, reference_count = compute_paired_thickness_errors(
            good, raw, pred, 2.0
        )
        self.assertEqual(reference_count, 4)
        self.assertEqual(int(valid.sum()), 3)
        np.testing.assert_allclose(unet[valid], [-2, 4, 0])
        np.testing.assert_allclose(commercial[valid], [2, -4, 0])
        self.assertTrue(np.array_equal(np.isnan(unet), np.isnan(commercial)))
        self.assertAlmostEqual(summarize_signed_error(unet)["mae_um"], 2.0)

    def test_shared_scale_and_rendered_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            array_path = root / "pair.npz"
            np.savez_compressed(
                array_path,
                unet_error_um=np.array([[0, -12], [np.nan, 4]]),
                commercial_error_um=np.array([[0, 24], [np.nan, -8]]),
            )
            limit = shared_color_limit([array_path])
            self.assertEqual(limit, 25.0)
            image_path = root / "map.png"
            render_signed_error_map(np.array([[0, -12], [np.nan, 4]]), image_path, "Pair", limit)
            with Image.open(image_path) as image:
                self.assertGreater(image.width, 100)
                self.assertGreater(image.height, 100)

    def test_evaluator_saves_only_independently_edited_held_out_pair(self):
        good = {"ILM": np.zeros((2, 2)), "NFL": np.full((2, 2), 10.0)}
        raw = {"ILM": np.zeros((2, 2)), "NFL": np.full((2, 2), 12.0)}
        prediction = SimpleNamespace(ilm_curve=np.zeros((2, 2)), nfl_curve=np.full((2, 2), 11.0))
        result = ScanEvaluationResult(
            subject="BEH0001", eye="OS", cohort="Validation", is_validation=True,
            unet_dice=0.9, unet_mabe=1.0, unet_p95=2.0, unet_cup_iou=0.9,
            audit_edited_columns=50,
        )
        volume = SimpleNamespace(
            subject="BEH0001", eye="OS", cohort_tag="Validation", is_validation=True,
            curves_good=good, curves_bad=raw,
        )
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = object.__new__(CohortEvaluatorPipeline)
            pipeline.assets_dir = temporary
            pipeline.biplanar_fusion = True
            pipeline.predictor = SimpleNamespace(predict=lambda *_args, **_kwargs: prediction)
            pipeline.metrics_calc = SimpleNamespace(evaluate_scan=lambda *_args: result)
            quality = SimpleNamespace(status="pass", flags=[], metrics={})
            with patch("batch_cohort_evaluator.assess_prediction", return_value=quality):
                _, gallery, deep_dive, entry = pipeline._evaluate_single_volume(volume, ["BEH0001"])
            self.assertIsNone(gallery)
            self.assertIsNone(deep_dive)
            self.assertEqual(entry["valid_columns"], 4)
            self.assertTrue((Path(temporary) / entry["array_filename"]).exists())

            result.is_mirror = True
            with patch("batch_cohort_evaluator.assess_prediction", return_value=quality):
                _, _, _, skipped = pipeline._evaluate_single_volume(volume, ["BEH0001"])
            self.assertIsNone(skipped)


if __name__ == "__main__":
    unittest.main()
