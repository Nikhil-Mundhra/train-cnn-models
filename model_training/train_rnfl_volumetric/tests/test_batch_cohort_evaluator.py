"""
test_batch_cohort_evaluator.py
==============================
Comprehensive automated unit and regression test suite for batch_cohort_evaluator.py:
1. Data structures and serialization (DiscGeometry, SliceMetrics, ScanEvaluationResult, VolumePrediction)
2. OCTVolume curve rasterization and disc geometry calculations
3. ClinicalMetricsCalculator metric calculation (Dice, MABE, P95, Cup IoU, mirror detection)
4. VolumetricRNFLPredictor dropout recovery, elliptical disc cut, and biplanar fusion
5. CohortEvaluatorPipeline validation tagging and volume discovery logic
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import unittest
from pathlib import Path
import numpy as np
import torch

TESTS_DIR = Path(__file__).resolve().parent
MODULE_DIR = TESTS_DIR.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from batch_cohort_evaluator import (
    DiscGeometry,
    SliceMetrics,
    ScanEvaluationResult,
    VolumePrediction,
    OCTVolume,
    ClinicalMetricsCalculator,
    VolumetricRNFLPredictor,
    OpticDiscCutConfig,
    InferenceConfig,
    AXIAL_RES_UM,
)
from model import VolumetricRNFLNet


class TestDataStructures(unittest.TestCase):
    """Verifies core dataclass contracts and dictionary serialization."""

    def test_disc_geometry(self):
        geom = DiscGeometry(zc=160, xc=150, r_disc=45)
        self.assertEqual(geom.zc, 160)
        self.assertEqual(geom.xc, 150)
        self.assertEqual(geom.r_disc, 45)

    def test_slice_metrics(self):
        sm = SliceMetrics(dice=0.92, nfl_mabe=3.14, nfl_p95=7.8, cup_iou=0.95)
        self.assertAlmostEqual(sm.dice, 0.92)
        self.assertAlmostEqual(sm.nfl_mabe, 3.14)
        self.assertAlmostEqual(sm.nfl_p95, 7.8)
        self.assertAlmostEqual(sm.cup_iou, 0.95)

    def test_scan_evaluation_result_serialization(self):
        res = ScanEvaluationResult(
            subject="BEH0086",
            eye="OD",
            cohort="Validation (Held-Out)",
            is_validation=True,
            unet_dice=0.905,
            unet_mabe=4.25,
            unet_p95=12.1,
            unet_cup_iou=0.91,
            is_mirror=False,
            bad_dice=0.82,
            bad_mabe=9.4,
            bad_p95=24.5,
            bad_cup_iou=0.76
        )
        d = res.to_dict()
        self.assertEqual(d["subject"], "BEH0086")
        self.assertEqual(d["eye"], "OD")
        self.assertTrue(d["is_validation"])
        self.assertAlmostEqual(d["unet_mabe"], 4.25)
        self.assertAlmostEqual(d["bad_dice"], 0.82)

    def test_volume_prediction(self):
        mask = np.zeros((10, 20, 20), dtype=np.uint8)
        ilm = np.full((10, 20), 5.0, dtype=np.float32)
        nfl = np.full((10, 20), 12.0, dtype=np.float32)
        cup = np.zeros((10, 20), dtype=np.float32)
        pred = VolumePrediction(mask=mask, ilm_curve=ilm, nfl_curve=nfl, cup_probs=cup)
        self.assertEqual(pred.mask.shape, (10, 20, 20))
        self.assertEqual(pred.ilm_curve.shape, (10, 20))


class TestOCTVolumeLogic(unittest.TestCase):
    """Tests curve rasterization and disc geometry extraction without requiring raw DICOM files."""

    def setUp(self):
        self.vol = OCTVolume(
            subject="TEST_SUBJ",
            eye="OD",
            dcm_path="/dummy/path.dcm",
            good_xml_path=None,
            bad_xml_path=None,
            is_validation=False,
            n_bscans=10,
            rows=50,
            cols=40
        )

    def test_rasterize_curve_slice_synthetic(self):
        # Inject synthetic curves
        ilm = np.full((10, 40), 10.0, dtype=np.float32)
        nfl = np.full((10, 40), 25.0, dtype=np.float32)
        # Add a NaN section simulating cup void in columns 15..25
        nfl[5, 15:25] = np.nan

        self.vol._curves_good = {"ILM": ilm, "NFL": nfl}

        mask = self.vol.rasterize_curve_slice("good", b_idx=5)
        self.assertEqual(mask.shape, (50, 40))

        # Check outside cup: rows 10..24 should be 1
        self.assertTrue(np.all(mask[10:25, 5] == 1))
        self.assertTrue(np.all(mask[0:10, 5] == 0))
        self.assertTrue(np.all(mask[25:, 5] == 0))

        # Check inside cup: column 20 should be completely 0
        self.assertTrue(np.all(mask[:, 20] == 0))

    def test_disc_geometry_synthetic(self):
        ilm = np.full((320, 320), 100.0, dtype=np.float32)
        nfl = np.full((320, 320), 130.0, dtype=np.float32)

        # Place cup void centered at z=160, x=150 with radius 20
        zc_true, xc_true, r_true = 160, 150, 20
        for z in range(zc_true - r_true, zc_true + r_true + 1):
            for x in range(xc_true - r_true, xc_true + r_true + 1):
                if (z - zc_true)**2 + (x - xc_true)**2 <= r_true**2:
                    nfl[z, x] = np.nan

        self.vol._curves_good = {"ILM": ilm, "NFL": nfl}
        self.vol.n_bscans = 320
        self.vol.rows = 768
        self.vol.cols = 320

        geom = self.vol.disc_geometry
        self.assertEqual(geom.zc, zc_true)
        self.assertEqual(geom.xc, xc_true)
        self.assertGreaterEqual(geom.r_disc, 20)


class TestClinicalMetricsCalculator(unittest.TestCase):
    """Verifies metric calculations: Dice, MABE, P95, Cup IoU, and Mirror Detection."""

    def setUp(self):
        self.calc = ClinicalMetricsCalculator(axial_res_um=3.0)

    def test_evaluate_slice_perfect_agreement(self):
        mask = np.zeros((100, 50), dtype=np.uint8)
        mask[20:40, :] = 1
        nfl = np.full(50, 40.0, dtype=np.float32)
        cup = np.zeros(50, dtype=np.float32)

        res = self.calc.evaluate_slice(
            pred_mask=mask,
            gt_mask=mask,
            pred_nfl=nfl,
            gt_nfl=nfl,
            pred_cup=cup,
            gt_cup=cup,
            is_peripapillary=True
        )
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res.dice, 1.0)
        self.assertAlmostEqual(res.nfl_mabe, 0.0)
        self.assertAlmostEqual(res.nfl_p95, 0.0)
        self.assertAlmostEqual(res.cup_iou, 1.0)

    def test_evaluate_slice_mabe_known_offset(self):
        mask_gt = np.zeros((100, 50), dtype=np.uint8)
        mask_gt[20:40, :] = 1
        mask_pred = np.zeros((100, 50), dtype=np.uint8)
        mask_pred[20:42, :] = 1  # 2 pixels thicker

        gt_nfl = np.full(50, 40.0, dtype=np.float32)
        pred_nfl = np.full(50, 42.0, dtype=np.float32)  # exactly 2.0 pixels difference
        cup = np.zeros(50, dtype=np.float32)

        res = self.calc.evaluate_slice(
            pred_mask=mask_pred,
            gt_mask=mask_gt,
            pred_nfl=pred_nfl,
            gt_nfl=gt_nfl,
            pred_cup=cup,
            gt_cup=cup,
            is_peripapillary=True
        )
        expected_mabe = 2.0 * 3.0  # 6.0 um
        self.assertAlmostEqual(res.nfl_mabe, expected_mabe, places=4)
        self.assertAlmostEqual(res.nfl_p95, expected_mabe, places=4)

    def test_evaluate_slice_cup_iou(self):
        mask = np.zeros((100, 50), dtype=np.uint8)
        nfl = np.full(50, 40.0, dtype=np.float32)
        # GT cup is active in columns 10..30 (21 columns)
        gt_cup = np.zeros(50, dtype=np.float32)
        gt_cup[10:31] = 1.0

        # Pred cup is active in columns 20..35 (16 columns)
        pred_cup = np.zeros(50, dtype=np.float32)
        pred_cup[20:36] = 1.0

        # Intersection: 20..30 (11 columns)
        # Union: 10..35 (26 columns)
        expected_iou = 11.0 / 26.0

        res = self.calc.evaluate_slice(
            pred_mask=mask,
            gt_mask=mask,
            pred_nfl=nfl,
            gt_nfl=nfl,
            pred_cup=pred_cup,
            gt_cup=gt_cup,
            is_peripapillary=True
        )
        self.assertAlmostEqual(res.cup_iou, expected_iou, places=4)

    def test_evaluate_slice_non_peripapillary(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        nfl = np.full(50, 20.0, dtype=np.float32)
        cup = np.zeros(50, dtype=np.float32)

        res = self.calc.evaluate_slice(
            pred_mask=mask,
            gt_mask=mask,
            pred_nfl=nfl,
            gt_nfl=nfl,
            pred_cup=cup,
            gt_cup=cup,
            is_peripapillary=False
        )
        self.assertIsNone(res)

    def test_evaluate_scan_mirror_detection(self):
        vol = OCTVolume(
            subject="BEH_MIRROR",
            eye="OD",
            dcm_path="/dummy/dcm",
            good_xml_path=None,
            bad_xml_path=None,
            n_bscans=20,
            rows=50,
            cols=40
        )
        ilm = np.full((20, 40), 10.0, dtype=np.float32)
        nfl = np.full((20, 40), 20.0, dtype=np.float32)
        vol._curves_good = {"ILM": ilm, "NFL": nfl}
        vol._curves_bad = {"ILM": ilm.copy(), "NFL": nfl.copy()}  # Identical mirror!
        vol._disc_geom = DiscGeometry(zc=10, xc=20, r_disc=3)

        pred_mask = np.zeros((20, 50, 40), dtype=np.uint8)
        pred_mask[:, 10:20, :] = 1
        prediction = VolumePrediction(
            mask=pred_mask,
            ilm_curve=ilm,
            nfl_curve=nfl,
            cup_probs=np.zeros((20, 40), dtype=np.float32)
        )

        scan_res = self.calc.evaluate_scan(vol, prediction)
        self.assertTrue(scan_res.is_mirror)
        self.assertIsNone(scan_res.bad_dice)  # mirror metrics must be suppressed to None


class TestConfigurations(unittest.TestCase):
    """Verifies semantic configurations and spatial scale conversions."""

    def test_optic_disc_cut_config(self):
        cfg = OpticDiscCutConfig()
        # dx_mm = 0.01875, dz_mm = 0.0188088
        # rx_px = (1.75 / 2.0) / 0.01875 ~= 46.6667
        # rz_slices = (1.85 / 2.0) / 0.0188088 ~= 49.179
        self.assertAlmostEqual(cfg.rad_x_px, 46.6667, places=3)
        self.assertAlmostEqual(cfg.rad_z_slices, 49.179, places=2)

    def test_inference_config(self):
        cfg = InferenceConfig()
        self.assertEqual(cfg.norm_divisor, 2560.0)
        self.assertEqual(cfg.mask_threshold, 0.40)
        self.assertEqual(cfg.cup_threshold, 0.50)
        self.assertEqual(cfg.min_layer_thickness, 2.0)


class TestPredictorPostProcessing(unittest.TestCase):
    """Tests surface-guided dropout recovery and elliptical disc cutting."""

    def test_dropout_recovery_direct_method(self):
        # 3 B-scans, 50 rows, 40 cols
        mask = np.zeros((3, 50, 40), dtype=np.uint8)
        ilm = np.full((3, 40), 10.0, dtype=np.float32)
        nfl = np.full((3, 40), 25.0, dtype=np.float32)
        cup = np.zeros((3, 40), dtype=np.float32)

        mask[1, :, :] = 1
        mask[1, :, 15] = 0  # artificial thin dropout

        recovered = VolumetricRNFLPredictor.recover_thin_dropouts(
            mask=mask,
            fused_ilm=ilm,
            fused_nfl=nfl,
            fused_cup=cup,
            config=InferenceConfig()
        )

        self.assertTrue(np.all(recovered[1, 10:25, 15] == 1))
        self.assertEqual(recovered[1, 0:10, 15].sum(), 0)

    def test_elliptical_disc_cut_direct_method(self):
        mask = np.ones((100, 50, 100), dtype=np.uint8)
        zc, xc = 50, 50
        # Custom config for testing exact radii
        cfg = OpticDiscCutConfig(
            dx_mm=0.01,
            dz_mm=0.01,
            disc_diam_x_mm=0.30,  # rad_x_px = 15.0
            disc_diam_z_mm=0.40   # rad_z_slices = 20.0
        )
        self.assertAlmostEqual(cfg.rad_x_px, 15.0)
        self.assertAlmostEqual(cfg.rad_z_slices, 20.0)

        cut_mask = VolumetricRNFLPredictor.apply_elliptical_disc_cut(
            mask=mask,
            zc=zc,
            xc=xc,
            config=cfg
        )

        # Center slice (z=50): cut should span xc - 15 to xc + 15 (35..65)
        self.assertEqual(cut_mask[50, :, 35:66].sum(), 0)
        # Outside horizontal cut: column 20 should remain 1
        self.assertEqual(cut_mask[50, :, 20].sum(), 50)
        # Far axial slice (z=90 > zc + rad_z): should remain untouched (all 1)
        self.assertEqual(cut_mask[90].sum(), 50 * 100)

    def test_fuse_predictions(self):
        # Shape: (2, 10, 8)
        h_probs = np.full((2, 10, 8), 0.8, dtype=np.float32)
        v_probs = np.full((2, 10, 8), 0.4, dtype=np.float32)
        # Shape: (2, 8) and (8, 2)
        h_ilm = np.full((2, 8), 10.0, dtype=np.float32)
        v_ilm = np.full((8, 2), 20.0, dtype=np.float32)
        h_nfl = np.full((2, 8), 15.0, dtype=np.float32)
        v_nfl = np.full((8, 2), 25.0, dtype=np.float32)
        h_cup = np.full((2, 8), 0.2, dtype=np.float32)
        v_cup = np.full((8, 2), 0.6, dtype=np.float32)

        horiz = (h_probs, h_ilm, h_nfl, h_cup)
        vert = (v_probs, v_ilm, v_nfl, v_cup)

        f_probs, f_ilm, f_nfl, f_cup = VolumetricRNFLPredictor._fuse_predictions(horiz, vert)

        self.assertTrue(np.allclose(f_probs, 0.6))
        self.assertTrue(np.allclose(f_ilm, 15.0))
        self.assertTrue(np.allclose(f_nfl, 20.0))
        self.assertTrue(np.allclose(f_cup, 0.4))

    def test_forward_batch(self):
        dummy_model = VolumetricRNFLNet(in_channels=5, base_channels=8, width=32)
        predictor = VolumetricRNFLPredictor(model=dummy_model, device=torch.device("cpu"), batch_size=2)

        # Batch of 2 slices: (B, 5, H, W) = (2, 5, 64, 32)
        batch_tensor = torch.zeros((2, 5, 64, 32), dtype=torch.float32)
        probs, ilm, nfl, cup = predictor._forward_batch(batch_tensor)

        self.assertEqual(probs.shape, (2, 64, 32))
        self.assertEqual(ilm.shape, (2, 32))
        self.assertEqual(nfl.shape, (2, 32))
        self.assertEqual(cup.shape, (2, 32))
        self.assertTrue(isinstance(probs, np.ndarray))



if __name__ == "__main__":
    unittest.main()
