"""
test_dataset_and_export.py
===========================
Automated unit tests verifying dataset discovery, curve matching,
and 3D Slicer export adapter functionality.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

TESTS_DIR = Path(__file__).resolve().parent
MODULE_DIR = TESTS_DIR.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from dataset import find_matching_curve_xml, find_subject_dicom_and_curves
from export_to_slicer import run_volumetric_inference, export_prediction
from batch_cohort_evaluator import (
    VolumetricRNFLPredictor,
    OCTVolume,
    InferenceConfig,
    VolumePrediction,
    DiscGeometry,
)
from model import VolumetricRNFLNet


class TestDatasetDiscovery(unittest.TestCase):
    """Tests curve matching and multi-arm volume discovery."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = self.temp_dir.name

        # Create realistic directory layout
        self.dicom_dir = os.path.join(self.root, "dicom", "BEH0099")
        self.tsv_good_dir = os.path.join(self.root, "tsv", "good")
        self.tsv_bad_dir = os.path.join(self.root, "tsv", "bad")
        os.makedirs(self.dicom_dir, exist_ok=True)
        os.makedirs(os.path.join(self.tsv_good_dir, "BEH0099", "curve"), exist_ok=True)
        os.makedirs(os.path.join(self.tsv_bad_dir, "BEH0099", "curve"), exist_ok=True)

        # Create dummy DICOM file with realistic timestamp in filename
        self.dcm_fname = "BEH0099_Disc Cube_OD_2026-09-20_10-30-00_OPT.dcm"
        self.dcm_path = os.path.join(self.dicom_dir, self.dcm_fname)
        with open(self.dcm_path, "wb") as f:
            f.write(b"DUMMY_DICOM_CONTENT")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_find_matching_curve_xml_single_candidate(self):
        # When only one candidate curve exists, it should match directly without master XML
        curve_path = os.path.join(self.tsv_good_dir, "BEH0099", "curve", "OD_Disc Cube_single.xml")
        with open(curve_path, "w") as f:
            f.write("<ScanCurve></ScanCurve>")

        matched = find_matching_curve_xml(
            tsv_dir=self.tsv_good_dir,
            subj="BEH0099",
            eye="OD",
            dcm_fname=self.dcm_fname,
            protocol="Disc Cube",
        )
        self.assertEqual(matched, curve_path)

    def test_find_matching_curve_xml_disambiguated_by_master_xml(self):
        # Two curves exist: master XML determines the correct one based on scan time
        curve1 = os.path.join(self.tsv_good_dir, "BEH0099", "curve", "OD_Disc Cube_morning.xml")
        curve2 = os.path.join(self.tsv_good_dir, "BEH0099", "curve", "OD_Disc Cube_afternoon.xml")
        with open(curve1, "w") as f:
            f.write("<ScanCurve></ScanCurve>")
        with open(curve2, "w") as f:
            f.write("<ScanCurve></ScanCurve>")

        master_xml = os.path.join(self.tsv_good_dir, "BEH0099", "MasterInfo.xml")
        with open(master_xml, "w") as f:
            f.write("""<?xml version="1.0"?>
<Patient>
    <Scan>
        <ScanType>Disc Cube</ScanType>
        <ScanTime>2026-09-20 10:30:00</ScanTime>
        <Eye>OD</Eye>
        <Curve>
            <File>./curve/OD_Disc Cube_morning.xml</File>
        </Curve>
    </Scan>
    <Scan>
        <ScanType>Disc Cube</ScanType>
        <ScanTime>2026-09-20 16:00:00</ScanTime>
        <Eye>OD</Eye>
        <Curve>
            <File>./curve/OD_Disc Cube_afternoon.xml</File>
        </Curve>
    </Scan>
</Patient>""")

        matched = find_matching_curve_xml(
            tsv_dir=self.tsv_good_dir,
            subj="BEH0099",
            eye="OD",
            dcm_fname=self.dcm_fname,
            protocol="Disc Cube",
        )
        self.assertEqual(matched, curve1)

    def test_find_matching_curve_xml_missing_directory(self):
        matched = find_matching_curve_xml(
            tsv_dir=self.tsv_good_dir,
            subj="NONEXISTENT_SUBJECT",
            eye="OD",
            dcm_fname=self.dcm_fname,
        )
        self.assertIsNone(matched)

    def test_find_subject_dicom_and_curves(self):
        # Place matching good curve and matching bad curve
        good_curve = os.path.join(self.tsv_good_dir, "BEH0099", "curve", "OD_Disc Cube_good.xml")
        bad_curve = os.path.join(self.tsv_bad_dir, "BEH0099", "curve", "OD_Disc Cube_bad.xml")
        with open(good_curve, "w") as f:
            f.write("<ScanCurve></ScanCurve>")
        with open(bad_curve, "w") as f:
            f.write("<ScanCurve></ScanCurve>")

        dcm, good, bad = find_subject_dicom_and_curves(
            dataset_root=self.root,
            subject="BEH0099",
            eye="OD",
            protocol="Disc Cube",
        )
        self.assertEqual(dcm, self.dcm_path)
        self.assertEqual(good, good_curve)
        self.assertEqual(bad, bad_curve)

    def test_find_subject_dicom_missing_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            find_subject_dicom_and_curves(
                dataset_root=self.root,
                subject="BEH_NO_DICOM",
                eye="OD",
            )


class TestExportToSlicer(unittest.TestCase):
    """Tests the refactored Slicer exporter and backward-compatible inference adapter."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_volumetric_predictor_on_unannotated_oct_volume(self):
        # Verify that an OCTVolume with good_xml_path=None does not throw FileNotFoundError
        # and defaults to the geometric center for disc cut
        vol = OCTVolume(
            subject="BEH_UNANNOTATED",
            eye="OD",
            dcm_path="/dummy/path.dcm",
            good_xml_path=None,
            bad_xml_path=None,
            n_bscans=10,
            rows=20,
            cols=20,
        )
        self.assertIsNone(vol.curves_good)
        self.assertEqual(vol.disc_geometry.zc, 5)
        self.assertEqual(vol.disc_geometry.xc, 10)

    def test_export_prediction_integration(self):
        # Create a lightweight dummy checkpoint with standard channels
        checkpoint_path = os.path.join(self.output_dir, "dummy_model.pt")
        dummy_model = VolumetricRNFLNet(
            in_channels=5,
            base_channels=16,
            channels=(16, 32, 64, 128, 256),
            width=320,
        )
        torch.save({
            'args': {'context_slices': 5, 'base_channels': 16},
            'model_state_dict': dummy_model.state_dict(),
        }, checkpoint_path)

        # Create a synthetic pseudo-DICOM file with pixel tag and full volume size
        dcm_path = os.path.join(self.output_dir, "TEST_Disc Cube_OD.dcm")
        with open(dcm_path, "wb") as f:
            f.write(b"PREAMBLE" * 16)
            f.write(b"\xe0\x7f\x10\x00")  # PixelData tag
            f.write(b"OB\x00\x00")
            # Create sparse file with exact length required for 320x768x320 uint16 volume
            f.seek(128 + 8 + 320 * 768 * 320 * 2 - 1)
            f.write(b"\x00")

        # Test export_prediction with launch_slicer=True
        from unittest.mock import patch
        mock_pred = VolumePrediction(
            mask=np.zeros((320, 768, 320), dtype=np.uint8),
            ilm_curve=np.zeros((320, 320), dtype=np.float32),
            nfl_curve=np.zeros((320, 320), dtype=np.float32),
            cup_probs=np.zeros((320, 320), dtype=np.float32),
        )
        with patch.object(VolumetricRNFLPredictor, "predict", return_value=mock_pred):
            out_npz = export_prediction(
                checkpoint_path=checkpoint_path,
                dcm_path=dcm_path,
                output_dir=self.output_dir,
                device="cpu",
                launch_slicer=True,
                biplanar_fusion=False,
            )

        self.assertTrue(os.path.exists(out_npz))
        loaded = np.load(out_npz)
        self.assertIn("rnfl_mask", loaded)
        self.assertEqual(loaded["rnfl_mask"].shape, (320, 768, 320))

        slicer_script = os.path.join(self.output_dir, "view_prediction_in_slicer.py")
        self.assertTrue(os.path.exists(slicer_script))
        with open(slicer_script, "r") as f:
            content = f.read()
            self.assertIn("ML_RNFL_Prediction", content)
            self.assertIn("CreateClosedSurfaceRepresentation", content)


if __name__ == "__main__":
    unittest.main()
