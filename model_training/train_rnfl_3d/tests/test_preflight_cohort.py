import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from preflight_cohort import extract_acquisition_time, run_preflight


class TestCohortPreflight(unittest.TestCase):
    def test_extract_acquisition_time(self):
        filename = "BEH0185_Disc Cube_OD_2025-03-21_10-29-41_OPT.dcm"
        self.assertEqual(extract_acquisition_time(filename), "2025-03-21 10:29:41")
        self.assertEqual(extract_acquisition_time("invalid_filename.dcm"), "")

    def test_preflight_synthetic_cohort(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "dataset"
            dicom_dir = root / "dicom"
            tsv_good = root / "tsv" / "good"
            tsv_bad = root / "tsv" / "bad"

            # Create mock subjects:
            # SUBJ01: in dicom, good, bad -> audited eligible
            # SUBJ02: in dicom, good -> phase 1 eligible
            # SUBJ03: in validation manifest -> held out
            # SUBJ04: in dicom only -> orphan
            for s in ["SUBJ01", "SUBJ02", "SUBJ03", "SUBJ04"]:
                (dicom_dir / s).mkdir(parents=True, exist_ok=True)
                (dicom_dir / s / f"{s}_Disc Cube_OD_2025-01-01_10-00-00_OPT.dcm").touch()
                (dicom_dir / s / f"{s}_Disc Cube_OS_2025-01-01_10-05-00_OPT.dcm").touch()

            for s in ["SUBJ01", "SUBJ02", "SUBJ03"]:
                curve_dir = tsv_good / s / "curve"
                curve_dir.mkdir(parents=True, exist_ok=True)
                (curve_dir / f"{s}_OD_Disc Cube.xml").touch()
                (curve_dir / f"{s}_OS_Disc Cube.xml").touch()

            curve_bad = tsv_bad / "SUBJ01" / "curve"
            curve_bad.mkdir(parents=True, exist_ok=True)
            (curve_bad / "SUBJ01_OD_Disc Cube.xml").touch()

            # Mock validation manifest
            val_manifest_path = Path(tmp_dir) / "val.json"
            val_manifest_path.write_text(
                json.dumps(
                    {
                        "subjects": [
                            {"subject": "SUBJ03", "split": "held_out", "annotation_tier": "accepted"}
                        ]
                    }
                )
            )

            out_dir = Path(tmp_dir) / "out"
            summary = run_preflight(
                dataset_root=root,
                validation_manifest_path=val_manifest_path,
                output_dir=out_dir,
                protocol="Disc Cube",
                verify_curves=False,  # mock files are empty
            )

            self.assertEqual(summary["directory_counts"]["dicom"], 4)
            self.assertEqual(summary["directory_counts"]["good"], 3)
            self.assertEqual(summary["directory_counts"]["bad"], 1)

            self.assertEqual(summary["held_out_validation"]["subject_count"], 1)
            self.assertEqual(summary["held_out_validation"]["subjects"], ["SUBJ03"])

            self.assertEqual(summary["phase1_train"]["subject_count"], 2)
            self.assertEqual(summary["phase1_train"]["subjects"], ["SUBJ01", "SUBJ02"])

            self.assertEqual(summary["phase2_audited_train"]["subject_count"], 1)
            self.assertEqual(summary["phase2_audited_train"]["subjects"], ["SUBJ01"])

            self.assertEqual(summary["orphans"]["count"], 1)
            self.assertEqual(summary["orphans"]["subjects"], ["SUBJ04"])


if __name__ == "__main__":
    unittest.main()
