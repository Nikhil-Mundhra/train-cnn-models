"""Training provenance checks for generated cohort reports."""

import json
from pathlib import Path
import unittest

from model_training.train_rnfl_volumetric.reporting.training_cohort import summarize_training_cohort


MANIFEST = Path(__file__).resolve().parents[2] / "train_rnfl_3d/manifests/phase1_train_manifest.json"
PHASE2_MANIFEST = MANIFEST.with_name("phase2_audited_train_manifest.json")
VALIDATION_MANIFEST = MANIFEST.with_name("stratified_held_out_v2.json")


class TrainingCohortReportTest(unittest.TestCase):
    def test_biplanar_training_tiers_and_subject_isolation(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        summary = summarize_training_cohort(manifest, [{"subject": "BEH0030"}])

        self.assertEqual(summary["subject_count"], 119)
        self.assertEqual(summary["scan_count"], 237)
        self.assertEqual(summary["audited_subject_count"], 20)
        self.assertEqual(summary["audited_scan_count"], 40)
        self.assertEqual(summary["accepted_subject_count"], 99)
        self.assertEqual(summary["accepted_scan_count"], 197)

        with self.assertRaisesRegex(ValueError, "subject overlap"):
            summarize_training_cohort(manifest, [{"subject": manifest["scans"][0]["subject"]}])

    def test_phase2_audited_manifest_is_disjoint_from_current_validation(self):
        phase2 = json.loads(PHASE2_MANIFEST.read_text(encoding="utf-8"))
        validation = json.loads(VALIDATION_MANIFEST.read_text(encoding="utf-8"))
        summary = summarize_training_cohort(phase2, validation["subjects"])

        self.assertEqual(summary["subject_count"], 20)
        self.assertEqual(summary["scan_count"], 40)
        self.assertEqual(summary["audited_subject_count"], 20)
        self.assertEqual(summary["accepted_subject_count"], 0)
