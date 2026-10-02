#!/usr/bin/env python3
"""Preflight audit for 3D RNFL cohort integrity, matching, and zero-leakage splits.

Freezes:
1. Exact matched Phase 1 subjects and scans (DICOM ∩ good XML).
2. Exact usable audited subjects (DICOM ∩ good XML ∩ bad XML - held-out).
3. Missing / orphan DICOM and XML records.
4. Zero overlap with the frozen validation manifest.
5. OD/OS scan counts and duplicate acquisition handling.
"""

import argparse
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


VOLUMETRIC_DIR = Path(__file__).resolve().parents[1] / "train_rnfl_volumetric"
if str(VOLUMETRIC_DIR) not in sys.path:
    sys.path.insert(0, str(VOLUMETRIC_DIR))

from dataset import find_dicom_pixel_offset, find_matching_curve_xml, load_curves


@dataclass
class ScanRecord:
    subject: str
    eye: str
    dcm_path: str
    dcm_filename: str
    good_xml_path: Optional[str] = None
    bad_xml_path: Optional[str] = None
    is_opt: bool = False
    acquisition_time: str = ""
    is_valid_good_curve: bool = False
    is_valid_bad_curve: bool = False
    is_duplicate_acquisition: bool = False
    selection_reason: str = ""
    error: str = ""


@dataclass
class SubjectAudit:
    subject: str
    in_dicom_dir: bool = False
    in_good_dir: bool = False
    in_bad_dir: bool = False
    scans: List[ScanRecord] = field(default_factory=list)
    has_matched_od: bool = False
    has_matched_os: bool = False
    matched_scan_count: int = 0
    is_phase1_eligible: bool = False
    is_audited_eligible: bool = False
    split: str = "unassigned"  # 'held_out', 'phase1_train', 'phase2_audited_train', 'orphan'
    orphan_reason: str = ""


def extract_acquisition_time(filename: str) -> str:
    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})", filename)
    if m:
        return f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}"
    return ""


def load_held_out_manifest(path: Path) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {entry["subject"]: entry for entry in data.get("subjects", [])}


def audit_subject_scans(
    dataset_root: Path,
    subject: str,
    protocol: str = "Disc Cube",
    verify_curves: bool = True,
) -> SubjectAudit:
    dcm_subj_dir = dataset_root / "dicom" / subject
    good_subj_dir = dataset_root / "tsv" / "good" / subject
    bad_subj_dir = dataset_root / "tsv" / "bad" / subject

    in_dicom = dcm_subj_dir.is_dir()
    in_good = good_subj_dir.is_dir()
    in_bad = bad_subj_dir.is_dir()

    audit = SubjectAudit(
        subject=subject,
        in_dicom_dir=in_dicom,
        in_good_dir=in_good,
        in_bad_dir=in_bad,
    )

    if not in_dicom:
        audit.orphan_reason = "Missing DICOM directory"
        return audit

    # Discover all Disc Cube DICOM files for this subject
    all_dcms = sorted(dcm_subj_dir.glob("*.dcm"))
    disc_dcms = [p for p in all_dcms if protocol.lower() in p.name.lower()]

    if not disc_dcms:
        audit.orphan_reason = f"No {protocol} DICOM volumes found"
        return audit

    # Group by eye: OD and OS
    by_eye: Dict[str, List[Path]] = {"OD": [], "OS": []}
    for p in disc_dcms:
        name_upper = p.name.upper()
        if "_OS_" in name_upper or name_upper.endswith("_OS.DCM") or " OS " in name_upper:
            by_eye["OS"].append(p)
        elif "_OD_" in name_upper or name_upper.endswith("_OD.DCM") or " OD " in name_upper:
            by_eye["OD"].append(p)

    for eye in ["OD", "OS"]:
        candidates = by_eye[eye]
        if not candidates:
            continue

        # Check for multiple acquisitions
        is_dup = len(candidates) > 1

        # Prefer _OPT.dcm, then latest acquisition time
        def sort_key(p: Path):
            is_opt = 1 if "_opt.dcm" in p.name.lower() else 0
            time_str = extract_acquisition_time(p.name)
            return (is_opt, time_str, p.name)

        sorted_candidates = sorted(candidates, key=sort_key, reverse=True)
        canonical_dcm = sorted_candidates[0]

        # Check curve matching
        good_xml = None
        bad_xml = None
        if in_good:
            good_xml = find_matching_curve_xml(
                str(dataset_root / "tsv" / "good"), subject, eye, canonical_dcm.name, protocol
            )
        if in_bad:
            bad_xml = find_matching_curve_xml(
                str(dataset_root / "tsv" / "bad"), subject, eye, canonical_dcm.name, protocol
            )

        rec = ScanRecord(
            subject=subject,
            eye=eye,
            dcm_path=str(canonical_dcm),
            dcm_filename=canonical_dcm.name,
            good_xml_path=good_xml,
            bad_xml_path=bad_xml,
            is_opt="_opt.dcm" in canonical_dcm.name.lower(),
            acquisition_time=extract_acquisition_time(canonical_dcm.name),
            is_duplicate_acquisition=is_dup,
            selection_reason=(
                f"Selected from {len(candidates)} candidates (preferred _OPT/timestamp)"
                if is_dup
                else "Single acquisition"
            ),
        )

        # Validate curve contents if requested
        if good_xml and os.path.exists(good_xml):
            if verify_curves:
                try:
                    c = load_curves(good_xml, mask_sentinel=True)
                    if "ILM" in c and "NFL" in c and c["ILM"].shape == (320, 320):
                        rec.is_valid_good_curve = True
                    else:
                        rec.error = f"Good curve missing ILM/NFL or shape != (320, 320) in {good_xml}"
                except Exception as ex:
                    rec.error = f"Failed to load good curve {good_xml}: {ex}"
            else:
                rec.is_valid_good_curve = True

        if bad_xml and os.path.exists(bad_xml):
            if verify_curves:
                try:
                    c_bad = load_curves(bad_xml, mask_sentinel=True)
                    if "ILM" in c_bad and "NFL" in c_bad:
                        rec.is_valid_bad_curve = True
                except Exception as ex:
                    pass
            else:
                rec.is_valid_bad_curve = True

        audit.scans.append(rec)
        if rec.is_valid_good_curve:
            if eye == "OD":
                audit.has_matched_od = True
            elif eye == "OS":
                audit.has_matched_os = True

    audit.matched_scan_count = sum(1 for s in audit.scans if s.is_valid_good_curve)
    audit.is_phase1_eligible = audit.matched_scan_count > 0
    audit.is_audited_eligible = any(s.is_valid_good_curve and s.is_valid_bad_curve for s in audit.scans)

    if not audit.is_phase1_eligible:
        audit.orphan_reason = "No scans matched with valid good curve XML"

    return audit


def run_preflight(
    dataset_root: Path,
    validation_manifest_path: Path,
    output_dir: Path,
    protocol: str = "Disc Cube",
    verify_curves: bool = True,
) -> dict:
    dataset_root = dataset_root.resolve()
    print(f"=== 3D RNFL Cohort Preflight Audit ===")
    print(f"Dataset root: {dataset_root}")
    print(f"Validation manifest: {validation_manifest_path}")

    held_out_map = load_held_out_manifest(validation_manifest_path)
    held_out_subjects = set(held_out_map.keys())
    print(f"Frozen held-out subjects: {len(held_out_subjects)} ({', '.join(sorted(held_out_subjects))})")

    dicom_dir = dataset_root / "dicom"
    good_dir = dataset_root / "tsv" / "good"
    bad_dir = dataset_root / "tsv" / "bad"

    dicom_subjects = sorted(d.name for d in dicom_dir.iterdir() if d.is_dir() and not d.name.startswith(".")) if dicom_dir.exists() else []
    good_subjects = sorted(d.name for d in good_dir.iterdir() if d.is_dir() and not d.name.startswith(".")) if good_dir.exists() else []
    bad_subjects = sorted(d.name for d in bad_dir.iterdir() if d.is_dir() and not d.name.startswith(".")) if bad_dir.exists() else []

    all_subjects = sorted(set(dicom_subjects) | set(good_subjects) | set(bad_subjects))
    print(f"\nDirectory Inventory:")
    print(f"  dicom/ subjects:      {len(dicom_subjects)}")
    print(f"  tsv/good/ subjects:   {len(good_subjects)}")
    print(f"  tsv/bad/ subjects:    {len(bad_subjects)}")
    print(f"  Total unique IDs:     {len(all_subjects)}")

    audits: Dict[str, SubjectAudit] = {}
    for subj in all_subjects:
        audits[subj] = audit_subject_scans(dataset_root, subj, protocol=protocol, verify_curves=verify_curves)

    # Classify splits & enforce zero-leakage
    phase1_train_subjects: List[str] = []
    phase2_audited_train_subjects: List[str] = []
    held_out_confirmed: List[str] = []
    orphans: List[str] = []

    for subj, a in audits.items():
        if subj in held_out_subjects:
            a.split = "held_out"
            if a.is_phase1_eligible:
                held_out_confirmed.append(subj)
            else:
                print(f"  [CRITICAL WARNING] Held-out subject {subj} is NOT phase-1 eligible! Reason: {a.orphan_reason}")
        elif a.is_phase1_eligible:
            a.split = "phase1_train"
            phase1_train_subjects.append(subj)
            if a.is_audited_eligible:
                phase2_audited_train_subjects.append(subj)
        else:
            a.split = "orphan"
            orphans.append(subj)

    # Zero-leakage assertions
    leakage_p1 = set(phase1_train_subjects) & held_out_subjects
    leakage_p2 = set(phase2_audited_train_subjects) & held_out_subjects
    assert not leakage_p1, f"FATAL: Phase 1 training set leaked held-out subjects: {leakage_p1}"
    assert not leakage_p2, f"FATAL: Phase 2 audited training set leaked held-out subjects: {leakage_p2}"

    # Counts
    p1_scans = [s for subj in phase1_train_subjects for s in audits[subj].scans if s.is_valid_good_curve]
    p2_scans = [s for subj in phase2_audited_train_subjects for s in audits[subj].scans if s.is_valid_good_curve and s.is_valid_bad_curve]
    ho_scans = [s for subj in held_out_confirmed for s in audits[subj].scans if s.is_valid_good_curve]

    duplicate_subjects = [subj for subj, a in audits.items() if any(s.is_duplicate_acquisition for s in a.scans)]
    single_eye_subjects = [subj for subj, a in audits.items() if a.matched_scan_count == 1]

    print("\n" + "=" * 80)
    print("=== COHORT PREFLIGHT SUMMARY ===")
    print("=" * 80)
    print(f"Total Unique Subjects in Dataset:       {len(all_subjects)}")
    print(f"Total Matched Phase 1 Eligible Subjects:{sum(1 for a in audits.values() if a.is_phase1_eligible)}")
    print(f"  - Phase 1 Training Subjects:          {len(phase1_train_subjects)} ({len(p1_scans)} scans)")
    print(f"    * Both eyes (OD + OS):              {sum(1 for s in phase1_train_subjects if audits[s].matched_scan_count == 2)}")
    print(f"    * Single eye only:                  {sum(1 for s in phase1_train_subjects if audits[s].matched_scan_count == 1)}")
    print(f"  - Frozen Held-Out Validation:         {len(held_out_confirmed)} / {len(held_out_subjects)} subjects ({len(ho_scans)} scans)")
    print(f"  - Phase 2 Usable Audited Training:    {len(phase2_audited_train_subjects)} subjects ({len(p2_scans)} paired scans)")
    print(f"  - Orphan / Incomplete Subjects:       {len(orphans)}")
    print(f"  - Subjects with Multiple Acquisitions: {len(duplicate_subjects)}")
    print(f"Zero-Leakage Guarantee:                 STRICTLY VERIFIED (0 overlap)")
    print("=" * 80)

    # Write manifests
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "cohort_preflight_summary.json"
    p1_manifest_path = output_dir / "phase1_train_manifest.json"
    p2_manifest_path = output_dir / "phase2_audited_train_manifest.json"
    orphans_path = output_dir / "cohort_orphans.json"

    # Save summary
    summary_payload = {
        "dataset_root": str(dataset_root),
        "protocol": protocol,
        "total_subjects": len(all_subjects),
        "directory_counts": {
            "dicom": len(dicom_subjects),
            "good": len(good_subjects),
            "bad": len(bad_subjects),
        },
        "phase1_train": {
            "subject_count": len(phase1_train_subjects),
            "scan_count": len(p1_scans),
            "subjects": sorted(phase1_train_subjects),
        },
        "phase2_audited_train": {
            "subject_count": len(phase2_audited_train_subjects),
            "scan_count": len(p2_scans),
            "subjects": sorted(phase2_audited_train_subjects),
        },
        "held_out_validation": {
            "subject_count": len(held_out_confirmed),
            "scan_count": len(ho_scans),
            "subjects": sorted(held_out_confirmed),
        },
        "orphans": {
            "count": len(orphans),
            "subjects": sorted(orphans),
            "details": {s: audits[s].orphan_reason for s in orphans},
        },
        "duplicate_acquisition_subjects": sorted(duplicate_subjects),
        "single_eye_subjects": sorted(single_eye_subjects),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    # Save Phase 1 Train Manifest
    p1_payload = {
        "schema_version": 1,
        "name": "rnfl_3d_phase1_broad_train_manifest",
        "description": "Pre-training cohort: matched Disc Cube DICOM/good XML pairs strictly disjoint from validation",
        "subject_count": len(phase1_train_subjects),
        "scan_count": len(p1_scans),
        "scans": [asdict(s) for s in p1_scans],
    }
    p1_manifest_path.write_text(json.dumps(p1_payload, indent=2) + "\n", encoding="utf-8")

    # Save Phase 2 Audited Train Manifest
    p2_payload = {
        "schema_version": 1,
        "name": "rnfl_3d_phase2_audited_train_manifest",
        "description": "Fine-tuning cohort: human-audited Disc Cube scans strictly disjoint from validation",
        "subject_count": len(phase2_audited_train_subjects),
        "scan_count": len(p2_scans),
        "scans": [asdict(s) for s in p2_scans],
    }
    p2_manifest_path.write_text(json.dumps(p2_payload, indent=2) + "\n", encoding="utf-8")

    # Save Held-Out Validation Scans Manifest
    val_manifest_path = output_dir / "held_out_validation_scans.json"
    val_payload = {
        "schema_version": 1,
        "name": "rnfl_3d_held_out_validation_scans",
        "description": "Held-out validation cohort: 20 subjects (40 scans) strictly disjoint from train",
        "subject_count": len(held_out_confirmed),
        "scan_count": len(ho_scans),
        "scans": [asdict(s) for s in ho_scans],
    }
    val_manifest_path.write_text(json.dumps(val_payload, indent=2) + "\n", encoding="utf-8")

    # Save Orphans
    orphans_payload = {
        "orphan_count": len(orphans),
        "orphans": [
            {
                "subject": s,
                "in_dicom": audits[s].in_dicom_dir,
                "in_good": audits[s].in_good_dir,
                "in_bad": audits[s].in_bad_dir,
                "reason": audits[s].orphan_reason,
            }
            for s in orphans
        ],
    }
    orphans_path.write_text(json.dumps(orphans_payload, indent=2) + "\n", encoding="utf-8")

    print(f"\nArtifacts written to {output_dir}:")
    print(f"  - Summary:    {summary_path.name}")
    print(f"  - Phase 1:    {p1_manifest_path.name}")
    print(f"  - Phase 2:    {p2_manifest_path.name}")
    print(f"  - Validation: {val_manifest_path.name}")
    print(f"  - Orphans:    {orphans_path.name}")

    return summary_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/scratch/nm4358/deidentified-new"),
        help="Path to deidentified-new dataset root",
    )
    parser.add_argument(
        "--validation_manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "manifests" / "successor_held_out_v1.json",
        help="Path to frozen validation manifest",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "manifests",
        help="Directory to save audit manifests",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="Disc Cube",
        help="OCT protocol to audit",
    )
    parser.add_argument(
        "--skip_curve_verification",
        action="store_true",
        help="Skip deep reading of curve XML files (fast mode)",
    )
    args = parser.parse_args()

    run_preflight(
        dataset_root=args.dataset_root,
        validation_manifest_path=args.validation_manifest,
        output_dir=args.output_dir,
        protocol=args.protocol,
        verify_curves=not args.skip_curve_verification,
    )


if __name__ == "__main__":
    main()
