"""Summarize the exact training manifest used by a cohort report."""

from collections import defaultdict


def summarize_training_cohort(manifest, evaluation_scans):
    scans = manifest.get("scans", [])
    if not scans:
        raise ValueError("Training manifest has no scans")

    subject_scans = defaultdict(list)
    for scan in scans:
        subject_scans[scan["subject"]].append(scan)

    if manifest.get("subject_count") != len(subject_scans):
        raise ValueError("Training manifest subject_count does not match its scans")
    if manifest.get("scan_count") != len(scans):
        raise ValueError("Training manifest scan_count does not match its scans")

    overlap = set(subject_scans) & {scan["subject"] for scan in evaluation_scans}
    if overlap:
        raise ValueError(f"Training/evaluation subject overlap: {sorted(overlap)}")

    audited_subjects = {
        subject for subject, rows in subject_scans.items()
        if any(row.get("bad_xml_path") for row in rows)
    }
    audited_scans = [scan for scan in scans if scan.get("bad_xml_path")]
    return {
        "subject_count": len(subject_scans),
        "scan_count": len(scans),
        "audited_subject_count": len(audited_subjects),
        "audited_scan_count": len(audited_scans),
        "accepted_subject_count": len(subject_scans) - len(audited_subjects),
        "accepted_scan_count": len(scans) - len(audited_scans),
        "manifest_name": manifest.get("name", "training manifest"),
    }
