"""Subject-disjoint cohort manifest parsing for internal and external evaluation."""

import csv
import json
from pathlib import Path
from typing import Dict, List


EVALUATION_SPLITS = {"validation", "held_out", "held-out", "test", "external", "external_test"}


def load_split_manifest(path: str) -> Dict[str, str]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

    if manifest_path.suffix.lower() == ".csv":
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        records = [(row.get("subject", "").strip(), row.get("split", "").strip()) for row in rows]
    else:
        with manifest_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if isinstance(payload, dict) and "subjects" in payload:
            payload = payload["subjects"]
        if isinstance(payload, dict):
            records = [(str(subject).strip(), str(split).strip()) for subject, split in payload.items()]
        elif isinstance(payload, list):
            records = [(str(row.get("subject", "")).strip(), str(row.get("split", "")).strip()) for row in payload]
        else:
            raise ValueError("Manifest must be a subject-to-split mapping or a list of records")

    result: Dict[str, str] = {}
    for subject, split in records:
        if not subject or not split:
            raise ValueError("Every manifest row requires non-empty subject and split values")
        if subject in result and result[subject] != split:
            raise ValueError(f"Subject {subject} is assigned to multiple splits")
        result[subject] = split
    if not result:
        raise ValueError("Split manifest is empty")
    return result


def is_evaluation_split(split: str) -> bool:
    return split.strip().lower() in EVALUATION_SPLITS


def evaluation_subjects(path: str) -> List[str]:
    """Return the sorted subjects assigned to an evaluation-only split."""
    split_map = load_split_manifest(path)
    return sorted(subject for subject, split in split_map.items() if is_evaluation_split(split))
