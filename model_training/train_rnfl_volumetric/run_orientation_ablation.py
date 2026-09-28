#!/usr/bin/env python3
"""Run controlled OS-orientation and fusion ablations on selected OCT volumes."""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from batch_cohort_evaluator import (
    AXIAL_RES_UM,
    ClinicalMetricsCalculator,
    CohortEvaluatorPipeline,
    VolumetricRNFLPredictor,
)
from quality_control import assess_prediction


VARIANTS: Dict[str, Tuple[str, bool]] = {
    "corrected_biplanar": ("corrected", True),
    "legacy_biplanar": ("legacy_vertical_mirror", True),
    "corrected_horizontal_only": ("corrected", False),
    "native_biplanar": ("native", True),
}


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["eye"])].append(row)

    summary = []
    for (variant, eye), items in sorted(grouped.items()):
        summary.append({
            "variant": variant,
            "eye": eye,
            "n": len(items),
            "median_dice": float(np.median([r["dice"] for r in items])),
            "median_mabe_um": float(np.median([r["mabe_um"] for r in items])),
            "median_p95_um": float(np.median([r["p95_um"] for r in items])),
            "median_cup_iou": float(np.median([r["cup_iou"] for r in items])),
            "manual_review_count": sum(r["qc_status"] == "manual_review" for r in items),
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--subjects", default="BEH0086,BEH0314,BEH0335")
    parser.add_argument("--eyes", default="OD,OS")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    subjects = [value.strip() for value in args.subjects.split(",") if value.strip()]
    eyes = {value.strip().upper() for value in args.eyes.split(",") if value.strip()}
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}; choices: {sorted(VARIANTS)}")

    dev_str = args.device if ("mps" in args.device and torch.backends.mps.is_available()) or "cuda" in args.device else "cpu"
    device = torch.device(dev_str)
    predictor = VolumetricRNFLPredictor.from_checkpoint(
        args.checkpoint,
        device=device,
        batch_size=args.batch_size,
        os_orientation_mode="corrected",
    )
    discovery = CohortEvaluatorPipeline.__new__(CohortEvaluatorPipeline)
    discovery.dataset_root = args.dataset_root
    discovery.val_subjects = subjects
    discovery.split_map = {}
    discovery.subject_filter = set(subjects)
    discovery.eye_filter = eyes
    volumes = discovery.discover_volumes()
    if not volumes:
        raise RuntimeError("No matching volumes found")

    metrics = ClinicalMetricsCalculator(axial_res_um=AXIAL_RES_UM)
    rows: List[Dict[str, Any]] = []
    for variant in variants:
        orientation_mode, biplanar = VARIANTS[variant]
        predictor.os_orientation_mode = orientation_mode
        for volume in volumes:
            print(f"[{variant}] {volume.subject} {volume.eye}", flush=True)
            prediction = predictor.predict(volume, biplanar_fusion=biplanar)
            result = metrics.evaluate_scan(volume, prediction)
            quality = assess_prediction(prediction, evaluation=result)
            rows.append({
                "variant": variant,
                "orientation_mode": orientation_mode,
                "biplanar": biplanar,
                "subject": volume.subject,
                "eye": volume.eye,
                "dice": result.unet_dice,
                "mabe_um": result.unet_mabe,
                "p95_um": result.unet_p95,
                "cup_iou": result.unet_cup_iou,
                "qc_status": quality.status,
                "qc_flags": quality.flags,
                "qc_metrics": quality.metrics,
            })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "dataset_root": os.path.abspath(args.dataset_root),
        "subjects": subjects,
        "eyes": sorted(eyes),
        "results": rows,
        "summary": summarize(rows),
    }
    json_path = output_dir / "orientation_ablation.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = output_dir / "orientation_ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["variant", "orientation_mode", "biplanar", "subject", "eye", "dice", "mabe_um", "p95_um", "cup_iou", "qc_status", "qc_flags"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            export = {key: row[key] for key in fields}
            export["qc_flags"] = ";".join(row["qc_flags"])
            writer.writerow(export)

    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
