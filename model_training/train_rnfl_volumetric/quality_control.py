"""Operational quality-control checks and manual-review routing for RNFL inference."""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class QualityControlThresholds:
    """Engineering defaults; these are not clinically validated decision limits."""

    max_surface_order_violation_fraction: float = 0.005
    max_mask_dropout_fraction: float = 0.02
    max_boundary_jump_p99_px: float = 30.0
    max_biplanar_nfl_disagreement_um: float = 15.0
    max_biplanar_probability_disagreement: float = 0.15
    min_cup_fraction: float = 0.002
    max_cup_fraction: float = 0.30
    reference_min_dice: float = 0.85
    reference_max_mabe_um: float = 5.0
    reference_min_cup_iou: float = 0.90


@dataclass
class QualityAssessment:
    status: str
    flags: List[str]
    metrics: Dict[str, float]
    threshold_profile: Dict[str, Any]

    @property
    def requires_manual_review(self) -> bool:
        return self.status != "pass"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _finite_percentile(values: np.ndarray, percentile: float, default: float = 0.0) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile)) if finite.size else default


def assess_prediction(
    prediction: Any,
    evaluation: Optional[Any] = None,
    thresholds: QualityControlThresholds = QualityControlThresholds(),
) -> QualityAssessment:
    """Assess prediction-only integrity and, when available, reference-based performance."""

    mask = np.asarray(prediction.mask)
    ilm = np.asarray(prediction.ilm_curve, dtype=float)
    nfl = np.asarray(prediction.nfl_curve, dtype=float)
    cup = np.asarray(prediction.cup_probs, dtype=float)
    diagnostics = getattr(prediction, "diagnostics", None) or {}

    tissue = np.isfinite(ilm) & np.isfinite(nfl) & (cup < 0.5)
    thickness = nfl - ilm
    order_violation_fraction = float(np.mean(tissue & (thickness <= 0)))

    expected_tissue = tissue & (thickness >= 2.0)
    mask_present = np.any(mask > 0, axis=1)
    expected_count = int(np.sum(expected_tissue))
    dropout_fraction = (
        float(np.sum(expected_tissue & ~mask_present) / expected_count)
        if expected_count
        else 1.0
    )

    valid_pairs = tissue[:, 1:] & tissue[:, :-1]
    boundary_jumps = np.abs(np.diff(nfl, axis=1))[valid_pairs]
    boundary_jump_p99 = _finite_percentile(boundary_jumps, 99.0)
    cup_fraction = float(np.mean(cup >= 0.5))

    metrics: Dict[str, float] = {
        "surface_order_violation_fraction": order_violation_fraction,
        "mask_dropout_fraction": dropout_fraction,
        "boundary_jump_p99_px": boundary_jump_p99,
        "cup_fraction": cup_fraction,
    }
    for key in (
        "biplanar_nfl_disagreement_um",
        "biplanar_ilm_disagreement_um",
        "biplanar_probability_disagreement",
    ):
        if key in diagnostics and diagnostics[key] is not None:
            metrics[key] = float(diagnostics[key])

    flags: List[str] = []
    if order_violation_fraction > thresholds.max_surface_order_violation_fraction:
        flags.append("surface_order_violation")
    if dropout_fraction > thresholds.max_mask_dropout_fraction:
        flags.append("mask_dropout")
    if boundary_jump_p99 > thresholds.max_boundary_jump_p99_px:
        flags.append("boundary_discontinuity")
    if not thresholds.min_cup_fraction <= cup_fraction <= thresholds.max_cup_fraction:
        flags.append("implausible_cup_fraction")
    if metrics.get("biplanar_nfl_disagreement_um", 0.0) > thresholds.max_biplanar_nfl_disagreement_um:
        flags.append("biplanar_boundary_disagreement")
    if metrics.get("biplanar_probability_disagreement", 0.0) > thresholds.max_biplanar_probability_disagreement:
        flags.append("biplanar_mask_disagreement")

    if evaluation is not None:
        metrics.update({
            "reference_dice": float(evaluation.unet_dice),
            "reference_mabe_um": float(evaluation.unet_mabe),
            "reference_cup_iou": float(evaluation.unet_cup_iou),
        })
        if evaluation.unet_dice < thresholds.reference_min_dice:
            flags.append("reference_dice_below_operational_limit")
        if evaluation.unet_mabe > thresholds.reference_max_mabe_um:
            flags.append("reference_mabe_above_operational_limit")
        if evaluation.unet_cup_iou < thresholds.reference_min_cup_iou:
            flags.append("reference_cup_iou_below_operational_limit")

    flags = list(dict.fromkeys(flags))
    return QualityAssessment(
        status="manual_review" if flags else "pass",
        flags=flags,
        metrics=metrics,
        threshold_profile=asdict(thresholds),
    )
