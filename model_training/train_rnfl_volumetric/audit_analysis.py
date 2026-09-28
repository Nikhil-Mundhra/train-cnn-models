"""Edit-focused analysis for raw commercial versus clinician-audited OCT curves."""

from typing import Any, Dict, Optional, Tuple

import numpy as np


AUDIT_EDIT_THRESHOLD_PX = 1.0
PRESERVATION_TOLERANCE_PX = 1.0
FAST_AXIS_RES_UM = 18.75


def _cup_edges(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    columns = np.flatnonzero(mask)
    if columns.size == 0:
        return None
    return int(columns[0]), int(columns[-1])


def compute_audit_correction_metrics(
    clinician_nfl: np.ndarray,
    raw_nfl: Optional[np.ndarray],
    predicted_nfl: np.ndarray,
    predicted_cup_probability: np.ndarray,
    peripapillary_slices: np.ndarray,
    *,
    axial_res_um: float,
    edit_threshold_px: float = AUDIT_EDIT_THRESHOLD_PX,
    preservation_tolerance_px: float = PRESERVATION_TOLERANCE_PX,
    fast_axis_res_um: float = FAST_AXIS_RES_UM,
) -> Dict[str, Any]:
    """Measure recovery of clinician edits separately from unchanged-region fidelity.

    A column is materially edited when the raw and audited NFL surfaces differ by at
    least ``edit_threshold_px``. Cup metrics use the horizontal extent of the NFL-
    absence region and are conditional on both masks containing a detectable region.
    """
    peri = np.asarray(peripapillary_slices, dtype=bool)
    if clinician_nfl.shape != predicted_nfl.shape or clinician_nfl.shape != predicted_cup_probability.shape:
        raise ValueError("Clinician, prediction, and cup arrays must have identical (B-scan, A-scan) shapes")
    if peri.shape != (clinician_nfl.shape[0],):
        raise ValueError("peripapillary_slices must contain one boolean per B-scan")

    result: Dict[str, Any] = {
        "audit_edit_threshold_px": float(edit_threshold_px),
        "audit_edited_columns": 0,
        "audit_unchanged_columns": 0,
        "audit_edit_fraction": None,
        "raw_edit_mabe_um": None,
        "unet_edit_mabe_um": None,
        "audit_correction_gain": None,
        "audit_edit_recovery_rate": None,
        "unet_unchanged_mabe_um": None,
        "audit_unchanged_preservation_rate": None,
        "unet_cup_presence_recall": None,
        "unet_cup_edge_valid_slices": 0,
        "unet_cup_left_edge_mae_um": None,
        "unet_cup_right_edge_mae_um": None,
        "unet_cup_width_mae_um": None,
        "raw_cup_edge_valid_slices": 0,
        "raw_cup_left_edge_mae_um": None,
        "raw_cup_right_edge_mae_um": None,
        "raw_cup_width_mae_um": None,
    }

    if raw_nfl is not None:
        if raw_nfl.shape != clinician_nfl.shape:
            raise ValueError("Raw and clinician NFL arrays must have identical shapes")
        valid = peri[:, None] & np.isfinite(clinician_nfl) & np.isfinite(raw_nfl) & np.isfinite(predicted_nfl)
        raw_error_px = np.abs(raw_nfl - clinician_nfl)
        unet_error_px = np.abs(predicted_nfl - clinician_nfl)
        edited = valid & (raw_error_px >= edit_threshold_px)
        unchanged = valid & (raw_error_px < edit_threshold_px)

        n_edited = int(np.sum(edited))
        n_unchanged = int(np.sum(unchanged))
        result["audit_edited_columns"] = n_edited
        result["audit_unchanged_columns"] = n_unchanged
        if n_edited + n_unchanged:
            result["audit_edit_fraction"] = n_edited / (n_edited + n_unchanged)

        if n_edited:
            raw_edit_mabe = float(np.mean(raw_error_px[edited]) * axial_res_um)
            unet_edit_mabe = float(np.mean(unet_error_px[edited]) * axial_res_um)
            result["raw_edit_mabe_um"] = raw_edit_mabe
            result["unet_edit_mabe_um"] = unet_edit_mabe
            result["audit_correction_gain"] = 1.0 - (unet_edit_mabe / raw_edit_mabe)
            result["audit_edit_recovery_rate"] = float(np.mean(unet_error_px[edited] < raw_error_px[edited]))

        if n_unchanged:
            result["unet_unchanged_mabe_um"] = float(np.mean(unet_error_px[unchanged]) * axial_res_um)
            result["audit_unchanged_preservation_rate"] = float(
                np.mean(unet_error_px[unchanged] <= preservation_tolerance_px)
            )

    gt_presence_count = 0
    pred_presence_hits = 0
    unet_left_errors = []
    unet_right_errors = []
    unet_width_errors = []
    raw_left_errors = []
    raw_right_errors = []
    raw_width_errors = []

    for b_idx in np.flatnonzero(peri):
        gt_edges = _cup_edges(np.isnan(clinician_nfl[b_idx]))
        if gt_edges is None:
            continue
        gt_presence_count += 1
        pred_edges = _cup_edges(predicted_cup_probability[b_idx] > 0.5)
        if pred_edges is not None:
            pred_presence_hits += 1
            unet_left_errors.append(abs(pred_edges[0] - gt_edges[0]) * fast_axis_res_um)
            unet_right_errors.append(abs(pred_edges[1] - gt_edges[1]) * fast_axis_res_um)
            unet_width_errors.append(
                abs((pred_edges[1] - pred_edges[0]) - (gt_edges[1] - gt_edges[0])) * fast_axis_res_um
            )
        if raw_nfl is not None:
            raw_edges = _cup_edges(np.isnan(raw_nfl[b_idx]))
            if raw_edges is not None:
                raw_left_errors.append(abs(raw_edges[0] - gt_edges[0]) * fast_axis_res_um)
                raw_right_errors.append(abs(raw_edges[1] - gt_edges[1]) * fast_axis_res_um)
                raw_width_errors.append(
                    abs((raw_edges[1] - raw_edges[0]) - (gt_edges[1] - gt_edges[0])) * fast_axis_res_um
                )

    if gt_presence_count:
        result["unet_cup_presence_recall"] = pred_presence_hits / gt_presence_count
    if unet_left_errors:
        result["unet_cup_edge_valid_slices"] = len(unet_left_errors)
        result["unet_cup_left_edge_mae_um"] = float(np.mean(unet_left_errors))
        result["unet_cup_right_edge_mae_um"] = float(np.mean(unet_right_errors))
        result["unet_cup_width_mae_um"] = float(np.mean(unet_width_errors))
    if raw_left_errors:
        result["raw_cup_edge_valid_slices"] = len(raw_left_errors)
        result["raw_cup_left_edge_mae_um"] = float(np.mean(raw_left_errors))
        result["raw_cup_right_edge_mae_um"] = float(np.mean(raw_right_errors))
        result["raw_cup_width_mae_um"] = float(np.mean(raw_width_errors))

    return result
