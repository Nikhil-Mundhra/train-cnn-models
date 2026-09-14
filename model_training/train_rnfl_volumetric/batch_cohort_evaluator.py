"""
batch_cohort_evaluator.py
==========================
Object-Oriented Comprehensive Batch Evaluator spanning all 11 Solix OCT subjects.
Computes quantitative boundary & volumetric metrics against clinician ground truth (good)
and commercial baseline (bad), and exports publication-ready comparative visual panels.
"""

import os
import sys
import glob
import json
import re
import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import scipy.ndimage

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import find_dicom_pixel_offset, load_curves
from model import VolumetricRNFLNet

AXIAL_RES_UM: float = 3.09


# ============================================================================
# 1. Data Structures & Entities
# ============================================================================

@dataclass
class DiscGeometry:
    """Encapsulates optic disc center coordinates and anatomical radius."""
    zc: int
    xc: int
    r_disc: int


@dataclass
class SliceMetrics:
    """Per-B-scan segmentation metrics."""
    dice: float
    nfl_mabe: float
    nfl_p95: float
    cup_iou: float


@dataclass
class ScanEvaluationResult:
    """Aggregated evaluation metrics for a single volumetric acquisition."""
    subject: str
    eye: str
    cohort: str
    is_validation: bool
    unet_dice: float
    unet_mabe: float
    unet_p95: float
    unet_cup_iou: float
    bad_dice: Optional[float] = None
    bad_mabe: Optional[float] = None
    bad_p95: Optional[float] = None
    bad_cup_iou: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VolumePrediction:
    """Container holding model predictions across an entire 3D volume."""
    mask: np.ndarray        # (320, 768, 320) uint8
    ilm_curve: np.ndarray   # (320, 320) float32
    nfl_curve: np.ndarray   # (320, 320) float32
    cup_probs: np.ndarray   # (320, 320) float32


# ============================================================================
# 2. Data Layer: OCTVolume
# ============================================================================

class OCTVolume:
    """
    Encapsulates a single patient OCT acquisition: DICOM raw data, ground truth
    clinician curves, commercial baseline curves, and spatial disc geometry.
    """

    def __init__(
        self,
        subject: str,
        eye: str,
        dcm_path: str,
        good_xml_path: Optional[str],
        bad_xml_path: Optional[str] = None,
        is_validation: bool = False,
        n_bscans: int = 320,
        rows: int = 768,
        cols: int = 320
    ):
        self.subject = subject
        self.eye = eye
        self.dcm_path = dcm_path
        self.good_xml_path = good_xml_path
        self.bad_xml_path = bad_xml_path
        self.is_validation = is_validation
        self.cohort_tag = "Validation (Held-Out)" if is_validation else "Training / Benchmark"

        self.n_bscans = n_bscans
        self.rows = rows
        self.cols = cols

        self._memmap: Optional[np.memmap] = None
        self._curves_good: Optional[Dict[str, np.ndarray]] = None
        self._curves_bad: Optional[Dict[str, np.ndarray]] = None
        self._disc_geom: Optional[DiscGeometry] = None

    @property
    def memmap(self) -> np.memmap:
        if self._memmap is None:
            offset = find_dicom_pixel_offset(self.dcm_path)
            self._memmap = np.memmap(
                self.dcm_path,
                dtype='<u2',
                mode='r',
                offset=offset,
                shape=(self.n_bscans, self.rows, self.cols)
            )
        return self._memmap

    @property
    def curves_good(self) -> Dict[str, np.ndarray]:
        if self._curves_good is None:
            if not self.good_xml_path or not os.path.exists(self.good_xml_path):
                raise FileNotFoundError(f"Good curves XML not found for {self.subject} ({self.eye})")
            self._curves_good = load_curves(self.good_xml_path, mask_sentinel=True)
        return self._curves_good

    @property
    def curves_bad(self) -> Optional[Dict[str, np.ndarray]]:
        if self._curves_bad is None and self.bad_xml_path and os.path.exists(self.bad_xml_path):
            self._curves_bad = load_curves(self.bad_xml_path, mask_sentinel=True)
        return self._curves_bad

    @property
    def disc_geometry(self) -> DiscGeometry:
        if self._disc_geom is None:
            ilm = self.curves_good.get('ILM')
            nfl = self.curves_good.get('NFL')
            if ilm is None or nfl is None:
                self._disc_geom = DiscGeometry(zc=160, xc=160, r_disc=45)
            else:
                cup_mask = np.isnan(nfl)
                cup_counts = np.sum(cup_mask, axis=1)
                zc = int(np.argmax(cup_counts)) if np.max(cup_counts) > 0 else 160
                cup_cols = np.where(cup_mask[zc])[0]
                xc = int(np.median(cup_cols)) if len(cup_cols) > 0 else 160
                active_slices = np.where(cup_counts > 5)[0]
                r_disc = int(max(len(active_slices) // 2, 25))
                self._disc_geom = DiscGeometry(zc=zc, xc=xc, r_disc=r_disc)
        return self._disc_geom

    def rasterize_curve_slice(self, curve_set: str, b_idx: int) -> np.ndarray:
        """Rasterizes ILM and NFL boundaries of slice b_idx into a binary 2D mask."""
        curves = self.curves_good if curve_set == "good" else self.curves_bad
        if curves is None:
            return np.zeros((self.rows, self.cols), dtype=np.uint8)

        ilm_curve = curves['ILM'][b_idx]
        nfl_curve = curves['NFL'][b_idx]

        mask = np.zeros((self.rows, self.cols), dtype=np.uint8)
        y_grid = np.arange(self.rows)[:, None]
        valid = ~np.isnan(ilm_curve) & ~np.isnan(nfl_curve) & (nfl_curve > ilm_curve)
        y0 = np.clip(np.round(np.nan_to_num(ilm_curve, nan=-1)), 0, self.rows)
        y1 = np.clip(np.round(np.nan_to_num(nfl_curve, nan=-1)), 0, self.rows)
        mask = ((y_grid >= y0) & (y_grid < y1) & valid).astype(np.uint8)
        return mask


# ============================================================================
# 3. Model & Inference Layer: VolumetricRNFLPredictor
# ============================================================================

class VolumetricRNFLPredictor:
    """
    Encapsulates PyTorch model weight loading, GPU/MPS execution, 2.5D context
    stacking, and 1D boundary regression decoding.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        batch_size: int = 8,
        half_ctx: int = 2
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.half_ctx = half_ctx
        self.model.eval()

        self.use_amp = "mps" in str(device) or "cuda" in str(device)
        self.autocast_device = "mps" if "mps" in str(device) else ("cuda" if "cuda" in str(device) else "cpu")

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: torch.device, batch_size: int = 8) -> "VolumetricRNFLPredictor":
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = VolumetricRNFLNet(in_channels=5, base_channels=16).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return cls(model=model, device=device, batch_size=batch_size)

    def predict(self, oct_volume: OCTVolume) -> VolumePrediction:
        """Performs full 3D volumetric segmentation across all slices."""
        memmap = oct_volume.memmap
        n_bscans = oct_volume.n_bscans
        rows = oct_volume.rows
        cols = oct_volume.cols

        full_mask = np.zeros((n_bscans, rows, cols), dtype=np.uint8)
        full_ilm = np.zeros((n_bscans, cols), dtype=np.float32)
        full_nfl = np.zeros((n_bscans, cols), dtype=np.float32)
        full_cup = np.zeros((n_bscans, cols), dtype=np.float32)

        for start_idx in range(0, n_bscans, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_bscans)
            batch_slices = []
            for b_idx in range(start_idx, end_idx):
                slice_indices = [min(max(b_idx + o, 0), n_bscans - 1) for o in range(-self.half_ctx, self.half_ctx + 1)]
                stack = [memmap[s] for s in slice_indices]
                img_stack = np.stack(stack, axis=0).astype(np.float32) / 2560.0
                batch_slices.append(img_stack)

            batch_tensor = torch.from_numpy(np.stack(batch_slices, axis=0)).to(self.device)

            with torch.no_grad():
                with torch.autocast(device_type=self.autocast_device, dtype=torch.bfloat16, enabled=self.use_amp):
                    preds = self.model(batch_tensor)
                    probs = torch.sigmoid(preds['mask_logits']).squeeze(1).float().cpu().numpy()
                    cup_probs = torch.sigmoid(preds['cup_logits']).float().cpu().numpy()
                    ilm_preds = preds['ilm_pred'].float().cpu().numpy()
                    nfl_preds = preds['nfl_pred'].float().cpu().numpy()

            b_masks = (probs > 0.5).astype(np.uint8)
            for i in range(b_masks.shape[0]):
                cup_cols = np.where(cup_probs[i] > 0.5)[0]
                if len(cup_cols) > 0:
                    b_masks[i, :, cup_cols[0]:cup_cols[-1] + 1] = 0

            full_mask[start_idx:end_idx] = b_masks
            full_ilm[start_idx:end_idx] = ilm_preds
            full_nfl[start_idx:end_idx] = nfl_preds
            full_cup[start_idx:end_idx] = cup_probs

        return VolumePrediction(
            mask=full_mask,
            ilm_curve=full_ilm,
            nfl_curve=full_nfl,
            cup_probs=full_cup
        )


# ============================================================================
# 4. Metric Evaluation Layer: ClinicalMetricsCalculator
# ============================================================================

class ClinicalMetricsCalculator:
    """
    Computes peripapillary dice, MABE, 95th-percentile error, and BMO cup IoU
    against ground truth and commercial baseline curves.
    """

    def __init__(self, axial_res_um: float = AXIAL_RES_UM):
        self.axial_res_um = axial_res_um

    def evaluate_slice(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        pred_nfl: np.ndarray,
        gt_nfl: np.ndarray,
        pred_cup: np.ndarray,
        gt_cup: np.ndarray,
        is_peripapillary: bool
    ) -> Optional[SliceMetrics]:
        if not is_peripapillary:
            return None

        # 1. Dice Score
        intersection = np.sum((pred_mask == 1) & (gt_mask == 1))
        total = np.sum(pred_mask == 1) + np.sum(gt_mask == 1)
        dice = (2.0 * intersection / total) if total > 0 else 1.0

        # 2. NFL Boundary Error (MABE & P95)
        valid_nfl = ~np.isnan(gt_nfl) & (gt_nfl > 0)
        if np.sum(valid_nfl) > 0:
            errors = np.abs(pred_nfl[valid_nfl] - gt_nfl[valid_nfl]) * self.axial_res_um
            mabe = float(np.mean(errors))
            p95 = float(np.percentile(errors, 95))
        else:
            mabe, p95 = 0.0, 0.0

        # 3. Cup Cavity IoU
        pred_cup_bin = (pred_cup > 0.5)
        gt_cup_bin = (gt_cup > 0.5)
        cup_inter = np.sum(pred_cup_bin & gt_cup_bin)
        cup_union = np.sum(pred_cup_bin | gt_cup_bin)
        cup_iou = (cup_inter / cup_union) if cup_union > 0 else 1.0

        return SliceMetrics(
            dice=float(dice),
            nfl_mabe=mabe,
            nfl_p95=p95,
            cup_iou=float(cup_iou)
        )

    def evaluate_scan(self, oct_volume: OCTVolume, prediction: VolumePrediction) -> ScanEvaluationResult:
        """Evaluates all peripapillary slices for both U-Net and Commercial baseline."""
        geom = oct_volume.disc_geometry
        curves_good = oct_volume.curves_good
        curves_bad = oct_volume.curves_bad

        slice_metrics_unet: List[SliceMetrics] = []
        slice_metrics_bad: List[SliceMetrics] = []

        for b_idx in range(oct_volume.n_bscans):
            is_peri = abs(b_idx - geom.zc) <= int(2.0 * geom.r_disc)
            if not is_peri:
                continue

            # Ground truth slice
            gt_ilm_b = curves_good['ILM'][b_idx]
            gt_nfl_b = curves_good['NFL'][b_idx]
            gt_mask_b = oct_volume.rasterize_curve_slice("good", b_idx)
            gt_cup_b = np.isnan(gt_nfl_b).astype(np.float32)

            # U-Net evaluation
            m_unet = self.evaluate_slice(
                prediction.mask[b_idx], gt_mask_b,
                prediction.nfl_curve[b_idx], gt_nfl_b,
                prediction.cup_probs[b_idx], gt_cup_b,
                is_peri
            )
            if m_unet:
                slice_metrics_unet.append(m_unet)

            # Commercial Baseline evaluation
            if curves_bad is not None:
                bad_nfl_b = curves_bad['NFL'][b_idx]
                bad_mask_b = oct_volume.rasterize_curve_slice("bad", b_idx)
                bad_cup_b = np.isnan(bad_nfl_b).astype(np.float32)

                m_bad = self.evaluate_slice(
                    bad_mask_b, gt_mask_b,
                    bad_nfl_b, gt_nfl_b,
                    bad_cup_b, gt_cup_b,
                    is_peri
                )
                if m_bad:
                    slice_metrics_bad.append(m_bad)

        return ScanEvaluationResult(
            subject=oct_volume.subject,
            eye=oct_volume.eye,
            cohort=oct_volume.cohort_tag,
            is_validation=oct_volume.is_validation,
            unet_dice=float(np.mean([m.dice for m in slice_metrics_unet])),
            unet_mabe=float(np.mean([m.nfl_mabe for m in slice_metrics_unet])),
            unet_p95=float(np.mean([m.nfl_p95 for m in slice_metrics_unet])),
            unet_cup_iou=float(np.mean([m.cup_iou for m in slice_metrics_unet])),
            bad_dice=float(np.mean([m.dice for m in slice_metrics_bad])) if slice_metrics_bad else None,
            bad_mabe=float(np.mean([m.nfl_mabe for m in slice_metrics_bad])) if slice_metrics_bad else None,
            bad_p95=float(np.mean([m.nfl_p95 for m in slice_metrics_bad])) if slice_metrics_bad else None,
            bad_cup_iou=float(np.mean([m.cup_iou for m in slice_metrics_bad])) if slice_metrics_bad else None,
        )


# ============================================================================
# 5. Visualization Layer: CohortVisualizer
# ============================================================================

class CohortVisualizer:
    """
    Renders high-contrast clinical figures: Side-by-side gallery panels,
    6-panel 3-arm deep dives, and publication-ready cohort summary charts.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def render_gallery_panel(self, oct_volume: OCTVolume, prediction: VolumePrediction) -> str:
        zc = oct_volume.disc_geometry.zc
        raw_bscan = oct_volume.memmap[zc]
        mask_cyan = oct_volume.rasterize_curve_slice("good", zc)
        mask_green = prediction.mask[zc]

        fig, axs = plt.subplots(1, 2, figsize=(16, 5), facecolor="black")
        for col_idx, (title, mask, color) in enumerate([
            ("Reference Algorithm (Cyan)", mask_cyan, [0.0, 0.8, 1.0]),
            ("Volumetric U-Net (Green)", mask_green, [0.1, 0.95, 0.3])
        ]):
            ax = axs[col_idx]
            ax.imshow(raw_bscan, cmap="gray", aspect="auto")

            overlay = np.zeros((*raw_bscan.shape, 4), dtype=np.float32)
            overlay[mask == 1] = [*color, 0.40]
            ax.imshow(overlay, aspect="auto")

            contours = scipy.ndimage.binary_dilation(mask) ^ mask
            overlay_c = np.zeros((*raw_bscan.shape, 4), dtype=np.float32)
            overlay_c[contours] = [*color, 1.0]
            ax.imshow(overlay_c, aspect="auto")

            ax.set_title(f"{title} | {oct_volume.subject} ({oct_volume.eye}) B-scan {zc}", color="white", fontsize=12, fontweight="bold")
            ax.set_ylim(440, 180)
            ax.axis("off")

        filename = f"gallery_bscan_{oct_volume.subject}_{oct_volume.eye}.png"
        out_path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="black")
        plt.close(fig)
        return filename

    def render_deep_dive_panel(self, oct_volume: OCTVolume, prediction: VolumePrediction) -> str:
        zc = oct_volume.disc_geometry.zc
        raw_bscan = oct_volume.memmap[zc]
        mask_cyan = oct_volume.rasterize_curve_slice("good", zc)
        mask_green = prediction.mask[zc]
        mask_red = oct_volume.rasterize_curve_slice("bad", zc) if oct_volume.curves_bad else None

        y_nfl = oct_volume.curves_good['NFL'][zc]
        y_enface = int(np.nanmedian(y_nfl)) if not np.isnan(np.nanmedian(y_nfl)) else 320
        y_enface = int(np.clip(y_enface, 100, 650))

        enface_raw = oct_volume.memmap[:, y_enface, :]
        enface_cyan = np.zeros((320, 320), dtype=np.uint8)
        enface_green = prediction.mask[:, y_enface, :]

        for b_i in range(320):
            m_b = oct_volume.rasterize_curve_slice("good", b_i)
            enface_cyan[b_i] = m_b[y_enface, :]

        fig, axs = plt.subplots(2, 3, figsize=(20, 12), facecolor="black")
        arms = [
            ("Reference Algorithm (Cyan)", mask_cyan, [0.0, 0.8, 1.0]),
            ("Commercial Solix (Red)", mask_red, [1.0, 0.2, 0.2]),
            ("Volumetric U-Net (Green)", mask_green, [0.1, 0.95, 0.3])
        ]

        # Row 0: Full B-Scans across 3 arms
        for col_idx, (title, mask, color) in enumerate(arms):
            ax = axs[0, col_idx]
            ax.imshow(raw_bscan, cmap="gray", aspect="auto")
            if mask is not None:
                overlay = np.zeros((*raw_bscan.shape, 4), dtype=np.float32)
                overlay[mask == 1] = [*color, 0.40]
                ax.imshow(overlay, aspect="auto")

                contours = scipy.ndimage.binary_dilation(mask) ^ mask
                overlay_c = np.zeros((*raw_bscan.shape, 4), dtype=np.float32)
                overlay_c[contours] = [*color, 1.0]
                ax.imshow(overlay_c, aspect="auto")

            ax.set_title(f"{title}\nCentral B-scan {zc}", color="white", fontsize=11, fontweight="bold")
            ax.set_ylim(440, 180)
            ax.axis("off")

        # Row 1, Col 0: Nasal Rim Zoom
        ax_left = axs[1, 0]
        ax_left.imshow(raw_bscan, cmap="gray", aspect="auto")
        over_l = np.zeros((*raw_bscan.shape, 4), dtype=np.float32)
        over_l[mask_cyan == 1] = [0.0, 0.8, 1.0, 0.3]
        over_l[mask_green == 1] = [0.1, 0.95, 0.3, 0.5]
        ax_left.imshow(over_l, aspect="auto")
        ax_left.set_xlim(60, 140)
        ax_left.set_ylim(370, 220)
        ax_left.set_title("Nasal Rim Zoom\n[Cyan: Ref vs Green: U-Net]", color="white", fontsize=11, fontweight="bold")
        ax_left.axis("off")

        # Row 1, Col 1: Temporal Rim Zoom
        ax_right = axs[1, 1]
        ax_right.imshow(raw_bscan, cmap="gray", aspect="auto")
        over_r = np.zeros((*raw_bscan.shape, 4), dtype=np.float32)
        over_r[mask_cyan == 1] = [0.0, 0.8, 1.0, 0.3]
        over_r[mask_green == 1] = [0.1, 0.95, 0.3, 0.5]
        ax_right.imshow(over_r, aspect="auto")
        ax_right.set_xlim(180, 260)
        ax_right.set_ylim(370, 220)
        ax_right.set_title("Temporal Rim Zoom\n[Cyan: Ref vs Green: U-Net]", color="white", fontsize=11, fontweight="bold")
        ax_right.axis("off")

        # Row 1, Col 2: En Face Mid-Rim
        ax_ef = axs[1, 2]
        ax_ef.imshow(enface_raw, cmap="gray", aspect="auto")
        over_ef = np.zeros((*enface_raw.shape, 4), dtype=np.float32)
        over_ef[enface_cyan == 1] = [0.0, 0.8, 1.0, 0.35]
        over_ef[enface_green == 1] = [0.1, 0.95, 0.3, 0.5]
        ax_ef.imshow(over_ef, aspect="auto")
        ax_ef.set_title(f"En Face Mid-Rim Plane (y={y_enface})\n[Cyan: Ref vs Green: U-Net]", color="white", fontsize=11, fontweight="bold")
        ax_ef.axis("off")

        filename = f"deep_dive_{oct_volume.subject}_{oct_volume.eye}.png"
        out_path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="black")
        plt.close(fig)
        return filename

    def render_cohort_summary_chart(self, scan_results: List[ScanEvaluationResult]) -> str:
        """Renders the executive summary chart with orange validation highlights."""
        od_scans = [s for s in scan_results if s.eye == "OD"]
        subjects = [s.subject for s in od_scans]
        unet_dices = [s.unet_dice for s in od_scans]
        bad_dices = [s.bad_dice if s.bad_dice is not None else 0.0 for s in od_scans]
        unet_mabes = [s.unet_mabe for s in od_scans]
        unet_cup_ious = [s.unet_cup_iou for s in od_scans]
        is_val = [s.is_validation for s in od_scans]

        mirror_subjects = {"BEH0181", "BEH0398", "BEH0410"}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.4), facecolor="#0f172a")
        x = np.arange(len(subjects))
        width = 0.38

        # --- Chart 1: Peripapillary Dice ---
        for idx, s_name in enumerate(subjects):
            if is_val[idx]:
                ax1.bar(x[idx] - width/2, unet_dices[idx], width, color="#f59e0b", alpha=0.95, edgecolor="white", linewidth=1.2, zorder=3)
            else:
                ax1.bar(x[idx] - width/2, unet_dices[idx], width, color="#10b981", alpha=0.92, edgecolor="white", linewidth=0.6, zorder=3)

        for idx, s_name in enumerate(subjects):
            b_dice = bad_dices[idx]
            if b_dice == 0.0:
                continue
            if s_name in mirror_subjects:
                ax1.bar(x[idx] + width/2, b_dice, width, color="#94a3b8", alpha=0.32, hatch="//", edgecolor="#f87171", linewidth=0.8, zorder=3)
                ax1.text(x[idx] + width/2, b_dice + 0.02, "†", color="#f87171", fontsize=13, fontweight="bold", ha="center")
            else:
                ax1.bar(x[idx] + width/2, b_dice, width, color="#ef4444", alpha=0.88, edgecolor="white", linewidth=0.6, zorder=3)

        for idx, s in enumerate(od_scans):
            if s.is_validation:
                y_pos = max(unet_dices[idx], bad_dices[idx]) + 0.05 if bad_dices[idx] < 0.95 else unet_dices[idx] + 0.04
                ax1.annotate("Held-Out Val", (x[idx], y_pos),
                             color="#f59e0b", fontweight="bold", ha="center", fontsize=8.5,
                             bbox=dict(boxstyle="round,pad=0.25", fc="#1e293b", ec="#f59e0b", lw=1.2))

        legend_elements1 = [
            Patch(facecolor="#10b981", edgecolor="white", label="U-Net (Train Fit)"),
            Patch(facecolor="#f59e0b", edgecolor="white", label="U-Net (Held-Out Val)"),
            Patch(facecolor="#ef4444", edgecolor="white", label="Commercial (Clinician GT)"),
            Patch(facecolor="#94a3b8", alpha=0.4, hatch="//", edgecolor="#f87171", label="Commercial (Mirror †)")
        ]

        ax1.set_title("Cohort Peripapillary Dice Score Comparison (OD)", color="white", fontsize=12.5, fontweight="bold", pad=28)
        ax1.set_xticks(x)
        xtick_labels_1 = ax1.set_xticklabels(subjects, rotation=42, fontsize=9.5)
        for idx, s in enumerate(od_scans):
            if s.is_validation:
                xtick_labels_1[idx].set_color("#f59e0b")
                xtick_labels_1[idx].set_fontweight("bold")
            else:
                xtick_labels_1[idx].set_color("#cbd5e1")

        ax1.set_ylabel("Peripapillary Dice Score", color="#cbd5e1", fontsize=10.5)
        ax1.set_ylim(0.0, 1.12)
        ax1.grid(axis="y", color="#334155", linestyle="--", alpha=0.7)
        ax1.legend(handles=legend_elements1, facecolor="#1e293b", edgecolor="#475569", labelcolor="white", fontsize=8.5,
                   loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=4, frameon=True)
        ax1.tick_params(colors="#cbd5e1")
        for spine in ax1.spines.values():
            spine.set_color("#475569")

        # --- Chart 2: MABE & Cup IoU ---
        train_mask = [not v for v in is_val]
        val_mask = is_val

        ax2.plot(x, unet_mabes, "-", color="#38bdf8", linewidth=2.2, alpha=0.85, zorder=2)
        train_x = [x[i] for i in range(len(x)) if train_mask[i]]
        train_mabes = [unet_mabes[i] for i in range(len(x)) if train_mask[i]]
        ax2.scatter(train_x, train_mabes, color="#38bdf8", s=60, zorder=3, edgecolors="white", linewidth=0.8, label="Train NFL MABE")

        val_x = [x[i] for i in range(len(x)) if val_mask[i]]
        val_mabes = [unet_mabes[i] for i in range(len(x)) if val_mask[i]]
        ax2.scatter(val_x, val_mabes, color="#f59e0b", s=120, zorder=5, edgecolors="white", linewidth=1.5, marker="D", label="Val MABE (Held-Out)")

        ax2.set_ylabel("Mean Absolute Boundary Error (µm)", color="#38bdf8", fontsize=10.5)
        ax2.set_xticks(x)
        xtick_labels_2 = ax2.set_xticklabels(subjects, rotation=42, fontsize=9.5)
        for idx, s in enumerate(od_scans):
            if s.is_validation:
                xtick_labels_2[idx].set_color("#f59e0b")
                xtick_labels_2[idx].set_fontweight("bold")
            else:
                xtick_labels_2[idx].set_color("#cbd5e1")

        ax2.tick_params(axis="y", labelcolor="#38bdf8", colors="#cbd5e1")
        ax2.tick_params(axis="x", colors="#cbd5e1")
        ax2.grid(color="#334155", linestyle="--", alpha=0.7)
        ax2.set_title("U-Net Boundary Accuracy & BMO Cup IoU (OD)", color="white", fontsize=12.5, fontweight="bold", pad=28)
        ax2.set_ylim(0, 108)
        ax2.axhspan(6.5, 9.5, color="#38bdf8", alpha=0.12, label="Train Benchmark (7-9 µm)")

        ax2_twin = ax2.twinx()
        ax2_twin.plot(x, unet_cup_ious, "--", color="#c084fc", linewidth=2.0, alpha=0.85, zorder=2)
        train_cup_ious = [unet_cup_ious[i] for i in range(len(x)) if train_mask[i]]
        ax2_twin.scatter(train_x, train_cup_ious, color="#c084fc", s=55, marker="s", zorder=3, edgecolors="white", linewidth=0.8, label="Train Cup IoU")

        val_cup_ious = [unet_cup_ious[i] for i in range(len(x)) if val_mask[i]]
        ax2_twin.scatter(val_x, val_cup_ious, color="#f59e0b", s=100, marker="s", zorder=5, edgecolors="white", linewidth=1.5, label="Val Cup IoU (Held-Out)")

        ax2_twin.set_ylabel("Cup Cavity IoU", color="#c084fc", fontsize=10.5)
        ax2_twin.tick_params(axis="y", labelcolor="#c084fc", colors="#cbd5e1")
        ax2_twin.set_ylim(0.65, 1.03)

        if "BEH0314" in subjects:
            idx_314 = subjects.index("BEH0314")
            ax2.annotate(f"Held-Out Val\n{unet_mabes[idx_314]:.1f} µm",
                         (idx_314, unet_mabes[idx_314]),
                         xytext=(idx_314, unet_mabes[idx_314] + 13),
                         color="#f59e0b", fontsize=8.2, fontweight="bold", ha="center",
                         arrowprops=dict(arrowstyle="->", color="#f59e0b", lw=1.2),
                         bbox=dict(boxstyle="round,pad=0.25", fc="#1e293b", ec="#f59e0b", lw=1.0))

        if "BEH0335" in subjects:
            idx_335 = subjects.index("BEH0335")
            ax2.annotate(f"Held-Out Val (Tilt)\n{unet_mabes[idx_335]:.1f} µm",
                         (idx_335, unet_mabes[idx_335]),
                         xytext=(idx_335, unet_mabes[idx_335] + 12),
                         color="#f59e0b", fontsize=8.2, fontweight="bold", ha="center",
                         arrowprops=dict(arrowstyle="->", color="#f59e0b", lw=1.2),
                         bbox=dict(boxstyle="round,pad=0.25", fc="#1e293b", ec="#f59e0b", lw=1.0))

        legend_elements2 = [
            Line2D([0], [0], color="#38bdf8", marker="o", markersize=6, label="Train MABE (µm)"),
            Line2D([0], [0], color="#f59e0b", marker="D", markersize=7, linestyle="None", label="Val MABE (Orange)"),
            Line2D([0], [0], color="#c084fc", marker="s", markersize=6, linestyle="--", label="Train Cup IoU"),
            Line2D([0], [0], color="#f59e0b", marker="s", markersize=7, linestyle="None", label="Val Cup IoU (Orange)"),
        ]
        ax2.legend(handles=legend_elements2, facecolor="#1e293b", edgecolor="#475569", labelcolor="white",
                   loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=4, fontsize=7.5, frameon=True)

        for spine in ax2.spines.values():
            spine.set_color("#475569")
        for spine in ax2_twin.spines.values():
            spine.set_color("#475569")

        fig.text(0.5, 0.015, "Orange color, text badges, and markers indicate held-out validation subjects (BEH0314, BEH0335) unseen during training.  † Unedited machine mirror.",
                 color="#cbd5e1", fontsize=8.5, ha="center", style="italic")

        out_path = os.path.join(self.output_dir, "cohort_summary_chart.png")
        plt.tight_layout(rect=[0, 0.035, 1, 0.98])
        plt.savefig(out_path, dpi=200, facecolor="#0f172a", bbox_inches="tight")
        plt.close(fig)
        return out_path


# ============================================================================
# 6. Pipeline Orchestration: CohortEvaluatorPipeline
# ============================================================================

class CohortEvaluatorPipeline:
    """
    Coordinates cohort volume discovery, neural network inference,
    clinical metric calculation, and report artifact generation.
    """

    def __init__(
        self,
        checkpoint_path: str,
        dataset_root: str,
        output_dir: str,
        val_subjects: List[str],
        device: str = "mps"
    ):
        self.dataset_root = dataset_root
        self.output_dir = output_dir
        self.val_subjects = val_subjects

        dev_str = device if ("mps" in device and torch.backends.mps.is_available()) or "cuda" in device else "cpu"
        self.device = torch.device(dev_str)

        self.assets_dir = os.path.join(self.output_dir, "assets", "executive_cohort_report")
        os.makedirs(self.assets_dir, exist_ok=True)

        print(f"[Cohort Pipeline] Initializing VolumetricRNFLPredictor on {self.device}...")
        self.predictor = VolumetricRNFLPredictor.from_checkpoint(checkpoint_path, self.device)
        self.metrics_calc = ClinicalMetricsCalculator(axial_res_um=AXIAL_RES_UM)
        self.visualizer = CohortVisualizer(output_dir=self.assets_dir)

    def discover_volumes(self) -> List[OCTVolume]:
        """Scans dataset root and pairs DICOM volumes with matching XML curves."""
        dicom_dir = os.path.join(self.dataset_root, "dicom")
        tsv_good_dir = os.path.join(self.dataset_root, "tsv", "good")
        tsv_bad_dir = os.path.join(self.dataset_root, "tsv", "bad")

        all_subjs = sorted([s for s in os.listdir(dicom_dir) if os.path.isdir(os.path.join(dicom_dir, s))])
        volumes: List[OCTVolume] = []

        for subj in all_subjs:
            is_val = subj in self.val_subjects
            subj_dcms = sorted(glob.glob(os.path.join(dicom_dir, subj, "*Disc Cube*_OPT.dcm")))
            for dcm_path in subj_dcms:
                fname = os.path.basename(dcm_path)
                eye = "OS" if "_OS_" in fname else "OD"

                good_xml = self._find_matching_curve_xml(tsv_good_dir, subj, eye, fname)
                if not good_xml:
                    continue
                bad_xml = self._find_matching_curve_xml(tsv_bad_dir, subj, eye, fname)

                vol = OCTVolume(
                    subject=subj,
                    eye=eye,
                    dcm_path=dcm_path,
                    good_xml_path=good_xml,
                    bad_xml_path=bad_xml,
                    is_validation=is_val
                )
                volumes.append(vol)

        return volumes

    @staticmethod
    def _find_matching_curve_xml(tsv_dir: str, subj: str, eye: str, dcm_fname: str, protocol: str = "Disc Cube") -> Optional[str]:
        subj_tsv = os.path.join(tsv_dir, subj)
        if not os.path.exists(subj_tsv):
            return None

        # 1. Timestamp matching via master XML
        m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})", dcm_fname)
        if m:
            target_time = f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}"
            master_xmls = glob.glob(os.path.join(subj_tsv, "*.xml"))
            for m_xml in master_xmls:
                try:
                    tree = ET.parse(m_xml)
                    for scan in tree.getroot().iter("Scan"):
                        stype = scan.findtext("ScanType", "").strip()
                        stime = scan.findtext("ScanTime", "").strip()
                        seye = scan.findtext("Eye", "").strip()
                        cfile = scan.findtext("Curve/File", "").strip()
                        if protocol.lower() in stype.lower() and seye == eye and stime == target_time and cfile:
                            rel_path = cfile.replace("\\", "/").lstrip("./")
                            full_path = os.path.join(subj_tsv, rel_path)
                            if os.path.exists(full_path):
                                return full_path
                except Exception:
                    pass

        # 2. Fallback glob
        xmls = glob.glob(os.path.join(subj_tsv, "curve", f"*{eye}*{protocol}*.xml"))
        if not xmls:
            xmls = glob.glob(os.path.join(subj_tsv, "curve", f"*{protocol}*{eye}*.xml"))
        return xmls[0] if xmls else None

    def run(self) -> List[ScanEvaluationResult]:
        volumes = self.discover_volumes()
        print(f"[Cohort Pipeline] Discovered {len(volumes)} valid volumes across cohort.")

        cohort_results: List[ScanEvaluationResult] = []
        gallery_manifest: List[Dict[str, Any]] = []
        deep_dive_manifest: List[Dict[str, Any]] = []

        for vol in volumes:
            print(f"\n---> Evaluating {vol.subject} ({vol.eye}) [{vol.cohort_tag}]...")

            # 1. Inference
            prediction = self.predictor.predict(vol)

            # 2. Metric Computation
            result = self.metrics_calc.evaluate_scan(vol, prediction)
            cohort_results.append(result)

            print(f"[{vol.subject} {vol.eye}] U-Net Dice: {result.unet_dice:.4f} | MABE: {result.unet_mabe:.2f} µm | P95: {result.unet_p95:.2f} µm | Cup IoU: {result.unet_cup_iou:.4f}")
            if result.bad_dice is not None:
                print(f"[{vol.subject} {vol.eye}] Bad   Dice: {result.bad_dice:.4f} | MABE: {result.bad_mabe:.2f} µm | P95: {result.bad_p95:.2f} µm | Cup IoU: {result.bad_cup_iou:.4f}")

            # 3. Visuals for OD acquisitions
            if vol.eye == "OD" or vol.subject not in [g['subject'] for g in gallery_manifest]:
                g_file = self.visualizer.render_gallery_panel(vol, prediction)
                gallery_manifest.append({
                    'subject': vol.subject,
                    'eye': vol.eye,
                    'cohort': vol.cohort_tag,
                    'filename': g_file,
                    'dice': result.unet_dice,
                    'mabe': result.unet_mabe
                })

                if vol.is_validation or vol.subject in ["BEH0181", "BEH0174"]:
                    dd_file = self.visualizer.render_deep_dive_panel(vol, prediction)
                    deep_dive_manifest.append({
                        'subject': vol.subject,
                        'eye': vol.eye,
                        'cohort': vol.cohort_tag,
                        'filename': dd_file
                    })

        # 4. Generate Cohort Summary Chart
        print("\n[Cohort Pipeline] Generating Publication Cohort Summary Chart...")
        chart_path = self.visualizer.render_cohort_summary_chart(cohort_results)
        print(f"[Cohort Pipeline] Summary chart saved to: {chart_path}")

        # 5. Export JSON Manifest
        json_path = os.path.join(self.assets_dir, "cohort_evaluation_metrics.json")
        with open(json_path, "w") as f:
            json.dump({
                'scans': [r.to_dict() for r in cohort_results],
                'gallery': gallery_manifest,
                'deep_dives': deep_dive_manifest
            }, f, indent=2)
        print(f"[Cohort Pipeline] Metrics JSON saved to: {json_path}")

        return cohort_results


# ============================================================================
# 7. CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Object-Oriented Volumetric RNFL Cohort Evaluator")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--dataset_root", type=str, default="/Users/nikhilmundhra/Library/CloudStorage/Box-Box/deidentified")
    parser.add_argument("--output_dir", type=str, default="/Users/nikhilmundhra/Documents/Github/Capstone/OCT-Analyser-Capstone/docs")
    parser.add_argument("--val_subjects", type=str, default="BEH0335,BEH0314")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    val_subjects = [s.strip() for s in args.val_subjects.split(",")]

    pipeline = CohortEvaluatorPipeline(
        checkpoint_path=args.checkpoint,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        val_subjects=val_subjects,
        device=args.device
    )
    pipeline.run()


if __name__ == "__main__":
    main()
