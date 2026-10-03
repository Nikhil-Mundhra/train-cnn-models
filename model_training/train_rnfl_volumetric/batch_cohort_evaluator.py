"""
batch_cohort_evaluator.py
==========================
Object-Oriented Comprehensive Batch Evaluator spanning all 11 Solix OCT subjects.
Computes quantitative boundary & volumetric metrics against the human-corrected reference (good)
and commercial baseline (bad), and exports publication-ready comparative visual panels.
"""

import os
import sys
import glob
import json
import re
import argparse
import csv
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
import scipy.ndimage

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
TRAIN_3D_DIR = SCRIPT_DIR.parent / "train_rnfl_3d"
if str(TRAIN_3D_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_3D_DIR))

from dataset import find_dicom_pixel_offset, load_curves, find_matching_curve_xml
from model import VolumetricRNFLNet
from orientation import (
    OS_ORIENTATION_MODES,
    horizontal_requires_flip,
    validate_orientation_mode,
    vertical_source_and_destination,
)
from quality_control import assess_prediction
from evaluation_manifest import evaluation_subjects, is_evaluation_split, load_split_manifest
from audit_analysis import (
    AUDIT_EDIT_THRESHOLD_PX,
    PRESERVATION_TOLERANCE_PX,
    compute_audit_correction_metrics,
)
from spatial import AXIAL_UM, FAST_UM, SLOW_UM
from evaluation_metrics import compute_volumetric_metrics

AXIAL_RES_UM: float = AXIAL_UM


def resolve_validation_subjects(
    checkpoint_path: str,
    requested: Optional[str] = None,
    split_manifest: Optional[str] = None,
) -> List[str]:
    """Resolve held-out subjects from override, manifest, or checkpoint metadata."""
    raw_subjects: Any = requested
    if not raw_subjects and split_manifest:
        subjects = evaluation_subjects(split_manifest)
        if not subjects:
            raise ValueError(f"Split manifest has no evaluation subjects: {split_manifest}")
        return subjects
    if not raw_subjects:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            raw_subjects = checkpoint.get("val_subjects") or (checkpoint.get("args") or {}).get("val_subjects")
    if not raw_subjects:
        raise ValueError(
            "Validation subjects are absent from the checkpoint metadata; "
            "provide --val_subjects or --split_manifest explicitly."
        )
    if isinstance(raw_subjects, str):
        subjects = [subject.strip() for subject in raw_subjects.split(",") if subject.strip()]
    else:
        subjects = [str(subject).strip() for subject in raw_subjects if str(subject).strip()]
    if not subjects:
        raise ValueError("Validation subject list resolved to an empty set")
    return subjects


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
    unet_dice_3d: Optional[float] = None
    unet_hd_um: Optional[float] = None
    unet_hd95_um: Optional[float] = None
    unet_asd_um: Optional[float] = None
    unet_volume_similarity: Optional[float] = None
    is_mirror: bool = False
    bad_dice: Optional[float] = None
    bad_mabe: Optional[float] = None
    bad_p95: Optional[float] = None
    bad_cup_iou: Optional[float] = None
    audit_edit_threshold_px: float = AUDIT_EDIT_THRESHOLD_PX
    audit_edited_columns: int = 0
    audit_unchanged_columns: int = 0
    audit_edit_fraction: Optional[float] = None
    raw_edit_mabe_um: Optional[float] = None
    unet_edit_mabe_um: Optional[float] = None
    delta_edit_mabe_um: Optional[float] = None
    audit_correction_gain: Optional[float] = None
    audit_ratio_gain: Optional[float] = None
    audit_symmetric_gain: Optional[float] = None
    audit_edit_recovery_rate: Optional[float] = None
    unet_unchanged_mabe_um: Optional[float] = None
    audit_unchanged_preservation_rate: Optional[float] = None
    unet_cup_presence_recall: Optional[float] = None
    unet_cup_edge_valid_slices: int = 0
    unet_cup_left_edge_mae_um: Optional[float] = None
    unet_cup_right_edge_mae_um: Optional[float] = None
    unet_cup_width_mae_um: Optional[float] = None
    raw_cup_edge_valid_slices: int = 0
    raw_cup_left_edge_mae_um: Optional[float] = None
    raw_cup_right_edge_mae_um: Optional[float] = None
    raw_cup_width_mae_um: Optional[float] = None
    qc_status: str = "not_assessed"
    qc_flags: List[str] = field(default_factory=list)
    qc_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VolumePrediction:
    """Container holding model predictions across an entire 3D volume."""
    mask: np.ndarray        # (320, 768, 320) uint8
    ilm_curve: np.ndarray   # (320, 320) float32
    nfl_curve: np.ndarray   # (320, 320) float32
    cup_probs: np.ndarray   # (320, 320) float32
    diagnostics: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class OpticDiscCutConfig:
    """Anatomical dimensions and scale factors for optic disc cup masking."""
    mode: str = "cup"             # "cup" (reaches optic cup, bounded by predicted cup void), "disc" (legacy 1.75mm disc cut), "none"
    dx_mm: float = FAST_UM / 1000.0  # Solix fast-axis resolution (mm/px)
    dz_mm: float = SLOW_UM / 1000.0  # Solix slow-axis resolution (mm/slice)
    disc_diam_x_mm: float = 1.75  # Mean anatomical horizontal optic disc diameter (mm)
    disc_diam_z_mm: float = 1.85  # Mean anatomical vertical optic disc diameter (mm)

    @property
    def rad_x_px(self) -> float:
        return (self.disc_diam_x_mm / 2.0) / self.dx_mm

    @property
    def rad_z_slices(self) -> float:
        return (self.disc_diam_z_mm / 2.0) / self.dz_mm


@dataclass(frozen=True)
class InferenceConfig:
    """Hyperparameters for neural network volume inference and post-processing."""
    norm_divisor: float = 2560.0
    mask_threshold: float = 0.40
    cup_threshold: float = 0.50
    min_layer_thickness: float = 2.0  # Minimum NFL - ILM thickness in px for valid tissue
    surface_axial_tolerance_px: float = 2.0  # Maximum axial margin beyond predicted surfaces



# ============================================================================
# 2. Data Layer: OCTVolume
# ============================================================================

class OCTVolume:
    """
    Encapsulates a single patient OCT acquisition: DICOM raw data, ground truth
    human-corrected curves, commercial baseline curves, and spatial disc geometry.
    """

    def __init__(
        self,
        subject: str,
        eye: str,
        dcm_path: str,
        good_xml_path: Optional[str],
        bad_xml_path: Optional[str] = None,
        is_validation: bool = False,
        cohort_tag: Optional[str] = None,
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
        self.cohort_tag = cohort_tag or ("Validation (Held-Out)" if is_validation else "Training / Benchmark")

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
    def curves_good(self) -> Optional[Dict[str, np.ndarray]]:
        if self._curves_good is None:
            if not self.good_xml_path or not os.path.exists(self.good_xml_path):
                return None
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
            curves = self.curves_good
            if curves is None:
                self._disc_geom = DiscGeometry(zc=self.n_bscans // 2, xc=self.cols // 2, r_disc=45)
            else:
                ilm = curves.get('ILM')
                nfl = curves.get('NFL')
                if ilm is None or nfl is None:
                    self._disc_geom = DiscGeometry(zc=self.n_bscans // 2, xc=self.cols // 2, r_disc=45)
                else:
                    cup_mask = np.isnan(nfl)
                    cup_counts = np.sum(cup_mask, axis=1)
                    zc = int(np.argmax(cup_counts)) if np.max(cup_counts) > 0 else self.n_bscans // 2
                    cup_cols = np.where(cup_mask[zc])[0]
                    xc = int(np.median(cup_cols)) if len(cup_cols) > 0 else self.cols // 2
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
        half_ctx: int = 2,
        config: InferenceConfig = InferenceConfig(),
        disc_cut_config: OpticDiscCutConfig = OpticDiscCutConfig(),
        os_orientation_mode: str = "corrected",
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.half_ctx = half_ctx
        self.config = config
        self.disc_cut_config = disc_cut_config
        self.os_orientation_mode = validate_orientation_mode(os_orientation_mode)
        self.model.eval()

        self.use_amp = "mps" in str(device) or "cuda" in str(device)
        self.autocast_device = "mps" if "mps" in str(device) else ("cuda" if "cuda" in str(device) else "cpu")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: torch.device,
        batch_size: int = 8,
        os_orientation_mode: str = "corrected",
        config: InferenceConfig = InferenceConfig(),
        disc_cut_config: OpticDiscCutConfig = OpticDiscCutConfig(),
    ) -> "VolumetricRNFLPredictor":
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        ckpt_args = ckpt.get('args', {})
        base_channels = ckpt_args.get('base_channels')
        state_dict = ckpt.get('model_state_dict', ckpt)
        if base_channels is None:
            if 'mask_head.weight' in state_dict:
                base_channels = state_dict['mask_head.weight'].shape[1]
            else:
                base_channels = 16

        raw_channels = ckpt_args.get('channels', None)
        if raw_channels and isinstance(raw_channels, str) and raw_channels.strip():
            channels = tuple(int(c.strip()) for c in raw_channels.split(','))
        elif raw_channels and isinstance(raw_channels, (list, tuple)) and len(raw_channels) >= 2:
            channels = tuple(raw_channels)
        else:
            channels = tuple(base_channels * (2**i) for i in range(5))

        num_res_units = ckpt_args.get('num_res_units', 2)
        use_laterality_embedding = ckpt_args.get('use_laterality_embedding', False)
        if not use_laterality_embedding:
            use_laterality_embedding = 'eye_emb.weight' in state_dict

        model = VolumetricRNFLNet(
            in_channels=5,
            base_channels=base_channels,
            channels=channels,
            num_res_units=num_res_units,
            use_laterality_embedding=use_laterality_embedding,
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        return cls(
            model=model,
            device=device,
            batch_size=batch_size,
            config=config,
            disc_cut_config=disc_cut_config,
            os_orientation_mode=os_orientation_mode,
        )

    def _forward_batch(self, batch_tensor: torch.Tensor, eye: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Executes a single forward pass under autocast and converts outputs to numpy arrays."""
        eye_idx = None
        if getattr(self.model, 'use_laterality_embedding', False) and eye is not None:
            eye_idx = torch.tensor([0 if eye.upper() == 'OD' else 1] * batch_tensor.size(0), dtype=torch.long, device=self.device)
        with torch.no_grad():
            with torch.autocast(device_type=self.autocast_device, dtype=torch.bfloat16, enabled=self.use_amp):
                if eye_idx is not None:
                    preds = self.model(batch_tensor, eye_idx=eye_idx)
                else:
                    preds = self.model(batch_tensor)
                probs = torch.sigmoid(preds['mask_logits']).squeeze(1).float().cpu().numpy()
                cup_probs = torch.sigmoid(preds['cup_logits']).float().cpu().numpy()
                ilm_preds = preds['ilm_pred'].float().cpu().numpy()
                nfl_preds = preds['nfl_pred'].float().cpu().numpy()
        return probs, ilm_preds, nfl_preds, cup_probs

    def _infer_horizontal_pane(
        self,
        oct_volume: OCTVolume
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Performs 2.5D context inference across horizontal B-scans (Z-axis slices)."""
        flip_os = horizontal_requires_flip(oct_volume.eye, self.os_orientation_mode)
        memmap = oct_volume.memmap
        n_bscans = oct_volume.n_bscans
        rows = oct_volume.rows
        cols = oct_volume.cols

        full_probs = np.zeros((n_bscans, rows, cols), dtype=np.float32)
        full_ilm = np.zeros((n_bscans, cols), dtype=np.float32)
        full_nfl = np.zeros((n_bscans, cols), dtype=np.float32)
        full_cup = np.zeros((n_bscans, cols), dtype=np.float32)

        for start_idx in range(0, n_bscans, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_bscans)
            batch_slices = []
            for b_idx in range(start_idx, end_idx):
                slice_indices = [min(max(b_idx + o, 0), n_bscans - 1) for o in range(-self.half_ctx, self.half_ctx + 1)]
                stack = [memmap[s] for s in slice_indices]
                img_stack = np.stack(stack, axis=0).astype(np.float32) / self.config.norm_divisor
                if flip_os:
                    img_stack = np.flip(img_stack, axis=-1).copy()
                batch_slices.append(img_stack)

            batch_tensor = torch.from_numpy(np.stack(batch_slices, axis=0)).to(self.device)
            probs, ilm_preds, nfl_preds, cup_probs = self._forward_batch(batch_tensor, eye=oct_volume.eye)

            if flip_os:
                probs = np.flip(probs, axis=-1).copy()
                ilm_preds = np.flip(ilm_preds, axis=-1).copy()
                nfl_preds = np.flip(nfl_preds, axis=-1).copy()
                cup_probs = np.flip(cup_probs, axis=-1).copy()

            full_probs[start_idx:end_idx] = probs
            full_ilm[start_idx:end_idx] = ilm_preds
            full_nfl[start_idx:end_idx] = nfl_preds
            full_cup[start_idx:end_idx] = cup_probs

        return full_probs, full_ilm, full_nfl, full_cup

    def _infer_vertical_pane(
        self,
        oct_volume: OCTVolume
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Performs 2.5D context inference across orthogonal vertical planes (X-axis slices)."""
        flip_os = horizontal_requires_flip(oct_volume.eye, self.os_orientation_mode)
        offset_sign = -1 if flip_os else 1
        memmap = oct_volume.memmap
        n_bscans = oct_volume.n_bscans
        rows = oct_volume.rows
        cols = oct_volume.cols

        full_probs = np.zeros((n_bscans, rows, cols), dtype=np.float32)
        full_ilm = np.zeros((cols, n_bscans), dtype=np.float32)
        full_nfl = np.zeros((cols, n_bscans), dtype=np.float32)
        full_cup = np.zeros((cols, n_bscans), dtype=np.float32)

        for start_a in range(0, cols, self.batch_size):
            end_a = min(start_a + self.batch_size, cols)
            batch_slices = []
            destinations = []
            for a_idx in range(start_a, end_a):
                eff_a, dest_a = vertical_source_and_destination(
                    a_idx, cols, oct_volume.eye, self.os_orientation_mode
                )
                slice_indices = [min(max(eff_a + offset_sign * o, 0), cols - 1) for o in range(-self.half_ctx, self.half_ctx + 1)]
                # Transpose (320, 768) -> (768, 320)
                stack = [memmap[:, :, s].T for s in slice_indices]
                img_stack = np.stack(stack, axis=0).astype(np.float32) / self.config.norm_divisor
                batch_slices.append(img_stack)
                destinations.append(dest_a)

            batch_tensor = torch.from_numpy(np.stack(batch_slices, axis=0)).to(self.device)
            v_probs, v_ilm, v_nfl, v_cup = self._forward_batch(batch_tensor, eye=oct_volume.eye)

            for i, dest_a in enumerate(destinations):
                # Transpose (768, 320) -> (320, 768) to map (Z, Y) back to (Y, Z)
                full_probs[:, :, dest_a] = v_probs[i].T
                full_ilm[dest_a, :] = v_ilm[i]
                full_nfl[dest_a, :] = v_nfl[i]
                full_cup[dest_a, :] = v_cup[i]

        return full_probs, full_ilm, full_nfl, full_cup

    @staticmethod
    def _fuse_predictions(
        horiz: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        vert: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fuses horizontal and orthogonal vertical predictions via biplanar averaging."""
        h_probs, h_ilm, h_nfl, h_cup = horiz
        v_probs, v_ilm, v_nfl, v_cup = vert
        fused_probs = 0.5 * (h_probs + v_probs)
        fused_ilm = 0.5 * (h_ilm + v_ilm.T)
        fused_nfl = 0.5 * (h_nfl + v_nfl.T)
        fused_cup = 0.5 * (h_cup + v_cup.T)
        return fused_probs, fused_ilm, fused_nfl, fused_cup

    @staticmethod
    def recover_thin_dropouts(
        mask: np.ndarray,
        fused_ilm: np.ndarray,
        fused_nfl: np.ndarray,
        fused_cup: np.ndarray,
        config: InferenceConfig = InferenceConfig()
    ) -> np.ndarray:
        """Recovers thin segmentation dropouts using continuous 1D surface boundaries."""
        out_mask = mask.copy()
        n_bscans, rows, _ = out_mask.shape
        for b in range(n_bscans):
            col_sums = out_mask[b].sum(axis=0)
            valid_tissue = (fused_cup[b] < config.cup_threshold) & (
                (fused_nfl[b] - fused_ilm[b]) >= config.min_layer_thickness
            )
            dropouts = np.where((col_sums == 0) & valid_tissue)[0]
            for x in dropouts:
                y0 = int(np.clip(np.round(fused_ilm[b, x]), 0, rows))
                y1 = int(np.clip(np.round(fused_nfl[b, x]), 0, rows))
                if y1 > y0:
                    out_mask[b, y0:y1, x] = 1
        return out_mask

    @staticmethod
    def clamp_mask_to_surfaces(
        mask: np.ndarray,
        fused_ilm: np.ndarray,
        fused_nfl: np.ndarray,
        fused_cup: np.ndarray,
        config: InferenceConfig = InferenceConfig(),
    ) -> np.ndarray:
        """
        Suppresses false-positive segmentation leakage into deep hyperreflective
        retinal layers (RPE / Bruch's Membrane / choroid) or vitreous by constraining
        dense voxel activations to the continuous 1D surface boundaries.
        """
        out_mask = mask.copy()
        n_bscans, rows, cols = out_mask.shape
        tol = config.surface_axial_tolerance_px
        y_grid = np.arange(rows, dtype=np.float32)[:, None]

        for b in range(n_bscans):
            # Suppress any column identified as cup void cavity or invalid
            cup_void = (fused_cup[b] >= config.cup_threshold) | np.isnan(fused_ilm[b]) | np.isnan(fused_nfl[b])
            out_mask[b, :, cup_void] = 0

            # Suppress pixels below NFL (RPE leakage) or above ILM (vitreous floaters)
            top_bound = fused_ilm[b] - tol
            bot_bound = fused_nfl[b] + tol

            out_of_bounds = (y_grid < top_bound) | (y_grid > bot_bound)
            out_mask[b][out_of_bounds] = 0

        return out_mask

    @staticmethod
    def apply_elliptical_disc_cut(
        mask: np.ndarray,
        zc: int,
        xc: int,
        config: OpticDiscCutConfig = OpticDiscCutConfig()
    ) -> np.ndarray:
        """
        Applies anatomical elliptical cut based on config.mode:
        - 'cup': Preserves the neuroretinal rim; masks reach the optic cup (bounded by cup void and surface clamp).
        - 'disc': Truncates at the outer 1.75mm optic disc margin (legacy baseline behavior).
        - 'none': Bypasses the cut entirely.
        """
        if config.mode in ("cup", "none"):
            return mask

        out_mask = mask.copy()
        n_bscans, _, cols = out_mask.shape
        rad_x_px = config.rad_x_px
        rad_z_slices = config.rad_z_slices

        for z in range(n_bscans):
            dz_val = abs(z - zc)
            if dz_val <= rad_z_slices:
                rx_z = rad_x_px * np.sqrt(max(0.0, 1.0 - (dz_val / rad_z_slices) ** 2))
                x_left = max(0, int(round(xc - rx_z)))
                x_right = min(cols - 1, int(round(xc + rx_z)))
                out_mask[z, :, x_left : x_right + 1] = 0
        return out_mask

    def predict(self, oct_volume: OCTVolume, biplanar_fusion: bool = True) -> VolumePrediction:
        """
        Performs full 3D volumetric segmentation.
        When biplanar_fusion=True, evaluates orthogonal 2.5D stacks along BOTH:
        1. Horizontal B-scan pane (Y-axis slicing)
        2. Vertical A-scan pane (X-axis slicing)
        Fuses predictions to eliminate directional blind spots and peripheral cutoffs.
        """
        horiz_preds = self._infer_horizontal_pane(oct_volume)
        diagnostics: Dict[str, float] = {}

        if biplanar_fusion:
            vert_preds = self._infer_vertical_pane(oct_volume)
            h_probs, h_ilm, h_nfl, _ = horiz_preds
            v_probs, v_ilm, v_nfl, _ = vert_preds
            diagnostics = {
                "biplanar_probability_disagreement": float(np.mean(np.abs(h_probs - v_probs))),
                "biplanar_ilm_disagreement_um": float(np.median(np.abs(h_ilm - v_ilm.T)) * AXIAL_RES_UM),
                "biplanar_nfl_disagreement_um": float(np.median(np.abs(h_nfl - v_nfl.T)) * AXIAL_RES_UM),
            }
            fused_probs, fused_ilm, fused_nfl, fused_cup = self._fuse_predictions(horiz_preds, vert_preds)
        else:
            fused_probs, fused_ilm, fused_nfl, fused_cup = horiz_preds

        # Dense Segmentation & Hybrid Surface-Guided Recovery
        full_mask = (fused_probs > self.config.mask_threshold).astype(np.uint8)
        full_mask = self.recover_thin_dropouts(
            full_mask, fused_ilm, fused_nfl, fused_cup, config=self.config
        )
        full_mask = self.clamp_mask_to_surfaces(
            full_mask, fused_ilm, fused_nfl, fused_cup, config=self.config
        )

        # Optic Disc Size & Anatomical Elliptical Cut
        full_mask = self.apply_elliptical_disc_cut(
            full_mask,
            zc=oct_volume.disc_geometry.zc,
            xc=oct_volume.disc_geometry.xc,
            config=self.disc_cut_config
        )

        return VolumePrediction(
            mask=full_mask,
            ilm_curve=fused_ilm,
            nfl_curve=fused_nfl,
            cup_probs=fused_cup,
            diagnostics=diagnostics,
        )


# ============================================================================
# 4. Metric Evaluation Layer: ClinicalMetricsCalculator
# ============================================================================

class ClinicalMetricsCalculator:
    """
    Computes peripapillary dice, MABE, 95th-percentile error, and BMO cup IoU
    against ground truth and commercial baseline curves.
    """

    def __init__(self, axial_res_um: float = AXIAL_RES_UM, include_volumetric_metrics: bool = False):
        self.axial_res_um = axial_res_um
        self.include_volumetric_metrics = include_volumetric_metrics

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
        valid_nfl = ~np.isnan(gt_nfl) & (gt_nfl > 0) & ~np.isnan(pred_nfl)
        if np.sum(valid_nfl) > 0:
            errors = np.abs(pred_nfl[valid_nfl] - gt_nfl[valid_nfl]) * self.axial_res_um
            mabe = float(np.nanmean(errors))
            p95 = float(np.nanpercentile(errors, 95))
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

        # Check if commercial curves are an unedited duplicate of ground truth (self-comparison tautology)
        is_mirror = False
        if curves_bad is not None and 'NFL' in curves_good and 'NFL' in curves_bad:
            diff = np.nanmean(np.abs(curves_good['NFL'] - curves_bad['NFL']))
            if np.isnan(diff) or diff < 1e-4:
                is_mirror = True

        has_valid_bad = (len(slice_metrics_bad) > 0 and not is_mirror)

        peripapillary_slices = np.array([
            abs(b_idx - geom.zc) <= int(2.0 * geom.r_disc)
            for b_idx in range(oct_volume.n_bscans)
        ])
        audit_metrics = compute_audit_correction_metrics(
            human_nfl=curves_good['NFL'],
            raw_nfl=curves_bad['NFL'] if (curves_bad is not None and not is_mirror) else None,
            predicted_nfl=prediction.nfl_curve,
            predicted_cup_probability=prediction.cup_probs,
            peripapillary_slices=peripapillary_slices,
            axial_res_um=self.axial_res_um,
        )

        volumetric = {}
        if self.include_volumetric_metrics:
            gt_mask_3d = np.stack(
                [oct_volume.rasterize_curve_slice("good", index) for index in range(oct_volume.n_bscans)]
            )
            metrics_3d = compute_volumetric_metrics(gt_mask_3d, prediction.mask)
            volumetric = {
                "unet_dice_3d": metrics_3d.dice,
                "unet_hd_um": metrics_3d.hd_um,
                "unet_hd95_um": metrics_3d.hd95_um,
                "unet_asd_um": metrics_3d.asd_um,
                "unet_volume_similarity": metrics_3d.volume_similarity,
            }

        return ScanEvaluationResult(
            subject=oct_volume.subject,
            eye=oct_volume.eye,
            cohort=oct_volume.cohort_tag,
            is_validation=oct_volume.is_validation,
            is_mirror=is_mirror,
            unet_dice=float(np.mean([m.dice for m in slice_metrics_unet])),
            unet_mabe=float(np.mean([m.nfl_mabe for m in slice_metrics_unet])),
            unet_p95=float(np.mean([m.nfl_p95 for m in slice_metrics_unet])),
            unet_cup_iou=float(np.mean([m.cup_iou for m in slice_metrics_unet])),
            bad_dice=float(np.mean([m.dice for m in slice_metrics_bad])) if has_valid_bad else None,
            bad_mabe=float(np.mean([m.nfl_mabe for m in slice_metrics_bad])) if has_valid_bad else None,
            bad_p95=float(np.mean([m.nfl_p95 for m in slice_metrics_bad])) if has_valid_bad else None,
            bad_cup_iou=float(np.mean([m.cup_iou for m in slice_metrics_bad])) if has_valid_bad else None,
            **volumetric,
            **audit_metrics,
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

        fig, axs = plt.subplots(1, 2, figsize=(16, 5), facecolor="black", constrained_layout=True)
        for col_idx, (title, mask, color) in enumerate([
            ("Reference Algorithm (Cyan)", mask_cyan, [0.0, 0.8, 1.0]),
            ("Volumetric U-Net (Green)", mask_green, [0.1, 0.95, 0.3])
        ]):
            ax = axs[col_idx]
            ax.imshow(raw_bscan, cmap="gray", aspect="auto")

            overlay = np.zeros((*raw_bscan.shape, 4), dtype=np.float32)
            overlay[mask == 1] = [*color, 0.16]
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
        plt.savefig(out_path, dpi=160, bbox_inches="tight", pad_inches=0.16, facecolor="black")
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

        has_distinct_red = (
            mask_red is not None
            and np.any(mask_red > 0)
            and not np.array_equal(mask_red, mask_cyan)
        )

        fig = plt.figure(figsize=(20, 12), facecolor="black", constrained_layout=True)
        gs = fig.add_gridspec(2, 6)

        def _draw_bscan(ax, title, mask, color, is_ref_arm=False):
            ax.imshow(raw_bscan, cmap="gray", aspect="auto")
            if mask is not None:
                overlay = np.zeros((*raw_bscan.shape, 4), dtype=np.float32)
                overlay[mask == 1] = [*color, 0.16]
                ax.imshow(overlay, aspect="auto")

                contours = scipy.ndimage.binary_dilation(mask) ^ mask
                overlay_c = np.zeros((*raw_bscan.shape, 4), dtype=np.float32)
                overlay_c[contours] = [*color, 1.0]
                ax.imshow(overlay_c, aspect="auto")

            ax.set_title(f"{title}\nCentral B-scan {zc}", color="white", fontsize=11, fontweight="bold")
            ax.set_ylim(440, 180)
            if is_ref_arm:
                ax.add_patch(Rectangle((60, 220), 80, 150, fill=False, edgecolor="#f8fafc", linewidth=1.4, linestyle="--"))
                ax.add_patch(Rectangle((180, 220), 80, 150, fill=False, edgecolor="#f8fafc", linewidth=1.4, linestyle="--"))
            ax.axis("off")

        if has_distinct_red:
            ax_cyan = fig.add_subplot(gs[0, 0:2])
            ax_red = fig.add_subplot(gs[0, 2:4])
            ax_green = fig.add_subplot(gs[0, 4:6])

            _draw_bscan(ax_cyan, "Reference Algorithm (Cyan)", mask_cyan, [0.0, 0.8, 1.0], is_ref_arm=True)
            _draw_bscan(ax_red, "Commercial Solix (Red)", mask_red, [1.0, 0.2, 0.2], is_ref_arm=False)
            _draw_bscan(ax_green, "Volumetric U-Net (Green)", mask_green, [0.1, 0.95, 0.3], is_ref_arm=False)
        else:
            # Commercial red mask is not present or identical to reference; expand blue and green across Row 0
            ax_cyan = fig.add_subplot(gs[0, 0:3])
            ax_green = fig.add_subplot(gs[0, 3:6])

            _draw_bscan(ax_cyan, "Reference Algorithm (Cyan) [Commercial Accepted]", mask_cyan, [0.0, 0.8, 1.0], is_ref_arm=True)
            _draw_bscan(ax_green, "Volumetric U-Net (Green)", mask_green, [0.1, 0.95, 0.3], is_ref_arm=False)

        # Row 1, Col 0: Nasal Rim Zoom
        ax_left = fig.add_subplot(gs[1, 0:2])
        ax_left.imshow(raw_bscan, cmap="gray", aspect="auto")
        over_l = np.zeros((*raw_bscan.shape, 4), dtype=np.float32)
        over_l[mask_cyan == 1] = [0.0, 0.8, 1.0, 0.16]
        over_l[mask_green == 1] = [0.1, 0.95, 0.3, 0.20]
        ax_left.imshow(over_l, aspect="auto")
        ax_left.set_xlim(60, 140)
        ax_left.set_ylim(370, 220)
        ax_left.set_title("Nasal Rim Zoom\n[Cyan: Ref vs Green: U-Net]", color="white", fontsize=11, fontweight="bold")
        ax_left.axis("off")

        # Row 1, Col 1: Temporal Rim Zoom
        ax_right = fig.add_subplot(gs[1, 2:4])
        ax_right.imshow(raw_bscan, cmap="gray", aspect="auto")
        over_r = np.zeros((*raw_bscan.shape, 4), dtype=np.float32)
        over_r[mask_cyan == 1] = [0.0, 0.8, 1.0, 0.16]
        over_r[mask_green == 1] = [0.1, 0.95, 0.3, 0.20]
        ax_right.imshow(over_r, aspect="auto")
        ax_right.set_xlim(180, 260)
        ax_right.set_ylim(370, 220)
        ax_right.set_title("Temporal Rim Zoom\n[Cyan: Ref vs Green: U-Net]", color="white", fontsize=11, fontweight="bold")
        ax_right.axis("off")

        # Row 1, Col 2: En Face Mid-Rim
        ax_ef = fig.add_subplot(gs[1, 4:6])
        ax_ef.imshow(enface_raw, cmap="gray", aspect="auto")
        over_ef = np.zeros((*enface_raw.shape, 4), dtype=np.float32)
        over_ef[enface_cyan == 1] = [0.0, 0.8, 1.0, 0.16]
        over_ef[enface_green == 1] = [0.1, 0.95, 0.3, 0.20]
        ax_ef.imshow(over_ef, aspect="auto")
        ax_ef.set_title(f"En Face Mid-Rim Plane (y={y_enface})\n[Cyan: Ref vs Green: U-Net]", color="white", fontsize=11, fontweight="bold")
        ax_ef.axis("off")

        filename = f"deep_dive_{oct_volume.subject}_{oct_volume.eye}.png"
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=160, bbox_inches="tight", pad_inches=0.18, facecolor="black")
        plt.close(fig)
        return filename

    def render_cohort_summary_chart(self, scan_results: List[ScanEvaluationResult], theme: str = "dark") -> str:
        """Renders the executive summary chart with orange validation highlights."""
        from build_cohort_report import render_cohort_summary_chart
        suffix = "_light.png" if theme == "light" else ".png"
        out_path = os.path.join(self.output_dir, f"cohort_summary_chart{suffix}")
        return render_cohort_summary_chart([r.to_dict() for r in scan_results], out_path, theme=theme)

    def render_statistical_raincloud_chart(self, scan_results: List[ScanEvaluationResult], theme: str = "dark") -> str:
        """Renders publication Figure 1: Raincloud distributions."""
        from build_cohort_report import render_statistical_raincloud_chart
        suffix = "_light.png" if theme == "light" else ".png"
        out_path = os.path.join(self.output_dir, f"cohort_raincloud_distributions{suffix}")
        return render_statistical_raincloud_chart([r.to_dict() for r in scan_results], out_path, theme=theme)

    def render_complete_scan_forest_chart(self, scan_results: List[ScanEvaluationResult], theme: str = "dark") -> str:
        """Renders publication Figure 2: Complete 46-scan ranked forest chart."""
        from build_cohort_report import render_complete_scan_forest_chart
        suffix = "_light.png" if theme == "light" else ".png"
        out_path = os.path.join(self.output_dir, f"cohort_per_scan_forest_plot{suffix}")
        return render_complete_scan_forest_chart([r.to_dict() for r in scan_results], out_path, theme=theme)

    def render_baseline_comparison_chart(self, scan_results: List[ScanEvaluationResult], theme: str = "dark") -> Optional[str]:
        """Renders publication Figure 3: edit-focused audit-correction analysis."""
        from build_cohort_report import render_baseline_comparison_chart
        suffix = "_light.png" if theme == "light" else ".png"
        out_path = os.path.join(self.output_dir, f"baseline_vs_unet_head_to_head{suffix}")
        return render_baseline_comparison_chart([r.to_dict() for r in scan_results], out_path, theme=theme)


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
        device: str = "mps",
        batch_size: int = 8,
        biplanar_fusion: bool = True,
        os_orientation_mode: str = "corrected",
        split_manifest: Optional[str] = None,
        subject_filter: Optional[List[str]] = None,
        eye_filter: Optional[List[str]] = None,
        disc_cut_mode: str = "cup",
        include_volumetric_metrics: bool = False,
        model_family: str = "auto",
        val_stride: int = 32,
    ):
        self.dataset_root = dataset_root
        self.output_dir = output_dir
        self.val_subjects = val_subjects
        self.batch_size = batch_size
        self.biplanar_fusion = biplanar_fusion
        self.os_orientation_mode = validate_orientation_mode(os_orientation_mode)
        self.split_map = load_split_manifest(split_manifest) if split_manifest else {}
        self.subject_filter = set(subject_filter or [])
        self.eye_filter = {eye.upper() for eye in (eye_filter or [])}
        self.disc_cut_mode = disc_cut_mode
        self.include_volumetric_metrics = include_volumetric_metrics
        self.model_family = model_family
        self.val_stride = val_stride

        dev_str = device if ("mps" in device and torch.backends.mps.is_available()) or "cuda" in device else "cpu"
        self.device = torch.device(dev_str)

        self.assets_dir = os.path.join(self.output_dir, "assets", "executive_cohort_report")
        os.makedirs(self.assets_dir, exist_ok=True)

        # Auto-detect or select model family
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        is_3d = False
        if model_family == "3d":
            is_3d = True
        elif model_family == "2.5d":
            is_3d = False
        else:
            if isinstance(ckpt, dict):
                if "config" in ckpt and hasattr(ckpt["config"], "strides"):
                    is_3d = True
                elif "model_state" in ckpt and any("encoder0" in k for k in ckpt["model_state"].keys()):
                    is_3d = True

        if is_3d:
            from predictor_3d import Anisotropic3DPredictor
            print(
                f"[Cohort Pipeline] Initializing Anisotropic3DPredictor on {self.device} "
                f"(stride={self.val_stride}, os_orientation_mode={self.os_orientation_mode}, "
                f"disc_cut_mode={self.disc_cut_mode})..."
            )
            self.predictor = Anisotropic3DPredictor.from_checkpoint(
                checkpoint_path,
                self.device,
                batch_size=self.batch_size,
                stride=(self.val_stride, self.val_stride),
                disc_cut_config=OpticDiscCutConfig(mode=self.disc_cut_mode),
                os_orientation_mode=self.os_orientation_mode,
            )
        else:
            print(
                f"[Cohort Pipeline] Initializing VolumetricRNFLPredictor on {self.device} "
                f"(batch_size={self.batch_size}, biplanar_fusion={self.biplanar_fusion}, "
                f"os_orientation_mode={self.os_orientation_mode}, disc_cut_mode={self.disc_cut_mode})..."
            )
            self.predictor = VolumetricRNFLPredictor.from_checkpoint(
                checkpoint_path,
                self.device,
                batch_size=self.batch_size,
                os_orientation_mode=self.os_orientation_mode,
                disc_cut_config=OpticDiscCutConfig(mode=self.disc_cut_mode),
            )
        self.metrics_calc = ClinicalMetricsCalculator(
            axial_res_um=AXIAL_RES_UM,
            include_volumetric_metrics=self.include_volumetric_metrics,
        )
        self.visualizer = CohortVisualizer(output_dir=self.assets_dir)

    def discover_volumes(self) -> List[OCTVolume]:
        """Scans dataset root and pairs DICOM volumes with matching XML curves."""
        dicom_dir = os.path.join(self.dataset_root, "dicom")
        tsv_good_dir = os.path.join(self.dataset_root, "tsv", "good")
        tsv_bad_dir = os.path.join(self.dataset_root, "tsv", "bad")

        all_subjs = sorted([s for s in os.listdir(dicom_dir) if os.path.isdir(os.path.join(dicom_dir, s))])
        volumes: List[OCTVolume] = []

        for subj in all_subjs:
            if self.subject_filter and subj not in self.subject_filter:
                continue
            if self.split_map and subj not in self.split_map:
                continue
            split = self.split_map.get(subj)
            is_val = is_evaluation_split(split) if split is not None else subj in self.val_subjects
            cohort_tag = split.replace("_", " ").title() if split is not None else None
            subj_dcms = sorted(glob.glob(os.path.join(dicom_dir, subj, "*Disc Cube*_OPT.dcm")))
            for dcm_path in subj_dcms:
                fname = os.path.basename(dcm_path)
                eye = "OS" if "_OS_" in fname else "OD"
                if self.eye_filter and eye not in self.eye_filter:
                    continue

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
                    is_validation=is_val,
                    cohort_tag=cohort_tag,
                )
                volumes.append(vol)

        return volumes

    @staticmethod
    def _find_matching_curve_xml(tsv_dir: str, subj: str, eye: str, dcm_fname: str, protocol: str = "Disc Cube") -> Optional[str]:
        return find_matching_curve_xml(tsv_dir, subj, eye, dcm_fname, protocol=protocol)

    def _evaluate_single_volume(
        self,
        vol: OCTVolume,
        seen_gallery_subjects: List[str]
    ) -> Tuple[ScanEvaluationResult, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Evaluates a single OCT volume and creates gallery/deep-dive entries if applicable."""
        print(f"\n---> Evaluating {vol.subject} ({vol.eye}) [{vol.cohort_tag}]...")

        # 1. Volumetric Inference
        prediction = self.predictor.predict(vol, biplanar_fusion=self.biplanar_fusion)

        # 2. Metric Computation
        result = self.metrics_calc.evaluate_scan(vol, prediction)
        quality = assess_prediction(prediction, evaluation=result)
        result.qc_status = quality.status
        result.qc_flags = quality.flags
        result.qc_metrics = quality.metrics

        print(f"[{vol.subject} {vol.eye}] U-Net Dice: {result.unet_dice:.4f} | MABE: {result.unet_mabe:.2f} µm | P95: {result.unet_p95:.2f} µm | Cup IoU: {result.unet_cup_iou:.4f}")
        print(f"[{vol.subject} {vol.eye}] QC: {result.qc_status} | Flags: {', '.join(result.qc_flags) if result.qc_flags else 'none'}")
        if result.bad_dice is not None:
            print(f"[{vol.subject} {vol.eye}] Bad   Dice: {result.bad_dice:.4f} | MABE: {result.bad_mabe:.2f} µm | P95: {result.bad_p95:.2f} µm | Cup IoU: {result.bad_cup_iou:.4f}")

        # 3. Visuals for OD acquisitions or unseen subjects
        gallery_entry = None
        deep_dive_entry = None
        if vol.eye == "OD" or vol.subject not in seen_gallery_subjects:
            g_file = self.visualizer.render_gallery_panel(vol, prediction)
            gallery_entry = {
                'subject': vol.subject,
                'eye': vol.eye,
                'cohort': vol.cohort_tag,
                'filename': g_file,
                'dice': result.unet_dice,
                'mabe': result.unet_mabe
            }

            if vol.is_validation or vol.subject in ["BEH0181", "BEH0174"]:
                dd_file = self.visualizer.render_deep_dive_panel(vol, prediction)
                deep_dive_entry = {
                    'subject': vol.subject,
                    'eye': vol.eye,
                    'cohort': vol.cohort_tag,
                    'filename': dd_file
                }

        return result, gallery_entry, deep_dive_entry

    def _export_reports(
        self,
        cohort_results: List[ScanEvaluationResult],
        gallery_manifest: List[Dict[str, Any]],
        deep_dive_manifest: List[Dict[str, Any]]
    ) -> str:
        """Renders publication summary charts (dual-theme) and exports JSON metrics manifest."""
        print("\n[Cohort Pipeline] Generating Publication Cohort Summary Charts (Dual-Theme)...")
        for theme in ("dark", "light"):
            self.visualizer.render_cohort_summary_chart(cohort_results, theme=theme)
            self.visualizer.render_statistical_raincloud_chart(cohort_results, theme=theme)
            self.visualizer.render_complete_scan_forest_chart(cohort_results, theme=theme)
            self.visualizer.render_baseline_comparison_chart(cohort_results, theme=theme)
        print("[Cohort Pipeline] Summary charts saved in both dark (.png) and light (_light.png) themes.")

        json_path = os.path.join(self.assets_dir, "cohort_evaluation_metrics.json")
        with open(json_path, "w") as f:
            json.dump({
                'metadata': {
                    'orientation_mode': self.os_orientation_mode,
                    'biplanar_fusion': self.biplanar_fusion,
                    'scan_count': len(cohort_results),
                    'validation_subjects': sorted(self.val_subjects),
                    'qc_threshold_profile': 'Operational engineering defaults; thresholds are not clinically validated.',
                    'audit_edit_threshold_px': AUDIT_EDIT_THRESHOLD_PX,
                    'audit_unchanged_preservation_tolerance_px': PRESERVATION_TOLERANCE_PX,
                },
                'scans': [r.to_dict() for r in cohort_results],
                'gallery': gallery_manifest,
                'deep_dives': deep_dive_manifest
            }, f, indent=2)
        print(f"[Cohort Pipeline] Metrics JSON saved to: {json_path}")

        paired_csv = os.path.join(self.assets_dir, "commercial_pairing_audit.csv")
        with open(paired_csv, "w", newline="") as f:
            fields = [
                "subject", "eye", "cohort", "pairing_status", "unet_dice", "commercial_dice", "dice_delta",
                "unet_mabe_um", "commercial_mabe_um", "mabe_delta_um", "unet_cup_iou", "commercial_cup_iou", "cup_iou_delta",
            ]
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for result in cohort_results:
                if result.is_mirror:
                    status = "excluded_unedited_mirror"
                elif result.bad_dice is None:
                    status = "missing_commercial_annotation"
                else:
                    status = "paired"
                writer.writerow({
                    "subject": result.subject,
                    "eye": result.eye,
                    "cohort": result.cohort,
                    "pairing_status": status,
                    "unet_dice": f"{result.unet_dice:.6f}",
                    "commercial_dice": "" if result.bad_dice is None else f"{result.bad_dice:.6f}",
                    "dice_delta": "" if result.bad_dice is None else f"{result.unet_dice - result.bad_dice:.6f}",
                    "unet_mabe_um": f"{result.unet_mabe:.6f}",
                    "commercial_mabe_um": "" if result.bad_mabe is None or not np.isfinite(result.bad_mabe) else f"{result.bad_mabe:.6f}",
                    "mabe_delta_um": "" if result.bad_mabe is None or not np.isfinite(result.bad_mabe) else f"{result.unet_mabe - result.bad_mabe:.6f}",
                    "unet_cup_iou": f"{result.unet_cup_iou:.6f}",
                    "commercial_cup_iou": "" if result.bad_cup_iou is None else f"{result.bad_cup_iou:.6f}",
                    "cup_iou_delta": "" if result.bad_cup_iou is None else f"{result.unet_cup_iou - result.bad_cup_iou:.6f}",
                })
        print(f"[Cohort Pipeline] Commercial pairing audit saved to: {paired_csv}")

        correction_csv = os.path.join(self.assets_dir, "audit_correction_analysis.csv")
        with open(correction_csv, "w", newline="") as f:
            fields = [
                "subject", "eye", "cohort", "pairing_status", "audit_edit_threshold_px",
                "audit_edited_columns", "audit_unchanged_columns", "audit_edit_fraction",
                "raw_edit_mabe_um", "unet_edit_mabe_um", "delta_edit_mabe_um",
                "audit_correction_gain", "audit_symmetric_gain", "audit_ratio_gain",
                "audit_edit_recovery_rate", "unet_unchanged_mabe_um",
                "audit_unchanged_preservation_rate", "unet_cup_presence_recall",
                "unet_cup_edge_valid_slices", "unet_cup_left_edge_mae_um",
                "unet_cup_right_edge_mae_um", "unet_cup_width_mae_um",
                "raw_cup_edge_valid_slices", "raw_cup_left_edge_mae_um",
                "raw_cup_right_edge_mae_um", "raw_cup_width_mae_um",
            ]
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for result in cohort_results:
                if result.is_mirror:
                    status = "unedited_mirror"
                elif result.bad_dice is None:
                    status = "missing_raw_annotation"
                elif result.audit_edited_columns >= 50:
                    status = "material_edits"
                else:
                    status = "no_material_edits"
                row = {field: getattr(result, field, None) for field in fields if field != "pairing_status"}
                row["pairing_status"] = status
                writer.writerow(row)
        print(f"[Cohort Pipeline] Audit-correction analysis saved to: {correction_csv}")

        review_rows = [r for r in cohort_results if r.qc_status == "manual_review"]
        review_json = os.path.join(self.assets_dir, "manual_review_queue.json")
        with open(review_json, "w") as f:
            json.dump({
                "notice": "Operational engineering flags; thresholds are not clinically validated.",
                "orientation_mode": self.os_orientation_mode,
                "total_scans": len(cohort_results),
                "review_count": len(review_rows),
                "scans": [r.to_dict() for r in review_rows],
            }, f, indent=2)

        review_csv = os.path.join(self.assets_dir, "manual_review_queue.csv")
        with open(review_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["subject", "eye", "cohort", "qc_status", "qc_flags", "dice", "mabe_um", "cup_iou"],
                lineterminator="\n",
            )
            writer.writeheader()
            for result in review_rows:
                writer.writerow({
                    "subject": result.subject,
                    "eye": result.eye,
                    "cohort": result.cohort,
                    "qc_status": result.qc_status,
                    "qc_flags": ";".join(result.qc_flags),
                    "dice": f"{result.unet_dice:.6f}",
                    "mabe_um": f"{result.unet_mabe:.6f}",
                    "cup_iou": f"{result.unet_cup_iou:.6f}",
                })
        print(f"[Cohort Pipeline] Manual-review queue: {len(review_rows)}/{len(cohort_results)} scans -> {review_json}")
        return json_path

    def run(self) -> List[ScanEvaluationResult]:
        """Executes full cohort evaluation across all discovered patient volumes."""
        volumes = self.discover_volumes()
        print(f"[Cohort Pipeline] Discovered {len(volumes)} valid volumes across cohort.")

        cohort_results: List[ScanEvaluationResult] = []
        gallery_manifest: List[Dict[str, Any]] = []
        deep_dive_manifest: List[Dict[str, Any]] = []

        for vol in volumes:
            seen_subjs = [g['subject'] for g in gallery_manifest]
            result, g_entry, dd_entry = self._evaluate_single_volume(vol, seen_subjs)
            cohort_results.append(result)
            if g_entry:
                gallery_manifest.append(g_entry)
            if dd_entry:
                deep_dive_manifest.append(dd_entry)

        self._export_reports(cohort_results, gallery_manifest, deep_dive_manifest)
        return cohort_results


# ============================================================================
# 7. CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Object-Oriented Volumetric RNFL Cohort Evaluator")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--dataset_root", type=str, default="/Users/nikhilmundhra/Library/CloudStorage/Box-Box/OCT_Segmentations_Solix/deidentified-new")
    parser.add_argument("--output_dir", type=str, default="/Users/nikhilmundhra/Documents/Github/Capstone/OCT-Analyser-Capstone/docs")
    parser.add_argument(
        "--val_subjects",
        type=str,
        default=None,
        help="Optional comma-separated override. Defaults to the validation subjects stored in the checkpoint.",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for volumetric inference")
    parser.add_argument("--disable_biplanar", dest="biplanar_fusion", action="store_false", help="Disable biplanar fusion")
    parser.set_defaults(biplanar_fusion=True)
    parser.add_argument(
        "--os_orientation_mode",
        choices=OS_ORIENTATION_MODES,
        default="corrected",
        help="OS coordinate policy. legacy_vertical_mirror is retained only for controlled regression experiments.",
    )
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument(
        "--disc_cut_mode",
        choices=["cup", "disc", "none"],
        default="cup",
        help="Optic nerve head boundary strategy: 'cup' allows masks to reach the optic cup down the neuroretinal rim; 'disc' truncates at fixed 1.75mm optic disc; 'none' bypasses.",
    )
    parser.add_argument("--split_manifest", type=str, default=None, help="Optional JSON/CSV subject-to-split manifest. Subjects absent from the manifest are excluded.")
    parser.add_argument("--subjects", type=str, default=None, help="Optional comma-separated subject filter for targeted or external evaluation.")
    parser.add_argument("--eyes", type=str, default=None, help="Optional comma-separated eye filter (OD,OS).")
    parser.add_argument(
        "--enable_volumetric_metrics",
        action="store_true",
        help="Compute CPU-intensive full-volume HD/HD95/ASD after inference.",
    )
    parser.add_argument(
        "--model_family",
        choices=["auto", "2.5d", "3d"],
        default="auto",
        help="Model architecture family: 'auto' (detect from checkpoint), '2.5d' (VolumetricRNFLNet), or '3d' (AnisotropicRNFLUNet3D).",
    )
    parser.add_argument(
        "--val_stride",
        type=int,
        default=32,
        help="Sliding-window stride along slow_z and fast_x for 3D model inference (default: 32 for 50%% Hann overlap).",
    )
    args = parser.parse_args()

    val_subjects = resolve_validation_subjects(
        args.checkpoint,
        args.val_subjects,
        split_manifest=args.split_manifest,
    )
    print(f"[Cohort Pipeline] Validation subjects: {', '.join(val_subjects)}")
    subject_filter = [s.strip() for s in args.subjects.split(",") if s.strip()] if args.subjects else None
    eye_filter = [eye.strip().upper() for eye in args.eyes.split(",") if eye.strip()] if args.eyes else None

    pipeline = CohortEvaluatorPipeline(
        checkpoint_path=args.checkpoint,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        val_subjects=val_subjects,
        device=args.device,
        batch_size=args.batch_size,
        biplanar_fusion=args.biplanar_fusion,
        os_orientation_mode=args.os_orientation_mode,
        split_manifest=args.split_manifest,
        subject_filter=subject_filter,
        eye_filter=eye_filter,
        disc_cut_mode=args.disc_cut_mode,
        include_volumetric_metrics=args.enable_volumetric_metrics,
        model_family=args.model_family,
        val_stride=args.val_stride,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
