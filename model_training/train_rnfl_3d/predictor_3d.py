"""
predictor_3d.py
================
Object-Oriented 3D Inference Predictor for Dense Anisotropic 3D U-Net.
Provides drop-in integration with the batch cohort evaluation pipeline,
producing VolumePrediction with continuous subpixel surfaces and cup detection.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
VOLUMETRIC_DIR = SCRIPT_DIR.parent / "train_rnfl_volumetric"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(VOLUMETRIC_DIR) not in sys.path:
    sys.path.insert(0, str(VOLUMETRIC_DIR))

from model_3d import AnisotropicRNFLUNet3D, AnisotropicUNetConfig
from dataset_3d import predict_whole_volume_mask
from orientation import validate_orientation_mode
from batch_cohort_evaluator import VolumePrediction, OpticDiscCutConfig, InferenceConfig, VolumetricRNFLPredictor


class Anisotropic3DPredictor:
    """
    Encapsulates sliding-window 3D volumetric inference, Hann overlap blending,
    and sub-voxel boundary decoding for AnisotropicRNFLUNet3D.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        batch_size: int = 4,
        stride: Tuple[int, int] = (64, 64),
        patch_shape: Tuple[int, int, int] = (64, 768, 64),
        threshold: float = 0.5,
        disc_cut_config: OpticDiscCutConfig = OpticDiscCutConfig(),
        os_orientation_mode: str = "corrected",
        config: InferenceConfig = InferenceConfig(),
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.stride = stride
        self.patch_shape = patch_shape
        self.threshold = threshold
        self.disc_cut_config = disc_cut_config
        self.os_orientation_mode = validate_orientation_mode(os_orientation_mode)
        self.config = config

        self.use_amp = "cuda" in str(device) or "mps" in str(device)
        self.autocast_device = "cuda" if "cuda" in str(device) else ("mps" if "mps" in str(device) else "cpu")
        self.amp_dtype = torch.bfloat16 if ("cuda" in str(device) and torch.cuda.is_bf16_supported()) else torch.float16
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: torch.device,
        batch_size: int = 4,
        stride: Tuple[int, int] = (64, 64),
        patch_shape: Tuple[int, int, int] = (64, 768, 64),
        threshold: float = 0.5,
        disc_cut_config: OpticDiscCutConfig = OpticDiscCutConfig(),
        os_orientation_mode: str = "corrected",
        config: InferenceConfig = InferenceConfig(),
    ) -> "Anisotropic3DPredictor":
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        unet_cfg = ckpt.get("config")
        if unet_cfg is None:
            unet_cfg = AnisotropicUNetConfig()

        model = AnisotropicRNFLUNet3D(unet_cfg).to(device)
        state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(state_dict)
        model.eval()

        return cls(
            model=model,
            device=device,
            batch_size=batch_size,
            stride=stride,
            patch_shape=patch_shape,
            threshold=threshold,
            disc_cut_config=disc_cut_config,
            os_orientation_mode=os_orientation_mode,
            config=config,
        )

    def predict(self, oct_volume: Any, biplanar_fusion: bool = False) -> VolumePrediction:
        """
        Runs sliding-window 3D volumetric inference on the patient volume,
        derives continuous 1D surface boundaries via subpixel interpolation,
        detects optic cup void cavity, and applies disc cut config.
        """
        raw_volume = oct_volume.memmap
        depth, axial, width = raw_volume.shape

        # Sliding window 3D inference with Hann overlap blending
        probs = predict_whole_volume_mask(
            model=self.model,
            volume=raw_volume,
            eye=oct_volume.eye,
            patch_shape=self.patch_shape,
            stride=self.stride,
            threshold=self.threshold,
            device=str(self.device),
            use_amp=self.use_amp,
            amp_dtype=self.amp_dtype,
            standardize_eye=(self.os_orientation_mode == "corrected"),
            return_probs=True,
            batch_size=self.batch_size,
        )

        mask = (probs > self.threshold).astype(np.uint8)

        # Continuous surface boundary extraction (vectorized)
        has_tissue = mask.sum(axis=1) >= self.config.min_layer_thickness
        ilm_curve = np.full((depth, width), np.nan, dtype=np.float32)
        nfl_curve = np.full((depth, width), np.nan, dtype=np.float32)
        cup_probs = (~has_tissue).astype(np.float32)

        if np.any(has_tissue):
            z_idx, x_idx = np.where(has_tissue)
            # Top boundary: first 1 along axial column
            y0 = np.argmax(mask[z_idx, :, x_idx] == 1, axis=1)
            # Bottom boundary: last 1 + 1 along axial column
            rev_first = np.argmax(mask[z_idx, ::-1, x_idx] == 1, axis=1)
            y1 = axial - rev_first

            # Sub-pixel boundary refinement using continuous probability sigmoids
            p_before = probs[z_idx, np.maximum(y0 - 1, 0), x_idx]
            p_at = probs[z_idx, y0, x_idx]
            ilm_sub = np.where(
                y0 > 0,
                (y0 - 1) + (0.5 - p_before) / np.maximum(p_at - p_before, 1e-6),
                y0.astype(np.float32),
            )

            p_in = probs[z_idx, np.maximum(y1 - 1, 0), x_idx]
            p_out = probs[z_idx, np.minimum(y1, axial - 1), x_idx]
            nfl_sub = np.where(
                y1 < axial,
                (y1 - 1) + (p_in - 0.5) / np.maximum(p_in - p_out, 1e-6),
                y1.astype(np.float32),
            )

            ilm_curve[z_idx, x_idx] = ilm_sub
            nfl_curve[z_idx, x_idx] = nfl_sub

            # Interpolate continuous surface across missing/cup columns per B-scan
            x_indices = np.arange(width, dtype=np.float32)
            for z in np.unique(z_idx):
                v_ilm = np.where(~np.isnan(ilm_curve[z]))[0]
                if len(v_ilm) >= 2:
                    ilm_curve[z] = np.interp(x_indices, v_ilm.astype(np.float32), ilm_curve[z, v_ilm])
                v_nfl = np.where(~np.isnan(nfl_curve[z]))[0]
                if len(v_nfl) >= 2:
                    nfl_curve[z] = np.interp(x_indices, v_nfl.astype(np.float32), nfl_curve[z, v_nfl])

        # Elliptical disc cut (if requested, though 'cup' mode preserves rim)
        mask = VolumetricRNFLPredictor.apply_elliptical_disc_cut(
            mask,
            zc=oct_volume.disc_geometry.zc,
            xc=oct_volume.disc_geometry.xc,
            config=self.disc_cut_config,
        )

        diagnostics: Dict[str, float] = {
            "mean_foreground_probability": float(np.mean(probs[mask == 1])) if np.any(mask == 1) else 0.0,
            "total_rnfl_volume_mm3": float(np.sum(mask) * (18.75 * 18.75 * 3.87) / 1e9),
        }

        return VolumePrediction(
            mask=mask,
            ilm_curve=ilm_curve,
            nfl_curve=nfl_curve,
            cup_probs=cup_probs,
            diagnostics=diagnostics,
        )
