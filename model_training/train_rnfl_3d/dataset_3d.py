"""Full-axial transverse-patch dataset for dense 3D RNFL training."""

import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


VOLUMETRIC_DIR = Path(__file__).resolve().parents[1] / "train_rnfl_volumetric"
if str(VOLUMETRIC_DIR) not in sys.path:
    sys.path.insert(0, str(VOLUMETRIC_DIR))

from dataset import find_dicom_pixel_offset, find_matching_curve_xml, load_curves


def extract_acquisition_time(filename: str) -> str:
    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})", filename)
    if m:
        return f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}"
    return ""


class SolixRNFL3DPatchDataset(Dataset):
    """Sample full-axial patches only along the slow and fast transverse axes."""

    def __init__(
        self,
        dataset_root: str,
        subjects: Optional[Sequence[str]] = None,
        manifest_path: Optional[str] = None,
        patch_shape: Tuple[int, int, int] = (64, 768, 64),
        patches_per_volume: int = 32,
        peripapillary_probability: float = 0.75,
        protocol: str = "Disc Cube",
        standardize_eye: bool = True,
        augment: bool = False,
    ):
        if patch_shape[1] != 768:
            raise ValueError("The Phase 1 reference dataset requires full axial depth (H=768)")
        if patch_shape[0] > 320 or patch_shape[2] > 320:
            raise ValueError("Transverse patch dimensions cannot exceed the 320x320 volume")
        if patches_per_volume <= 0:
            raise ValueError("patches_per_volume must be positive")
        if not 0.0 <= peripapillary_probability <= 1.0:
            raise ValueError("peripapillary_probability must lie in [0, 1]")

        self.dataset_root = dataset_root
        self.subjects = list(subjects) if subjects is not None else None
        self.manifest_path = manifest_path
        self.patch_shape = patch_shape
        self.patches_per_volume = patches_per_volume
        self.peripapillary_probability = peripapillary_probability
        self.protocol = protocol
        self.standardize_eye = standardize_eye
        self.augment = augment
        self.scans = self._discover_scans()
        if not self.scans:
            raise ValueError("No matched Disc Cube DICOM/XML pairs were found")

    def _discover_scans(self):
        scans = []
        target_subjects = self.subjects

        if self.manifest_path and os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            scan_records = payload.get("scans", [])
            if scan_records:
                for rec in scan_records:
                    subj = rec["subject"]
                    if target_subjects is not None and subj not in target_subjects:
                        continue
                    eye = rec["eye"]
                    dcm_filename = rec.get("dcm_filename") or os.path.basename(rec["dcm_path"])
                    dcm_path = os.path.join(self.dataset_root, "dicom", subj, dcm_filename)
                    if not os.path.exists(dcm_path):
                        dcm_path = rec["dcm_path"]
                    if not os.path.exists(dcm_path):
                        continue

                    xml_path = rec.get("good_xml_path")
                    if xml_path and not os.path.exists(xml_path):
                        xml_filename = os.path.basename(xml_path)
                        xml_path = os.path.join(self.dataset_root, "tsv", "good", subj, "curve", xml_filename)
                    if not xml_path or not os.path.exists(xml_path):
                        xml_path = find_matching_curve_xml(
                            os.path.join(self.dataset_root, "tsv", "good"), subj, eye, dcm_filename, self.protocol
                        )
                    if not xml_path or not os.path.exists(xml_path):
                        continue

                    curves = load_curves(xml_path, mask_sentinel=True)
                    pixel_offset = find_dicom_pixel_offset(dcm_path)
                    volume = np.memmap(
                        dcm_path,
                        dtype="<u2",
                        mode="r",
                        offset=pixel_offset,
                        shape=(320, 768, 320),
                    )
                    absent = np.isnan(curves["NFL"])
                    if absent.any():
                        z_center, x_center = np.mean(np.where(absent), axis=1)
                    else:
                        z_center = x_center = 159.5
                    scans.append(
                        {
                            "subject": subj,
                            "eye": eye,
                            "volume": volume,
                            "curves": curves,
                            "disc_center": (float(z_center), float(x_center)),
                            "dcm_path": dcm_path,
                            "xml_path": xml_path,
                        }
                    )
                return scans
            elif "subjects" in payload:
                manifest_subjects = [s["subject"] if isinstance(s, dict) else s for s in payload["subjects"]]
                target_subjects = target_subjects or manifest_subjects

        dcm_root = os.path.join(self.dataset_root, "dicom")
        tsv_good = os.path.join(self.dataset_root, "tsv", "good")
        target_subjects = target_subjects or sorted([d for d in os.listdir(dcm_root) if not d.startswith(".")])

        for subject in target_subjects:
            subj_dcm_dir = os.path.join(dcm_root, subject)
            if not os.path.isdir(subj_dcm_dir):
                continue
            disc_dcms = [
                os.path.join(subj_dcm_dir, f)
                for f in os.listdir(subj_dcm_dir)
                if self.protocol.lower() in f.lower() and f.lower().endswith(".dcm")
            ]
            by_eye: Dict[str, List[str]] = {"OD": [], "OS": []}
            for p in disc_dcms:
                name_upper = os.path.basename(p).upper()
                if "_OS_" in name_upper or name_upper.endswith("_OS.DCM") or " OS " in name_upper:
                    by_eye["OS"].append(p)
                elif "_OD_" in name_upper or name_upper.endswith("_OD.DCM") or " OD " in name_upper:
                    by_eye["OD"].append(p)

            for eye in ["OD", "OS"]:
                candidates = by_eye[eye]
                if not candidates:
                    continue

                def sort_key(p: str):
                    is_opt = 1 if "_opt.dcm" in p.lower() else 0
                    time_str = extract_acquisition_time(os.path.basename(p))
                    return (is_opt, time_str, p)

                canonical_dcm = sorted(candidates, key=sort_key, reverse=True)[0]
                filename = os.path.basename(canonical_dcm)
                xml_path = find_matching_curve_xml(tsv_good, subject, eye, filename, self.protocol)
                if not xml_path or not os.path.exists(xml_path):
                    continue

                curves = load_curves(xml_path, mask_sentinel=True)
                pixel_offset = find_dicom_pixel_offset(canonical_dcm)
                volume = np.memmap(
                    canonical_dcm,
                    dtype="<u2",
                    mode="r",
                    offset=pixel_offset,
                    shape=(320, 768, 320),
                )
                absent = np.isnan(curves["NFL"])
                if absent.any():
                    z_center, x_center = np.mean(np.where(absent), axis=1)
                else:
                    z_center = x_center = 159.5
                scans.append(
                    {
                        "subject": subject,
                        "eye": eye,
                        "volume": volume,
                        "curves": curves,
                        "disc_center": (float(z_center), float(x_center)),
                        "dcm_path": canonical_dcm,
                        "xml_path": xml_path,
                    }
                )
        return scans

    def __len__(self) -> int:
        return len(self.scans) * self.patches_per_volume

    @staticmethod
    def _start_containing(center: float, patch_size: int, full_size: int) -> int:
        minimum = max(0, int(np.ceil(center - patch_size + 1)))
        maximum = min(int(np.floor(center)), full_size - patch_size)
        if minimum > maximum:
            return int(np.clip(round(center - patch_size / 2), 0, full_size - patch_size))
        return int(np.random.randint(minimum, maximum + 1))

    def _sample_origins(self, scan: Dict[str, object]) -> Tuple[int, int]:
        depth, _, width = self.patch_shape
        full_depth, _, full_width = scan["volume"].shape
        if np.random.random() < self.peripapillary_probability:
            z_center, x_center_native = scan["disc_center"]
            x_center = full_width - 1.0 - x_center_native if self.standardize_eye and scan["eye"] == "OS" else x_center_native
            return (
                self._start_containing(z_center, depth, full_depth),
                self._start_containing(x_center, width, full_width),
            )
        return (
            int(np.random.randint(0, full_depth - depth + 1)),
            int(np.random.randint(0, full_width - width + 1)),
        )

    def _extract_patch(self, scan: Dict[str, object], z_start: int, x_start: int) -> Dict[str, object]:
        depth, axial, width = self.patch_shape
        _, _, full_width = scan["volume"].shape
        z_slice = slice(z_start, z_start + depth)
        is_os = self.standardize_eye and scan["eye"] == "OS"
        if is_os:
            native_x_slice = slice(full_width - x_start - width, full_width - x_start)
        else:
            native_x_slice = slice(x_start, x_start + width)

        image = np.asarray(scan["volume"][z_slice, :axial, native_x_slice], dtype=np.float32)
        ilm = np.asarray(scan["curves"]["ILM"][z_slice, native_x_slice], dtype=np.float32)
        nfl = np.asarray(scan["curves"]["NFL"][z_slice, native_x_slice], dtype=np.float32)
        if is_os:
            image = np.flip(image, axis=-1).copy()
            ilm = np.flip(ilm, axis=-1).copy()
            nfl = np.flip(nfl, axis=-1).copy()

        image = np.clip(image / 2560.0, 0.0, 1.0)
        if self.augment:
            scale = np.random.uniform(0.85, 1.15)
            gamma = np.random.uniform(0.90, 1.10)
            image = np.clip(np.power(np.clip(image * scale, 0.0, 1.0), gamma), 0.0, 1.0)
            if np.random.random() > 0.5:
                noise = np.random.normal(0.0, 0.012, image.shape).astype(np.float32)
                image = np.clip(image + noise, 0.0, 1.0)

        ilm_valid = np.isfinite(ilm) & (ilm > 0)
        nfl_valid = np.isfinite(nfl) & (nfl > 0)
        tissue_valid = ilm_valid & nfl_valid & (nfl > ilm)
        y = np.arange(axial, dtype=np.float32)[None, :, None]
        mask = (
            (y >= np.nan_to_num(ilm, nan=-1.0)[:, None, :])
            & (y < np.nan_to_num(nfl, nan=-1.0)[:, None, :])
            & tissue_valid[:, None, :]
        ).astype(np.float32)

        return {
            "image": torch.from_numpy(image).unsqueeze(0),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "ilm_surface": torch.from_numpy(np.nan_to_num(ilm, nan=0.0)),
            "nfl_surface": torch.from_numpy(np.nan_to_num(nfl, nan=0.0)),
            "ilm_valid": torch.from_numpy(ilm_valid),
            "nfl_valid": torch.from_numpy(nfl_valid),
            "nfl_absent": torch.from_numpy(~nfl_valid),
            "subject": scan["subject"],
            "eye": scan["eye"],
            "patch_origin": torch.tensor((z_start, 0, x_start), dtype=torch.int32),
        }

    def __getitem__(self, index: int) -> Dict[str, object]:
        scan = self.scans[index // self.patches_per_volume]
        return self._extract_patch(scan, *self._sample_origins(scan))


def build_ground_truth_mask(curves: Dict[str, np.ndarray], shape: Tuple[int, int, int] = (320, 768, 320)) -> np.ndarray:
    """Build binary RNFL mask (slow_z, axial_y, fast_x) from ILM and NFL curves."""
    depth, axial, width = shape
    ilm = curves["ILM"][:depth, :width]
    nfl = curves["NFL"][:depth, :width]
    ilm_valid = np.isfinite(ilm) & (ilm > 0)
    nfl_valid = np.isfinite(nfl) & (nfl > 0)
    tissue_valid = ilm_valid & nfl_valid & (nfl > ilm)
    y = np.arange(axial, dtype=np.float32)[None, :, None]
    mask = (
        (y >= np.nan_to_num(ilm, nan=-1.0)[:, None, :])
        & (y < np.nan_to_num(nfl, nan=-1.0)[:, None, :])
        & tissue_valid[:, None, :]
    ).astype(np.uint8)
    return mask


def predict_whole_volume_mask(
    model: torch.nn.Module,
    volume: np.ndarray,
    eye: str = "OD",
    patch_shape: Tuple[int, int, int] = (64, 768, 64),
    stride: Tuple[int, int] = (32, 32),
    threshold: float = 0.5,
    device: str = "cuda",
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    standardize_eye: bool = True,
) -> np.ndarray:
    """Run sliding-window 3D inference across full 320x768x320 volume with overlap blending."""
    model.eval()
    depth, axial, width = volume.shape
    patch_d, patch_h, patch_w = patch_shape
    stride_z, stride_x = stride

    is_os = standardize_eye and eye == "OS"

    # Coordinates along slow_z and fast_x
    z_starts = list(range(0, depth - patch_d + 1, stride_z))
    if z_starts[-1] + patch_d < depth:
        z_starts.append(depth - patch_d)
    x_starts = list(range(0, width - patch_w + 1, stride_x))
    if x_starts[-1] + patch_w < width:
        x_starts.append(width - patch_w)

    prob_accum = np.zeros((depth, axial, width), dtype=np.float32)
    count_accum = np.zeros((depth, axial, width), dtype=np.float32)

    # Hann window along transverse axes for smooth boundary transitions
    w_z = np.hanning(patch_d) if stride_z < patch_d else np.ones(patch_d)
    w_x = np.hanning(patch_w) if stride_x < patch_w else np.ones(patch_w)
    weight_patch = (w_z[:, None, None] * w_x[None, None, :]).astype(np.float32)

    autocast_device = "cuda" if "cuda" in str(device) else ("mps" if "mps" in str(device) else "cpu")

    with torch.no_grad():
        for z_start in z_starts:
            for x_start in x_starts:
                z_slice = slice(z_start, z_start + patch_d)
                if is_os:
                    native_x_slice = slice(width - x_start - patch_w, width - x_start)
                else:
                    native_x_slice = slice(x_start, x_start + patch_w)

                patch_raw = np.asarray(volume[z_slice, :patch_h, native_x_slice], dtype=np.float32)
                if is_os:
                    patch_raw = np.flip(patch_raw, axis=-1).copy()

                patch_norm = np.clip(patch_raw / 2560.0, 0.0, 1.0)
                input_tensor = torch.from_numpy(patch_norm).unsqueeze(0).unsqueeze(0).to(device)

                with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
                    preds = model(input_tensor)
                    logits = preds["mask_logits"]
                    probs = torch.sigmoid(logits).squeeze(0).squeeze(0).float().cpu().numpy()

                if is_os:
                    probs = np.flip(probs, axis=-1).copy()
                    native_x_slice_target = slice(width - x_start - patch_w, width - x_start)
                else:
                    native_x_slice_target = slice(x_start, x_start + patch_w)

                prob_accum[z_slice, :patch_h, native_x_slice_target] += probs * weight_patch
                count_accum[z_slice, :patch_h, native_x_slice_target] += weight_patch

    final_probs = prob_accum / np.maximum(count_accum, 1e-6)
    return (final_probs > threshold).astype(np.uint8)
