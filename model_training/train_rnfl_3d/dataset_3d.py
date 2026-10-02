"""Full-axial transverse-patch dataset for dense 3D RNFL training."""

import glob
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


VOLUMETRIC_DIR = Path(__file__).resolve().parents[1] / "train_rnfl_volumetric"
if str(VOLUMETRIC_DIR) not in sys.path:
    sys.path.insert(0, str(VOLUMETRIC_DIR))

from dataset import find_dicom_pixel_offset, find_matching_curve_xml, load_curves


class SolixRNFL3DPatchDataset(Dataset):
    """Sample full-axial patches only along the slow and fast transverse axes."""

    def __init__(
        self,
        dataset_root: str,
        subjects: Sequence[str],
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
        self.subjects = list(subjects)
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
        dcm_root = os.path.join(self.dataset_root, "dicom")
        tsv_good = os.path.join(self.dataset_root, "tsv", "good")
        for subject in self.subjects:
            pattern = os.path.join(dcm_root, subject, f"*{self.protocol}*_OPT.dcm")
            for dcm_path in sorted(glob.glob(pattern)):
                filename = os.path.basename(dcm_path)
                eye = "OS" if "_OS_" in filename else "OD"
                xml_path = find_matching_curve_xml(tsv_good, subject, eye, filename, self.protocol)
                if not xml_path:
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
                        "subject": subject,
                        "eye": eye,
                        "volume": volume,
                        "curves": curves,
                        "disc_center": (float(z_center), float(x_center)),
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
