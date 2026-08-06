"""
tests/test_dicom_loader.py

Unit tests for core_ml.dicom_loader and DICOMVolumeOCTDataset.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "image-classification-model-training"))

from core_ml.dicom_loader import load_dicom_volume, normalize_intensity, resample_volume_isotropic
from data.dataset import DICOMVolumeOCTDataset


def test_normalize_intensity_basic():
    arr = np.array([0, 50, 100, 200, 255], dtype=np.float32)
    norm = normalize_intensity(arr)
    assert norm.min() == 0.0
    assert norm.max() == 1.0
    assert norm.dtype == np.float32


def test_normalize_intensity_constant():
    arr = np.zeros((10, 10, 10), dtype=np.float32)
    norm = normalize_intensity(arr)
    assert norm.shape == (10, 10, 10)
    assert np.all(norm == 0.0)


def test_resample_volume_isotropic():
    vol = np.random.rand(10, 64, 64).astype(np.float32)
    spacing = {"z_spacing": 2.0, "y_spacing": 1.0, "x_spacing": 1.0}
    resampled = resample_volume_isotropic(vol, spacing, target_spacing=1.0)
    # Z dimension should double from 10 to 20
    assert resampled.shape[0] == 20
    assert resampled.shape[1] == 64
    assert resampled.shape[2] == 64


def test_load_dicom_volume_from_image_directory(tmp_path):
    vol_dir = tmp_path / "volume_scan"
    vol_dir.mkdir(parents=True, exist_ok=True)

    # Create 5 slice image files
    for i in range(5):
        img_arr = (np.ones((32, 32)) * (i * 50)).astype(np.uint8)
        img = Image.fromarray(img_arr)
        img.save(vol_dir / f"slice_{i:03d}.png")

    volume, spacing = load_dicom_volume(vol_dir)
    assert volume.shape == (5, 32, 32)
    assert spacing["z_spacing"] == 1.0
    assert spacing["y_spacing"] == 1.0
    assert spacing["x_spacing"] == 1.0


def test_dicom_dataset_2d_slice_mode(tmp_path):
    vol_dir = tmp_path / "vol1"
    vol_dir.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        img_arr = (np.random.rand(16, 16) * 255).astype(np.uint8)
        Image.fromarray(img_arr).save(vol_dir / f"slice_{i:02d}.png")

    dataset = DICOMVolumeOCTDataset(volume_paths=[vol_dir], mode="2d_slice")
    assert len(dataset) == 3

    sample_tensor = dataset[0]
    # Should yield 3-channel 2D tensor: (3, 16, 16)
    assert isinstance(sample_tensor, torch.Tensor)
    assert sample_tensor.shape == (3, 16, 16)
    assert sample_tensor.dtype == torch.float32


def test_dicom_dataset_3d_cube_mode(tmp_path):
    vol_dir = tmp_path / "vol1"
    vol_dir.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        img_arr = (np.random.rand(16, 16) * 255).astype(np.uint8)
        Image.fromarray(img_arr).save(vol_dir / f"slice_{i:02d}.png")

    dataset = DICOMVolumeOCTDataset(volume_paths=[vol_dir], mode="3d_cube")
    assert len(dataset) == 1

    sample_tensor = dataset[0]
    # Should yield 3D tensor: (1, 4, 16, 16)
    assert isinstance(sample_tensor, torch.Tensor)
    assert sample_tensor.shape == (1, 4, 16, 16)
    assert sample_tensor.dtype == torch.float32
