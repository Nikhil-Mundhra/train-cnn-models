"""
core_ml/dicom_loader.py

Utility module for 3D DICOM & Volumetric OCT image processing.
Handles reading DICOM files/directories, intensity normalization, and 3D isotropic voxel resampling.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np
from PIL import Image
from scipy.ndimage import zoom


def load_dicom_volume(path_or_dir: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Loads a 3D volumetric array (Z, Height, Width) and physical spacing dictionary from a DICOM file,
    a folder of DICOM slices, or a directory of 2D image slices.

    Args:
        path_or_dir: Path to a DICOM file (.dcm) or directory of slice images.

    Returns:
        Tuple containing:
            - volume: 3D float32 NumPy array with shape (Z, H, W).
            - spacing: Dict with keys 'z_spacing', 'y_spacing', 'x_spacing' in mm.
    """
    target_path = Path(path_or_dir)
    
    if not target_path.exists():
        raise FileNotFoundError(f"Path does not exist: {target_path}")

    spacing = {"z_spacing": 1.0, "y_spacing": 1.0, "x_spacing": 1.0}

    # Case 1: Single DICOM file (.dcm)
    if target_path.is_file() and target_path.suffix.lower() in [".dcm", ".dicom"]:
        try:
            import pydicom
        except ImportError as err:
            raise ImportError("pydicom is required to parse native DICOM files. Please run 'pip install pydicom'.") from err
        
        ds = pydicom.dcmread(str(target_path))
        volume = ds.pixel_array.astype(np.float32)
        
        # Apply DICOM Rescale Slope & Intercept if specified
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        volume = volume * slope + intercept
        
        # Extract spacing metadata if available
        pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
        slice_thickness = getattr(ds, "SliceThickness", getattr(ds, "SpacingBetweenSlices", 1.0))
        
        spacing = {
            "z_spacing": float(slice_thickness),
            "y_spacing": float(pixel_spacing[0]),
            "x_spacing": float(pixel_spacing[1]),
        }
        
        if volume.ndim == 2:
            volume = np.expand_dims(volume, axis=0)

    # Case 2: Directory of DICOM files or standard slice images
    elif target_path.is_dir():
        files = sorted([f for f in target_path.iterdir() if f.is_file() and not f.name.startswith(".")])
        if not files:
            raise ValueError(f"No valid image or DICOM files found in directory: {target_path}")
        
        dcm_files = [f for f in files if f.suffix.lower() in [".dcm", ".dicom"]]
        if dcm_files:
            try:
                import pydicom
            except ImportError as err:
                raise ImportError("pydicom is required to parse DICOM directories. Please run 'pip install pydicom'.") from err
            
            slices = []
            for f in dcm_files:
                ds = pydicom.dcmread(str(f))
                arr = ds.pixel_array.astype(np.float32)
                slope = float(getattr(ds, "RescaleSlope", 1.0))
                intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                slices.append(arr * slope + intercept)
            volume = np.stack(slices, axis=0)
        else:
            # Standard image stack (.png, .jpg, .tif, .bmp)
            slices = []
            for f in files:
                img = Image.open(f).convert("L")
                slices.append(np.array(img, dtype=np.float32))
            volume = np.stack(slices, axis=0)

    else:
        # Single 2D image file loaded as 1-slice volume
        img = Image.open(target_path).convert("L")
        volume = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

    return volume, spacing


def normalize_intensity(volume: np.ndarray, clip_percentiles: Tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
    """
    Normalizes 3D volume pixel values to [0.0, 1.0] using percentile intensity clipping.
    Handles homogeneous arrays gracefully without division-by-zero errors.

    Args:
        volume: 3D NumPy array.
        clip_percentiles: Tuple of (low_percentile, high_percentile).

    Returns:
        Normalized float32 NumPy array with values in [0.0, 1.0].
    """
    if volume.size == 0:
        return volume.astype(np.float32)

    p_low, p_high = np.percentile(volume, clip_percentiles)
    if p_high <= p_low:
        min_val, max_val = volume.min(), volume.max()
        if max_val > min_val:
            return ((volume - min_val) / (max_val - min_val)).astype(np.float32)
        return np.zeros_like(volume, dtype=np.float32)

    clipped = np.clip(volume, p_low, p_high)
    norm = (clipped - p_low) / (p_high - p_low + 1e-8)
    return norm.astype(np.float32)


def resample_volume_isotropic(
    volume: np.ndarray,
    current_spacing: Dict[str, float],
    target_spacing: float = 1.0
) -> np.ndarray:
    """
    Resamples a 3D volume (Z, H, W) to uniform isotropic voxel spacing (in mm).

    Args:
        volume: 3D NumPy array with shape (Z, H, W).
        current_spacing: Dict with keys 'z_spacing', 'y_spacing', 'x_spacing'.
        target_spacing: Desired isotropic voxel spacing in mm (default: 1.0).

    Returns:
        Resampled 3D float32 NumPy array.
    """
    zoom_factors = (
        current_spacing["z_spacing"] / target_spacing,
        current_spacing["y_spacing"] / target_spacing,
        current_spacing["x_spacing"] / target_spacing,
    )
    if all(abs(f - 1.0) < 1e-3 for f in zoom_factors):
        return volume.astype(np.float32)

    resampled = zoom(volume, zoom=zoom_factors, order=1)
    return resampled.astype(np.float32)
