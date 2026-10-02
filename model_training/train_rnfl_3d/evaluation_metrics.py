"""Pinned 3D volumetric metric contract for RNFL masks.

The public function deliberately wraps ``seg-metrics`` so callers cannot
accidentally change array order, spacing, connectivity, or empty-mask policy.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Sequence

import numpy as np


SOLIX_SPACING_UM = (18.81, 3.12367, 18.75)  # (slow_z, axial_y, fast_x)


@dataclass(frozen=True)
class VolumetricMetrics:
    dice: float
    hd_um: float
    hd95_um: float
    asd_um: float
    volume_similarity: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _binary_volume(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D array in (slow_z, axial_y, fast_x) order")
    if not np.all(np.isin(array, (0, 1, False, True))):
        raise ValueError(f"{name} must contain only binary values")
    return array.astype(np.uint8, copy=False)


def _scalar(record: Dict[str, object], key: str) -> float:
    value = record[key]
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 1:
        raise ValueError(f"Expected one value for {key}, received {value!r}")
    return float(array[0])


def compute_volumetric_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    spacing_um: Sequence[float] = SOLIX_SPACING_UM,
) -> VolumetricMetrics:
    """Compute whole-volume metrics using seg-metrics 1.2.8.

    Empty-mask policy is explicit: two empty masks are a perfect match; when
    only one mask is empty, overlap/volume similarity are zero and undefined
    surface distances are reported as NaN. Surface extraction is face-connected
    (6-neighbourhood in 3D).
    """
    gt = _binary_volume("ground_truth", ground_truth)
    pred = _binary_volume("prediction", prediction)
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: ground_truth={gt.shape}, prediction={pred.shape}")

    spacing = tuple(float(v) for v in spacing_um)
    if len(spacing) != 3 or any(v <= 0 for v in spacing):
        raise ValueError("spacing_um must contain three positive values")

    gt_nonempty = bool(gt.any())
    pred_nonempty = bool(pred.any())
    if not gt_nonempty and not pred_nonempty:
        return VolumetricMetrics(1.0, 0.0, 0.0, 0.0, 1.0)
    if gt_nonempty != pred_nonempty:
        return VolumetricMetrics(0.0, np.nan, np.nan, np.nan, 0.0)

    try:
        import seg_metrics.seg_metrics as sg
    except ImportError as exc:
        raise RuntimeError(
            "seg-metrics==1.2.8 is required; install requirements-evaluation.txt"
        ) from exc

    output = sg.write_metrics(
        labels=[1],
        gdth_img=gt,
        pred_img=pred,
        csv_file=None,
        spacing=spacing,
        metrics=["dice", "hd", "hd95", "msd"],
        fully_connected=False,
        verbose=False,
    )
    record = output[0] if isinstance(output, list) else output
    gt_volume = int(gt.sum())
    pred_volume = int(pred.sum())
    # seg-metrics 1.2.8 exposes ``vs`` as a signed relative volume
    # difference, despite the public name "volume similarity". Compute the
    # conventional bounded similarity explicitly so 1 means equal volume.
    volume_similarity = 1.0 - abs(pred_volume - gt_volume) / (pred_volume + gt_volume)
    return VolumetricMetrics(
        dice=_scalar(record, "dice"),
        hd_um=_scalar(record, "hd"),
        hd95_um=_scalar(record, "hd95"),
        asd_um=_scalar(record, "msd"),
        volume_similarity=float(volume_similarity),
    )
