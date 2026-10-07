"""Full-cube RNFL thickness differences for paired audited OCT scans."""

from pathlib import Path

import numpy as np


def compute_paired_thickness_errors(good, commercial, prediction, axial_um):
    """Return signed micrometre errors on one shared, reference-valid en face grid."""
    arrays = [
        np.asarray(good[name], dtype=np.float32) for name in ("ILM", "NFL")
    ] + [
        np.asarray(commercial[name], dtype=np.float32) for name in ("ILM", "NFL")
    ] + [
        np.asarray(prediction.ilm_curve, dtype=np.float32),
        np.asarray(prediction.nfl_curve, dtype=np.float32),
    ]
    if len({array.shape for array in arrays}) != 1 or arrays[0].ndim != 2:
        raise ValueError("All ILM and NFL surfaces must share one two-dimensional cube grid")

    good_ilm, good_nfl, raw_ilm, raw_nfl, pred_ilm, pred_nfl = arrays
    reference_valid = (
        np.isfinite(good_ilm) & np.isfinite(good_nfl) & (good_nfl > good_ilm)
    )
    valid = (
        reference_valid
        & np.isfinite(raw_ilm) & np.isfinite(raw_nfl)
        & np.isfinite(pred_ilm) & np.isfinite(pred_nfl)
    )
    if not np.any(valid):
        raise ValueError("No common valid columns for the paired thickness comparison")

    reference_thickness = (good_nfl - good_ilm) * axial_um
    unet_error = ((pred_nfl - pred_ilm) * axial_um - reference_thickness).astype(np.float32)
    commercial_error = ((raw_nfl - raw_ilm) * axial_um - reference_thickness).astype(np.float32)
    unet_error[~valid] = np.nan
    commercial_error[~valid] = np.nan
    return unet_error, commercial_error, valid, int(reference_valid.sum())


def summarize_signed_error(error):
    values = np.asarray(error)[np.isfinite(error)]
    if not len(values):
        raise ValueError("Cannot summarize an empty thickness error map")
    return {
        "mean_um": float(np.mean(values)),
        "std_um": float(np.std(values)),
        "mae_um": float(np.mean(np.abs(values))),
        "p95_abs_um": float(np.percentile(np.abs(values), 95)),
    }


def shared_color_limit(map_paths):
    """One robust symmetric limit for every pair in a report."""
    values = []
    for path in map_paths:
        with np.load(path) as maps:
            for key in ("unet_error_um", "commercial_error_um"):
                error = maps[key]
                values.append(np.abs(error[np.isfinite(error)]))
    if not values:
        raise ValueError("No thickness error maps were supplied")
    p98 = float(np.percentile(np.concatenate(values), 98))
    return float(max(10, 5 * np.ceil(p98 / 5)))


def render_signed_error_map(error, destination, title, color_limit_um):
    """Render one full-grid en face image without cropping invalid or cup columns."""
    import matplotlib.pyplot as plt

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("RdBu_r").with_extremes(bad="#d1d5db")
    fig, ax = plt.subplots(figsize=(5.2, 5.4), facecolor="white", constrained_layout=True)
    image = ax.imshow(
        np.ma.masked_invalid(error),
        cmap=cmap,
        vmin=-color_limit_um,
        vmax=color_limit_um,
        interpolation="nearest",
        origin="upper",
    )
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlabel("A-scan column")
    ax.set_ylabel("B-scan index")
    fig.colorbar(image, ax=ax, label="Thickness difference (µm)", extend="both", shrink=0.8)
    fig.savefig(destination, dpi=170, facecolor="white")
    plt.close(fig)
    return destination.name
