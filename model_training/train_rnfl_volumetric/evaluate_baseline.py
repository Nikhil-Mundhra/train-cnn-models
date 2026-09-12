"""
evaluate_baseline.py
====================
Computes the quantitative baseline error of the uncorrected Solix machine segmentation ('bad')
against the human clinician-reviewed reference ground truth ('good') across all subjects
in /Users/nikhilmundhra/Library/CloudStorage/Box-Box/deidentified.

Per README Section 5:
- The difference between bad and good is strongly concentrated within 0 to 2 disc radii
  from the optic disc center (peripapillary region).
- Outside 2 disc radii, difference is single-row quantization noise (3.12 um).
- This script measures Mean Absolute Boundary Error (MABE), P95 error, and Max error
  in microns for ILM and NFL (the boundaries of the RNFL layer).
"""

import os
import glob
import re
import numpy as np
import pandas as pd

AXIAL_UM = 3.12367
FAST_UM = 18.7500
SLOW_UM = 18.8100
SENTINEL = 3000
DROP = {'N/A', 'RPE Ref'}

_D = re.compile(rb'<D>\s*(-?\d+)\s*</D>')
_TYPE = re.compile(rb'<Type>\s*([^<]+?)\s*</Type>')
_ARRAY = re.compile(rb'<ARRAY>\s*(\d+)\s*</ARRAY>')
_IMG = re.compile(rb'<IMAGE_Number>\s*(\d+)\s*</IMAGE_Number>')


def load_curves(path, mask_sentinel=True):
    """
    Parses curve XML file into dictionary of 2D numpy arrays (B-scans x A-scans).
    Returns row depths in pixels, or NaN where absent / masked.
    """
    with open(path, 'rb') as fh:
        blob = fh.read()
    types = [t.decode().strip() for t in _TYPE.findall(blob)]
    widths = {int(a) for a in _ARRAY.findall(blob)}
    n_img = len(_IMG.findall(blob))
    assert len(widths) == 1, f"Ragged array lengths in {path}"
    width = widths.pop()
    per = len(types) // n_img
    order = types[:per]

    cube = np.array(_D.findall(blob), dtype=np.int32).reshape(n_img, per, width)
    out = {}
    for j, name in enumerate(order):
        if name in DROP:
            continue
        a = cube[:, j, :].astype(float)
        if mask_sentinel:
            a[a >= SENTINEL] = np.nan
            a[a <= 0] = np.nan
        out[name] = a
    return out


def compute_disc_geometry(nfl_good):
    """
    Derives the optic disc center (zc, xc) and equivalent radius (pixels)
    from the Bruch's Membrane Opening / cup absence mask (NaNs in NFL).
    """
    nan_mask = np.isnan(nfl_good)
    z_coords, x_coords = np.where(nan_mask)
    if len(z_coords) < 100:
        # Fallback to center if disc is very small or unsegmented
        return 160.0, 160.0, 45.0
    zc = float(np.mean(z_coords))
    xc = float(np.mean(x_coords))
    radius = float(np.sqrt(len(z_coords) / np.pi))
    return zc, xc, radius


def evaluate_pair(bad_xml_path, good_xml_path):
    """
    Evaluates one bad vs good pair of Disc Cube XML curves.
    Returns dictionary of metrics for ILM and NFL over full volume and peripapillary zone.
    """
    bad_curves = load_curves(bad_xml_path)
    good_curves = load_curves(good_xml_path)

    zc, xc, r_disc = compute_disc_geometry(good_curves['NFL'])
    nz, nx = good_curves['ILM'].shape

    # 2D distance grid from disc center in pixels
    zz, xx = np.indices((nz, nx))
    dist_map = np.sqrt((zz - zc) ** 2 + (xx - xc) ** 2)

    # Peripapillary mask: within 2 disc radii (excluding the cup itself where NFL is absent)
    peripapillary_mask = dist_map <= (2.0 * r_disc)
    cup_mask = np.isnan(good_curves['NFL'])
    peripapillary_tissue_mask = peripapillary_mask & (~cup_mask)

    results = {
        'zc': zc,
        'xc': xc,
        'r_disc_px': r_disc,
        'r_disc_um': r_disc * FAST_UM,
    }

    for layer in ['ILM', 'NFL']:
        b_layer = bad_curves[layer]
        g_layer = good_curves[layer]

        # Valid in both
        valid = (~np.isnan(b_layer)) & (~np.isnan(g_layer))
        diff_um = np.abs(b_layer - g_layer) * AXIAL_UM

        # Full volume
        v_diff = diff_um[valid]
        results[f'{layer}_full_mean_um'] = float(np.mean(v_diff)) if len(v_diff) > 0 else 0.0
        results[f'{layer}_full_p95_um'] = float(np.percentile(v_diff, 95)) if len(v_diff) > 0 else 0.0
        results[f'{layer}_full_max_um'] = float(np.max(v_diff)) if len(v_diff) > 0 else 0.0

        # Peripapillary tissue region (r <= 2.0 radii)
        peri_valid = valid & peripapillary_tissue_mask
        p_diff = diff_um[peri_valid]
        results[f'{layer}_peri_mean_um'] = float(np.mean(p_diff)) if len(p_diff) > 0 else 0.0
        results[f'{layer}_peri_p95_um'] = float(np.percentile(p_diff, 95)) if len(p_diff) > 0 else 0.0
        results[f'{layer}_peri_max_um'] = float(np.max(p_diff)) if len(p_diff) > 0 else 0.0
        results[f'{layer}_peri_frac_gt10um'] = float(np.mean(p_diff > 10.0)) * 100.0 if len(p_diff) > 0 else 0.0

    # Optic Cup Concordance (Absence detection)
    bad_cup = np.isnan(bad_curves['NFL'])
    good_cup = cup_mask
    intersection = np.logical_and(bad_cup, good_cup).sum()
    union = np.logical_or(bad_cup, good_cup).sum()
    results['cup_iou'] = float(intersection / union) if union > 0 else 1.0

    return results


def run_benchmark(dataset_root="/Users/nikhilmundhra/Library/CloudStorage/Box-Box/deidentified"):
    """
    Runs the full baseline benchmark across all available bad vs good pairs.
    """
    tsv_bad_dir = os.path.join(dataset_root, "tsv", "bad")
    tsv_good_dir = os.path.join(dataset_root, "tsv", "good")

    subjects = sorted([d for d in os.listdir(tsv_bad_dir) if not d.startswith('.')])
    records = []

    print("==========================================================================================")
    print("=== SOLIX MACHINE SEGMENTATION BASELINE EVALUATION ('bad' vs 'good' Ground Truth)     ===")
    print("==========================================================================================")

    for subj in subjects:
        bad_xmls = sorted(glob.glob(os.path.join(tsv_bad_dir, subj, "curve", "*Disc Cube*.xml")))
        for bad_path in bad_xmls:
            fname = os.path.basename(bad_path)
            # Match in good
            good_match = glob.glob(os.path.join(tsv_good_dir, subj, "curve", fname))
            if not good_match:
                eye = "_OD_" if "_OD_" in fname else "_OS_"
                good_match = glob.glob(os.path.join(tsv_good_dir, subj, "curve", f"*{eye}Disc Cube*.xml"))

            if good_match:
                good_path = good_match[0]
                eye = "OD" if "_OD_" in fname else "OS"
                res = evaluate_pair(bad_path, good_path)
                res['subject'] = subj
                res['eye'] = eye
                res['filename'] = fname
                records.append(res)

    df = pd.DataFrame(records)

    # Display Per-Subject Peripapillary Summary
    summary_cols = [
        'subject', 'eye',
        'NFL_peri_mean_um', 'NFL_peri_p95_um', 'NFL_peri_max_um', 'NFL_peri_frac_gt10um',
        'ILM_peri_mean_um', 'ILM_peri_max_um', 'cup_iou'
    ]
    print(df[summary_cols].to_string(index=False))

    print("\n------------------------------------------------------------------------------------------")
    print("=== OVERALL BENCHMARK BASELINE (Target for Machine Learning Model to Beat)             ===")
    print("------------------------------------------------------------------------------------------")
    print(f"Total Scans Evaluated: {len(df)}")
    print(f"Peripapillary NFL MABE:  {df['NFL_peri_mean_um'].mean():.2f} +/- {df['NFL_peri_mean_um'].std():.2f} um")
    print(f"Peripapillary NFL P95:   {df['NFL_peri_p95_um'].mean():.2f} um")
    print(f"Peripapillary NFL Max:   {df['NFL_peri_max_um'].max():.2f} um  (Worst-case Solix error)")
    print(f"A-scans with >10 um err: {df['NFL_peri_frac_gt10um'].mean():.1f}%")
    print(f"Peripapillary ILM MABE:  {df['ILM_peri_mean_um'].mean():.2f} +/- {df['ILM_peri_mean_um'].std():.2f} um")
    print(f"Cup Absence IoU:         {df['cup_iou'].mean():.4f}")
    print("==========================================================================================\n")

    return df


if __name__ == "__main__":
    run_benchmark()
