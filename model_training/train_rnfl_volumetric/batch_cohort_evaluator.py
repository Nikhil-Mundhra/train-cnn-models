"""
batch_cohort_evaluator.py
==========================
Comprehensive batch evaluator spanning all 11 Solix OCT subjects.
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
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.ndimage

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import find_dicom_pixel_offset, load_curves
from model import VolumetricRNFLNet

AXIAL_RES_UM = 3.09


def extract_disc_geometry(curves, n_bscans=320):
    ilm = curves.get('ILM')
    nfl = curves.get('NFL')
    if ilm is None or nfl is None:
        return 160, 160, 45
    cup_mask = np.isnan(nfl)
    cup_counts = np.sum(cup_mask, axis=1)
    zc = int(np.argmax(cup_counts)) if np.max(cup_counts) > 0 else 160
    cup_cols = np.where(cup_mask[zc])[0]
    xc = int(np.median(cup_cols)) if len(cup_cols) > 0 else 160
    active_slices = np.where(cup_counts > 5)[0]
    r_disc = int(max(len(active_slices) // 2, 25))
    return zc, xc, r_disc


def rasterize_curves_to_mask(ilm_curve, nfl_curve, rows=768, cols=320):
    mask = np.zeros((rows, cols), dtype=np.uint8)
    y_grid = np.arange(rows)[:, None]
    valid = ~np.isnan(ilm_curve) & ~np.isnan(nfl_curve) & (nfl_curve > ilm_curve)
    y0 = np.clip(np.round(np.nan_to_num(ilm_curve, nan=-1)), 0, rows)
    y1 = np.clip(np.round(np.nan_to_num(nfl_curve, nan=-1)), 0, rows)
    mask = ((y_grid >= y0) & (y_grid < y1) & valid).astype(np.uint8)
    return mask


def run_volume_inference(model, memmap, device="cpu", batch_size=8, n_bscans=320, rows=768, cols=320):
    model.eval()
    half_ctx = 2
    full_mask = np.zeros((n_bscans, rows, cols), dtype=np.uint8)
    full_ilm = np.zeros((n_bscans, cols), dtype=np.float32)
    full_nfl = np.zeros((n_bscans, cols), dtype=np.float32)
    full_cup = np.zeros((n_bscans, cols), dtype=np.float32)

    use_amp = "mps" in str(device) or "cuda" in str(device)
    autocast_device = "mps" if "mps" in str(device) else ("cuda" if "cuda" in str(device) else "cpu")

    for start_idx in range(0, n_bscans, batch_size):
        end_idx = min(start_idx + batch_size, n_bscans)
        batch_slices = []
        for b_idx in range(start_idx, end_idx):
            slice_indices = [min(max(b_idx + o, 0), n_bscans - 1) for o in range(-half_ctx, half_ctx + 1)]
            stack = [memmap[s] for s in slice_indices]
            img_stack = np.stack(stack, axis=0).astype(np.float32) / 2560.0
            batch_slices.append(img_stack)

        batch_tensor = torch.from_numpy(np.stack(batch_slices, axis=0)).to(device)

        with torch.no_grad():
            with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=use_amp):
                preds = model(batch_tensor)
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

    return full_mask, full_ilm, full_nfl, full_cup


def compute_metrics_slice(pred_mask, gt_mask, pred_nfl, gt_nfl, pred_cup, gt_cup, is_peripapillary):
    if not is_peripapillary:
        return None

    # Dice
    intersection = np.sum((pred_mask == 1) & (gt_mask == 1))
    total = np.sum(pred_mask == 1) + np.sum(gt_mask == 1)
    dice = (2.0 * intersection / total) if total > 0 else 1.0

    # NFL Boundary error
    valid_nfl = ~np.isnan(gt_nfl) & (gt_nfl > 0)
    if np.sum(valid_nfl) > 0:
        errors = np.abs(pred_nfl[valid_nfl] - gt_nfl[valid_nfl]) * AXIAL_RES_UM
        mabe = float(np.mean(errors))
        p95 = float(np.percentile(errors, 95))
    else:
        mabe, p95 = 0.0, 0.0

    # Cup IoU
    pred_cup_bin = (pred_cup > 0.5)
    gt_cup_bin = (gt_cup > 0.5)
    cup_inter = np.sum(pred_cup_bin & gt_cup_bin)
    cup_union = np.sum(pred_cup_bin | gt_cup_bin)
    cup_iou = (cup_inter / cup_union) if cup_union > 0 else 1.0

    return {
        'dice': float(dice),
        'nfl_mabe': mabe,
        'nfl_p95': p95,
        'cup_iou': float(cup_iou)
    }


def render_gallery_panel(raw_bscan, mask_cyan, mask_green, subj, eye, z_idx, out_path):
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

        ax.set_title(f"{title} | {subj} ({eye}) B-scan {z_idx}", color="white", fontsize=12, fontweight="bold")
        ax.set_ylim(440, 180)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def render_deep_dive_panel(raw_bscan, mask_cyan, mask_green, mask_red, enface_img, enface_cyan, enface_green, subj, eye, z_idx, y_enface, out_path):
    fig, axs = plt.subplots(2, 3, figsize=(20, 12), facecolor="black")

    arms = [
        ("Reference Algorithm (Cyan)", mask_cyan, [0.0, 0.8, 1.0]),
        ("Commercial Solix (Red)", mask_red, [1.0, 0.2, 0.2]),
        ("Volumetric U-Net (Green)", mask_green, [0.1, 0.95, 0.3])
    ]

    # Row 0: Full Central B-scans across 3 arms
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

        ax.set_title(f"{title}\nCentral B-scan {z_idx}", color="white", fontsize=11, fontweight="bold")
        ax.set_ylim(440, 180)
        ax.axis("off")

    # Row 1, Col 0: Left Rim Zoom (Algorithm vs UNet)
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

    # Row 1, Col 1: Right Rim Zoom (Algorithm vs UNet)
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

    # Row 1, Col 2: En Face Mid-Rim Axial Comparison
    ax_ef = axs[1, 2]
    ax_ef.imshow(enface_img, cmap="gray", aspect="auto")
    over_ef = np.zeros((*enface_img.shape, 4), dtype=np.float32)
    if enface_cyan is not None:
        over_ef[enface_cyan == 1] = [0.0, 0.8, 1.0, 0.35]
    if enface_green is not None:
        over_ef[enface_green == 1] = [0.1, 0.95, 0.3, 0.5]
    ax_ef.imshow(over_ef, aspect="auto")
    ax_ef.set_title(f"En Face Mid-Rim Plane (y={y_enface})\n[Cyan: Ref vs Green: U-Net]", color="white", fontsize=11, fontweight="bold")
    ax_ef.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def find_matching_curve_xml(tsv_dir, subj, eye, dcm_fname, protocol="Disc Cube"):
    """
    Finds the exact XML curve corresponding to a DICOM volume.
    Matches via timestamp in the master XML if available, otherwise falls back to glob.
    """
    subj_tsv = os.path.join(tsv_dir, subj)
    if not os.path.exists(subj_tsv):
        return None

    # Try matching by timestamp via master XML
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

    # Fallback to globbing by eye and protocol if timestamp matching yields no result
    xmls = glob.glob(os.path.join(subj_tsv, "curve", f"*{eye}*{protocol}*.xml"))
    if not xmls:
        xmls = glob.glob(os.path.join(subj_tsv, "curve", f"*{protocol}*{eye}*.xml"))
    return xmls[0] if xmls else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default="/Users/nikhilmundhra/Library/CloudStorage/Box-Box/deidentified")
    parser.add_argument("--output_dir", type=str, default="/Users/nikhilmundhra/Documents/Github/Capstone/OCT-Analyser-Capstone/docs")
    parser.add_argument("--val_subjects", type=str, default="BEH0335,BEH0314")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    dev = torch.device(args.device if ("mps" in args.device and torch.backends.mps.is_available()) or "cuda" in args.device else "cpu")
    print(f"[Cohort Evaluator] Using device: {dev}")

    assets_dir = os.path.join(args.output_dir, "assets", "executive_cohort_report")
    os.makedirs(assets_dir, exist_ok=True)

    print(f"[Cohort Evaluator] Loading model checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    model = VolumetricRNFLNet(in_channels=5, base_channels=16).to(dev)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    val_subjects = [s.strip() for s in args.val_subjects.split(",")]

    dicom_dir = os.path.join(args.dataset_root, "dicom")
    tsv_good_dir = os.path.join(args.dataset_root, "tsv", "good")
    tsv_bad_dir = os.path.join(args.dataset_root, "tsv", "bad")

    all_subjects = sorted([s for s in os.listdir(dicom_dir) if os.path.isdir(os.path.join(dicom_dir, s))])
    print(f"[Cohort Evaluator] Found {len(all_subjects)} subjects across cohort: {all_subjects}")

    cohort_results = []
    gallery_manifest = []
    deep_dive_manifest = []

    for subj in all_subjects:
        is_val = subj in val_subjects
        cohort_tag = "Validation (Held-Out)" if is_val else "Training / Benchmark"

        subj_dcms = sorted(glob.glob(os.path.join(dicom_dir, subj, "*Disc Cube*_OPT.dcm")))
        if not subj_dcms:
            continue

        for dcm_path in subj_dcms:
            fname = os.path.basename(dcm_path)
            eye = "OS" if "_OS_" in fname else "OD"

            # Find matching XML curves with timestamp fidelity
            good_xml = find_matching_curve_xml(tsv_good_dir, subj, eye, fname, protocol="Disc Cube")
            if not good_xml:
                continue

            bad_xml = find_matching_curve_xml(tsv_bad_dir, subj, eye, fname, protocol="Disc Cube")

            print(f"\n---> Evaluating {subj} ({eye}) [{cohort_tag}]...")

            # Load curves
            curves_good = load_curves(good_xml, mask_sentinel=True)
            zc, xc, r_disc = extract_disc_geometry(curves_good)
            curves_bad = load_curves(bad_xml, mask_sentinel=True) if bad_xml else None

            # Load DICOM via memory-map
            offset = find_dicom_pixel_offset(dcm_path)
            memmap = np.memmap(dcm_path, dtype='<u2', mode='r', offset=offset, shape=(320, 768, 320))

            # Run Volumetric U-Net inference
            pred_vol_mask, pred_ilm, pred_nfl, pred_cup = run_volume_inference(model, memmap, device=dev)

            # Compute slice-by-slice metrics in peripapillary region
            slice_metrics_unet = []
            slice_metrics_bad = []

            for b_idx in range(320):
                is_peri = abs(b_idx - zc) <= int(2.0 * r_disc)
                if not is_peri:
                    continue

                # Ground truth mask & curves
                gt_ilm_b = curves_good['ILM'][b_idx]
                gt_nfl_b = curves_good['NFL'][b_idx]
                gt_mask_b = rasterize_curves_to_mask(gt_ilm_b, gt_nfl_b)
                gt_cup_b = np.isnan(gt_nfl_b).astype(np.float32)

                # UNet metrics
                m_unet = compute_metrics_slice(
                    pred_vol_mask[b_idx], gt_mask_b,
                    pred_nfl[b_idx], gt_nfl_b,
                    pred_cup[b_idx], gt_cup_b,
                    is_peri
                )
                if m_unet:
                    slice_metrics_unet.append(m_unet)

                # Bad arm metrics
                if curves_bad is not None:
                    bad_ilm_b = curves_bad['ILM'][b_idx]
                    bad_nfl_b = curves_bad['NFL'][b_idx]
                    bad_mask_b = rasterize_curves_to_mask(bad_ilm_b, bad_nfl_b)
                    bad_cup_b = np.isnan(bad_nfl_b).astype(np.float32)

                    m_bad = compute_metrics_slice(
                        bad_mask_b, gt_mask_b,
                        bad_nfl_b, gt_nfl_b,
                        bad_cup_b, gt_cup_b,
                        is_peri
                    )
                    if m_bad:
                        slice_metrics_bad.append(m_bad)

            # Aggregate per-scan metrics
            avg_dice_unet = float(np.mean([m['dice'] for m in slice_metrics_unet]))
            avg_mabe_unet = float(np.mean([m['nfl_mabe'] for m in slice_metrics_unet]))
            avg_p95_unet = float(np.mean([m['nfl_p95'] for m in slice_metrics_unet]))
            avg_cup_unet = float(np.mean([m['cup_iou'] for m in slice_metrics_unet]))

            avg_dice_bad = float(np.mean([m['dice'] for m in slice_metrics_bad])) if slice_metrics_bad else None
            avg_mabe_bad = float(np.mean([m['nfl_mabe'] for m in slice_metrics_bad])) if slice_metrics_bad else None
            avg_p95_bad = float(np.mean([m['nfl_p95'] for m in slice_metrics_bad])) if slice_metrics_bad else None
            avg_cup_bad = float(np.mean([m['cup_iou'] for m in slice_metrics_bad])) if slice_metrics_bad else None

            print(f"[{subj} {eye}] U-Net Dice: {avg_dice_unet:.4f} | MABE: {avg_mabe_unet:.2f} um | P95: {avg_p95_unet:.2f} um | Cup IoU: {avg_cup_unet:.4f}")
            if avg_dice_bad is not None:
                print(f"[{subj} {eye}] Bad   Dice: {avg_dice_bad:.4f} | MABE: {avg_mabe_bad:.2f} um | P95: {avg_p95_bad:.2f} um | Cup IoU: {avg_cup_bad:.4f}")

            scan_record = {
                'subject': subj,
                'eye': eye,
                'cohort': cohort_tag,
                'is_validation': is_val,
                'unet_dice': avg_dice_unet,
                'unet_mabe': avg_mabe_unet,
                'unet_p95': avg_p95_unet,
                'unet_cup_iou': avg_cup_unet,
                'bad_dice': avg_dice_bad,
                'bad_mabe': avg_mabe_bad,
                'bad_p95': avg_p95_bad,
                'bad_cup_iou': avg_cup_bad,
            }
            cohort_results.append(scan_record)

            # Generate visual outputs for OD (or first available eye)
            if eye == "OD" or subj not in [g['subject'] for g in gallery_manifest]:
                central_bscan = memmap[zc]
                gt_central_mask = rasterize_curves_to_mask(curves_good['ILM'][zc], curves_good['NFL'][zc])
                pred_central_mask = pred_vol_mask[zc]
                bad_central_mask = rasterize_curves_to_mask(curves_bad['ILM'][zc], curves_bad['NFL'][zc]) if curves_bad else None

                # 1. Gallery Panel (Side-by-Side Cyan vs Green)
                gallery_filename = f"gallery_bscan_{subj}_{eye}.png"
                gallery_path = os.path.join(assets_dir, gallery_filename)
                render_gallery_panel(central_bscan, gt_central_mask, pred_central_mask, subj, eye, zc, gallery_path)
                gallery_manifest.append({
                    'subject': subj,
                    'eye': eye,
                    'cohort': cohort_tag,
                    'filename': gallery_filename,
                    'dice': avg_dice_unet,
                    'mabe': avg_mabe_unet
                })

                # 2. Deep Dive Panel (for validation + key benchmarks)
                if is_val or subj in ["BEH0181", "BEH0174"]:
                    y_enface = int(np.nanmedian(curves_good['NFL'][zc])) if not np.isnan(np.nanmedian(curves_good['NFL'][zc])) else 320
                    y_enface = int(np.clip(y_enface, 100, 650))

                    enface_raw = memmap[:, y_enface, :]
                    enface_cyan = np.zeros((320, 320), dtype=np.uint8)
                    enface_green = pred_vol_mask[:, y_enface, :]

                    for b_i in range(320):
                        m_b = rasterize_curves_to_mask(curves_good['ILM'][b_i], curves_good['NFL'][b_i])
                        enface_cyan[b_i] = m_b[y_enface, :]

                    deep_dive_filename = f"deep_dive_{subj}_{eye}.png"
                    deep_dive_path = os.path.join(assets_dir, deep_dive_filename)
                    render_deep_dive_panel(
                        central_bscan, gt_central_mask, pred_central_mask, bad_central_mask,
                        enface_raw, enface_cyan, enface_green,
                        subj, eye, zc, y_enface, deep_dive_path
                    )
                    deep_dive_manifest.append({
                        'subject': subj,
                        'eye': eye,
                        'cohort': cohort_tag,
                        'filename': deep_dive_filename
                    })

    # Save cohort results JSON
    json_path = os.path.join(assets_dir, "cohort_evaluation_metrics.json")
    with open(json_path, "w") as f:
        json.dump({
            'scans': cohort_results,
            'gallery': gallery_manifest,
            'deep_dives': deep_dive_manifest
        }, f, indent=2)

    print("\n==========================================================================================")
    print(f"=== COHORT EVALUATION COMPLETED: {len(cohort_results)} scans evaluated across 11 subjects ===")
    print(f"=== Results saved to: {json_path} ===")
    print("==========================================================================================")


if __name__ == "__main__":
    main()
