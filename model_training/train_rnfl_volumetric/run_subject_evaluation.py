"""
run_subject_evaluation.py
==========================
Automated end-to-end evaluation runner for a single Solix OCT subject:
1. Performs 2.5D context volumetric U-Net inference with continuous 1D surface boundary decoding.
2. Applies anatomical optic disc vertical radius cut (1.85 mm vertical x 1.75 mm horizontal ellipse).
3. Computes quantitative metrics (Dice, NFL MABE, NFL P95, Cup IoU) against clinician ground truth.
4. Exports full 3D volumetric masks (.npz) for U-Net, Clinician GT, and Commercial Baseline.
5. Renders clinical comparative figures (Gallery and 6-panel Deep Dive).
6. Generates 3D Slicer interactive scene script and optionally launches 3D Slicer.
"""

import os
import sys
import glob
import re
import argparse
import subprocess
from pathlib import Path
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import find_dicom_pixel_offset, load_curves
from model import VolumetricRNFLNet
from batch_cohort_evaluator import (
    OCTVolume,
    VolumetricRNFLPredictor,
    ClinicalMetricsCalculator,
    CohortVisualizer,
    DiscGeometry,
    ScanEvaluationResult,
)


def find_files_for_subject(dataset_root: str, subject: str, eye: str = "OD"):
    """Locates DICOM file and XML curves for the given subject and eye."""
    dicom_dir = os.path.join(dataset_root, "dicom", subject)
    good_tsv_dir = os.path.join(dataset_root, "tsv", "good", subject, "curve")
    bad_tsv_dir = os.path.join(dataset_root, "tsv", "bad", subject, "curve")

    # Locate Disc Cube DICOM
    dcm_candidates = sorted(glob.glob(os.path.join(dicom_dir, f"*{eye}*.dcm")))
    disc_dcms = [f for f in dcm_candidates if "disc cube" in f.lower()]
    # Prefer _OPT.dcm if available
    opt_dcms = [f for f in disc_dcms if "_opt.dcm" in f.lower()]
    dcm_path = opt_dcms[0] if opt_dcms else (disc_dcms[0] if disc_dcms else (dcm_candidates[0] if dcm_candidates else None))

    if not dcm_path:
        raise FileNotFoundError(f"No DICOM found for {subject} {eye} in {dicom_dir}")

    # Locate Good XML
    good_candidates = sorted(glob.glob(os.path.join(good_tsv_dir, f"*{eye}*Disc Cube*.xml")) +
                             glob.glob(os.path.join(good_tsv_dir, f"*Disc Cube*{eye}*.xml")) +
                             glob.glob(os.path.join(good_tsv_dir, f"*{eye}*.xml")))
    good_xml = good_candidates[0] if good_candidates else None

    # Locate Bad XML
    bad_candidates = sorted(glob.glob(os.path.join(bad_tsv_dir, f"*{eye}*Disc Cube*.xml")) +
                            glob.glob(os.path.join(bad_tsv_dir, f"*Disc Cube*{eye}*.xml")) +
                            glob.glob(os.path.join(bad_tsv_dir, f"*{eye}*.xml")))
    bad_xml = bad_candidates[0] if bad_candidates else None

    return dcm_path, good_xml, bad_xml


def rasterize_full_volume(oct_volume: OCTVolume, curve_set: str) -> np.ndarray:
    """Rasterizes all 320 B-scans from XML curves into a full (320, 768, 320) uint8 3D volume."""
    vol_mask = np.zeros((oct_volume.n_bscans, oct_volume.rows, oct_volume.cols), dtype=np.uint8)
    for b_idx in range(oct_volume.n_bscans):
        vol_mask[b_idx] = oct_volume.rasterize_curve_slice(curve_set, b_idx)
    return vol_mask


def generate_slicer_script(
    slicer_script_path: str,
    subject: str,
    eye: str,
    dcm_path: str,
    pred_npz: str,
    good_npz: str,
    bad_npz: str,
):
    """Generates the interactive multi-arm 3D Slicer script with closed surface representations."""
    script_content = f'''"""
view_prediction_in_slicer.py
============================
Interactive 3D Slicer loader for Subject {subject} ({eye}) with Volumetric U-Net RNFL Segmentation.
Includes optional toggleable Clinician Ground Truth and Commercial Solix Baseline layers.
"""

import os
import numpy as np
import slicer

print("=" * 70)
print("=== Launching 3D Slicer Session: Subject {subject} ({eye}) ===")
print("=== Volumetric U-Net RNFL Mask & Multi-Arm Comparative Display ===")
print("=" * 70)

# Paths
DCM_PATH = r"{dcm_path}"
PRED_NPZ = r"{pred_npz}"
GOOD_NPZ = r"{good_npz}"
BAD_NPZ = r"{bad_npz}"

# 1. Load Reference Structural OCT DICOM Volume
print(f"Loading reference structural OCT volume: {{os.path.basename(DCM_PATH)}}...")
disc_node = slicer.util.loadVolume(DCM_PATH)
if not disc_node:
    raise RuntimeError(f"Failed to load DICOM volume from {{DCM_PATH}}")

disc_node.SetName("{subject}_Disc_Cube_{eye}")

# Optimize window / level for OCT tissue contrast
disp_node = disc_node.GetDisplayNode()
if disp_node:
    disp_node.AutoWindowLevelOff()
    disp_node.SetWindow(1800)
    disp_node.SetLevel(900)

# 2. Load and Configure Multi-Task Volumetric U-Net RNFL Segmentation (Primary Arm - Green)
print("Loading U-Net predicted RNFL mask...")
pred_data = np.load(PRED_NPZ)
unet_mask = pred_data["rnfl_mask"]

label_vol_unet = slicer.modules.volumes.logic().CreateAndAddLabelVolume(
    slicer.mrmlScene, disc_node, "Temp_Label_UNet"
)
slicer.util.updateVolumeFromArray(label_vol_unet, unet_mask.astype(np.int16))

unet_seg_node = slicer.mrmlScene.AddNewNodeByClass(
    "vtkMRMLSegmentationNode", "UNet_RNFL_Prediction"
)
slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
    label_vol_unet, unet_seg_node
)
slicer.mrmlScene.RemoveNode(label_vol_unet)

unet_seg = unet_seg_node.GetSegmentation().GetNthSegment(0)
unet_seg.SetName("RNFL (U-Net Prediction)")
unet_seg.SetColor(0.1, 0.95, 0.3)  # Clinical Green

# Create 3D closed surface representation
unet_seg_node.CreateClosedSurfaceRepresentation()
unet_disp = unet_seg_node.GetDisplayNode()
if unet_disp:
    unet_disp.SetVisibility(True)
    unet_disp.SetVisibility2D(True)
    unet_disp.SetVisibility3D(True)
    unet_disp.SetOpacity3D(0.85)

# 3. Load Clinician-Corrected Ground Truth (Good Reference Arm - Cyan, toggleable)
if os.path.exists(GOOD_NPZ):
    print("Loading Clinician Ground Truth mask...")
    gt_data = np.load(GOOD_NPZ)
    gt_mask = gt_data["rnfl_mask"]

    label_vol_gt = slicer.modules.volumes.logic().CreateAndAddLabelVolume(
        slicer.mrmlScene, disc_node, "Temp_Label_GT"
    )
    slicer.util.updateVolumeFromArray(label_vol_gt, gt_mask.astype(np.int16))

    gt_seg_node = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLSegmentationNode", "Clinician_Ground_Truth"
    )
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
        label_vol_gt, gt_seg_node
    )
    slicer.mrmlScene.RemoveNode(label_vol_gt)

    gt_seg = gt_seg_node.GetSegmentation().GetNthSegment(0)
    gt_seg.SetName("RNFL (Clinician Ground Truth - Good)")
    gt_seg.SetColor(0.0, 0.8, 1.0)  # Vibrant Cyan

    gt_seg_node.CreateClosedSurfaceRepresentation()
    gt_disp = gt_seg_node.GetDisplayNode()
    if gt_disp:
        gt_disp.SetVisibility(False)
        gt_disp.SetVisibility2D(False)
        gt_disp.SetVisibility3D(False)

# 4. Load Commercial Solix Baseline (Bad Baseline Arm - Red, toggleable)
if os.path.exists(BAD_NPZ):
    print("Loading Commercial Solix Baseline mask...")
    bad_data = np.load(BAD_NPZ)
    bad_mask = bad_data["rnfl_mask"]

    label_vol_bad = slicer.modules.volumes.logic().CreateAndAddLabelVolume(
        slicer.mrmlScene, disc_node, "Temp_Label_Bad"
    )
    slicer.util.updateVolumeFromArray(label_vol_bad, bad_mask.astype(np.int16))

    bad_seg_node = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLSegmentationNode", "Commercial_Solix_Baseline"
    )
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
        label_vol_bad, bad_seg_node
    )
    slicer.mrmlScene.RemoveNode(label_vol_bad)

    bad_seg = bad_seg_node.GetSegmentation().GetNthSegment(0)
    bad_seg.SetName("RNFL (Commercial Solix - Bad)")
    bad_seg.SetColor(1.0, 0.2, 0.2)  # High-visibility Red

    bad_seg_node.CreateClosedSurfaceRepresentation()
    bad_disp = bad_seg_node.GetDisplayNode()
    if bad_disp:
        bad_disp.SetVisibility(False)
        bad_disp.SetVisibility2D(False)
        bad_disp.SetVisibility3D(False)

# 5. Set Layout & Camera / Slice Positioning
layout_mgr = slicer.app.layoutManager()
if layout_mgr:
    layout_mgr.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    slicer.util.resetSliceViews()
    slicer.util.resetThreeDViews()

    red_widget = layout_mgr.sliceWidget("Red")
    if red_widget:
        red_logic = red_widget.sliceLogic()
        bounds = [0.0] * 6
        disc_node.GetRASBounds(bounds)
        center_y = (bounds[2] + bounds[3]) / 2.0
        red_logic.SetSliceOffset(center_y)

print("=" * 70)
print("SUCCESS: {subject} {eye} loaded live in 3D Slicer with U-Net mask (Green)!")
print("Clinician Ground Truth (Cyan) and Commercial Solix (Red) available in Segmentations module.")
print("=" * 70)
'''
    with open(slicer_script_path, "w") as f:
        f.write(script_content)
    print(f"[Export] Saved 3D Slicer scene loader script: {slicer_script_path}")


def main():
    parser = argparse.ArgumentParser(description="Run full U-Net volumetric evaluation on single subject")
    parser.add_argument("--subject", type=str, default="BEH0354", help="Subject ID")
    parser.add_argument("--eye", type=str, default="OD", choices=["OD", "OS"], help="Eye")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/Users/nikhilmundhra/Documents/Github/Capstone/train-cnn-models/checkpoints/rnfl_optical_finetuned/best_volumetric_rnfl_net.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/Users/nikhilmundhra/Library/CloudStorage/Box-Box/deidentified",
        help="Path to deidentified root directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory (defaults to train-cnn-models/predictions/<SUBJECT>)",
    )
    parser.add_argument("--device", type=str, default="mps", help="Compute device (mps, cuda, cpu)")
    parser.add_argument("--launch_slicer", action="store_true", help="Launch 3D Slicer GUI with scene")
    args = parser.parse_args()

    subject = args.subject.upper()
    eye = args.eye.upper()

    if not args.output_dir:
        output_dir = os.path.abspath(
            os.path.join(SCRIPT_DIR, "../../predictions", subject)
        )
    else:
        output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(f"=== Volumetric RNFL Clinical System Execution: {subject} ({eye}) ===")
    print("=" * 80)

    # 1. Discover data files
    print("\n[1/6] Discovering acquisition files...")
    dcm_path, good_xml, bad_xml = find_files_for_subject(args.dataset_root, subject, eye)
    print(f"  DICOM : {dcm_path}")
    print(f"  Good  : {good_xml}")
    print(f"  Bad   : {bad_xml}")

    oct_volume = OCTVolume(
        subject=subject,
        eye=eye,
        dcm_path=dcm_path,
        good_xml_path=good_xml,
        bad_xml_path=bad_xml,
        is_validation=True,
    )

    # 2. Load Model & Predict
    print(f"\n[2/6] Loading checkpoint on {args.device} and running volumetric 2.5D inference...")
    dev = torch.device(args.device if torch.backends.mps.is_available() or args.device == "cpu" else "cpu")
    predictor = VolumetricRNFLPredictor.from_checkpoint(args.checkpoint, device=dev, batch_size=8)
    prediction = predictor.predict(oct_volume)
    print(f"  Total U-Net segmented RNFL voxels: {np.sum(prediction.mask == 1):,}")

    # 3. Quantitative Evaluation
    print("\n[3/6] Computing quantitative metrics against clinician ground truth...")
    calc = ClinicalMetricsCalculator()
    eval_result = calc.evaluate_scan(oct_volume, prediction)

    print("-" * 60)
    print(f"METRICS REPORT: {subject} ({eye})")
    print("-" * 60)
    print(f"  U-Net Dice Similarity Coefficient : {eval_result.unet_dice:.4f}")
    print(f"  U-Net NFL MABE (µm)               : {eval_result.unet_mabe:.2f} µm")
    print(f"  U-Net NFL P95 Error (µm)          : {eval_result.unet_p95:.2f} µm")
    print(f"  U-Net Optic Cup IoU               : {eval_result.unet_cup_iou:.4f}")

    if eval_result.bad_dice is not None:
        print(f"  Commercial Solix Baseline Dice    : {eval_result.bad_dice:.4f}")
        print(f"  Commercial Solix Baseline MABE    : {eval_result.bad_mabe:.2f} µm")
        print(f"  Commercial Solix Baseline P95     : {eval_result.bad_p95:.2f} µm")
        print(f"  Commercial Solix Baseline Cup IoU : {eval_result.bad_cup_iou:.4f}")
        mabe_delta = eval_result.bad_mabe - eval_result.unet_mabe
        dice_delta = eval_result.unet_dice - eval_result.bad_dice
        print(f"  -> U-Net MABE Improvement         : +{mabe_delta:.2f} µm ({mabe_delta/eval_result.bad_mabe*100:.1f}% reduction)")
        print(f"  -> U-Net Dice Improvement         : +{dice_delta:.4f}")
    print("-" * 60)

    # 4. Rasterize and export 3D volumes
    print("\n[4/6] Exporting 3D volumetric masks (.npz)...")
    base_prefix = f"{subject}_Disc Cube_{eye}"
    pred_npz = os.path.join(output_dir, f"{base_prefix}_rnfl_pred.npz")
    good_npz = os.path.join(output_dir, f"{base_prefix}_good_ground_truth.npz")
    bad_npz = os.path.join(output_dir, f"{base_prefix}_bad_raw_solix.npz")

    np.savez_compressed(pred_npz, rnfl_mask=prediction.mask)
    print(f"  Saved U-Net mask: {pred_npz}")

    if good_xml:
        gt_mask = rasterize_full_volume(oct_volume, "good")
        np.savez_compressed(good_npz, rnfl_mask=gt_mask)
        print(f"  Saved Clinician GT mask: {good_npz} (voxels: {np.sum(gt_mask == 1):,})")

    if bad_xml:
        bad_mask = rasterize_full_volume(oct_volume, "bad")
        np.savez_compressed(bad_npz, rnfl_mask=bad_mask)
        print(f"  Saved Solix Baseline mask: {bad_npz} (voxels: {np.sum(bad_mask == 1):,})")

    # 5. Render Visualization Figures
    print("\n[5/6] Rendering comparative visual panels...")
    visualizer = CohortVisualizer(output_dir=output_dir)
    gallery_file = visualizer.render_gallery_panel(oct_volume, prediction)
    deep_dive_file = visualizer.render_deep_dive_panel(oct_volume, prediction)
    print(f"  Gallery panel  : {os.path.join(output_dir, gallery_file)}")
    print(f"  Deep dive panel: {os.path.join(output_dir, deep_dive_file)}")

    # 6. Generate 3D Slicer Script & Optionally Launch
    print("\n[6/6] Generating 3D Slicer interactive scene loader...")
    slicer_script = os.path.join(output_dir, "view_prediction_in_slicer.py")
    generate_slicer_script(
        slicer_script_path=slicer_script,
        subject=subject,
        eye=eye,
        dcm_path=dcm_path,
        pred_npz=pred_npz,
        good_npz=good_npz,
        bad_npz=bad_npz,
    )

    if args.launch_slicer:
        slicer_app = "/Applications/Slicer.app"
        if os.path.exists(slicer_app):
            print(f"\n[Slicer] Launching 3D Slicer for {subject} ({eye})...")
            cmd = ["open", "-a", slicer_app, "--args", "--python-script", slicer_script]
            subprocess.Popen(cmd)
            print("[Slicer] Process launched successfully in background.")
        else:
            print(f"[Slicer] 3D Slicer not found at {slicer_app}")

    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETE FOR {subject} ({eye})")
    print("=" * 80)


if __name__ == "__main__":
    main()
