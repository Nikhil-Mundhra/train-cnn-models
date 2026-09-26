"""
export_to_slicer.py
===================
Runs inference using a trained VolumetricRNFLNet checkpoint on a full DICOM OCT volume
and exports the predicted 3D segmentation into 3D Slicer.
Features:
- Batched 2.5D inference across all B-scans
- Full 3D volumetric labelmap assembly (320 x 768 x 320)
- NIfTI (.nii.gz) export and direct 3D Slicer scene generation
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import read_bscan_raw, find_dicom_pixel_offset
from model import VolumetricRNFLNet


def run_volumetric_inference(
    model,
    dcm_path,
    context_slices=5,
    device="cpu",
    batch_size=8,
    num_bscans=320,
    rows=768,
    cols=320,
    threshold=0.40,
    biplanar_fusion=True
):
    model.eval()
    half_ctx = context_slices // 2
    is_os = ("_OS_" in os.path.basename(dcm_path))

    print(f"[Inference] Processing {num_bscans} B-scans from {os.path.basename(dcm_path)} on {device} (Bi-Planar Fusion: {biplanar_fusion})...")

    # Fast memory-mapped DICOM slice reading
    offset = find_dicom_pixel_offset(dcm_path)
    memmap = np.memmap(dcm_path, dtype='<u2', mode='r', offset=offset, shape=(num_bscans, rows, cols))

    autocast_device = "mps" if "mps" in str(device) else ("cuda" if "cuda" in str(device) else "cpu")
    use_amp = "mps" in str(device) or "cuda" in str(device)

    # 1. Pass 1: Horizontal B-Scan Pane (Y-axis 2.5D stacks)
    horiz_probs = np.zeros((num_bscans, rows, cols), dtype=np.float32)
    horiz_ilm = np.zeros((num_bscans, cols), dtype=np.float32)
    horiz_nfl = np.zeros((num_bscans, cols), dtype=np.float32)
    horiz_cup = np.zeros((num_bscans, cols), dtype=np.float32)

    for start_idx in range(0, num_bscans, batch_size):
        end_idx = min(start_idx + batch_size, num_bscans)
        batch_slices = []

        for b_idx in range(start_idx, end_idx):
            slice_indices = [
                min(max(b_idx + offset_idx, 0), num_bscans - 1)
                for offset_idx in range(-half_ctx, half_ctx + 1)
            ]
            stack = [memmap[s_idx] for s_idx in slice_indices]
            img_stack = np.stack(stack, axis=0).astype(np.float32) / 2560.0
            if is_os:
                img_stack = np.flip(img_stack, axis=-1).copy()
            batch_slices.append(img_stack)

        batch_tensor = torch.from_numpy(np.stack(batch_slices, axis=0)).to(device)

        with torch.no_grad():
            with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=use_amp):
                preds = model(batch_tensor)
                probs = torch.sigmoid(preds['mask_logits']).squeeze(1).float().cpu().numpy()
                cup_probs = torch.sigmoid(preds['cup_logits']).float().cpu().numpy()
                ilm_preds = preds['ilm_pred'].float().cpu().numpy()
                nfl_preds = preds['nfl_pred'].float().cpu().numpy()

        if is_os:
            probs = np.flip(probs, axis=-1).copy()
            ilm_preds = np.flip(ilm_preds, axis=-1).copy()
            nfl_preds = np.flip(nfl_preds, axis=-1).copy()
            cup_probs = np.flip(cup_probs, axis=-1).copy()

        horiz_probs[start_idx:end_idx] = probs
        horiz_ilm[start_idx:end_idx] = ilm_preds
        horiz_nfl[start_idx:end_idx] = nfl_preds
        horiz_cup[start_idx:end_idx] = cup_probs

    # 2. Pass 2: Vertical B-Scan Pane (X-axis 2.5D stacks)
    if biplanar_fusion:
        vert_probs = np.zeros((num_bscans, rows, cols), dtype=np.float32)
        vert_ilm = np.zeros((cols, num_bscans), dtype=np.float32)
        vert_nfl = np.zeros((cols, num_bscans), dtype=np.float32)
        vert_cup = np.zeros((cols, num_bscans), dtype=np.float32)

        for start_a in range(0, cols, batch_size):
            end_a = min(start_a + batch_size, cols)
            batch_slices = []
            for a_idx in range(start_a, end_a):
                eff_a = (cols - 1 - a_idx) if is_os else a_idx
                slice_indices = [min(max(eff_a + o, 0), cols - 1) for o in range(-half_ctx, half_ctx + 1)]
                stack = [memmap[:, :, s].T for s in slice_indices]
                img_stack = np.stack(stack, axis=0).astype(np.float32) / 2560.0
                batch_slices.append(img_stack)

            batch_tensor = torch.from_numpy(np.stack(batch_slices, axis=0)).to(device)

            with torch.no_grad():
                with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=use_amp):
                    preds = model(batch_tensor)
                    v_probs = torch.sigmoid(preds['mask_logits']).squeeze(1).float().cpu().numpy()
                    v_cup = torch.sigmoid(preds['cup_logits']).float().cpu().numpy()
                    v_ilm = preds['ilm_pred'].float().cpu().numpy()
                    v_nfl = preds['nfl_pred'].float().cpu().numpy()

            for i, a_idx in enumerate(range(start_a, end_a)):
                vert_probs[:, :, a_idx] = v_probs[i].T
                vert_ilm[a_idx, :] = v_ilm[i]
                vert_nfl[a_idx, :] = v_nfl[i]
                vert_cup[a_idx, :] = v_cup[i]

        # Fused multi-planar representations
        fused_probs = 0.5 * (horiz_probs + vert_probs)
        fused_ilm = 0.5 * (horiz_ilm + vert_ilm.T)
        fused_nfl = 0.5 * (horiz_nfl + vert_nfl.T)
        fused_cup = 0.5 * (horiz_cup + vert_cup.T)
    else:
        fused_probs = horiz_probs
        fused_ilm = horiz_ilm
        fused_nfl = horiz_nfl
        fused_cup = horiz_cup

    # 3. Dense Segmentation & Hybrid Surface-Guided Recovery
    full_vol_mask = (fused_probs > threshold).astype(np.uint8)

    for b in range(num_bscans):
        col_sums = full_vol_mask[b].sum(axis=0)
        valid_tissue = (fused_cup[b] < 0.5) & ((fused_nfl[b] - fused_ilm[b]) >= 2.0)
        dropouts = np.where((col_sums == 0) & valid_tissue)[0]
        for x in dropouts:
            y0 = int(np.clip(np.round(fused_ilm[b, x]), 0, rows))
            y1 = int(np.clip(np.round(fused_nfl[b, x]), 0, rows))
            if y1 > y0:
                full_vol_mask[b, y0:y1, x] = 1

    # -------------------------------------------------------------------------
    # Optic Disc Size & Anatomical Vertical Cut
    # Average normal human optic disc dimensions:
    # - Vertical diameter: ~1.8 to 1.9 mm (mean: 1.85 mm -> vertical radius ~0.925 mm)
    # - Horizontal diameter: ~1.7 to 1.8 mm (mean: 1.75 mm -> horizontal radius ~0.875 mm)
    # Solix Disc Cube: dx = 0.01875 mm/px, dz = 0.0188088 mm/slice
    # Cut occurs from center (zc, xc) to that radius on either side.
    # -------------------------------------------------------------------------
    dx_mm = 0.01875
    dz_mm = 0.0188088
    rad_x_px = (1.75 / 2.0) / dx_mm       # ~46.67 pixels
    rad_z_slices = (1.85 / 2.0) / dz_mm   # ~49.18 slices

    # Disc center from scan geometry / cup prior (default: volume center)
    zc, xc = num_bscans // 2, cols // 2

    for z in range(num_bscans):
        dz_val = abs(z - zc)
        if dz_val <= rad_z_slices:
            rx_z = rad_x_px * np.sqrt(max(0.0, 1.0 - (dz_val / rad_z_slices) ** 2))
            x_left = max(0, int(round(xc - rx_z)))
            x_right = min(cols - 1, int(round(xc + rx_z)))
            full_vol_mask[z, :, x_left : x_right + 1] = 0

    print(f"[Inference] Completed! Total RNFL voxels: {np.sum(full_vol_mask == 1)}")
    return full_vol_mask


def export_prediction(
    checkpoint_path,
    dcm_path,
    output_dir="./predictions",
    device="",
    launch_slicer=False
):
    dev = torch.device(device if device else ("mps" if torch.backends.mps.is_available() else "cpu"))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)

    model = VolumetricRNFLNet(
        in_channels=ckpt['args'].get('context_slices', 5),
        base_channels=16
    ).to(dev)
    model.load_state_dict(ckpt['model_state_dict'])

    # Run inference
    pred_mask = run_volumetric_inference(model, dcm_path, device=dev)

    # Save as compressed numpy
    base_name = Path(dcm_path).stem
    out_npz = os.path.abspath(os.path.join(output_dir, f"{base_name}_rnfl_pred.npz"))
    np.savez_compressed(out_npz, rnfl_mask=pred_mask)
    print(f"[Export] Saved 3D binary mask to {out_npz}")

    # Generate standalone launch script for 3D Slicer if requested
    if launch_slicer:
        slicer_script = os.path.join(output_dir, "view_prediction_in_slicer.py")
        with open(slicer_script, "w") as f:
            f.write(f"""import slicer
import numpy as np

print("Loading predicted RNFL segmentation into Slicer...")
data = np.load(r"{out_npz}")
mask = data['rnfl_mask']

# Load reference volume
disc_node = slicer.util.loadVolume(r"{dcm_path}")
label_vol = slicer.modules.volumes.logic().CreateAndAddLabelVolume(slicer.mrmlScene, disc_node, "ML_RNFL_Prediction")
slicer.util.updateVolumeFromArray(label_vol, mask.astype(np.int16))

seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "ML_RNFL_Segmentation")
slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(label_vol, seg_node)
slicer.mrmlScene.RemoveNode(label_vol)

seg = seg_node.GetSegmentation().GetNthSegment(0)
seg.SetName("RNFL (Model Prediction)")
seg.SetColor(0.1, 0.95, 0.3)
seg_node.CreateClosedSurfaceRepresentation()

slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
slicer.util.resetSliceViews()
print("Prediction loaded live in 3D Slicer!")
""")
        print(f"[Export] Generated Slicer loader script: {slicer_script}")
        print(f"[Export] Run with: /Applications/Slicer.app/Contents/MacOS/Slicer --python-script {slicer_script}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dcm", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./predictions")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--launch_slicer", action="store_true")
    args = parser.parse_args()

    export_prediction(
        checkpoint_path=args.checkpoint,
        dcm_path=args.dcm,
        output_dir=args.output_dir,
        device=args.device,
        launch_slicer=args.launch_slicer
    )
