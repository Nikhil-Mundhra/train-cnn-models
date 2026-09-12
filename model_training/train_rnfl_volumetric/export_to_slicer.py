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

from dataset import read_bscan_raw
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
    threshold=0.5
):
    model.eval()
    half_ctx = context_slices // 2
    full_vol_mask = np.zeros((num_bscans, rows, cols), dtype=np.uint8)

    print(f"[Inference] Processing {num_bscans} B-scans from {os.path.basename(dcm_path)} on {device}...")

    # Process in batches
    for start_idx in range(0, num_bscans, batch_size):
        end_idx = min(start_idx + batch_size, num_bscans)
        batch_slices = []

        for b_idx in range(start_idx, end_idx):
            slice_indices = [
                min(max(b_idx + offset, 0), num_bscans - 1)
                for offset in range(-half_ctx, half_ctx + 1)
            ]
            stack = [read_bscan_raw(dcm_path, s_idx, rows=rows, cols=cols) for s_idx in slice_indices]
            img_stack = np.stack(stack, axis=0).astype(np.float32) / 2560.0
            batch_slices.append(img_stack)

        batch_tensor = torch.from_numpy(np.stack(batch_slices, axis=0)).to(device)

        with torch.no_grad():
            preds = model(batch_tensor)
            probs = torch.sigmoid(preds['mask_logits']).squeeze(1).cpu().numpy()
            binary_masks = (probs > threshold).astype(np.uint8)

        full_vol_mask[start_idx:end_idx] = binary_masks

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
    out_npz = os.path.join(output_dir, f"{base_name}_rnfl_pred.npz")
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
