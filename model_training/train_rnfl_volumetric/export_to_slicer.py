"""
export_to_slicer.py
===================
Runs inference using a trained VolumetricRNFLNet checkpoint on a full DICOM OCT volume
and exports the predicted 3D segmentation into 3D Slicer.
Features:
- Batched 2.5D inference across all B-scans via VolumetricRNFLPredictor
- Full 3D volumetric labelmap assembly (320 x 768 x 320)
- NIfTI / compressed NumPy (.npz) export and direct 3D Slicer scene generation
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from batch_cohort_evaluator import (
    OCTVolume,
    VolumetricRNFLPredictor,
    InferenceConfig,
    OpticDiscCutConfig,
)
from orientation import OS_ORIENTATION_MODES, validate_orientation_mode


def run_volumetric_inference(
    model: torch.nn.Module,
    dcm_path: str,
    context_slices: int = 5,
    device: str = "cpu",
    batch_size: int = 8,
    num_bscans: int = 320,
    rows: int = 768,
    cols: int = 320,
    threshold: float = 0.40,
    biplanar_fusion: bool = True,
    os_orientation_mode: str = "corrected",
) -> np.ndarray:
    """
    Backward-compatible adapter that performs 3D volumetric RNFL segmentation
    by delegating to VolumetricRNFLPredictor.
    """
    dev = torch.device(device)
    config = InferenceConfig(mask_threshold=threshold)
    predictor = VolumetricRNFLPredictor(
        model=model,
        device=dev,
        batch_size=batch_size,
        half_ctx=context_slices // 2,
        config=config,
        os_orientation_mode=os_orientation_mode,
    )

    eye = "OS" if "_OS_" in os.path.basename(dcm_path) else "OD"
    subject = Path(dcm_path).stem.split("_")[0]
    oct_volume = OCTVolume(
        subject=subject,
        eye=eye,
        dcm_path=dcm_path,
        good_xml_path=None,
        bad_xml_path=None,
        n_bscans=num_bscans,
        rows=rows,
        cols=cols,
    )

    prediction = predictor.predict(oct_volume, biplanar_fusion=biplanar_fusion)
    return prediction.mask


def export_prediction(
    checkpoint_path: str,
    dcm_path: str,
    output_dir: str = "./predictions",
    device: str = "",
    launch_slicer: bool = False,
    os_orientation_mode: str = "corrected",
    biplanar_fusion: bool = True,
) -> str:
    dev = torch.device(device if device else ("mps" if torch.backends.mps.is_available() else "cpu"))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading checkpoint with VolumetricRNFLPredictor: {checkpoint_path}")
    predictor = VolumetricRNFLPredictor.from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=dev,
        batch_size=8,
        os_orientation_mode=os_orientation_mode,
    )

    eye = "OS" if "_OS_" in os.path.basename(dcm_path) else "OD"
    subject = Path(dcm_path).stem.split("_")[0]
    oct_volume = OCTVolume(
        subject=subject,
        eye=eye,
        dcm_path=dcm_path,
        good_xml_path=None,
        bad_xml_path=None,
    )

    print(f"[Export] Running 3D volumetric prediction on {dcm_path} (biplanar={biplanar_fusion}, mode={os_orientation_mode})...")
    prediction = predictor.predict(oct_volume, biplanar_fusion=biplanar_fusion)
    pred_mask = prediction.mask
    print(f"[Export] Completed! Total RNFL voxels: {np.sum(pred_mask == 1):,}")

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

    return out_npz


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Volumetric RNFL predictions to 3D Slicer")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--dcm", type=str, required=True, help="Path to target DICOM volume")
    parser.add_argument("--output_dir", type=str, default="./predictions", help="Directory to save predictions")
    parser.add_argument("--device", type=str, default="", help="Torch device (e.g. cpu, mps, cuda)")
    parser.add_argument("--launch_slicer", action="store_true", help="Generate helper script to launch 3D Slicer")
    parser.add_argument("--disable_biplanar", dest="biplanar_fusion", action="store_false", help="Disable biplanar fusion")
    parser.set_defaults(biplanar_fusion=True)
    parser.add_argument(
        "--os_orientation_mode",
        choices=OS_ORIENTATION_MODES,
        default="corrected",
        help="OS coordinate policy (corrected or legacy_vertical_mirror)",
    )
    args = parser.parse_args()

    export_prediction(
        checkpoint_path=args.checkpoint,
        dcm_path=args.dcm,
        output_dir=args.output_dir,
        device=args.device,
        launch_slicer=args.launch_slicer,
        os_orientation_mode=args.os_orientation_mode,
        biplanar_fusion=args.biplanar_fusion,
    )
