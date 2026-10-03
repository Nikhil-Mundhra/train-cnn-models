"""Dense Anisotropic 3D U-Net Training Pipeline for RNFL Segmentation.

Features:
- Pure 3D anisotropic convolutional architecture (AnisotropicRNFLUNet3D)
- Full-axial transverse patch sampling (64, 768, 64)
- Mixed precision training (bfloat16 / AMP)
- Micro-batch size 1 with gradient accumulation steps 8
- Combined BCE + Soft Dice voxel loss
- Zero-leakage verification against frozen validation cohort
- Periodic full-volume sliding-window validation with whole-volume metrics
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

VOLUMETRIC_DIR = SCRIPT_DIR.parent / "train_rnfl_volumetric"
if str(VOLUMETRIC_DIR) not in sys.path:
    sys.path.insert(0, str(VOLUMETRIC_DIR))

from model_3d import AnisotropicRNFLUNet3D, AnisotropicUNetConfig
from dataset_3d import (
    SolixRNFL3DPatchDataset,
    build_ground_truth_mask,
    predict_whole_volume_mask,
)
from evaluation_metrics import compute_volumetric_metrics


class DiceBCELoss(nn.Module):
    """Combined Binary Cross Entropy and Soft-Dice Loss for 3D binary masks."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, eps: float = 1e-5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        # Sum over spatial dimensions (D, H, W)
        intersection = (probs * targets).sum(dim=(2, 3, 4))
        cardinality = probs.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4))
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        dice_loss = 1.0 - dice.mean()
        total_loss = self.bce_weight * bce + self.dice_weight * dice_loss
        return total_loss, bce, dice_loss, dice.mean()


def evaluate_validation_patches(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: DiceBCELoss,
    device: str,
    autocast_device: str,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> Dict[str, float]:
    """Fast validation across sampled 3D patches."""
    model.eval()
    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0
    total_dice = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
                preds = model(images)
                loss, bce, dice_loss, dice = criterion(preds["mask_logits"], masks)

            total_loss += loss.item()
            total_bce += bce.item()
            total_dice_loss += dice_loss.item()
            total_dice += dice.item()
            num_batches += 1

    if num_batches == 0:
        return {"loss": 0.0, "bce": 0.0, "dice_loss": 0.0, "dice": 0.0}

    return {
        "loss": total_loss / num_batches,
        "bce": total_bce / num_batches,
        "dice_loss": total_dice_loss / num_batches,
        "dice": total_dice / num_batches,
    }


def evaluate_whole_volumes(
    model: nn.Module,
    val_scans: List[Dict[str, object]],
    device: str,
    amp_dtype: torch.dtype,
    use_amp: bool,
    stride: Tuple[int, int] = (64, 64),
    max_volumes: Optional[int] = None,
) -> Dict[str, float]:
    """Whole-volume 3D evaluation computing whole-volume Dice and surface metrics."""
    model.eval()
    dices = []
    vss = []
    scans_to_eval = val_scans[:max_volumes] if max_volumes else val_scans

    for i, scan in enumerate(scans_to_eval):
        vol = scan["volume"]
        eye = scan["eye"]
        gt_mask = build_ground_truth_mask(scan["curves"], shape=vol.shape)
        pred_mask = predict_whole_volume_mask(
            model=model,
            volume=vol,
            eye=eye,
            patch_shape=(64, 768, 64),
            stride=stride,
            threshold=0.5,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        try:
            metrics = compute_volumetric_metrics(gt_mask, pred_mask)
            dices.append(metrics.dice)
            vss.append(metrics.volume_similarity)
        except Exception as e:
            # Fallback direct Dice calculation if seg-metrics is unavailable
            intersection = np.logical_and(gt_mask == 1, pred_mask == 1).sum()
            cardinality = (gt_mask == 1).sum() + (pred_mask == 1).sum()
            dice_val = (2.0 * intersection + 1e-5) / (cardinality + 1e-5)
            dices.append(float(dice_val))

    return {
        "volume_dice": float(np.mean(dices)) if dices else 0.0,
        "volume_similarity": float(np.mean(vss)) if vss else 0.0,
        "evaluated_scans": len(scans_to_eval),
    }


def train(args):
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Determine device and precision
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        print("[Warning] MPS requested but not available. Falling back to CPU.")
        device = "cpu"

    autocast_device = "cuda" if "cuda" in device else ("mps" if "mps" in device else "cpu")
    use_amp = args.amp and (autocast_device in ("cuda", "cpu"))
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    if autocast_device == "cpu":
        amp_dtype = torch.bfloat16

    print("==================================================================")
    print("=== Training Dense Anisotropic 3D RNFL U-Net ===")
    print("==================================================================")
    print(f"Device:                 {device} (Autocast: {autocast_device}, AMP: {use_amp}, dtype: {amp_dtype})")
    print(f"Dataset root:           {args.dataset_root}")
    print(f"Train manifest:         {args.train_manifest}")
    print(f"Validation manifest:    {args.val_manifest}")
    print(f"Patch shape:            {args.patch_shape}")
    print(f"Micro-batch size:       {args.batch_size}")
    print(f"Grad accumulation:      {args.grad_accum_steps} (Effective batch: {args.batch_size * args.grad_accum_steps})")
    print(f"Epochs:                 {args.epochs}")
    print(f"Base Learning Rate:     {args.lr}")
    print(f"Checkpoint directory:   {args.checkpoint_dir}")
    print("==================================================================")

    # 1. Datasets & Manifest Loading
    train_dataset = SolixRNFL3DPatchDataset(
        dataset_root=args.dataset_root,
        manifest_path=args.train_manifest,
        patch_shape=tuple(args.patch_shape),
        patches_per_volume=args.patches_per_volume,
        peripapillary_probability=0.75,
        standardize_eye=True,
        augment=True,
    )

    val_dataset = SolixRNFL3DPatchDataset(
        dataset_root=args.dataset_root,
        manifest_path=args.val_manifest,
        patch_shape=tuple(args.patch_shape),
        patches_per_volume=16,
        peripapillary_probability=0.75,
        standardize_eye=True,
        augment=False,
    )

    train_subjects = sorted(list({s["subject"] for s in train_dataset.scans}))
    val_subjects = sorted(list({s["subject"] for s in val_dataset.scans}))

    # 2. Strict Zero-Leakage Assertion
    overlap = set(train_subjects) & set(val_subjects)
    print(f"\n[Audit] Training subjects:   {len(train_subjects)} ({len(train_dataset.scans)} scans)")
    print(f"[Audit] Validation subjects: {len(val_subjects)} ({len(val_dataset.scans)} scans)")
    if overlap:
        raise RuntimeError(
            f"CRITICAL ZERO-LEAKAGE FAILURE: Overlapping subjects detected between train and val: {sorted(list(overlap))}"
        )
    print(f"[Audit] Zero-Leakage Guarantee: STRICTLY VERIFIED (0 overlapping subjects)\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    # 3. Model, Loss, Optimizer, Scheduler
    model_config = AnisotropicUNetConfig(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        patch_shape=tuple(args.patch_shape),
    )
    model = AnisotropicRNFLUNet3D(config=model_config).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] AnisotropicRNFLUNet3D initialized with {total_params:,} trainable parameters")

    criterion = DiceBCELoss(bce_weight=1.0, dice_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 1
    best_val_dice = -1.0
    history = []

    # Resume if requested
    if args.resume and os.path.exists(args.resume):
        try:
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_dice = ckpt.get("best_val_dice", -1.0)
        history = ckpt.get("history", [])

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 4. Training Loop
    scaler = torch.amp.GradScaler(device=autocast_device, enabled=(use_amp and amp_dtype == torch.float16))

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        running_bce = 0.0
        running_dice_loss = 0.0
        running_dice = 0.0
        optimizer.zero_grad()

        num_steps = len(train_loader)
        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
                preds = model(images)
                loss, bce, dice_loss, dice = criterion(preds["mask_logits"], masks)
                loss_scaled = loss / args.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            running_loss += loss.item()
            running_bce += bce.item()
            running_dice_loss += dice_loss.item()
            running_dice += dice.item()

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == num_steps:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()

            if (step + 1) % args.log_interval == 0 or (step + 1) == num_steps:
                curr_loss = running_loss / (step + 1)
                curr_dice = running_dice / (step + 1)
                lr = optimizer.param_groups[0]["lr"]
                mem_str = ""
                if device == "cuda":
                    mem_alloc = torch.cuda.memory_allocated() / (1024**3)
                    mem_res = torch.cuda.memory_reserved() / (1024**3)
                    mem_str = f" | GPU Mem: {mem_alloc:.1f}/{mem_res:.1f} GB"
                print(
                    f"Epoch [{epoch:02d}/{args.epochs:02d}] Step [{step + 1:04d}/{num_steps:04d}] "
                    f"Loss: {curr_loss:.4f} | Dice: {curr_dice:.4f} | LR: {lr:.2e}{mem_str}",
                    end="\r",
                    flush=True,
                )

        train_loss = running_loss / num_steps
        train_dice = running_dice / num_steps
        scheduler.step()

        # Validation
        print(f"\n[Validation] Running validation for epoch {epoch}...")
        val_patch_metrics = evaluate_validation_patches(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            autocast_device=autocast_device,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
        )

        vol_metrics = {}
        if epoch % args.vol_eval_interval == 0 or epoch == args.epochs:
            print(f"[Validation] Running whole-volume evaluation on held-out scans (stride={args.val_stride})...")
            vol_metrics = evaluate_whole_volumes(
                model=model,
                val_scans=val_dataset.scans,
                device=device,
                amp_dtype=amp_dtype,
                use_amp=use_amp,
                stride=(args.val_stride, args.val_stride),
                max_volumes=args.max_val_volumes,
            )

        epoch_time = time.time() - epoch_start_time
        val_dice = val_patch_metrics["dice"]
        vol_dice = vol_metrics.get("volume_dice", None)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) "
            f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_patch_metrics['loss']:.4f} | Val Patch Dice: {val_dice:.4f}"
            + (f" | Whole-Vol Dice: {vol_dice:.4f}" if vol_dice is not None else "")
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_loss": val_patch_metrics["loss"],
            "val_dice": val_dice,
            "volume_dice": vol_dice,
            "volume_similarity": vol_metrics.get("volume_similarity", None),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_seconds": epoch_time,
        }
        history.append(epoch_record)

        # Save latest checkpoint
        latest_path = os.path.join(args.checkpoint_dir, "latest_rnfl_3d_net.pt")
        ckpt_data = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_dice": best_val_dice,
            "history": history,
            "config": model_config,
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "args": vars(args),
        }
        torch.save(ckpt_data, latest_path)

        # Check for best model
        target_eval_dice = vol_dice if vol_dice is not None else val_dice
        if target_eval_dice > best_val_dice:
            best_val_dice = target_eval_dice
            best_path = os.path.join(args.checkpoint_dir, "best_rnfl_3d_net.pt")
            torch.save(ckpt_data, best_path)
            print(f"  >>> New best validation Dice: {best_val_dice:.4f}! Saved to {best_path}")

        # Save metrics json
        with open(os.path.join(args.checkpoint_dir, "training_metrics.json"), "w") as f:
            json.dump(history, f, indent=2)

    print("\n==================================================================")
    print(f"Training completed successfully! Best Validation Dice: {best_val_dice:.4f}")
    print("==================================================================")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Dense Anisotropic 3D RNFL U-Net Training")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/scratch/nm4358/deidentified-new",
        help="Path to Solix expanded cohort root",
    )
    parser.add_argument(
        "--train_manifest",
        type=str,
        default="model_training/train_rnfl_3d/manifests/phase1_train_manifest.json",
        help="Path to Phase 1 training manifest",
    )
    parser.add_argument(
        "--val_manifest",
        type=str,
        default="model_training/train_rnfl_3d/manifests/stratified_held_out_v2.json",
        help="Path to held-out validation manifest",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/rnfl_3d",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--patch_shape",
        nargs=3,
        type=int,
        default=[64, 768, 64],
        help="Input patch shape (D, H, W)",
    )
    parser.add_argument(
        "--patches_per_volume",
        type=int,
        default=32,
        help="Number of patches sampled per 3D volume per epoch",
    )
    parser.add_argument(
        "--base_channels",
        type=int,
        default=32,
        help="Initial channels for 3D U-Net encoder",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Micro-batch size per forward/backward pass",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Total training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Initial learning rate for AdamW",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device (cuda, mps, cpu)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Enable automatic mixed precision",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker count",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Steps between terminal log lines",
    )
    parser.add_argument(
        "--vol_eval_interval",
        type=int,
        default=5,
        help="Epochs between whole-volume evaluation passes",
    )
    parser.add_argument(
        "--val_stride",
        type=int,
        default=64,
        help="Transverse sliding window stride for whole-volume evaluation (64=25 patches fast, 32=81 patches blended)",
    )
    parser.add_argument(
        "--max_val_volumes",
        type=int,
        default=None,
        help="Optional maximum number of validation volumes to evaluate whole-volume metrics on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    train(args)
