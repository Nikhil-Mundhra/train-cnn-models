"""
train.py
========
Training pipeline for the Multi-Task 2.5D Volumetric RNFL Network.
Features:
- Strict subject-level train/val splitting (no B-scan leakage)
- Gradient accumulation for stable GPU memory on Apple Silicon (MPS)
- Peripapillary-focused clinical validation metrics (MABE in um, P95 in um, Dice)
- Model checkpointing based on peripapillary NFL boundary accuracy
"""

import os
import sys
import time
import argparse
from pathlib import Path

# OpenMP guard for macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import SolixRNFLDataset, AXIAL_UM
from model import VolumetricRNFLNet
from losses import VolumetricRNFLLoss


def compute_batch_metrics(preds, targets):
    """
    Computes validation metrics on a batch:
    - Dice Score on binary RNFL mask
    - ILM and NFL MABE in microns
    - Cup absence classification IoU
    """
    with torch.no_grad():
        # 1. Mask Dice
        prob = torch.sigmoid(preds['mask_logits'])
        pred_mask = (prob > 0.5).float()
        true_mask = targets['mask']

        intersection = (pred_mask * true_mask).sum().item()
        union = pred_mask.sum().item() + true_mask.sum().item()
        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)

        # 2. Boundary Absolute Errors in Microns
        ilm_err = torch.abs(preds['ilm_pred'] - targets['ilm_surface']) * AXIAL_UM
        ilm_mabe = ilm_err.mean().item()

        cup_absent = targets['cup_absent']
        tissue_mask = (1.0 - cup_absent)
        nfl_err = torch.abs(preds['nfl_pred'] - targets['nfl_surface']) * AXIAL_UM
        nfl_valid_err = nfl_err * tissue_mask
        nfl_mabe = (nfl_valid_err.sum() / (tissue_mask.sum() + 1e-6)).item()

        # 3. Cup Absence IoU
        cup_prob = torch.sigmoid(preds['cup_logits'])
        pred_cup = (cup_prob > 0.5).float()
        cup_inter = (pred_cup * cup_absent).sum().item()
        cup_union = (pred_cup + cup_absent).clamp(0, 1).sum().item()
        cup_iou = (cup_inter + 1e-5) / (cup_union + 1e-5)

        return {
            'dice': dice,
            'ilm_mabe': ilm_mabe,
            'nfl_mabe': nfl_mabe,
            'cup_iou': cup_iou
        }


def validate(model, val_loader, criterion, device, autocast_device="cpu", amp_dtype=torch.float32, use_amp=False):
    model.eval()
    val_loss = 0.0
    dices = []
    ilm_mabes = []
    nfl_mabes = []
    cup_ious = []

    # Filtered peripapillary errors
    peri_nfl_errs = []

    with torch.no_grad():
        with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
            for batch in val_loader:
                for k in ['image', 'mask', 'ilm_surface', 'nfl_surface', 'cup_absent', 'is_peripapillary']:
                    batch[k] = batch[k].to(device)

                preds = model(batch['image'])
                losses = criterion(preds, batch)
                val_loss += losses['loss'].item()

                metrics = compute_batch_metrics(preds, batch)
                dices.append(metrics['dice'])
                ilm_mabes.append(metrics['ilm_mabe'])
                nfl_mabes.append(metrics['nfl_mabe'])
                cup_ious.append(metrics['cup_iou'])

                # Collect peripapillary slice metrics
                is_peri = batch['is_peripapillary']
                if is_peri.any():
                    cup_absent = batch['cup_absent'][is_peri]
                    tissue = (1.0 - cup_absent)
                    nfl_err = torch.abs(preds['nfl_pred'][is_peri] - batch['nfl_surface'][is_peri]) * AXIAL_UM
                    valid_err = nfl_err[tissue > 0.5].cpu().numpy()
                    if len(valid_err) > 0:
                        peri_nfl_errs.extend(valid_err)

    n_batches = len(val_loader)
    avg_loss = val_loss / n_batches
    avg_dice = np.mean(dices)
    avg_ilm_mabe = np.mean(ilm_mabes)
    avg_nfl_mabe = np.mean(nfl_mabes)
    avg_cup_iou = np.mean(cup_ious)

    peri_mabe = float(np.mean(peri_nfl_errs)) if peri_nfl_errs else avg_nfl_mabe
    peri_p95 = float(np.percentile(peri_nfl_errs, 95)) if peri_nfl_errs else 0.0

    return {
        'loss': avg_loss,
        'dice': avg_dice,
        'ilm_mabe': avg_ilm_mabe,
        'nfl_mabe': avg_nfl_mabe,
        'cup_iou': avg_cup_iou,
        'peri_nfl_mabe': peri_mabe,
        'peri_nfl_p95': peri_p95
    }


def train(args):
    device = torch.device(args.device if args.device else ("mps" if torch.backends.mps.is_available() else "cpu"))
    autocast_device = "mps" if device.type == "mps" else ("cuda" if device.type == "cuda" else "cpu")
    use_amp = args.precision in ["bf16", "fp16"]
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else (torch.float16 if args.precision == "fp16" else torch.float32)

    print(f"[Training] Using compute device: {device} | Precision: {args.precision.upper()} (AMP: {use_amp}, dtype: {amp_dtype})")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Subject-level Dataset Partitioning
    all_subjects = [
        'BEH0174', 'BEH0181', 'BEH0310', 'BEH0314', 'BEH0321',
        'BEH0335', 'BEH0349', 'BEH0354', 'BEH0364', 'BEH0398', 'BEH0410'
    ]
    val_subjects = args.val_subjects.split(',')
    train_subjects = [s for s in all_subjects if s not in val_subjects]

    print(f"[Training] Training Subjects ({len(train_subjects)}): {train_subjects}")
    print(f"[Training] Validation Subjects ({len(val_subjects)}): {val_subjects}")

    train_ds = SolixRNFLDataset(
        dataset_root=args.dataset_root,
        subjects=train_subjects,
        context_slices=args.context_slices
    )
    val_ds = SolixRNFLDataset(
        dataset_root=args.dataset_root,
        subjects=val_subjects,
        context_slices=args.context_slices
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )

    # 2. Instantiate Model, Loss, Optimizer
    model = VolumetricRNFLNet(
        in_channels=args.context_slices,
        base_channels=16,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    ).to(device)

    criterion = VolumetricRNFLLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # AMP Autocast & GradScaler compatibility across PyTorch 1.8 -> 2.6+
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            scaler = torch.amp.GradScaler(autocast_device, enabled=(args.precision == "fp16"))
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=(args.precision == "fp16" and device.type == "cuda"))
    elif hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        scaler = torch.cuda.amp.GradScaler(enabled=(args.precision == "fp16" and device.type == "cuda"))
    else:
        scaler = None

    def get_autocast_context():
        if hasattr(torch, "autocast"):
            return torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp)
        elif hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
            return torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda"))
        else:
            from contextlib import nullcontext
            return nullcontext()

    best_peri_mabe = float('inf')
    best_checkpoint_path = os.path.join(args.checkpoint_dir, "best_volumetric_rnfl_net.pt")
    training_start_time = time.time()

    if args.resume_checkpoint and os.path.isfile(args.resume_checkpoint):
        print(f"[Training] Initializing weights from checkpoint: {args.resume_checkpoint}")
        ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'val_metrics' in ckpt and 'peri_nfl_mabe' in ckpt['val_metrics']:
            baseline_mabe = ckpt['val_metrics']['peri_nfl_mabe']
            print(f"[Training] Baseline peripapillary NFL MABE from checkpoint: {baseline_mabe:.2f} um")

    print("\n==========================================================================================")
    print("=== STARTING VOLUMETRIC RNFL MODEL TRAINING                                           ===")
    print("==========================================================================================")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            for k in ['image', 'mask', 'ilm_surface', 'nfl_surface', 'cup_absent', 'is_peripapillary']:
                batch[k] = batch[k].to(device)

            with get_autocast_context():
                preds = model(batch['image'])
                loss_dict = criterion(preds, batch)
                loss = loss_dict['loss'] / args.accum_steps

            if args.precision == "fp16":
                scaler.scale(loss).backward()
                if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_loss += loss_dict['loss'].item()

            if (step + 1) % args.log_interval == 0:
                step_elapsed = time.time() - t0
                speed = ((step + 1) * args.batch_size) / step_elapsed
                edge_val = loss_dict.get('loss_edge', torch.tensor(0.0)).item()
                print(
                    f"Epoch [{epoch}/{args.epochs}] Step [{step+1}/{len(train_loader)}] "
                    f"Loss: {loss_dict['loss'].item():.2f} "
                    f"(Dice: {loss_dict['loss_dice'].item():.3f}, Bnd: {loss_dict['loss_boundary'].item():.1f} um, Edge: {edge_val:.3f}) | "
                    f"Speed: {speed:.1f} slices/s",
                    flush=True
                )

        scheduler.step()
        train_loss = epoch_loss / len(train_loader)
        train_time = time.time() - t0

        # Run Validation
        t_val_start = time.time()
        val_metrics = validate(model, val_loader, criterion, device, autocast_device, amp_dtype, use_amp)
        val_time = time.time() - t_val_start

        epoch_total_time = time.time() - t0
        total_elapsed = time.time() - training_start_time
        avg_epoch_time = total_elapsed / epoch
        remaining_eta = avg_epoch_time * (args.epochs - epoch)

        print("------------------------------------------------------------------------------------------", flush=True)
        print(
            f"Epoch [{epoch}/{args.epochs}] Total Time: {epoch_total_time:.1f}s (Train: {train_time:.1f}s, Val: {val_time:.1f}s) | "
            f"Train Loss: {train_loss:.2f} | Val Loss: {val_metrics['loss']:.2f} | "
            f"Val Dice: {val_metrics['dice']:.4f} | "
            f"Val Peri NFL MABE: {val_metrics['peri_nfl_mabe']:.2f} um | "
            f"Val Peri NFL P95: {val_metrics['peri_nfl_p95']:.2f} um",
            flush=True
        )
        print(
            f"[Telemetry] Throughput: {len(train_ds)/train_time:.1f} train slices/s | "
            f"Cumulative Elapsed: {total_elapsed/60:.1f}m | Remaining ETA: {remaining_eta/60:.1f}m",
            flush=True
        )
        print("------------------------------------------------------------------------------------------", flush=True)

        # Always save latest checkpoint
        latest_checkpoint_path = os.path.join(args.checkpoint_dir, "latest_volumetric_rnfl_net.pt")
        ckpt_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
            'args': vars(args)
        }
        torch.save(ckpt_data, latest_checkpoint_path)

        # Save Best Checkpoint
        if val_metrics['peri_nfl_mabe'] < best_peri_mabe:
            best_peri_mabe = val_metrics['peri_nfl_mabe']
            torch.save(ckpt_data, best_checkpoint_path)
            print(f"[Checkpoint] New best peripapillary NFL MABE ({best_peri_mabe:.2f} um) saved to {best_checkpoint_path}\n", flush=True)

    print("\n==========================================================================================")
    print(f"=== TRAINING COMPLETE. Best Peripapillary NFL MABE: {best_peri_mabe:.2f} um ===")
    print("==========================================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="/Users/nikhilmundhra/Library/CloudStorage/Box-Box/deidentified")
    parser.add_argument("--val_subjects", type=str, default="BEH0335,BEH0314")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--context_slices", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
                        help="Training precision: bf16 (bfloat16, recommended for MPS), fp16 (float16 with GradScaler), or fp32")
    parser.add_argument("--resume_checkpoint", type=str, default="",
                        help="Path to pre-trained checkpoint to resume or fine-tune from")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/train_rnfl_volumetric")
    parser.add_argument("--log_interval", type=int, default=100)
    args = parser.parse_args()

    train(args)
