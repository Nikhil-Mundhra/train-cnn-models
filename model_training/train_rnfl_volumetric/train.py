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

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import SolixRNFLDataset, AXIAL_UM
from model import VolumetricRNFLNet, CanonicalVolumetricRNFLNet, TransUNetRNFLNet
from augmentations import OCTRobustnessAugmenter

from losses import VolumetricRNFLLoss
from evaluation_manifest import evaluation_subjects


def compute_batch_metrics(preds, targets):
    """
    Computes validation metrics on a batch:
    - Dice Score on binary RNFL mask
    - ILM and NFL MABE in microns
    - Cup absence classification IoU
    """
    with torch.no_grad():
        # 1. Mask Dice with calibrated threshold (0.40) to preserve thin peripheral sheets
        prob = torch.sigmoid(preds['mask_logits'])
        pred_mask = (prob > 0.40).float()
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
    peri_nfl_errs_od = []
    peri_nfl_errs_os = []

    with torch.no_grad():
        with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
            for batch in val_loader:
                for k in ['image', 'mask', 'ilm_surface', 'nfl_surface', 'cup_absent', 'is_peripapillary']:
                    batch[k] = batch[k].to(device)

                eye_idx = batch.get('eye_idx', None)
                if eye_idx is not None and getattr(model, 'use_laterality_embedding', False):
                    preds = model(batch['image'], eye_idx=eye_idx.to(device))
                else:
                    preds = model(batch['image'])

                losses = criterion(preds, batch)
                val_loss += losses['loss'].item()

                metrics = compute_batch_metrics(preds, batch)
                dices.append(metrics['dice'])
                ilm_mabes.append(metrics['ilm_mabe'])
                nfl_mabes.append(metrics['nfl_mabe'])
                cup_ious.append(metrics['cup_iou'])

                # Collect peripapillary slice metrics (overall, OD, and OS)
                is_peri = batch['is_peripapillary']
                if is_peri.any():
                    cup_absent = batch['cup_absent'][is_peri]
                    tissue = (1.0 - cup_absent)
                    nfl_err = torch.abs(preds['nfl_pred'][is_peri] - batch['nfl_surface'][is_peri]) * AXIAL_UM
                    eyes = [batch['eye'][i] for i in range(len(batch['eye'])) if is_peri[i]]
                    for i_peri, eye_str in enumerate(eyes):
                        t = tissue[i_peri]
                        e = nfl_err[i_peri]
                        valid_err = e[t > 0.5].cpu().numpy()
                        if len(valid_err) > 0:
                            peri_nfl_errs.extend(valid_err)
                            if eye_str.upper() == 'OD':
                                peri_nfl_errs_od.extend(valid_err)
                            else:
                                peri_nfl_errs_os.extend(valid_err)

    n_batches = len(val_loader)
    avg_loss = val_loss / n_batches
    avg_dice = np.mean(dices)
    avg_ilm_mabe = np.mean(ilm_mabes)
    avg_nfl_mabe = np.mean(nfl_mabes)
    avg_cup_iou = np.mean(cup_ious)

    peri_mabe = float(np.mean(peri_nfl_errs)) if peri_nfl_errs else avg_nfl_mabe
    peri_p95 = float(np.percentile(peri_nfl_errs, 95)) if peri_nfl_errs else 0.0
    peri_od_mabe = float(np.mean(peri_nfl_errs_od)) if peri_nfl_errs_od else peri_mabe
    peri_os_mabe = float(np.mean(peri_nfl_errs_os)) if peri_nfl_errs_os else peri_mabe

    return {
        'loss': avg_loss,
        'dice': avg_dice,
        'ilm_mabe': avg_ilm_mabe,
        'nfl_mabe': avg_nfl_mabe,
        'cup_iou': avg_cup_iou,
        'peri_nfl_mabe': peri_mabe,
        'peri_nfl_p95': peri_p95,
        'peri_od_mabe': peri_od_mabe,
        'peri_os_mabe': peri_os_mabe
    }


def train(args):
    # Setup Distributed Data Parallel (DDP) if launched via torchrun / SLURM
    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))

    if is_distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    else:
        device = torch.device(args.device if args.device else ("mps" if torch.backends.mps.is_available() else "cpu"))

    autocast_device = "mps" if device.type == "mps" else ("cuda" if device.type == "cuda" else "cpu")
    use_amp = args.precision in ["bf16", "fp16"]
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else (torch.float16 if args.precision == "fp16" else torch.float32)

    if rank == 0:
        print(f"[Training] Using compute device: {device} (Distributed: {is_distributed}, World Size: {dist.get_world_size() if is_distributed else 1}) | Precision: {args.precision.upper()} (AMP: {use_amp}, dtype: {amp_dtype})")
        os.makedirs(args.checkpoint_dir, exist_ok=True)


    # 1. Subject-level Dataset Partitioning
    import glob

    tsv_good_dir = os.path.join(args.dataset_root, "tsv", "good")
    dcm_dir = os.path.join(args.dataset_root, "dicom")
    if os.path.exists(tsv_good_dir):
        raw_subjects = sorted([d for d in os.listdir(tsv_good_dir) if os.path.isdir(os.path.join(tsv_good_dir, d)) and not d.startswith('.')])
    else:
        raw_subjects = [
            'BEH0086', 'BEH0090', 'BEH0096', 'BEH0174', 'BEH0181', 'BEH0185',
            'BEH0241', 'BEH0249', 'BEH0259', 'BEH0264', 'BEH0282', 'BEH0284',
            'BEH0287', 'BEH0294', 'BEH0310', 'BEH0314', 'BEH0321', 'BEH0335',
            'BEH0349', 'BEH0354', 'BEH0364', 'BEH0398', 'BEH0410'
        ]
    all_subjects = [
        s for s in raw_subjects
        if glob.glob(os.path.join(dcm_dir, s, f"*{args.protocol}*_OPT.dcm"))
    ]
    if not all_subjects:
        all_subjects = raw_subjects

    if args.split_manifest:
        val_subjects = evaluation_subjects(args.split_manifest)
    else:
        val_subjects = [s.strip() for s in args.val_subjects.split(',') if s.strip()]

    if args.train_manifest:
        with open(args.train_manifest, "r", encoding="utf-8") as f:
            t_data = json.load(f)
        t_subjs = t_data.get("subjects") or [s["subject"] for s in t_data.get("scans", [])]
        train_subjects = sorted(set(t_subjs))
    elif args.train_subjects:
        train_subjects = [s.strip() for s in args.train_subjects.split(',') if s.strip()]
    else:
        train_subjects = [s for s in all_subjects if s not in val_subjects]

    leakage = set(train_subjects) & set(val_subjects)
    assert not leakage, f"FATAL DATA LEAKAGE: Subject(s) {leakage} appear in both train and validation sets!"

    if rank == 0:
        print(f"[Training] Protocol: {args.protocol} | Augmentation: {args.augment} | Laterality Emb: {args.use_laterality_embedding}")
        print(f"[Training] Total Cohort Pool: {len(all_subjects)} subjects with valid {args.protocol} scans")
        print(f"[Training] Training Subjects ({len(train_subjects)}): {train_subjects}")
        print(f"[Training] Validation Subjects ({len(val_subjects)}): {val_subjects}")

    train_ds = SolixRNFLDataset(
        dataset_root=args.dataset_root,
        subjects=train_subjects,
        protocol=args.protocol,
        context_slices=args.context_slices,
        enable_orthogonal=args.enable_orthogonal,
        augment=args.augment
    )
    val_ds = SolixRNFLDataset(
        dataset_root=args.dataset_root,
        subjects=val_subjects,
        protocol=args.protocol,
        context_slices=args.context_slices,
        enable_orthogonal=False,  # Keep validation on standard clinical B-scan planes
        augment=False
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )


    # 2. Instantiate Model, Loss, Optimizer
    if args.channels:
        model_channels = tuple(int(c.strip()) for c in args.channels.split(','))
    else:
        b = args.base_channels
        model_channels = (b, b * 2, b * 4, b * 8, b * 16)

    if getattr(args, 'arch', 'unet') == 'transunet':
        print(f"[Architecture] Initializing TransUNetRNFLNet (base_channels={args.base_channels}, hidden_size={args.transunet_hidden_size}, layers={args.transunet_layers}, heads={args.transunet_heads}, laterality_emb={args.use_laterality_embedding}, use_stn={getattr(args, 'use_stn', False)})...")
        model = TransUNetRNFLNet(
            in_channels=args.context_slices,
            base_channels=args.base_channels,
            hidden_size=args.transunet_hidden_size,
            num_layers=args.transunet_layers,
            num_heads=args.transunet_heads,
            mlp_dim=args.transunet_mlp_dim,
            use_laterality_embedding=args.use_laterality_embedding,
            use_stn=getattr(args, 'use_stn', False),
        ).to(device)
    elif getattr(args, 'use_stn', False):
        print(f"[Architecture] Initializing CanonicalVolumetricRNFLNet with STN (base_channels={args.base_channels}, channels={model_channels}, res_units={args.num_res_units}, laterality_emb={args.use_laterality_embedding})...")
        model = CanonicalVolumetricRNFLNet(
            in_channels=args.context_slices,
            base_channels=args.base_channels,
            channels=model_channels,
            strides=(2, 2, 2, 2),
            num_res_units=args.num_res_units,
            use_laterality_embedding=args.use_laterality_embedding
        ).to(device)
    else:
        print(f"[Architecture] Initializing VolumetricRNFLNet (base_channels={args.base_channels}, channels={model_channels}, res_units={args.num_res_units}, laterality_emb={args.use_laterality_embedding})...")
        model = VolumetricRNFLNet(
            in_channels=args.context_slices,
            base_channels=args.base_channels,
            channels=model_channels,
            strides=(2, 2, 2, 2),
            num_res_units=args.num_res_units,
            use_laterality_embedding=args.use_laterality_embedding
        ).to(device)


    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Architecture] Trainable Parameters: {total_params:,}")

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
    best_score = float('inf')
    best_checkpoint_path = os.path.join(args.checkpoint_dir, "best_volumetric_rnfl_net.pt")
    training_start_time = time.time()
    start_epoch = 1

    if args.resume_checkpoint and os.path.isfile(args.resume_checkpoint):
        if rank == 0:
            print(f"[Training] Initializing weights from checkpoint: {args.resume_checkpoint}")

        ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        if getattr(args, 'use_stn', False) and not any(k.startswith('backbone_net.') for k in state_dict):
            # Checkpoint from baseline model without STN: map keys to backbone_net
            adapted_dict = {f"backbone_net.{k}": v for k, v in state_dict.items()}
            missing, unexpected = model.load_state_dict(adapted_dict, strict=False)
            if rank == 0:
                print(f"[Training] Loaded baseline weights into CanonicalVolumetricRNFLNet backbone. STN initialized to identity.")

        elif not getattr(args, 'use_stn', False) and any(k.startswith('backbone_net.') for k in state_dict):
            # Checkpoint from STN model into standard baseline: strip backbone_net. prefix
            adapted_dict = {k.replace('backbone_net.', ''): v for k, v in state_dict.items() if not k.startswith('stn.')}
            missing, unexpected = model.load_state_dict(adapted_dict, strict=False)
            if rank == 0:
                print(f"[Training] Loaded STN backbone weights into VolumetricRNFLNet.")
        else:
            model.load_state_dict(state_dict)

        if isinstance(ckpt, dict):
            if 'epoch' in ckpt:
                start_epoch = ckpt['epoch'] + 1
                if rank == 0:
                    print(f"[Training] Resuming from epoch {start_epoch} (completed epoch {ckpt['epoch']})")
            if 'optimizer_state_dict' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    if rank == 0:
                        print("[Training] Optimizer state successfully restored.")
                except Exception as e:
                    if rank == 0:
                        print(f"[Training] Notice: Could not restore optimizer state: {e}")
            if 'scheduler_state_dict' in ckpt:
                try:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                except Exception:
                    pass
            elif start_epoch > 1:
                # Advance scheduler to current epoch
                for _ in range(start_epoch - 1):
                    scheduler.step()

            if 'val_metrics' in ckpt and 'peri_nfl_mabe' in ckpt['val_metrics']:
                baseline_mabe = ckpt['val_metrics']['peri_nfl_mabe']
                best_peri_mabe = baseline_mabe
                best_score = baseline_mabe + 50.0 * max(0.0, 0.75 - ckpt['val_metrics'].get('dice', 0.8))
                if rank == 0:
                    print(f"[Training] Baseline peripapillary NFL MABE from checkpoint: {baseline_mabe:.2f} um")

    # Wrap model with DistributedDataParallel if running multi-GPU
    raw_model = model
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if rank == 0:
        print("\n==========================================================================================")
        print(f"=== STARTING VOLUMETRIC RNFL MODEL TRAINING (Epochs {start_epoch} -> {args.epochs})   ===")
        print("==========================================================================================")


    for epoch in range(start_epoch, args.epochs + 1):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        t0 = time.time()


        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            for k in ['image', 'mask', 'ilm_surface', 'nfl_surface', 'cup_absent', 'is_peripapillary']:
                batch[k] = batch[k].to(device, non_blocking=(device.type == "cuda"))

            if getattr(args, 'robust_augment', False):
                batch = OCTRobustnessAugmenter().to(device)(batch)
            elif args.augment:
                # Fast GPU Vectorized Augmentations (< 0.2ms total for batch of 32 on A100)
                B = batch['image'].shape[0]
                scales = torch.empty(B, 1, 1, 1, device=device).uniform_(0.85, 1.15)
                gammas = torch.empty(B, 1, 1, 1, device=device).uniform_(0.90, 1.10)
                batch['image'] = (batch['image'] * scales).clamp(0.0, 1.0).pow(gammas).clamp(0.0, 1.0)
                if torch.rand(1, device=device).item() > 0.5:
                    noise = torch.randn_like(batch['image']) * 0.012
                    batch['image'] = (batch['image'] + noise).clamp(0.0, 1.0)

            eye_idx = batch.get('eye_idx', None)
            if eye_idx is not None and args.use_laterality_embedding:
                eye_idx = eye_idx.to(device)
            else:
                eye_idx = None

            with get_autocast_context():
                if eye_idx is not None:
                    preds = model(batch['image'], eye_idx=eye_idx)
                else:
                    preds = model(batch['image'])
                loss_dict = criterion(preds, batch)
                loss = loss_dict['loss'] / args.accum_steps
                if 'predicted_tilt_deg' in preds:
                    # Mild L2 regularization to anchor STN when scan is naturally horizontal
                    loss = loss + (1e-4 * (preds['predicted_tilt_deg'] ** 2).mean()) / args.accum_steps


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

            if rank == 0 and (step + 1) % args.log_interval == 0:
                step_elapsed = time.time() - t0
                world_sz = dist.get_world_size() if is_distributed else 1
                speed = ((step + 1) * args.batch_size * world_sz) / step_elapsed
                edge_val = loss_dict.get('loss_edge', torch.tensor(0.0)).item()
                overlap_val = loss_dict.get('loss_overlap', loss_dict.get('loss_dice', torch.tensor(0.0))).item()
                thick_val = loss_dict.get('loss_thick', torch.tensor(0.0)).item()
                print(
                    f"Epoch [{epoch}/{args.epochs}] Step [{step+1}/{len(train_loader)}] "
                    f"Loss: {loss_dict['loss'].item():.2f} "
                    f"(Overlap: {overlap_val:.3f}, Thick: {thick_val:.1f} um, Bnd: {loss_dict['loss_boundary'].item():.1f} um, Edge: {edge_val:.3f}) | "
                    f"Speed: {speed:.1f} slices/s",
                    flush=True
                )

        scheduler.step()
        train_loss = epoch_loss / len(train_loader)
        train_time = time.time() - t0

        # Run Validation (evaluated on rank 0 or all ranks)
        t_val_start = time.time()
        eval_model = model.module if is_distributed else model
        val_metrics = validate(eval_model, val_loader, criterion, device, autocast_device, amp_dtype, use_amp)
        val_time = time.time() - t_val_start

        epoch_total_time = time.time() - t0
        total_elapsed = time.time() - training_start_time
        avg_epoch_time = total_elapsed / epoch
        remaining_eta = avg_epoch_time * (args.epochs - epoch)

        if rank == 0:
            print("------------------------------------------------------------------------------------------", flush=True)
            print(
                f"Epoch [{epoch}/{args.epochs}] Total Time: {epoch_total_time:.1f}s (Train: {train_time:.1f}s, Val: {val_time:.1f}s) | "
                f"Train Loss: {train_loss:.2f} | Val Loss: {val_metrics['loss']:.2f} | "
                f"Val Dice: {val_metrics['dice']:.4f} | "
                f"Val Peri NFL MABE: {val_metrics['peri_nfl_mabe']:.2f} um (OD: {val_metrics['peri_od_mabe']:.2f} um, OS: {val_metrics['peri_os_mabe']:.2f} um) | "
                f"Val Peri NFL P95: {val_metrics['peri_nfl_p95']:.2f} um",
                flush=True
            )
            print(
                f"[Telemetry] Throughput: {len(train_ds)/train_time:.1f} train slices/s | "
                f"Cumulative Elapsed: {total_elapsed/60:.1f}m | Remaining ETA: {remaining_eta/60:.1f}m",
                flush=True
            )
            print("------------------------------------------------------------------------------------------", flush=True)

            # Always save latest checkpoint without DDP module prefix
            latest_checkpoint_path = os.path.join(args.checkpoint_dir, "latest_volumetric_rnfl_net.pt")
            saved_state_dict = eval_model.state_dict()
            ckpt_data = {
                'epoch': epoch,
                'model_state_dict': saved_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'args': vars(args)
            }
            torch.save(ckpt_data, latest_checkpoint_path)

            # Save Best Checkpoint: balanced score guarding against thin mask collapse
            # Penalizes if Dice drops below 0.75 while optimizing peripapillary MABE
            score = val_metrics['peri_nfl_mabe'] + 50.0 * max(0.0, 0.75 - val_metrics['dice'])
            if score < best_score:
                best_score = score
                best_peri_mabe = val_metrics['peri_nfl_mabe']
                torch.save(ckpt_data, best_checkpoint_path)
                print(f"[Checkpoint] New best balanced model (Score: {score:.2f} | Peri NFL MABE: {best_peri_mabe:.2f} um [OD: {val_metrics['peri_od_mabe']:.2f}, OS: {val_metrics['peri_os_mabe']:.2f}] | Dice: {val_metrics['dice']:.4f}) saved to {best_checkpoint_path}\n", flush=True)


    print("\n==========================================================================================")
    print(f"=== TRAINING COMPLETE. Best Peripapillary NFL MABE: {best_peri_mabe:.2f} um ===")
    print("==========================================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="/Users/nikhilmundhra/Library/CloudStorage/Box-Box/OCT_Segmentations_Solix/deidentified-new")
    parser.add_argument("--val_subjects", type=str, default="BEH0335,BEH0314")
    parser.add_argument("--train_subjects", type=str, default="", help="Optional explicit comma-separated training subjects (for fine-tuning on audited subsets)")
    parser.add_argument("--split_manifest", type=str, default="", help="Optional JSON split manifest file containing held_out validation subjects")
    parser.add_argument("--train_manifest", type=str, default="", help="Optional JSON manifest file containing explicit training subjects")
    parser.add_argument("--protocol", type=str, default="Disc Cube", help="OCT scan protocol to train on (e.g. 'Disc Cube')")
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
    parser.add_argument("--base_channels", type=int, default=16, help="Base channel multiplier (16 -> ~1.67M params, 32 -> ~6.6M params)")
    parser.add_argument("--channels", type=str, default="", help="Comma-separated channel counts across stages (e.g. '32,64,128,256,512')")
    parser.add_argument("--num_res_units", type=int, default=2, help="Number of residual units per stage")
    parser.add_argument("--enable_orthogonal", action="store_true", default=True, help="Enable orthogonal bi-planar vertical slicing during training")
    parser.add_argument("--disable_orthogonal", dest="enable_orthogonal", action="store_false", help="Disable orthogonal bi-planar training")
    parser.add_argument("--augment", action="store_true", default=True, help="Enable training data augmentations (speckle, contrast, depth jitter)")
    parser.add_argument("--no_augment", dest="augment", action="store_false", help="Disable training data augmentations")
    parser.add_argument("--use_stn", action="store_true", default=False, help="Enable ConstrainedSpatialTransformer STN canonicalizer")
    parser.add_argument("--robust_augment", action="store_true", default=False, help="Enable physics-informed OCT robustness augmentations (speckle, shadows, roll-off, dropout)")
    parser.add_argument("--use_laterality_embedding", action="store_true", default=False, help="Enable explicit OD/OS conditioning embedding")
    parser.add_argument("--arch", type=str, default="unet", choices=["unet", "transunet"],
                        help="Network architecture: 'unet' (standard VolumetricRNFLNet) or 'transunet' (hybrid CNN-Transformer TransUNetRNFLNet)")
    parser.add_argument("--transunet_hidden_size", type=int, default=256, help="Transformer embedding dimension for TransUNet")
    parser.add_argument("--transunet_layers", type=int, default=6, help="Number of Transformer encoder layers for TransUNet")
    parser.add_argument("--transunet_heads", type=int, default=8, help="Number of attention heads for TransUNet")
    parser.add_argument("--transunet_mlp_dim", type=int, default=512, help="Feedforward MLP dimension for TransUNet")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/train_rnfl_volumetric")
    parser.add_argument("--log_interval", type=int, default=100)
    args = parser.parse_args()


    train(args)
