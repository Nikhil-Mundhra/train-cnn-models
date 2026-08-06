"""
train.py

Unified Training CLI Entry Point for OCT Deep Learning CNN Models.
Supports training multi-head disease classification models, standard single-head classifiers,
and hierarchical retinal layer segmentation U-Nets with customizable model architectures and image sizes.

Preserves all CLI arguments of `train_convnext.py`.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Prevent PyTorch DataLoader multiprocessing deadlocks with OpenCV/ITK & Matplotlib warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['MPLCONFIGDIR'] = os.path.expanduser('~/.cache/matplotlib_antigravity')



import cv2
cv2.setNumThreads(0)

import torch
import torch.nn as nn
import torch.nn.init

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
try:
    import timm.layers.weight_init
    timm.layers.weight_init.trunc_normal_ = lambda tensor, mean=0., std=1., a=-2., b=2.: torch.nn.init.normal_(tensor, mean=mean, std=std)
except ImportError:
    pass

# Ensure project root and classification package are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
CLS_TRAIN_DIR = PROJECT_ROOT / "image-classification-model-training"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CLS_TRAIN_DIR))

from models.multi_head_convnext import build_multi_head_model, MultiHeadConvNeXt
from core_ml.segmentation.models.unet import HierarchicalUNet
from data.dataset import build_kfold_dataloaders, MultiHeadOCTDataset
from data.transforms import get_transforms
from training.multi_head_trainer import MultiHeadTrainer
from training.losses import FocalLoss
from utils.device import ComputeManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose third-party logs
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
for noisy_logger in ["httpx", "urllib3", "huggingface_hub"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# Architecture to default image size map
DEFAULT_IMAGE_SIZES = {
    "convnextv2_base": 224,
    "convnext_small": 224,
    "convnext_tiny": 224,
    "convnext_large": 224,
    "resnet18": 224,
    "resnet34": 224,
    "resnet50": 224,
    "resnet101": 224,
    "efficientnet_b0": 224,
    "efficientnet_b2": 260,
    "efficientnet_b4": 380,
    "densenet121": 224,
    "densenet201": 224,
    "vit_base_patch16_224": 224,
    "swin_tiny_patch4_window7_224": 224,
    "hierarchical_unet": 512,
    "unet": 512,
}

def resolve_image_size(arch: str, requested_size: int) -> int:
    """Returns requested image size if > 0, otherwise defaults to optimal size for target architecture."""
    if requested_size > 0:
        return requested_size
    arch_lower = arch.lower()
    return DEFAULT_IMAGE_SIZES.get(arch_lower, 224)

def parse_args():
    parser = argparse.ArgumentParser(description="Unified CNN Model Training Application")
    
    # --- Selection Arguments ---
    parser.add_argument("--task", type=str, default="multi_head",
                        choices=["multi_head", "classification", "segmentation", "unified"],
                        help="Training task type (default: multi_head)")
    parser.add_argument("--arch", "--model", "--backbone", dest="arch", type=str, default="convnextv2_base",
                        help="Model architecture or timm backbone (e.g. convnextv2_base, convnext_small, resnet50, efficientnet_b0, hierarchical_unet)")
    parser.add_argument("--img-size", "--image-size", dest="img_size", type=int, default=0,
                        help="Input image height/width resolution (default: 0 = auto-select per architecture)")
    parser.add_argument("--n-coarse-classes", type=int, default=3, help="Coarse segmentation classes for U-Net (default: 3)")
    parser.add_argument("--n-granular-classes", type=int, default=15, help="Granular segmentation classes for U-Net (default: 15)")

    # --- Retained train_convnext.py Arguments ---
    default_config = str(CLS_TRAIN_DIR / "config" / "hierarchy.yaml") if (CLS_TRAIN_DIR / "config" / "hierarchy.yaml").exists() else "config/hierarchy.yaml"
    parser.add_argument("--config", type=str, default=default_config, help="Path to dataset hierarchy.yaml")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU/device")
    parser.add_argument("--epochs-warmup", type=int, default=10, help="Number of frozen-backbone warmup epochs")
    parser.add_argument("--epochs-finetune", type=int, default=10, help="Number of stage-wise fine-tuning epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--lr-head", type=float, default=1e-4, help="Learning rate for heads / decoder")
    parser.add_argument("--lr-backbone", type=float, default=1e-6, help="Learning rate for backbone stages")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 epoch per phase to verify pipeline")
    parser.add_argument("--w-h1", type=float, default=1.0, help="Weight for Head 1 (Binary) loss")
    parser.add_argument("--w-h2", type=float, default=1.0, help="Weight for Head 2 (12-Class) loss")
    default_ckpt_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "checkpoints"
    parser.add_argument("--checkpoint-dir", type=str, default=default_ckpt_dir, help="Directory to save model checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training")
    parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face Hub repository ID for cloud backup")
    parser.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--save-steps", type=int, default=2250, help="Checkpoint save step frequency")
    parser.add_argument("--use-data-parallel", action="store_true", help="Enable PyTorch DataParallel")
    parser.add_argument("--use-ddp", action="store_true", help="Enable PyTorch DistributedDataParallel")
    parser.add_argument("--use-weighted-sampler", action="store_true", help="Enable WeightedRandomSampler for class balancing")

    return parser.parse_args()

def main():
    args = parse_args()
    resolved_img_size = resolve_image_size(args.arch, args.img_size)
    
    is_ddp_env = ("LOCAL_RANK" in os.environ) or args.use_ddp
    if is_ddp_env and torch.cuda.is_available() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    compute_manager = ComputeManager(use_data_parallel=args.use_data_parallel, use_ddp=args.use_ddp)

    if compute_manager.is_main_process:
        logger.info("=====================================================")
        logger.info("           UNIFIED CNN TRAINING PIPELINE            ")
        logger.info("=====================================================")
        logger.info(f"  Task         : {args.task}")
        logger.info(f"  Architecture : {args.arch}")
        logger.info(f"  Input Size   : {resolved_img_size}x{resolved_img_size}")
        logger.info(f"  Batch Size   : {args.batch_size}")
        logger.info(f"  Warmup Ep    : {args.epochs_warmup}")
        logger.info(f"  Finetune Ep  : {args.epochs_finetune}")
        logger.info(f"  Checkpoint   : {args.checkpoint_dir}")
        logger.info("=====================================================")

    # Handle Segmentation Architecture Selection
    if args.task == "segmentation" or args.arch in ["hierarchical_unet", "unet"]:
        if compute_manager.is_main_process:
            logger.info("Building Hierarchical U-Net Segmentation Model...")
        model = HierarchicalUNet(
            n_channels=1,
            n_coarse_classes=args.n_coarse_classes,
            n_granular_classes=args.n_granular_classes
        )
        # Note: Segmentation pipeline uses 512x512 resolution transforms
        train_transforms = get_transforms("train", img_size=resolved_img_size)
        val_transforms = get_transforms("val", img_size=resolved_img_size)
        
        # Build dataloaders
        fold_loaders = build_kfold_dataloaders(
            config_path=args.config,
            mode="multi_head",
            n_splits=5,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_transform=train_transforms,
            val_transform=val_transforms,
            use_weighted_sampler=args.use_weighted_sampler,
            is_ddp=compute_manager.is_ddp,
            rank=compute_manager.rank,
            world_size=compute_manager.world_size
        )
        
        for fold_id, (train_loader, val_loader) in enumerate(fold_loaders):
            if compute_manager.is_main_process:
                logger.info(f"=== Starting Fold {fold_id} (Segmentation) ===")
            model_prepared = compute_manager.prepare_model(model)
            # Run training using MultiHeadTrainer in segmentation mode
            trainer = MultiHeadTrainer(
                model=model_prepared,
                criterions={'h1': nn.BCEWithLogitsLoss(), 'h2': nn.CrossEntropyLoss()},
                loss_weights={'h1': args.w_h1, 'h2': args.w_h2},
                checkpoint_dir=args.checkpoint_dir,
                compute_manager=compute_manager,
                mode="multi_head"
            )
            trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                warmup_epochs=args.epochs_warmup,
                warmup_lr=args.lr_head,
                finetune_epochs=args.epochs_finetune,
                head_lr=args.lr_head,
                backbone_lr=args.lr_backbone,
                fold_id=fold_id,
                smoke_test=args.smoke_test,
                resume_path=args.resume,
                hf_repo=args.hf_repo,
                accum_steps=args.accum_steps,
                save_steps=args.save_steps,
                patience=args.patience,
                use_weighted_sampler=args.use_weighted_sampler
            )
        return

    # Handle Classification / Multi-Head Architecture Selection
    full_ds = MultiHeadOCTDataset(config_path=args.config, transform=None)
    h2_alpha = full_ds.compute_class_weights("h2")
    
    if compute_manager.is_main_process:
        class_names = full_ds.get_class_names("h2")
        logger.info("=== H2 Bounded FocalLoss Alpha Weights ===")
        for idx, (c_name, w_val) in enumerate(zip(class_names, h2_alpha.tolist())):
            logger.info(f"  {c_name:<15} : {w_val:.2f}")
        logger.info("==========================================")

    criterions = {
        'h1': nn.BCEWithLogitsLoss(),
        'h2': FocalLoss(gamma=2.0, alpha=h2_alpha, reduction="mean", label_smoothing=0.1)
    }
    loss_weights = {'h1': args.w_h1, 'h2': args.w_h2}

    train_transforms = get_transforms("train", img_size=resolved_img_size)
    val_transforms = get_transforms("val", img_size=resolved_img_size)

    fold_loaders = build_kfold_dataloaders(
        config_path=args.config,
        mode="multi_head",
        n_splits=5,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=train_transforms,
        val_transform=val_transforms,
        use_weighted_sampler=args.use_weighted_sampler,
        is_ddp=compute_manager.is_ddp,
        rank=compute_manager.rank,
        world_size=compute_manager.world_size
    )

    for fold_id, (train_loader, val_loader) in enumerate(fold_loaders):
        if compute_manager.is_main_process:
            logger.info(f"=== Starting Fold {fold_id} ({args.arch}) ===")
        
        model = build_multi_head_model(pretrained=True, warmup=True, backbone_name=args.arch)
        model = compute_manager.prepare_model(model)
        
        sub_dir = "WeightedRandomSampler" if args.use_weighted_sampler else None
        trainer = MultiHeadTrainer(
            model=model,
            criterions=criterions,
            loss_weights=loss_weights,
            checkpoint_dir=args.checkpoint_dir,
            compute_manager=compute_manager,
            mode="multi_head",
            sub_dir=sub_dir
        )
        
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            warmup_epochs=args.epochs_warmup,
            warmup_lr=args.lr_head,
            finetune_epochs=args.epochs_finetune,
            head_lr=args.lr_head,
            backbone_lr=args.lr_backbone,
            fold_id=fold_id,
            smoke_test=args.smoke_test,
            resume_path=args.resume,
            hf_repo=args.hf_repo,
            accum_steps=args.accum_steps,
            save_steps=args.save_steps,
            patience=args.patience,
            use_weighted_sampler=args.use_weighted_sampler
        )

if __name__ == "__main__":
    main()
