# Volumetric RNFL Deep Learning Pipeline & Benchmark

Automated deep learning model architecture and training pipeline for volumetric segmentation and boundary extraction of the **Retinal Nerve Fiber Layer (RNFL)** directly from Optovue Solix OCT DICOM volumes (`Disc Cube` and `Retina Cube`).

Trained and benchmarked against a human-reviewed reference from:
`/Users/nikhilmundhra/Library/CloudStorage/Box-Box/OCT_Segmentations_Solix/deidentified-new`

---

## 1. Overview & Clinical Context

In ophthalmic imaging (glaucoma, optic neuritis, multiple sclerosis), accurate quantification of the **Retinal Nerve Fiber Layer (RNFL)** is the primary structural biomarker of ganglion cell axonal loss.

Standard commercial segmentation algorithms (such as the Optovue Solix automated output, `bad`) frequently produce severe localized errors around the **peripapillary optic disc margin ($0 - 2$ disc radii from center)**:
- Bridging horizontally across the steep optic cup excavation ("pit fall").
- Inaccurate tracking of the neuroretinal rim slopes.
- Misidentifying the RNFL-GCL interface under blood vessel shadows.

In this dataset, a human reviewer manually corrected the disc margin, generating the reference (`good`). This deep learning pipeline trains a 2.5D multi-task residual network to learn the human-corrected boundaries and improve on the commercial machine output.

---

## 2. Quantitative Solix Baseline Benchmark

Evaluated across all 21 paired `Disc Cube` scans (`tsv/bad` vs `tsv/good`):

| Metric | Machine Baseline | Clinical Meaning |
| :--- | :--- | :--- |
| **Peripapillary NFL MABE** | **$7.87 \pm 11.64\ \mu\text{m}$** | Mean Absolute Boundary Error around the optic disc |
| **Peripapillary NFL P95 Error** | **$36.29\ \mu\text{m}$** | 95th percentile error across peripapillary A-scans |
| **Peripapillary NFL Max Error** | **$237.40\ \mu\text{m}$** | Worst-case Solix rim failure (observed on `BEH0335 OD`) |
| **A-scans with $> 10\ \mu\text{m}$ Error** | **$13.6\%$** | Up to **$50.4\%$** on severely affected scans (`BEH0314 OD`) |
| **Cup Absence IoU** | **$0.8372$** | Solix misclassifies the cup boundary (down to **$0.1789$**) |
| **Peripapillary ILM MABE** | **$0.19 \pm 0.31\ \mu\text{m}$** | Vitreous-retina interface is accurate; NFL is the primary failure |

To run the baseline evaluation:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
.venv/bin/python model_training/train_rnfl_volumetric/evaluate_baseline.py
```

---

## 3. Pipeline Architecture

```
                               ┌────────────────────────────────┐
                               │ Raw OCT Volume (320x768x320)   │
                               └───────────────┬────────────────┘
                                               │
                                 Fast Binary Buffer Seeking
                                   (1.5 ms / B-scan slice)
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │ 2.5D Multi-Slice Input Tensor    │
                              │ (5 Channels: Z-2, Z-1, Z, Z+1, Z+2)
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │   VolumetricRNFLNet Backbone     │
                              │   MONAI Residual U-Net (2D)      │
                              │   Channels: 16, 32, 64, 128, 256 │
                              └────┬───────────┬──────────────┬──┘
                                   │           │              │
                   ┌───────────────┘           │              └───────────────┐
                   ▼                           ▼                              ▼
      ┌─────────────────────────┐ ┌─────────────────────────┐  ┌─────────────────────────┐
      │ Dense RNFL Voxel Mask   │ │ 1D Boundary Regression  │  │ 1D Cup Absence Head     │
      │ Output: (B, 1, 768, 320)│ │ ILM & NFL Depths in px │  │ Optic Cup Cavity Flag   │
      │ Loss: Dice + BCE        │ │ Loss: Smooth L1 in µm   │  │ Loss: BCEWithLogits     │
      └─────────────────────────┘ └─────────────────────────┘  └─────────────────────────┘
                   │                           │                              │
                   └───────────────────────────┼──────────────────────────────┘
                                               ▼
                               ┌────────────────────────────────┐
                               │ Multi-Task Loss + Topo Penalty │
                               │ L_dice + L_bce + L_bnd + L_topo│
                               └────────────────────────────────┘
```

### Key Technical Innovations:
1. **2.5D Volumetric Context**: Stacking 5 adjacent B-scans ($z-2, z-1, z, z+1, z+2$) gives the network 3D inter-slice spatial continuity (spaced $18.8\ \mu\text{m}$ apart) without the prohibitive memory cost of full 3D dense volumes.
2. **Standardized Eye Orientation**: Anatomically normalizes left eyes (OS) by horizontal flipping, ensuring nasal and temporal bundles are uniformly oriented for the neural network.
3. **Multi-Task Dense + Boundary Learning**: Combines pixel-level volumetric segmentation (for 3D mesh reconstruction in Slicer) with direct continuous boundary regression in microns.
4. **Topological Surface Ordering**: Enforces $\text{ReLU}(\hat{y}_{\text{ILM}} - \hat{y}_{\text{NFL}})$ penalty, mathematically preventing the inner retinal surface from crossing below the outer boundary.
5. **Anisotropic TransUNet Hybrid Architecture (`transunet.py`)**:
   - **Anisotropic Axial-Preserving Tokenization**: Decouples axial ($Z$) vs. lateral ($X$) downsampling, preserving 96–192 depth rows (bin size $\sim 12.5\,\mu\text{m}$ vs. $50\,\mu\text{m}$ in naive ViTs). Prevents blurring of the thin sub-millimeter RNFL-GCL interface.
   - **Multi-Scale Boundary Injection**: Direct lateral skip connections from the high-resolution CNN stem ($1/1$ scale) feed directly into the continuous 1D boundary regression heads, ensuring physical voxel edge alignment.
   - **Constrained Spatial Transformer (STN)**: Pre-aligns variable patient tilt ($\theta, t_y$) to a horizontal coordinate frame prior to self-attention tokenization, eliminating diagonal grid artifacts.
   - **Cascaded Upsampler (CUP) Decoder**: Progressively restores spatial dimensions through dynamic bilinear interpolation and skip concatenation.

---

## 4. Directory Structure

```
train-cnn-models/model_training/train_rnfl_volumetric/
├── README.md              <- This documentation
├── evaluate_baseline.py   <- Computes quantitative Solix machine baseline vs ground truth
├── dataset.py             <- SolixRNFLDataset: DICOM seeker, 2.5D batching, ground-truth parser
├── model.py               <- VolumetricRNFLNet: 2.5D multi-task residual network
├── transunet.py           <- TransUNetRNFLNet: Hybrid CNN-ViT encoder with Cascaded Upsampler (CUP)
├── losses.py              <- VolumetricRNFLLoss: Dice, boundary Huber, cup BCE, topo penalty
├── train.py               <- Subject-grouped training loop with validation & checkpointing (--arch transunet support)
├── export_to_slicer.py    <- Full-volume inference and interactive 3D Slicer exporter
├── batch_cohort_evaluator.py <- Subject-disjoint cohort evaluation and report assets
├── build_cohort_report.py <- Executive cohort report CLI orchestrator
├── train_rnfl_transunet_jubail.slurm <- SLURM script for A100 HPC cluster training of TransUNet
├── reporting/             <- Modular visualization and report generation suite
│   ├── theme.py           <- Dual-theme color palettes (light for PDF, dark for markdown)
│   ├── charts.py          <- High-res publication chart generators
│   ├── markdown.py        <- List normalization and markdown report templating
│   └── compiler.py        <- Headless browser discovery and HTML/PDF compiler
├── run_orientation_ablation.py <- Controlled OS orientation/fusion experiment
├── orientation.py         <- Shared OD/OS coordinate transforms
├── quality_control.py     <- Operational failure detection and review routing
├── evaluation_manifest.py <- JSON/CSV subject split validation
├── EVALUATION_AND_QC.md   <- Evaluation, QC, and comparator protocol
└── tests/
    ├── test_pipeline.py   <- Unit tests for dataset, model, loss, and disc geometry
    ├── test_transunet.py  <- Unit tests for TransUNet forward pass, tensor dimensions, and backward gradient flow
    ├── test_orientation_qc.py <- Orientation, manifest, and QC regression tests
    ├── test_dataset_and_export.py <- Curve matching, multi-arm discovery, and Slicer export tests
    └── test_reporting.py  <- Unit tests for charts, markdown normalization, and PDF compiler
```

---

## 5. How to Run Training & Inference

### Environment Setup
Ensure the virtual environment is activated and the OpenMP guard is set:
```bash
cd /Users/nikhilmundhra/Documents/Github/Capstone/train-cnn-models
export KMP_DUPLICATE_LIB_OK=TRUE
```

### Running Unit Tests
```bash
.venv/bin/python model_training/train_rnfl_volumetric/tests/test_pipeline.py
```

### Launching Multi-Epoch Training
```bash
.venv/bin/python model_training/train_rnfl_volumetric/train.py \
    --dataset_root "/Users/nikhilmundhra/Library/CloudStorage/Box-Box/OCT_Segmentations_Solix/deidentified-new" \
    --val_subjects "BEH0335,BEH0314,BEH0086" \
    --epochs 15 \
    --batch_size 4 \
    --accum_steps 2 \
    --lr 3e-4 \
    --checkpoint_dir "./checkpoints/rnfl_production"
```

#### Training Arguments:
- `--dataset_root`: Path to deidentified Box dataset folder.
- `--val_subjects`: Optional comma-separated validation override. Evaluation defaults to the subject-disjoint validation list stored in the checkpoint (`BEH0335,BEH0314,BEH0086` for job 18223981).
- `--epochs`: Number of full passes through the training dataset (default: 15).
- `--batch_size`: Physical batch size per step on Apple Silicon GPU (`mps`).
- `--accum_steps`: Gradient accumulation steps (effective batch size = `batch_size * accum_steps = 8`).
- `--lr`: Initial learning rate (default: `3e-4` with Cosine Annealing scheduler).
- `--context_slices`: Number of adjacent B-scans in 2.5D tensor (default: 5).
- `--checkpoint_dir`: Directory to save `best_volumetric_rnfl_net.pt`.

#### Training with the TransUNet Architecture:
To train using the hybrid CNN-Transformer architecture:
```bash
.venv/bin/python model_training/train_rnfl_volumetric/train.py \
    --dataset_root "/Users/nikhilmundhra/Library/CloudStorage/Box-Box/OCT_Segmentations_Solix/deidentified-new" \
    --split_manifest "model_training/train_rnfl_3d/manifests/stratified_held_out_v2.json" \
    --train_manifest "model_training/train_rnfl_3d/manifests/phase1_train_manifest.json" \
    --arch "transunet" \
    --transunet_hidden_size 256 \
    --transunet_layers 6 \
    --transunet_heads 8 \
    --transunet_mlp_dim 512 \
    --epochs 20 \
    --batch_size 16 \
    --accum_steps 2 \
    --lr 3e-4 \
    --checkpoint_dir "./checkpoints/rnfl_transunet"
```

### Running Volumetric Inference & Exporting to 3D Slicer
To segment an entire 320-slice DICOM volume and visualize the 3D surface model in Slicer:
```bash
.venv/bin/python model_training/train_rnfl_volumetric/export_to_slicer.py \
    --checkpoint "./checkpoints/rnfl_production/best_volumetric_rnfl_net.pt" \
    --dcm "/Users/nikhilmundhra/Library/CloudStorage/Box-Box/deidentified/dicom/BEH0181/BEH0181_Disc Cube_OD_2025-05-20_12-15-20_OPT.dcm" \
    --output_dir "./predictions/BEH0181" \
    --launch_slicer
```
This generates:
1. `*_rnfl_pred.npz`: Compressed 3D binary labelmap `(320 x 768 x 320)`.
2. `view_prediction_in_slicer.py`: Script that launches 3D Slicer, loads the master OCT volume, imports the predicted RNFL layer, and generates the closed 3D surface mesh.

### Cohort evaluation and quality control

Use `batch_cohort_evaluator.py` for corrected biplanar cohort inference. It
supports a frozen subject-split manifest, targeted subject filters, explicit OS
orientation modes, automatic failure flags, and commercial-annotation
availability auditing. See [EVALUATION_AND_QC.md](EVALUATION_AND_QC.md) for the
commands and interpretation rules.

---

## 6. Running Training on NYUAD HPC (Jubail Cluster)

Training runs directly on the NYUAD Jubail HPC cluster using SLURM, NVIDIA GPU acceleration, and the dedicated PyTorch CUDA environment on scratch.

### Cluster Architecture & Environment
* **Python & PyTorch Environment**: `/scratch/nm4358/envs/oct-env/bin/python` (PyTorch 2.8.0 + CUDA 12.8).
* **Data Storage (`/scratch/`)**: Full 23-subject cohort (46 paired volumes) at `/scratch/nm4358/deidentified/`. Output checkpoints saved to `/scratch/nm4358/checkpoints/`.
* **Repository Location**: `~/train-cnn-models` (code resides in `/home`, fast I/O data resides on `/scratch`).
* **Partition & Resources**: `#SBATCH -p nvidia` with `--gres=gpu:a100:1` (or `--gres=gpu:1`), `-c 8` CPU cores, and `--mem=32G`.

### Step 1: Transfer Dataset to Jubail Scratch (Direct Cloud-to-Cluster Streaming)
Do **not** download unhydrated files to your local Mac SSD via Box Drive, as this can exhaust local disk cache. Instead, stream directly in-memory from Box API to Jubail SFTP using `rclone` (or the helper script):

```bash
# Recommended: Stream directly via rclone (zero local Mac disk usage)
rclone copy "rnfl-box:OCT_Segmentations_Solix/deidentified-new" "jubail:/scratch/nm4358/deidentified-new" \
    --exclude "*Retina Cube*" \
    --transfers 4 \
    --checkers 8 \
    -P -v

# Or use the helper wrapper:
./scripts/sync_jubail.sh push-new-data
```

### Step 2: Connect to Jubail & Pull Latest Code
SSH into Jubail login node and update repository:
```bash
ssh nm4358@jubail.abudhabi.nyu.edu
cd ~/train-cnn-models && git pull origin main
```

### Step 3: Interactive GPU Smoke Test (Recommended)
Before launching a multi-hour batch job, verify CUDA acceleration and data loading interactively:
```bash
# 1. Allocate a temporary interactive GPU node (15 min)
salloc -p nvidia --gres=gpu:1 -c 4 -t 00:15:00

# 2. Run quick sanity test
/scratch/nm4358/envs/oct-env/bin/python -c "
import torch
print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
"

# 3. Exit interactive node
exit
```

### Step 4: Submit the SLURM Batch Job
```bash
cd ~/train-cnn-models/model_training/train_rnfl_volumetric
sbatch train_rnfl_jubail.slurm
```

### Step 5: Monitor the Job & GPU Utilization
```bash
# Check queue status
squeue -u nm4358

# Monitor real-time GPU utilization (NYUAD CRC tool)
# Note: Jubail auto-terminates jobs if GPU utilization drops below 5% for 2 consecutive hours!
gutil <JOB_ID>

# Follow live training logs
tail -f slurm_rnfl_*.out
```

### Step 6: Retrieve Checkpoints to Local Machine
Once training completes, download the best model weights back to your local machine:
```bash
mkdir -p ./checkpoints/jubail_run
rsync -avhP nm4358@jubail.abudhabi.nyu.edu:/scratch/nm4358/checkpoints/rnfl_volumetric_*/best_volumetric_rnfl_net.pt ./checkpoints/jubail_run/
```

