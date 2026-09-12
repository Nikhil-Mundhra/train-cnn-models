# Volumetric RNFL Deep Learning Pipeline & Benchmark

Automated deep learning model architecture and training pipeline for volumetric segmentation and boundary extraction of the **Retinal Nerve Fiber Layer (RNFL)** directly from Optovue Solix OCT DICOM volumes (`Disc Cube` and `Retina Cube`).

Trained and benchmarked against human clinician-reviewed reference ground truth from:
`/Users/nikhilmundhra/Library/CloudStorage/Box-Box/deidentified`

---

## 1. Overview & Clinical Context

In ophthalmic imaging (glaucoma, optic neuritis, multiple sclerosis), accurate quantification of the **Retinal Nerve Fiber Layer (RNFL)** is the primary structural biomarker of ganglion cell axonal loss.

Standard commercial segmentation algorithms (such as the Optovue Solix automated output, `bad`) frequently produce severe localized errors around the **peripapillary optic disc margin ($0 - 2$ disc radii from center)**:
- Bridging horizontally across the steep optic cup excavation ("pit fall").
- Inaccurate tracking of the neuroretinal rim slopes.
- Misidentifying the RNFL-GCL interface under blood vessel shadows.

In this dataset, clinical experts reviewed and manually corrected the disc margin, generating the ground-truth reference (`good`). This deep learning pipeline trains a 2.5D multi-task residual network to learn the clinician-corrected boundaries and beat the commercial machine output.

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

---

## 4. Directory Structure

```
train-cnn-models/model_training/train_rnfl_volumetric/
├── README.md              <- This documentation
├── evaluate_baseline.py   <- Computes quantitative Solix machine baseline vs ground truth
├── dataset.py             <- SolixRNFLDataset: DICOM seeker, 2.5D batching, ground-truth parser
├── model.py               <- VolumetricRNFLNet: 2.5D multi-task residual network
├── losses.py              <- VolumetricRNFLLoss: Dice, boundary Huber, cup BCE, topo penalty
├── train.py               <- Subject-grouped training loop with validation & checkpointing
├── export_to_slicer.py    <- Full-volume inference and interactive 3D Slicer exporter
└── tests/
    └── test_pipeline.py   <- Unit tests for dataset, model, loss, and disc geometry
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
    --dataset_root "/Users/nikhilmundhra/Library/CloudStorage/Box-Box/deidentified" \
    --val_subjects "BEH0335,BEH0314" \
    --epochs 15 \
    --batch_size 4 \
    --accum_steps 2 \
    --lr 3e-4 \
    --checkpoint_dir "./checkpoints/rnfl_production"
```

#### Training Arguments:
- `--dataset_root`: Path to deidentified Box dataset folder.
- `--val_subjects`: Comma-separated subject IDs for validation. Must be split by subject to prevent B-scan data leakage across scans. Recommended: `BEH0335,BEH0314` (the subjects with the largest clinical disc margin corrections).
- `--epochs`: Number of full passes through the training dataset (default: 15).
- `--batch_size`: Physical batch size per step on Apple Silicon GPU (`mps`).
- `--accum_steps`: Gradient accumulation steps (effective batch size = `batch_size * accum_steps = 8`).
- `--lr`: Initial learning rate (default: `3e-4` with Cosine Annealing scheduler).
- `--context_slices`: Number of adjacent B-scans in 2.5D tensor (default: 5).
- `--checkpoint_dir`: Directory to save `best_volumetric_rnfl_net.pt`.

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

---

## 6. Running Training on NYUAD HPC (Jubail Cluster)

For high-throughput training across large volumes, the training job runs on the NYUAD Jubail HPC cluster using SLURM, NVIDIA GPU acceleration, and **Singularity containers** (no `pip install` needed).

### Cluster Guidelines & Best Practices (CRC Policy)
* **Zero `pip install` / Container-First**: Users should not `pip install` into home directories or login nodes, which rapidly consumes file quotas (inodes) on Lustre. Instead, we use `module load singularity` with pre-built GPU containers that bundle PyTorch, MONAI, CUDA, and dependencies.
* **Data Storage (`/scratch/`)**: All datasets, Singularity container images (`.sif`), and checkpoints **must** reside in `/scratch/<NetID>/`. The `/home` directory has strict quota limits and must only store code.
* **Partition**: Request the **`nvidia`** partition for GPU jobs.
* **Modules**: Loaded via Lmod (`module purge`, `module load singularity`).

### Step 1: Transfer Dataset to Jubail Scratch
From your local terminal, transfer the deidentified Box directory directly to your Jubail scratch directory:
```bash
# Replace nm4358 with your NYU NetID
rsync -avP /Users/nikhilmundhra/Library/CloudStorage/Box-Box/deidentified/ \
    nm4358@jubail.abudhabi.nyu.edu:/scratch/nm4358/deidentified/
```

### Step 2: Connect to Jubail & Pull Code
SSH into Jubail:
```bash
ssh nm4358@jubail.abudhabi.nyu.edu
```

Clone or pull the repository into your home directory:
```bash
cd ~
git clone https://github.com/Nikhil-Mundhra/train-cnn-models.git
# or if already cloned:
cd ~/train-cnn-models && git pull origin main
```

### Step 3: Prepare the Singularity Image (Zero `pip install`)
Load the Singularity module and pull the official MONAI GPU container image into `/scratch` (takes ~2-3 minutes, run once):
```bash
module purge
module load singularity

# Set Singularity cache to /scratch to preserve home quota
export SINGULARITY_CACHEDIR="/scratch/${USER}/.singularity_cache"
export SINGULARITY_TMPDIR="/scratch/${USER}/.singularity_tmp"
mkdir -p "/scratch/${USER}/singularityimages" "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}"

# Pull official MONAI/PyTorch CUDA container into scratch
singularity pull /scratch/${USER}/singularityimages/monai.sif docker://projectmonai/monai:latest
```
*(Note: If you skip this manual pull, `train_rnfl_jubail.slurm` will automatically detect that the image is missing and pull it on first launch).*

### Step 4: Submit the SLURM Batch Job
Submit [`train_rnfl_jubail.slurm`](file:///Users/nikhilmundhra/Documents/Github/Capstone/train-cnn-models/model_training/train_rnfl_volumetric/train_rnfl_jubail.slurm):
```bash
cd ~/train-cnn-models/model_training/train_rnfl_volumetric
sbatch train_rnfl_jubail.slurm
```

The batch script automatically:
1. Loads `singularity` via Lmod.
2. Mounts `/scratch` into the container.
3. Passes the GPU hardware via `--nv`.
4. Executes the multi-epoch training script using the container's built-in PyTorch and MONAI.

### Step 5: Monitor the Job
```bash
# Check job queue status
squeue -u $USER

# Follow the live training logs
tail -f slurm_rnfl_*.out
```

### Step 6: Retrieve Checkpoints to Local Machine
Once training completes, copy the best model weights back to your local machine:
```bash
# Run on your local machine:
mkdir -p ./checkpoints/jubail_run
rsync -avP nm4358@jubail.abudhabi.nyu.edu:/scratch/nm4358/checkpoints/rnfl_volumetric_*/best_volumetric_rnfl_net.pt ./checkpoints/jubail_run/
```


