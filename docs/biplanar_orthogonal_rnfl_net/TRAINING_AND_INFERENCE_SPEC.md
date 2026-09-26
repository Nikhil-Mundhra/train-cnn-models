# Training and Inference Specification

---

## 1. High-Performance Cluster (NYUAD Jubail) Training

### SLURM Submission Parameters
* **Job Script**: `model_training/train_rnfl_volumetric/train_rnfl_biplanar_jubail.slurm`
* **Partition**: `#SBATCH -p nvidia`
* **GPU**: `#SBATCH --gres=gpu:a100:1` (NVIDIA A100 40GB VRAM)
* **CPUs**: `#SBATCH -c 8`
* **Memory**: `#SBATCH --mem=32G`
* **Environment**: `/scratch/nm4358/envs/oct-env/bin/python` (PyTorch 2.8.0 + CUDA 12.8)
* **Dataset**: `/scratch/nm4358/deidentified` (23 subjects, 46 paired volumes)

```bash
# To submit on Jubail:
ssh nm4358@jubail.abudhabi.nyu.edu
cd /home/nm4358/train-cnn-models
sbatch model_training/train_rnfl_volumetric/train_rnfl_biplanar_jubail.slurm
```

---

## 2. Command Line Arguments for `train.py`

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset_root` | `str` | `.../deidentified` | Path to deidentified DICOM & TSV directory |
| `--val_subjects` | `str` | `BEH0335,BEH0314,BEH0086` | Comma-separated validation subject IDs |
| `--epochs` | `int` | `20` | Number of training epochs |
| `--base_channels` | `int` | `16` (Light) or `32` (Heavy) | Channel multiplier |
| `--batch_size` | `int` | `32` (A100) or `4` (MPS) | Batch size per step |
| `--accum_steps` | `int` | `2` | Gradient accumulation steps |
| `--lr` | `float` | `5e-4` | Peak learning rate (Cosine Annealed to 1e-6) |
| `--precision` | `str` | `bf16` | Mixed precision (`bf16` recommended) |
| `--enable_orthogonal` | `flag` | `True` | Samples both horizontal and vertical planes |
| `--checkpoint_dir` | `str` | `./checkpoints/...` | Target directory for model weights |

---

## 3. End-to-End Inference & Cohort Evaluation

### Running Batch Cohort Evaluation
```bash
python model_training/train_rnfl_volumetric/batch_cohort_evaluator.py \
    --checkpoint ./checkpoints/best_volumetric_rnfl_net.pt \
    --dataset_root /Users/nikhilmundhra/Library/CloudStorage/Box-Box/deidentified \
    --arm good \
    --batch_size 8 \
    --device mps
```

### Running 3D Slicer Interactive Scene Export
```bash
python model_training/train_rnfl_volumetric/export_to_slicer.py \
    --dcm /path/to/subject_Disc_Cube_OPT.dcm \
    --checkpoint ./checkpoints/best_volumetric_rnfl_net.pt \
    --output ./predictions/subject_export.nii.gz
```
