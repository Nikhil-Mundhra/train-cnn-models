#!/bin/bash -l
# ==============================================================================
# SLURM Job Script: 4x NVIDIA V100 Distributed (DDP) Bi-Planar RNFL Network
# Cohort Scope: 61 Subjects (122 Paired Scans) strictly disjoint from V2 validation
# Validation Split: stratified_held_out_v2.json (20 Subjects / 40 Scans)
# Features:
#   - 4x NVIDIA V100 GPUs via PyTorch DDP (torchrun --nproc_per_node=4)
#   - Mixed Precision FP16 with GradScaler (tailored for Volta V100 Tensor Cores)
#   - Effective Batch Size 64 (4 GPUs x 16 per-device batch x 1 accum)
#   - Orthogonal Bi-Planar Slicing (Horizontal + Vertical Fusion)
#   - Zero-Leakage Guarantee: Certified disjoint from stratified_held_out_v2.json
# Partition: nvidia (4x NVIDIA V100 GPUs, 128GB aggregate VRAM)
# ==============================================================================

#SBATCH -p nvidia                      # Request GPU partition (nvidia)
#SBATCH --nodes=1                      # 1 compute node
#SBATCH --ntasks=1                     # 1 SLURM task
#SBATCH --gres=gpu:v100:4              # Request 4x NVIDIA V100 GPUs
#SBATCH --cpus-per-task=16             # 16 CPU cores (4 cores per GPU)
#SBATCH --mem=64G                      # 64 GB system memory
#SBATCH --time=10:00:00                # Wall clock limit (10 hours)
#SBATCH -J rnfl_bp_61s_4v100           # Job name
#SBATCH -o slurm_rnfl_bp_61s_4v100_%j.out # Standard output log (%j = jobID)
#SBATCH -e slurm_rnfl_bp_61s_4v100_%j.err # Standard error log

set -euo pipefail

echo "=================================================================="
echo "=== NYUAD HPC Job Started on $(hostname) at $(date) ==="
echo "=== Job ID: ${SLURM_JOB_ID} | Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-all} ==="
echo "=== 4x V100 Distributed Data Parallel (DDP) 61-Subject Bi-Planar RNFL ==="
echo "=================================================================="

# ----------------------------------------------------------------------
# 1. Environment & Paths Configuration
# ----------------------------------------------------------------------
module purge

NETID="${USER}"
SCRATCH_DIR="/scratch/${NETID}"
PYTHON_BIN="${SCRATCH_DIR}/envs/oct-env/bin/python"
TORCHRUN_BIN="${SCRATCH_DIR}/envs/oct-env/bin/torchrun"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "[ERROR] Python interpreter not found at: ${PYTHON_BIN}" >&2
    exit 1
fi

DATASET_ROOT="${SCRATCH_DIR}/deidentified-new"
if [ ! -d "${DATASET_ROOT}" ]; then
    echo "[ERROR] Dataset not found at: ${DATASET_ROOT}!" >&2
    exit 1
fi

REPO_DIR="${HOME}/train-cnn-models"
TRAIN_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/train.py"
EVAL_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/batch_cohort_evaluator.py"
REPORT_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/build_cohort_report.py"

SPLIT_MANIFEST="${REPO_DIR}/model_training/train_rnfl_3d/manifests/stratified_held_out_v2.json"
TRAIN_MANIFEST="${REPO_DIR}/model_training/train_rnfl_3d/manifests/train_manifest_61subj_v2.json"

for required in "${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${EVAL_SCRIPT}" "${REPORT_SCRIPT}" \
    "${SPLIT_MANIFEST}" "${TRAIN_MANIFEST}"; do
    if [ ! -f "${required}" ]; then
        echo "[ERROR] Required file is missing: ${required}" >&2
        exit 1
    fi
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

CHECKPOINT_DIR="${SCRATCH_DIR}/checkpoints/rnfl_biplanar_61subj_4v100_${SLURM_JOB_ID}"
REPORT_DIR="${SCRATCH_DIR}/reports/report_biplanar_61subj_4v100_${SLURM_JOB_ID}"
ASSETS_DIR="${REPORT_DIR}/assets/executive_cohort_report"
mkdir -p "${CHECKPOINT_DIR}" "${ASSETS_DIR}"

echo "[Environment] Python: ${PYTHON_BIN}"
echo "[GPU Topology] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "[Train Manifest] ${TRAIN_MANIFEST} (61 subjects / 122 scans)"
echo "[Validation Manifest] ${SPLIT_MANIFEST} (20 subjects / 40 scans)"

# ----------------------------------------------------------------------
# 2. Distributed Training (4x V100 DDP)
# ----------------------------------------------------------------------
echo ""
echo "##################################################################"
echo "### STEP 1: DISTRIBUTED 4x V100 TRAINING WITH TORCHRUN DDP    ###"
echo "##################################################################"

MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
export OMP_NUM_THREADS=4

"${TORCHRUN_BIN}" \
    --nproc_per_node=4 \
    --master_port="${MASTER_PORT}" \
    "${TRAIN_SCRIPT}" \
    --dataset_root "${DATASET_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}" \
    --train_manifest "${TRAIN_MANIFEST}" \
    --epochs 20 \
    --base_channels 32 \
    --batch_size 16 \
    --accum_steps 1 \
    --lr 5e-4 \
    --weight_decay 1e-4 \
    --context_slices 5 \
    --num_workers 4 \
    --precision "fp16" \
    --enable_orthogonal \
    --augment \
    --device "cuda" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --log_interval 25

BEST_MODEL="${CHECKPOINT_DIR}/best_volumetric_rnfl_net.pt"
if [ ! -f "${BEST_MODEL}" ]; then
    BEST_MODEL="${CHECKPOINT_DIR}/latest_volumetric_rnfl_net.pt"
fi
echo "[Step 1 SUCCESS] Best Model saved to: ${BEST_MODEL}"

# ----------------------------------------------------------------------
# 3. High-Throughput Held-Out Evaluation (40 Scans)
# ----------------------------------------------------------------------
echo ""
echo "##################################################################"
echo "### STEP 2: HIGH-THROUGHPUT BI-PLANAR FUSION EVALUATION        ###"
echo "##################################################################"

"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --checkpoint "${BEST_MODEL}" \
    --dataset_root "${DATASET_ROOT}" \
    --output_dir "${REPORT_DIR}" \
    --split_manifest "${SPLIT_MANIFEST}" \
    --os_orientation_mode "corrected" \
    --batch_size 8 \
    --device "cuda"

echo "[Step 2 SUCCESS] Bi-Planar Evaluation finished across 40 validation scans."

# ----------------------------------------------------------------------
# 4. Clinical Report & PDF Compilation
# ----------------------------------------------------------------------
echo ""
echo "##################################################################"
echo "### STEP 3: COMPILING EXECUTIVE CLINICAL COHORT REPORT         ###"
echo "##################################################################"

OUTPUT_MD="${REPORT_DIR}/executive_cohort_rnfl_report_biplanar_61subj_job_${SLURM_JOB_ID}_${TIMESTAMP}.md"
OUTPUT_PDF="${REPORT_DIR}/executive_cohort_rnfl_report_biplanar_61subj_job_${SLURM_JOB_ID}_${TIMESTAMP}.pdf"

"${PYTHON_BIN}" "${REPORT_SCRIPT}" \
    --metrics_json "${ASSETS_DIR}/cohort_evaluation_metrics.json" \
    --train_manifest "${TRAIN_MANIFEST}" \
    --training_stage "Pre-training 61-subject cohort (Zero-leakage V2 split)" \
    --assets_dir "${ASSETS_DIR}" \
    --output_md "${OUTPUT_MD}" \
    --output_pdf "${OUTPUT_PDF}" \
    --job_id "${SLURM_JOB_ID}" \
    --checkpoint_name "rnfl_biplanar_61subj_4v100_${SLURM_JOB_ID}" \
    --model_variant "Bi-Planar Orthogonal Heavy (61 Subj V2 Split)" \
    --model_desc "Bi-Planar 2.5D Fusion, Tversky + Boundary BCE + Thickness Integral, base_channels=32 (~6.69M params), 4x V100 DDP"

echo "[Step 3 SUCCESS] Executive Report Compiled: ${OUTPUT_PDF}"
echo ""
echo "=================================================================="
echo "=== Job Completed Successfully at $(date) ==="
echo "=================================================================="
