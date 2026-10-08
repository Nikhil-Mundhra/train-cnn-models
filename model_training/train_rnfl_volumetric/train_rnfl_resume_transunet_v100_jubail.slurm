#!/bin/bash -l
# Resume TransUNet training from previous job checkpoint and run full cohort evaluation.
# Usage:
#   sbatch model_training/train_rnfl_volumetric/train_rnfl_resume_transunet_v100_jubail.slurm [PREV_JOB_ID]
# Default PREV_JOB_ID is 18703790.

#SBATCH -p nvidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=rnfl_trans_res
#SBATCH --output=slurm_rnfl_trans_res_%j.out
#SBATCH --error=slurm_rnfl_trans_res_%j.err

set -euo pipefail

PREV_JOB_ID="${1:-18703790}"

SCRATCH_DIR="/scratch/${USER}"
PYTHON_BIN="${SCRATCH_DIR}/envs/oct-env/bin/python"
DATASET_ROOT="${SCRATCH_DIR}/deidentified-new"
REPO_DIR="${HOME}/train-cnn-models"
TRAIN_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/train.py"
EVAL_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/batch_cohort_evaluator.py"
REPORT_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/build_cohort_report.py"
MODEL_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/transunet.py"
SPLIT_MANIFEST="${REPO_DIR}/model_training/train_rnfl_3d/manifests/stratified_held_out_v2.json"
TRAIN_MANIFEST="${REPO_DIR}/model_training/train_rnfl_3d/manifests/phase1_train_manifest.json"

PREV_CKPT_DIR="${SCRATCH_DIR}/checkpoints/rnfl_transunet_v100_${PREV_JOB_ID}"
RESUME_CKPT="${PREV_CKPT_DIR}/latest_volumetric_rnfl_net.pt"

if [ ! -f "${RESUME_CKPT}" ]; then
    RESUME_CKPT="${PREV_CKPT_DIR}/best_volumetric_rnfl_net.pt"
fi

if [ ! -f "${RESUME_CKPT}" ]; then
    echo "[ERROR] Cannot find resume checkpoint in: ${PREV_CKPT_DIR}" >&2
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CHECKPOINT_DIR="${SCRATCH_DIR}/checkpoints/rnfl_transunet_v100_${SLURM_JOB_ID}"
REPORT_DIR="${SCRATCH_DIR}/reports/report_transunet_v100_${SLURM_JOB_ID}"
ASSETS_DIR="${REPORT_DIR}/assets/executive_cohort_report"
mkdir -p "${CHECKPOINT_DIR}" "${ASSETS_DIR}"

# Seed the new checkpoint directory with the previous best checkpoint
if [ -f "${PREV_CKPT_DIR}/best_volumetric_rnfl_net.pt" ]; then
    cp "${PREV_CKPT_DIR}/best_volumetric_rnfl_net.pt" "${CHECKPOINT_DIR}/best_volumetric_rnfl_net.pt"
fi

echo "=================================================================="
echo "[Job] ${SLURM_JOB_ID} resuming from Job ${PREV_JOB_ID}"
echo "[Checkpoint] Resuming from: ${RESUME_CKPT}"
echo "[Node] $(hostname) at $(date)"
echo "[GPU] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=================================================================="

# Two DDP ranks on 2x V100. Effective batch size = 2 * 2 * 8 = 32.
export OMP_NUM_THREADS=4
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 "${TRAIN_SCRIPT}" \
    --dataset_root "${DATASET_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}" \
    --train_manifest "${TRAIN_MANIFEST}" \
    --resume_checkpoint "${RESUME_CKPT}" \
    --arch transunet \
    --transunet_hidden_size 256 \
    --transunet_layers 6 \
    --transunet_heads 8 \
    --transunet_mlp_dim 512 \
    --epochs 20 \
    --base_channels 16 \
    --batch_size 2 \
    --accum_steps 8 \
    --lr 3e-4 \
    --weight_decay 1e-4 \
    --context_slices 5 \
    --num_workers 4 \
    --precision fp16 \
    --enable_orthogonal \
    --device cuda \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --log_interval 50

BEST_MODEL="${CHECKPOINT_DIR}/best_volumetric_rnfl_net.pt"
if [ ! -f "${BEST_MODEL}" ]; then
    BEST_MODEL="${CHECKPOINT_DIR}/latest_volumetric_rnfl_net.pt"
fi

echo ""
echo "##################################################################"
echo "### STEP 2: HIGH-THROUGHPUT BATCH COHORT EVALUATION             ###"
echo "##################################################################"

"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --checkpoint "${BEST_MODEL}" \
    --dataset_root "${DATASET_ROOT}" \
    --output_dir "${REPORT_DIR}" \
    --split_manifest "${SPLIT_MANIFEST}" \
    --os_orientation_mode corrected \
    --disc_cut_mode cup \
    --batch_size 4 \
    --device cuda

echo ""
echo "##################################################################"
echo "### STEP 3: COMPILING EXECUTIVE CLINICAL COHORT REPORT         ###"
echo "##################################################################"

OUTPUT_MD="${REPORT_DIR}/executive_cohort_rnfl_report_transunet_v100_job_${SLURM_JOB_ID}_${TIMESTAMP}.md"
OUTPUT_PDF="${REPORT_DIR}/executive_cohort_rnfl_report_transunet_v100_job_${SLURM_JOB_ID}_${TIMESTAMP}.pdf"

"${PYTHON_BIN}" "${REPORT_SCRIPT}" \
    --metrics_json "${ASSETS_DIR}/cohort_evaluation_metrics.json" \
    --train_manifest "${TRAIN_MANIFEST}" \
    --training_stage "Phase 1 broad-cohort TransUNet training (resumed)" \
    --assets_dir "${ASSETS_DIR}" \
    --output_md "${OUTPUT_MD}" \
    --output_pdf "${OUTPUT_PDF}" \
    --job_id "${SLURM_JOB_ID}" \
    --checkpoint_name "rnfl_transunet_v100_${SLURM_JOB_ID}" \
    --model_variant "TransUNet Hybrid (2x V100, Resumed)" \
    --model_desc "Resumed TransUNet hybrid CNN-ViT architecture (~3.76M params) across full 20 epochs"

echo "[SUCCESS] Resumed training, evaluation, and reporting complete!"
