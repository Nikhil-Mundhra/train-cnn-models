#!/bin/bash -l
# Independent TransUNet architecture experiment on two V100s in one node.
# Submit from ~/train-cnn-models:
#   sbatch model_training/train_rnfl_volumetric/train_rnfl_transunet_v100_jubail.slurm

#SBATCH -p nvidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=rnfl_trans_v100
#SBATCH --output=slurm_rnfl_trans_v100_%j.out
#SBATCH --error=slurm_rnfl_trans_v100_%j.err

set -euo pipefail

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

for required in "${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${EVAL_SCRIPT}" "${REPORT_SCRIPT}" \
    "${MODEL_SCRIPT}" "${SPLIT_MANIFEST}" "${TRAIN_MANIFEST}"; do
    if [ ! -f "${required}" ]; then
        echo "[ERROR] Required file is missing: ${required}" >&2
        exit 1
    fi
done
if [ ! -d "${DATASET_ROOT}" ]; then
    echo "[ERROR] Dataset is missing: ${DATASET_ROOT}" >&2
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CHECKPOINT_DIR="${SCRATCH_DIR}/checkpoints/rnfl_transunet_v100_${SLURM_JOB_ID}"
REPORT_DIR="${SCRATCH_DIR}/reports/report_transunet_v100_${SLURM_JOB_ID}"
ASSETS_DIR="${REPORT_DIR}/assets/executive_cohort_report"
mkdir -p "${CHECKPOINT_DIR}" "${ASSETS_DIR}"

echo "[Job] ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "[GPU] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "[CUDA_VISIBLE_DEVICES] ${CUDA_VISIBLE_DEVICES:-unset}"
echo "[Train manifest] ${TRAIN_MANIFEST}"
echo "[Validation manifest] ${SPLIT_MANIFEST}"

# Two DDP ranks, one per V100. Effective batch = 2 x 8 x 2 = 32.
# FP16 with GradScaler is supported by train.py and by the V100.
export OMP_NUM_THREADS=4
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 "${TRAIN_SCRIPT}" \
    --dataset_root "${DATASET_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}" \
    --train_manifest "${TRAIN_MANIFEST}" \
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
if [ ! -f "${BEST_MODEL}" ]; then
    echo "[ERROR] Training completed without a checkpoint" >&2
    exit 1
fi

"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --checkpoint "${BEST_MODEL}" \
    --dataset_root "${DATASET_ROOT}" \
    --output_dir "${REPORT_DIR}" \
    --split_manifest "${SPLIT_MANIFEST}" \
    --os_orientation_mode corrected \
    --disc_cut_mode cup \
    --batch_size 4 \
    --device cuda

OUTPUT_MD="${REPORT_DIR}/executive_cohort_rnfl_report_transunet_v100_job_${SLURM_JOB_ID}_${TIMESTAMP}.md"
OUTPUT_PDF="${REPORT_DIR}/executive_cohort_rnfl_report_transunet_v100_job_${SLURM_JOB_ID}_${TIMESTAMP}.pdf"
"${PYTHON_BIN}" "${REPORT_SCRIPT}" \
    --metrics_json "${ASSETS_DIR}/cohort_evaluation_metrics.json" \
    --train_manifest "${TRAIN_MANIFEST}" \
    --training_stage "Phase 1 broad-cohort TransUNet training" \
    --assets_dir "${ASSETS_DIR}" \
    --output_md "${OUTPUT_MD}" \
    --output_pdf "${OUTPUT_PDF}" \
    --job_id "${SLURM_JOB_ID}" \
    --checkpoint_name "rnfl_transunet_v100_${SLURM_JOB_ID}" \
    --model_variant "TransUNet Hybrid (2x V100)" \
    --model_desc "Hybrid CNN-Transformer encoder and cascaded decoder with RNFL mask and boundary heads; 20 broad-cohort epochs on two V100 GPUs, effective batch 32"

echo "[Complete] Checkpoint: ${BEST_MODEL}"
echo "[Complete] Report: ${OUTPUT_PDF}"
