#!/bin/bash -l
# Separate Phase 2 experiment: two V100s on one node, using audited subjects.
# Submit from ~/train-cnn-models with:
#   sbatch model_training/train_rnfl_volumetric/train_rnfl_finetune_canonical_v100_jubail.slurm

#SBATCH -p nvidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=rnfl_ft_v100
#SBATCH --output=slurm_rnfl_ft_v100_%j.out
#SBATCH --error=slurm_rnfl_ft_v100_%j.err

set -euo pipefail

SCRATCH_DIR="/scratch/${USER}"
PYTHON_BIN="${SCRATCH_DIR}/envs/oct-env/bin/python"
DATASET_ROOT="${SCRATCH_DIR}/deidentified-new"
REPO_DIR="${HOME}/train-cnn-models"
TRAIN_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/train.py"
EVAL_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/batch_cohort_evaluator.py"
REPORT_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/build_cohort_report.py"
SPLIT_MANIFEST="${REPO_DIR}/model_training/train_rnfl_3d/manifests/stratified_held_out_v2.json"
PRETRAIN_MANIFEST="${REPO_DIR}/model_training/train_rnfl_3d/manifests/phase1_train_manifest.json"
TRAIN_MANIFEST="${REPO_DIR}/model_training/train_rnfl_3d/manifests/phase2_audited_train_manifest.json"
BASE_CHECKPOINT="${1:-${SCRATCH_DIR}/checkpoints/rnfl_canonical_robust_18697275/best_volumetric_rnfl_net.pt}"

for required in "${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${EVAL_SCRIPT}" "${REPORT_SCRIPT}" \
    "${SPLIT_MANIFEST}" "${PRETRAIN_MANIFEST}" "${TRAIN_MANIFEST}" "${BASE_CHECKPOINT}"; do
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
CHECKPOINT_DIR="${SCRATCH_DIR}/checkpoints/rnfl_canonical_finetuned_v100_${SLURM_JOB_ID}"
REPORT_DIR="${SCRATCH_DIR}/reports/report_canonical_finetuned_v100_${SLURM_JOB_ID}"
ASSETS_DIR="${REPORT_DIR}/assets/executive_cohort_report"
mkdir -p "${CHECKPOINT_DIR}" "${ASSETS_DIR}"

echo "[Job] ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "[GPU] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "[CUDA_VISIBLE_DEVICES] ${CUDA_VISIBLE_DEVICES:-unset}"
echo "[Base checkpoint] ${BASE_CHECKPOINT}"
echo "[Train manifest] ${TRAIN_MANIFEST}"
echo "[Validation manifest] ${SPLIT_MANIFEST}"

# Each rank sees one V100. Global effective batch = 4 x 8 x 2 = 64.
# V100 supports FP16 Tensor Cores; BF16 requires a newer GPU generation.
# This five-epoch, unaugmented run is independent of the existing H200 job.
export OMP_NUM_THREADS=4
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 "${TRAIN_SCRIPT}" \
    --dataset_root "${DATASET_ROOT}" \
    --resume_checkpoint "${BASE_CHECKPOINT}" \
    --train_manifest "${TRAIN_MANIFEST}" \
    --split_manifest "${SPLIT_MANIFEST}" \
    --epochs 5 \
    --base_channels 32 \
    --batch_size 4 \
    --accum_steps 8 \
    --lr 2e-5 \
    --weight_decay 1e-4 \
    --context_slices 5 \
    --num_workers 4 \
    --precision fp16 \
    --enable_orthogonal \
    --use_stn \
    --no_augment \
    --device cuda \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --log_interval 50

BEST_MODEL="${CHECKPOINT_DIR}/best_volumetric_rnfl_net.pt"
if [ ! -f "${BEST_MODEL}" ]; then
    BEST_MODEL="${CHECKPOINT_DIR}/latest_volumetric_rnfl_net.pt"
fi
if [ ! -f "${BEST_MODEL}" ]; then
    echo "[ERROR] Fine-tuning completed without a checkpoint" >&2
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

OUTPUT_MD="${REPORT_DIR}/executive_cohort_rnfl_report_canonical_v100_job_${SLURM_JOB_ID}_${TIMESTAMP}.md"
OUTPUT_PDF="${REPORT_DIR}/executive_cohort_rnfl_report_canonical_v100_job_${SLURM_JOB_ID}_${TIMESTAMP}.pdf"
"${PYTHON_BIN}" "${REPORT_SCRIPT}" \
    --metrics_json "${ASSETS_DIR}/cohort_evaluation_metrics.json" \
    --train_manifest "${PRETRAIN_MANIFEST}" \
    --fine_tune_manifest "${TRAIN_MANIFEST}" \
    --training_stage "Phase 1 pretraining and Phase 2 audited fine-tuning" \
    --assets_dir "${ASSETS_DIR}" \
    --output_md "${OUTPUT_MD}" \
    --output_pdf "${OUTPUT_PDF}" \
    --job_id "${SLURM_JOB_ID}" \
    --checkpoint_name "rnfl_canonical_finetuned_v100_${SLURM_JOB_ID}" \
    --model_variant "Canonical Robust Fine-Tuned (2x V100)" \
    --model_desc "Bi-Planar 2.5D STN; Phase 1 pretrained; five-epoch audited fine-tuning on two V100 GPUs, global effective batch 64"

echo "[Complete] Checkpoint: ${BEST_MODEL}"
echo "[Complete] Report: ${OUTPUT_PDF}"
