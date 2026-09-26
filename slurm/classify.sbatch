#!/bin/bash -l
# Classify notes with one or both VetNutriLM adapters. Edit ACCOUNT and paths first.
#SBATCH --account=ACCOUNT
#SBATCH --partition=gpu_cuda
#SBATCH --qos=short
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=vetnutrilm-classify
#SBATCH --output=vetnutrilm-classify-%j.out
set -euo pipefail
umask 077
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
NOTES=${NOTES:?CSV with id and text}
CHECKPOINT=${CHECKPOINT:?local gemma-4-31B-it directory}
vetnutrilm classify --task both --input "$NOTES" --checkpoint "$CHECKPOINT" \
  --binary-adapter "${BINARY_ADAPTER:?}" --category-adapter "${CATEGORY_ADAPTER:?}" \
  --output "runs/classify-${SLURM_JOB_ID}"
