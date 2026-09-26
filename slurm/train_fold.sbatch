#!/bin/bash -l
# Train one VetNutriLM fold adapter and predict its held-out fold.
# Edit the ACCOUNT, paths and email before use. Published runs used one H100 80 GB;
# five binary folds took 1 h 49 min in total.
#SBATCH --account=ACCOUNT
#SBATCH --partition=gpu_cuda
#SBATCH --qos=short
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=vetnutrilm-fold
#SBATCH --output=vetnutrilm-fold-%j.out
set -euo pipefail
umask 077
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
TASK=${TASK:-binary}          # binary or categories
FOLD=${FOLD:?set FOLD=1..5}
DATA=${DATA:?labelled CSV}
FOLDS=${FOLDS:?folds CSV from vetnutrilm prepare}
CHECKPOINT=${CHECKPOINT:?local gemma-4-31B-it directory}
vetnutrilm train --task "$TASK" --input "$DATA" --folds "$FOLDS" --fold "$FOLD" \
  --checkpoint "$CHECKPOINT" --output "runs/${TASK}-fold${FOLD}-${SLURM_JOB_ID}"
