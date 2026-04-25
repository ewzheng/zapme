#!/bin/bash
# SLURM submission script for the zapme slouch classifier training job.
#
# Targets the U of A HPC Puma cluster's A100 MIG slices. The training is
# tiny (a few thousand parameters, a few thousand windows) so any GPU is
# overkill — a single MIG slice is plenty and uses fewer allocation hours
# than asking for a full V100. The job almost always finishes in under
# a minute of GPU time; the wall clock is dominated by job queue plus
# Python startup.
#
# Submit from anywhere on the HPC:
#
#     sbatch scripts/train_classifier.slurm
#
# Assumes the repo lives at ~/Hackathon/zapme with a venv at .venv that
# already has requirements.txt installed.

#SBATCH --job-name=zapme-train
#SBATCH --account=duey
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ezheng@arizona.edu

set -euo pipefail

VAL_SESSION="${VAL_SESSION:-eric_session_03}"

cd ~/Hackathon/zapme
module load python/3.11
source .venv/bin/activate

mkdir -p logs models

# --- Job preamble -----------------------------------------------------
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "GPU(s):"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
echo "Val session:   $VAL_SESSION"
echo "----------------------------------------------------------------"

# --- Train ------------------------------------------------------------
python scripts/train_classifier.py \
    --val-session "$VAL_SESSION" \
    --device cuda

echo "----------------------------------------------------------------"
echo "Done. Trained model at: models/slouch_cnn.onnx"
ls -lh models/slouch_cnn.onnx || true
