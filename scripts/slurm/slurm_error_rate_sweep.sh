#!/bin/bash
#SBATCH --job-name=patbinn_error_sweep
#SBATCH --output=patbinn_error_sweep_%j.out
#SBATCH --error=patbinn_error_sweep_%j.err
# Partition/account are intentionally not hardcoded so this runs on any cluster.
# Set them at submit time, e.g.:  SBATCH_PARTITION=gpu sbatch <script>
# (Slurm honours the SBATCH_PARTITION / SBATCH_ACCOUNT environment variables,
#  or pass --partition=... --account=... directly to sbatch.)
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
IMAGE_SIF="$WORKDIR/patbinn.sif"

mkdir -p "$WORKDIR/results"

# Pure inference on a small binarized model - no GPU requested (deliberately, so
# this can run alongside other GPU jobs on the shared node instead of queuing
# behind them for the single GPU).
apptainer exec \
    --bind "$WORKDIR":/workspace \
    --pwd /workspace \
    "$IMAGE_SIF" \
    python3 src/sweep_error_rates.py \
    --checkpoint checkpoints/final/moe_final_1024x128.pth \
    --sample_manifest data/manifests/final_train_manifest.csv \
    --split val \
    --device cpu \
    --out_csv results/error_rate_sweep.csv
