#!/bin/bash
#SBATCH --job-name=patbinn_train_moe
#SBATCH --output=patbinn_train_moe_%j.out
#SBATCH --error=patbinn_train_moe_%j.err
# Partition/account are intentionally not hardcoded so this runs on any cluster.
# Set them at submit time, e.g.:  SBATCH_PARTITION=gpu sbatch <script>
# (Slurm honours the SBATCH_PARTITION / SBATCH_ACCOUNT environment variables,
#  or pass --partition=... --account=... directly to sbatch.)
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
IMAGE_SIF="$WORKDIR/patbinn.sif"

mkdir -p "$WORKDIR/checkpoints"

if [[ ! -f "$IMAGE_SIF" || "$WORKDIR/patbinn.def" -nt "$IMAGE_SIF" ]]; then
    apptainer build --fakeroot "$IMAGE_SIF" "$WORKDIR/patbinn.def"
fi

apptainer exec --nv \
    --bind "$WORKDIR":/workspace \
    --pwd /workspace \
    "$IMAGE_SIF" \
    python3 src/train_moe.py \
    --sample_manifest data/manifests/sample_manifest.csv \
    --router_epochs 30 \
    --expert_epochs 30 \
    --save_model "checkpoints/moe_bnn_best_${SLURM_JOB_ID}.pth"
