#!/bin/bash
#SBATCH --job-name=patbinn_read_count_sweep
#SBATCH --output=patbinn_read_count_sweep_%j.out
#SBATCH --error=patbinn_read_count_sweep_%j.err
# Partition/account are intentionally not hardcoded so this runs on any cluster.
# Set them at submit time, e.g.:  SBATCH_PARTITION=gpu sbatch <script>
# (Slurm honours the SBATCH_PARTITION / SBATCH_ACCOUNT environment variables,
#  or pass --partition=... --account=... directly to sbatch.)
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#
# Pure inference over already-trained CV fold checkpoints (run_cv.py must have
# completed all 5 folds first) - no GPU needed.

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
IMAGE_SIF="$WORKDIR/patbinn.sif"

mkdir -p "$WORKDIR/results"

apptainer exec \
    --bind "$WORKDIR":/workspace \
    --pwd /workspace \
    "$IMAGE_SIF" \
    python3 src/sweep_read_counts.py \
    --checkpoints_dir checkpoints/cv_hierarchical_binary \
    --manifests_dir data/manifests_hierarchical_binary \
    --k_folds 5 \
    --out_csv results/read_count_sweep.csv
