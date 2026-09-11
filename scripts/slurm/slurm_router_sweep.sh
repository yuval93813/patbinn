#!/bin/bash
#SBATCH --job-name=patbinn_router_sweep
#SBATCH --output=patbinn_router_sweep_%j.out
#SBATCH --error=patbinn_router_sweep_%j.err
# Partition/account are intentionally not hardcoded so this runs on any cluster.
# Set them at submit time, e.g.:  SBATCH_PARTITION=gpu sbatch <script>
# (Slurm honours the SBATCH_PARTITION / SBATCH_ACCOUNT environment variables,
#  or pass --partition=... --account=... directly to sbatch.)
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

set -euo pipefail
WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
IMAGE_SIF="$WORKDIR/patbinn.sif"
mkdir -p "$WORKDIR/checkpoints/router_sweep" "$WORKDIR/results" "$WORKDIR/data/vocab"

# Router-only retraining: the eight experts are restored from the fold-1 CV
# checkpoint rather than retrained, which is exact because experts never
# consume router output during training.
apptainer exec --nv --bind "$WORKDIR":/workspace --pwd /workspace "$IMAGE_SIF" \
    python3 src/sweep_router.py \
    --cv_sample_manifest data/manifests/cv_sample_manifest.csv \
    --k_folds 5 --fold 1 \
    --base_checkpoint checkpoints/cv_hierarchical_binary/fold1.pth \
    --vocab_cache data/vocab_hierarchical_binary/kmer_vocab_fold1.json \
    --router_epochs 30 --max_kmers 1024 \
    --checkpoints_dir checkpoints/router_sweep \
    --out_csv results/router_sweep.csv
