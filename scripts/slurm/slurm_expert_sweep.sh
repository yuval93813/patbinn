#!/bin/bash
#SBATCH --job-name=patbinn_expert_sweep
#SBATCH --output=patbinn_expert_sweep_%j.out
#SBATCH --error=patbinn_expert_sweep_%j.err
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
mkdir -p "$WORKDIR/checkpoints/expert_sweep" "$WORKDIR/results" "$WORKDIR/data/vocab"

# Expert-only retraining on fold 1, not fold 0: the router is restored from
# the fold-1 CV checkpoint, so only the eight experts are retrained at each
# shape. Fold 0 is where twelve singly-represented classes contribute no
# training material, which is why the original joint sweep's expert-side
# result was uninformative there.
apptainer exec --nv --bind "$WORKDIR":/workspace --pwd /workspace "$IMAGE_SIF" \
    python3 src/sweep_experts.py \
    --cv_sample_manifest data/manifests/cv_sample_manifest.csv \
    --k_folds 5 --fold 1 \
    --base_checkpoint checkpoints/cv_hierarchical_binary/fold1.pth \
    --vocab_cache data/vocab_hierarchical_binary/kmer_vocab_fold1.json \
    --expert_epochs 30 --max_kmers 1024 \
    --checkpoints_dir checkpoints/expert_sweep \
    --out_csv results/expert_sweep.csv
