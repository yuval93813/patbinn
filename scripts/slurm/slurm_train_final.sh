#!/bin/bash
#SBATCH --job-name=patbinn_final
#SBATCH --output=patbinn_final_%j.out
#SBATCH --error=patbinn_final_%j.err
# Partition/account are intentionally not hardcoded so this runs on any cluster.
# Set them at submit time, e.g.:  SBATCH_PARTITION=gpu sbatch <script>
# (Slurm honours the SBATCH_PARTITION / SBATCH_ACCOUNT environment variables,
#  or pass --partition=... --account=... directly to sbatch.)
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
IMAGE_SIF="$WORKDIR/patbinn.sif"

mkdir -p "$WORKDIR/checkpoints/final" "$WORKDIR/data/vocab"

if [[ ! -f "$IMAGE_SIF" || "$WORKDIR/patbinn.def" -nt "$IMAGE_SIF" ]]; then
    apptainer build --fakeroot "$IMAGE_SIF" "$WORKDIR/patbinn.def"
fi

# The actual deployable model: full dataset, no held-out test fold (that's what
# run_cv.py's 5-fold CV is for - this is the "ship it" run). 1024x128 for both
# router and experts - the shape already validated across the whole CV.
apptainer exec --nv \
    --bind "$WORKDIR":/workspace \
    --pwd /workspace \
    "$IMAGE_SIF" \
    python3 src/train_final_model.py \
    --cv_sample_manifest data/manifests/cv_sample_manifest.csv \
    --val_fold 4 \
    --precision binary \
    --router_hidden_sizes 1024 128 \
    --expert_hidden_sizes 1024 128 \
    --router_epochs 30 --expert_epochs 30 \
    --max_kmers 1024 \
    --vocab_cache data/vocab/kmer_vocab_final.json \
    --save_model checkpoints/final/moe_final_1024x128.pth
