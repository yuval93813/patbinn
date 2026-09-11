#!/bin/bash
#SBATCH --job-name=patbinn_k4
#SBATCH --output=patbinn_k4_%j.out
#SBATCH --error=patbinn_k4_%j.err
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
mkdir -p "$WORKDIR/checkpoints/encoding" "$WORKDIR/results" "$WORKDIR/data/vocab"

# k=4: only 256 distinct 4-mers exist, so a 256-bit vector holds the complete
# 4-mer spectrum - a quarter of the baseline input width, with nothing
# discarded. Tests how far the input layer can shrink before the encoding
# stops separating 58 classes.
apptainer exec --nv --bind "$WORKDIR":/workspace --pwd /workspace "$IMAGE_SIF" \
    python3 src/run_encoding_experiment.py \
    --label k4_256 --k 4 --max_kmers 256 --fold 1 \
    --vocab_cache data/vocab/kmer_vocab_fold1_k4.json \
    --save_model checkpoints/encoding/k4_256.pth \
    --out_csv results/encoding_variants.csv
