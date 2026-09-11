#!/bin/bash
#SBATCH --job-name=patbinn_encoding
#SBATCH --output=patbinn_encoding_%j.out
#SBATCH --error=patbinn_encoding_%j.err
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

RUN() { apptainer exec --nv --bind "$WORKDIR":/workspace --pwd /workspace "$IMAGE_SIF" "$@"; }

# B. k=6 at the same 1024-wide vector: 4096 possible 6-mers against 1024 slots,
#    so document-frequency selection does real work for the first time.
RUN python3 src/run_encoding_experiment.py \
    --label k6_1024 --k 6 --max_kmers 1024 --fold 1 \
    --vocab_cache data/vocab/kmer_vocab_fold1_k6.json \
    --save_model checkpoints/encoding/k6_1024.pth \
    --out_csv results/encoding_variants.csv

# C. Canonical k=5: strand-invariant, and the 5-mer space halves to exactly 512,
#    so the input layer halves.
RUN python3 src/run_encoding_experiment.py \
    --label canonical_k5_512 --k 5 --max_kmers 512 --canonical --fold 1 \
    --vocab_cache data/vocab/kmer_vocab_fold1_k5canon.json \
    --save_model checkpoints/encoding/canonical_k5_512.pth \
    --out_csv results/encoding_variants.csv
