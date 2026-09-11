#!/bin/bash
#SBATCH --job-name=patbinn_data_prep
#SBATCH --output=patbinn_data_prep_%j.out
#SBATCH --error=patbinn_data_prep_%j.err
# Partition/account are intentionally not hardcoded so this runs on any cluster.
# Set them at submit time, e.g.:  SBATCH_PARTITION=gpu sbatch <script>
# (Slurm honours the SBATCH_PARTITION / SBATCH_ACCOUNT environment variables,
#  or pass --partition=... --account=... directly to sbatch.)
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
IMAGE_SIF="$WORKDIR/patbinn.sif"

# CPU-only: genome download + ART read simulation don't need a GPU.
if [[ ! -f "$IMAGE_SIF" || "$WORKDIR/patbinn.def" -nt "$IMAGE_SIF" ]]; then
    apptainer build --fakeroot "$IMAGE_SIF" "$WORKDIR/patbinn.def"
fi

apptainer exec \
    --bind "$WORKDIR":/workspace \
    --pwd /workspace \
    "$IMAGE_SIF" \
    python3 data/download_genomes.py \
    --manifest data/manifests/full_manifest.csv \
    --out_dir data/refs \
    --genome_manifest_out data/manifests/genome_manifest.csv

apptainer exec \
    --bind "$WORKDIR":/workspace \
    --pwd /workspace \
    "$IMAGE_SIF" \
    python3 data/simulate_reads.py \
    --genome_manifest data/manifests/genome_manifest.csv \
    --reads_dir data/reads \
    --sample_manifest_out data/manifests/sample_manifest.csv \
    --num_replicates 6 \
    --read_length 150 \
    --coverage 30
