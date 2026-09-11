#!/bin/bash
#SBATCH --job-name=patbinn_cv_data_prep
#SBATCH --output=patbinn_cv_data_prep_%j.out
#SBATCH --error=patbinn_cv_data_prep_%j.err
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

# Multi-isolate download for the genome-level generalization benchmark: up to 10
# independent isolates per (family,variant), across the 58 classes with enough
# real diversity (6 rare classes dropped - see generalization_manifest.csv).
apptainer exec \
    --bind "$WORKDIR":/workspace \
    --pwd /workspace \
    "$IMAGE_SIF" \
    python3 data/download_genomes.py \
    --manifest data/manifests/generalization_manifest.csv \
    --out_dir data/refs \
    --genome_manifest_out data/manifests/genome_manifest_multi.csv \
    --num_isolates 10 --k_folds 5

apptainer exec \
    --bind "$WORKDIR":/workspace \
    --pwd /workspace \
    "$IMAGE_SIF" \
    python3 data/simulate_reads.py \
    --genome_manifest data/manifests/genome_manifest_multi.csv \
    --reads_dir data/reads_multi \
    --sample_manifest_out data/manifests/cv_sample_manifest.csv \
    --num_replicates 6 --read_length 150 --coverage 30
