#!/bin/bash
#SBATCH --job-name=patbinn_resim
#SBATCH --output=patbinn_resim_%j.out
#SBATCH --error=patbinn_resim_%j.err
# Partition/account are intentionally not hardcoded so this runs on any cluster.
# Set them at submit time, e.g.:  SBATCH_PARTITION=gpu sbatch <script>
# (Slurm honours the SBATCH_PARTITION / SBATCH_ACCOUNT environment variables,
#  or pass --partition=... --account=... directly to sbatch.)
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail
WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
IMAGE_SIF="$WORKDIR/patbinn.sif"

# Re-simulate reads over the corrected genome manifest (12 leaves fixed: an
# NCBI query-construction bug and an ICTV-rename validation bug were silently
# capping them at 1 isolate; see download_genomes.py history 2026-08-09).
# CPU-only (ART), no GPU needed. Regenerates cv_sample_manifest.csv, which
# every downstream training/sweep script consumes.
apptainer exec \
    --bind "$WORKDIR":/workspace \
    --pwd /workspace \
    "$IMAGE_SIF" \
    python3 data/simulate_reads.py \
    --genome_manifest data/manifests/genome_manifest_multi.csv \
    --reads_dir data/reads_multi \
    --sample_manifest_out data/manifests/cv_sample_manifest.csv \
    --num_replicates 6 --read_length 150 --coverage 30
