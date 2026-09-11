#!/bin/bash
#SBATCH --job-name=patbinn_sweep_arch_extra
#SBATCH --output=patbinn_sweep_arch_extra_%j.out
#SBATCH --error=patbinn_sweep_arch_extra_%j.err
# Partition/account are intentionally not hardcoded so this runs on any cluster.
# Set them at submit time, e.g.:  SBATCH_PARTITION=gpu sbatch <script>
# (Slurm honours the SBATCH_PARTITION / SBATCH_ACCOUNT environment variables,
#  or pass --partition=... --account=... directly to sbatch.)
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
IMAGE_SIF="$WORKDIR/patbinn.sif"

mkdir -p "$WORKDIR/checkpoints/sweep" "$WORKDIR/results"

# Supplementary single-hidden-layer (128) shape, added after the main 6x2 grid
# (slurm_sweep_architecture.sh / job 66) was already running. Reuses that job's
# vocab cache and fold-0 manifest, and fixes the router at (1024,256) - the
# shape job 66's router sweep actually chose (verified from a completed expert
# checkpoint's router_hidden_sizes) - so this extra data point is directly
# comparable to the rest of results/size_sweep.csv instead of re-running the
# whole grid just to add one shape.
apptainer exec --nv \
    --bind "$WORKDIR":/workspace \
    --pwd /workspace \
    "$IMAGE_SIF" \
    python3 src/sweep_architecture.py \
    --cv_sample_manifest data/manifests/cv_sample_manifest.csv \
    --k_folds 5 --sweep_fold 0 \
    --router_epochs 30 --expert_epochs 30 \
    --max_kmers 1024 \
    --skip_grid --extra_shapes 128 \
    --router_shape_override 1024,256 \
    --checkpoints_dir checkpoints/sweep \
    --vocab_cache_sweep data/vocab/kmer_vocab_sweep.json \
    --out_csv results/size_sweep_extra.csv
