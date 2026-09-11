#!/bin/bash
#SBATCH --job-name=patbinn_sweep_cross
#SBATCH --output=patbinn_sweep_cross_%j.out
#SBATCH --error=patbinn_sweep_cross_%j.err
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

if [[ ! -f "$IMAGE_SIF" || "$WORKDIR/patbinn.def" -nt "$IMAGE_SIF" ]]; then
    apptainer build --fakeroot "$IMAGE_SIF" "$WORKDIR/patbinn.def"
fi

# Tests the two "opposite corner" router/expert size combos that the independent
# sweep (job 66) never jointly trained: small router + big expert, and vice versa.
# Checks whether router/expert size interact (job 66 assumed they don't).
apptainer exec --nv \
    --bind "$WORKDIR":/workspace \
    --pwd /workspace \
    "$IMAGE_SIF" \
    python3 src/sweep_cross_combos.py \
    --cv_sample_manifest data/manifests/cv_sample_manifest.csv \
    --k_folds 5 --sweep_fold 0 \
    --router_epochs 30 --expert_epochs 30 \
    --max_kmers 1024 \
    --checkpoints_dir checkpoints/sweep \
    --vocab_cache_sweep data/vocab/kmer_vocab_sweep.json \
    --out_csv results/size_sweep_cross.csv
