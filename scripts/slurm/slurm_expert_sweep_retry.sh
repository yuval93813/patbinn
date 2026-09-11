#!/bin/bash
#SBATCH --job-name=patbinn_expert_sweep_retry
#SBATCH --output=patbinn_expert_sweep_retry_%j.out
#SBATCH --error=patbinn_expert_sweep_retry_%j.err
# Partition/account are intentionally not hardcoded so this runs on any cluster.
# Set them at submit time, e.g.:  SBATCH_PARTITION=gpu sbatch <script>
# (Slurm honours the SBATCH_PARTITION / SBATCH_ACCOUNT environment variables,
#  or pass --partition=... --account=... directly to sbatch.)
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32

set -euo pipefail
WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
IMAGE_SIF="$WORKDIR/patbinn.sif"
mkdir -p "$WORKDIR/checkpoints/expert_sweep" "$WORKDIR/results"

# Retry of the 227 expert sweep: the original concurrent run raced with
# sweep_router.py and the Kraken2 task over the shared default
# data/manifests/cv_fold1_sample_manifest.csv path and read it mid-write,
# seeing only 4 of 8 families and crashing on the checkpoint-compatibility
# check. Fixed here (and in slurm_parallel_batch.sh going forward) with a
# dedicated --manifests_dir so this can never collide with a concurrent writer.
apptainer exec --nv --bind "$WORKDIR":/workspace --pwd /workspace "$IMAGE_SIF" \
    python3 src/sweep_experts.py \
    --cv_sample_manifest data/manifests/cv_sample_manifest.csv \
    --k_folds 5 --fold 1 \
    --manifests_dir data/manifests_expert_sweep \
    --base_checkpoint checkpoints/cv_hierarchical_binary/fold1.pth \
    --vocab_cache data/vocab_hierarchical_binary/kmer_vocab_fold1.json \
    --expert_epochs 30 --max_kmers 1024 \
    --checkpoints_dir checkpoints/expert_sweep \
    --out_csv results/expert_sweep.csv
