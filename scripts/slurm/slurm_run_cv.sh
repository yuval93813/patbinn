#!/bin/bash
#SBATCH --job-name=patbinn_run_cv
#SBATCH --output=patbinn_run_cv_%A_%a.out
#SBATCH --error=patbinn_run_cv_%A_%a.err
# Partition/account are intentionally not hardcoded so this runs on any cluster.
# Set them at submit time, e.g.:  SBATCH_PARTITION=gpu sbatch <script>
# (Slurm honours the SBATCH_PARTITION / SBATCH_ACCOUNT environment variables,
#  or pass --partition=... --account=... directly to sbatch.)
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32
#SBATCH --array=0-4
#
# One Slurm array task per CV fold (5 folds, --array=0-4 matches --k_folds 5 below).
# Set CLASSIFIER=flat and/or PRECISION=float32 in the environment
# (sbatch --export=CLASSIFIER=flat,PRECISION=float32 ...) to run the flat-classifier
# and/or float32-precision baselines instead of the hierarchical/binary default.
# Each (classifier,precision) combo gets its own vocab/checkpoint/results dirs to
# avoid concurrent-job races on a shared vocab cache file.
# After all 5 tasks of a combo complete, run:
#   python3 src/aggregate_cv_results.py --results_dir results/cv_<tag> --out_csv results/cv_results_<tag>.csv

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
IMAGE_SIF="$WORKDIR/patbinn.sif"
CLASSIFIER="${CLASSIFIER:-hierarchical}"
PRECISION="${PRECISION:-binary}"
TAG="${CLASSIFIER}_${PRECISION}"

mkdir -p "$WORKDIR/checkpoints/cv_$TAG" "$WORKDIR/results/cv_$TAG" "$WORKDIR/data/vocab_$TAG" "$WORKDIR/data/manifests_$TAG"

if [[ ! -f "$IMAGE_SIF" || "$WORKDIR/patbinn.def" -nt "$IMAGE_SIF" ]]; then
    apptainer build --fakeroot "$IMAGE_SIF" "$WORKDIR/patbinn.def"
fi

apptainer exec --nv \
    --bind "$WORKDIR":/workspace \
    --pwd /workspace \
    "$IMAGE_SIF" \
    python3 src/run_cv.py \
    --cv_sample_manifest data/manifests/cv_sample_manifest.csv \
    --k_folds 5 \
    --only_fold "${SLURM_ARRAY_TASK_ID}" \
    --classifier "$CLASSIFIER" \
    --precision "$PRECISION" \
    --max_kmers 1024 --router_hidden_sizes 1024 128 --expert_hidden_sizes 1024 128 \
    --router_epochs 30 --expert_epochs 30 \
    --checkpoints_dir "checkpoints/cv_$TAG" \
    --vocab_dir "data/vocab_$TAG" \
    --manifests_dir "data/manifests_$TAG" \
    --results_dir "results/cv_$TAG"
