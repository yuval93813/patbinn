#!/bin/bash
#SBATCH --job-name=patbinn_kraken_compare
#SBATCH --output=patbinn_kraken_compare_%j.out
#SBATCH --error=patbinn_kraken_compare_%j.err
# Partition/account are intentionally not hardcoded so this runs on any cluster.
# Set them at submit time, e.g.:  SBATCH_PARTITION=gpu sbatch <script>
# (Slurm honours the SBATCH_PARTITION / SBATCH_ACCOUNT environment variables,
#  or pass --partition=... --account=... directly to sbatch.)
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

set -euo pipefail
WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PATBINN_SIF="$WORKDIR/patbinn.sif"
KRAKEN_SIF="$WORKDIR/containers/kraken2.sif"

mkdir -p "$WORKDIR/results" "$WORKDIR/data/manifests"

# Fold 1's plain (non-tagged) manifest is a byproduct of sweep_router.py /
# sweep_experts.py / run_encoding_experiment.py, all of which default to
# --manifests_dir data/manifests. Regenerate it explicitly here too, so this
# job doesn't silently depend on run order relative to those.
apptainer exec --bind "$WORKDIR":/workspace --pwd /workspace "$PATBINN_SIF" \
    python3 -c "
from run_cv import load_cv_manifest, write_fold_manifest
rows = load_cv_manifest('data/manifests/cv_sample_manifest.csv')
write_fold_manifest(rows, 1, 5, 'data/manifests/cv_fold1_sample_manifest.csv')
"

# Kraken2 side: build a DB from exactly fold 1's non-test genomes, then classify
# the same corrupted reads PatBiNN sees, across the same error sweep.
apptainer exec --bind "$WORKDIR":/workspace --pwd /workspace "$KRAKEN_SIF" \
    python3 src/compare_kraken2.py --build \
    --db data/kraken2_db_fold1 \
    --genome_manifest data/manifests/genome_manifest_multi.csv \
    --sample_manifest data/manifests/cv_fold1_sample_manifest.csv \
    --test_fold 1

apptainer exec --bind "$WORKDIR":/workspace --pwd /workspace "$KRAKEN_SIF" \
    python3 src/compare_kraken2.py --classify \
    --db data/kraken2_db_fold1 \
    --genome_manifest data/manifests/genome_manifest_multi.csv \
    --sample_manifest data/manifests/cv_fold1_sample_manifest.csv \
    --test_fold 1 \
    --seed 0 \
    --out_csv results/kraken2_error_sweep.csv

# PatBiNN side: the fold-1 CV checkpoint (not the deployed model) on the
# identical fold-1 test reads and the identical seed, so the two methods see
# byte-identical input at every error rate.
apptainer exec --nv --bind "$WORKDIR":/workspace --pwd /workspace "$PATBINN_SIF" \
    python3 src/sweep_error_rates.py \
    --checkpoint checkpoints/cv_hierarchical_binary/fold1.pth \
    --sample_manifest data/manifests/cv_fold1_sample_manifest.csv \
    --split test \
    --seed 0 \
    --out_csv results/patbinn_fold1_error_sweep.csv
