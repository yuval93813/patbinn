#!/bin/bash
# Standalone helper for the Kraken2-comparison leg of slurm_parallel_batch.sh's
# phase 2. Pulled out into its own file rather than inlined, so it doesn't need
# to survive being embedded inside an eval'd, already-quoted array element.
set -euo pipefail
WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PATBINN_SIF="$WORKDIR/patbinn.sif"
KRAKEN_SIF="$WORKDIR/containers/kraken2.sif"

apptainer exec --bind "$WORKDIR":/workspace --pwd /workspace "$PATBINN_SIF" \
    python3 -c "
from run_cv import load_cv_manifest, write_fold_manifest
rows = load_cv_manifest('data/manifests/cv_sample_manifest.csv')
write_fold_manifest(rows, 1, 5, 'data/manifests_kraken/cv_fold1_sample_manifest.csv')
"

apptainer exec --bind "$WORKDIR":/workspace --pwd /workspace "$KRAKEN_SIF" \
    python3 src/compare_kraken2.py --build \
    --db data/kraken2_db_fold1 \
    --genome_manifest data/manifests/genome_manifest_multi.csv \
    --sample_manifest data/manifests_kraken/cv_fold1_sample_manifest.csv \
    --test_fold 1

apptainer exec --bind "$WORKDIR":/workspace --pwd /workspace "$KRAKEN_SIF" \
    python3 src/compare_kraken2.py --classify \
    --db data/kraken2_db_fold1 \
    --genome_manifest data/manifests/genome_manifest_multi.csv \
    --sample_manifest data/manifests_kraken/cv_fold1_sample_manifest.csv \
    --test_fold 1 \
    --seed 0 \
    --out_csv results/kraken2_error_sweep.csv

apptainer exec --nv --bind "$WORKDIR":/workspace --pwd /workspace "$PATBINN_SIF" \
    python3 src/sweep_error_rates.py \
    --checkpoint checkpoints/cv_hierarchical_binary/fold1.pth \
    --sample_manifest data/manifests_kraken/cv_fold1_sample_manifest.csv \
    --split test \
    --seed 0 \
    --out_csv results/patbinn_fold1_error_sweep.csv
