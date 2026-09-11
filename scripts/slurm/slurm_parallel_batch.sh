#!/bin/bash
#SBATCH --job-name=patbinn_parallel
#SBATCH --output=patbinn_parallel_%j.out
#SBATCH --error=patbinn_parallel_%j.err
# Partition/account are intentionally not hardcoded so this runs on any cluster.
# Set them at submit time, e.g.:  SBATCH_PARTITION=gpu sbatch <script>
# (Slurm honours the SBATCH_PARTITION / SBATCH_ACCOUNT environment variables,
#  or pass --partition=... --account=... directly to sbatch.)
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=64

set -euo pipefail
WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
IMAGE_SIF="$WORKDIR/patbinn.sif"
KRAKEN_SIF="$WORKDIR/containers/kraken2.sif"

RUN() { apptainer exec --nv --bind "$WORKDIR":/workspace --pwd /workspace "$IMAGE_SIF" "$@"; }
RUNK() { apptainer exec --bind "$WORKDIR":/workspace --pwd /workspace "$KRAKEN_SIF" "$@"; }

mkdir -p checkpoints/cv_hierarchical_binary checkpoints/cv_hierarchical_float32 checkpoints/cv_flat_binary \
         checkpoints/encoding checkpoints/final checkpoints/router_sweep checkpoints/expert_sweep \
         data/vocab_hierarchical_binary data/vocab_hierarchical_float32 data/vocab_flat_binary data/vocab \
         data/manifests_hierarchical_binary data/manifests_hierarchical_float32 data/manifests_flat_binary \
         data/manifests_k6 data/manifests_canon5 data/manifests_k4 \
         results/cv_hierarchical_binary results/cv_hierarchical_float32 results/cv_flat_binary results

# Bounded-concurrency runner: launches each command in the array in the
# background, capping simultaneous jobs at $1. Peak memory measured for one
# full router+8-expert fold is ~45GB, so with a 128GB budget, 2 concurrent
# heavy (full-pipeline) jobs is the safe ceiling - GPU compute is not the
# constraint here (utilization was ~12% during serial runs), memory is.
run_batch() {
    local max_concurrent="$1"; shift
    local pids=()
    for cmd in "$@"; do
        while [[ ${#pids[@]} -ge $max_concurrent ]]; do
            wait -n
            local alive=()
            for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && alive+=("$pid"); done
            pids=("${alive[@]}")
        done
        eval "$cmd" &
        pids+=($!)
    done
    wait
}

echo "=== PHASE 1: independent full-pipeline jobs, concurrency=5 (memory-bound, ~45GB each against 256GB budget; 64 CPUs shared/oversubscribed across them) ==="
phase1=(
    "RUN python3 src/run_cv.py --cv_sample_manifest data/manifests/cv_sample_manifest.csv --k_folds 5 --only_fold 0 --classifier hierarchical --precision binary --max_kmers 1024 --router_hidden_sizes 1024 128 --expert_hidden_sizes 1024 128 --router_epochs 30 --expert_epochs 30 --checkpoints_dir checkpoints/cv_hierarchical_binary --vocab_dir data/vocab_hierarchical_binary --manifests_dir data/manifests_hierarchical_binary --results_dir results/cv_hierarchical_binary > patbinn_parallel_223_fold0.log 2>&1"
    "RUN python3 src/run_cv.py --cv_sample_manifest data/manifests/cv_sample_manifest.csv --k_folds 5 --only_fold 1 --classifier hierarchical --precision binary --max_kmers 1024 --router_hidden_sizes 1024 128 --expert_hidden_sizes 1024 128 --router_epochs 30 --expert_epochs 30 --checkpoints_dir checkpoints/cv_hierarchical_binary --vocab_dir data/vocab_hierarchical_binary --manifests_dir data/manifests_hierarchical_binary --results_dir results/cv_hierarchical_binary > patbinn_parallel_223_fold1.log 2>&1"
    "RUN python3 src/run_cv.py --cv_sample_manifest data/manifests/cv_sample_manifest.csv --k_folds 5 --only_fold 2 --classifier hierarchical --precision binary --max_kmers 1024 --router_hidden_sizes 1024 128 --expert_hidden_sizes 1024 128 --router_epochs 30 --expert_epochs 30 --checkpoints_dir checkpoints/cv_hierarchical_binary --vocab_dir data/vocab_hierarchical_binary --manifests_dir data/manifests_hierarchical_binary --results_dir results/cv_hierarchical_binary > patbinn_parallel_223_fold2.log 2>&1"
    "RUN python3 src/run_cv.py --cv_sample_manifest data/manifests/cv_sample_manifest.csv --k_folds 5 --only_fold 3 --classifier hierarchical --precision binary --max_kmers 1024 --router_hidden_sizes 1024 128 --expert_hidden_sizes 1024 128 --router_epochs 30 --expert_epochs 30 --checkpoints_dir checkpoints/cv_hierarchical_binary --vocab_dir data/vocab_hierarchical_binary --manifests_dir data/manifests_hierarchical_binary --results_dir results/cv_hierarchical_binary > patbinn_parallel_223_fold3.log 2>&1"
    "RUN python3 src/run_cv.py --cv_sample_manifest data/manifests/cv_sample_manifest.csv --k_folds 5 --only_fold 4 --classifier hierarchical --precision binary --max_kmers 1024 --router_hidden_sizes 1024 128 --expert_hidden_sizes 1024 128 --router_epochs 30 --expert_epochs 30 --checkpoints_dir checkpoints/cv_hierarchical_binary --vocab_dir data/vocab_hierarchical_binary --manifests_dir data/manifests_hierarchical_binary --results_dir results/cv_hierarchical_binary > patbinn_parallel_223_fold4.log 2>&1"
    "RUN python3 src/run_cv.py --cv_sample_manifest data/manifests/cv_sample_manifest.csv --k_folds 5 --only_fold 0 --classifier hierarchical --precision float32 --max_kmers 1024 --router_hidden_sizes 1024 128 --expert_hidden_sizes 1024 128 --router_epochs 30 --expert_epochs 30 --checkpoints_dir checkpoints/cv_hierarchical_float32 --vocab_dir data/vocab_hierarchical_float32 --manifests_dir data/manifests_hierarchical_float32 --results_dir results/cv_hierarchical_float32 > patbinn_parallel_224_fold0.log 2>&1"
    "RUN python3 src/run_cv.py --cv_sample_manifest data/manifests/cv_sample_manifest.csv --k_folds 5 --only_fold 1 --classifier hierarchical --precision float32 --max_kmers 1024 --router_hidden_sizes 1024 128 --expert_hidden_sizes 1024 128 --router_epochs 30 --expert_epochs 30 --checkpoints_dir checkpoints/cv_hierarchical_float32 --vocab_dir data/vocab_hierarchical_float32 --manifests_dir data/manifests_hierarchical_float32 --results_dir results/cv_hierarchical_float32 > patbinn_parallel_224_fold1.log 2>&1"
    "RUN python3 src/run_cv.py --cv_sample_manifest data/manifests/cv_sample_manifest.csv --k_folds 5 --only_fold 2 --classifier hierarchical --precision float32 --max_kmers 1024 --router_hidden_sizes 1024 128 --expert_hidden_sizes 1024 128 --router_epochs 30 --expert_epochs 30 --checkpoints_dir checkpoints/cv_hierarchical_float32 --vocab_dir data/vocab_hierarchical_float32 --manifests_dir data/manifests_hierarchical_float32 --results_dir results/cv_hierarchical_float32 > patbinn_parallel_224_fold2.log 2>&1"
    "RUN python3 src/run_cv.py --cv_sample_manifest data/manifests/cv_sample_manifest.csv --k_folds 5 --only_fold 3 --classifier hierarchical --precision float32 --max_kmers 1024 --router_hidden_sizes 1024 128 --expert_hidden_sizes 1024 128 --router_epochs 30 --expert_epochs 30 --checkpoints_dir checkpoints/cv_hierarchical_float32 --vocab_dir data/vocab_hierarchical_float32 --manifests_dir data/manifests_hierarchical_float32 --results_dir results/cv_hierarchical_float32 > patbinn_parallel_224_fold3.log 2>&1"
    "RUN python3 src/run_cv.py --cv_sample_manifest data/manifests/cv_sample_manifest.csv --k_folds 5 --only_fold 4 --classifier hierarchical --precision float32 --max_kmers 1024 --router_hidden_sizes 1024 128 --expert_hidden_sizes 1024 128 --router_epochs 30 --expert_epochs 30 --checkpoints_dir checkpoints/cv_hierarchical_float32 --vocab_dir data/vocab_hierarchical_float32 --manifests_dir data/manifests_hierarchical_float32 --results_dir results/cv_hierarchical_float32 > patbinn_parallel_224_fold4.log 2>&1"
    "RUN python3 src/run_cv.py --cv_sample_manifest data/manifests/cv_sample_manifest.csv --k_folds 5 --only_fold 0 --classifier flat --precision binary --max_kmers 1024 --router_hidden_sizes 1024 128 --expert_hidden_sizes 1024 128 --router_epochs 30 --expert_epochs 30 --checkpoints_dir checkpoints/cv_flat_binary --vocab_dir data/vocab_flat_binary --manifests_dir data/manifests_flat_binary --results_dir results/cv_flat_binary > patbinn_parallel_225_fold0.log 2>&1"
    "RUN python3 src/run_cv.py --cv_sample_manifest data/manifests/cv_sample_manifest.csv --k_folds 5 --only_fold 1 --classifier flat --precision binary --max_kmers 1024 --router_hidden_sizes 1024 128 --expert_hidden_sizes 1024 128 --router_epochs 30 --expert_epochs 30 --checkpoints_dir checkpoints/cv_flat_binary --vocab_dir data/vocab_flat_binary --manifests_dir data/manifests_flat_binary --results_dir results/cv_flat_binary > patbinn_parallel_225_fold1.log 2>&1"
    "RUN python3 src/run_cv.py --cv_sample_manifest data/manifests/cv_sample_manifest.csv --k_folds 5 --only_fold 2 --classifier flat --precision binary --max_kmers 1024 --router_hidden_sizes 1024 128 --expert_hidden_sizes 1024 128 --router_epochs 30 --expert_epochs 30 --checkpoints_dir checkpoints/cv_flat_binary --vocab_dir data/vocab_flat_binary --manifests_dir data/manifests_flat_binary --results_dir results/cv_flat_binary > patbinn_parallel_225_fold2.log 2>&1"
    "RUN python3 src/run_cv.py --cv_sample_manifest data/manifests/cv_sample_manifest.csv --k_folds 5 --only_fold 3 --classifier flat --precision binary --max_kmers 1024 --router_hidden_sizes 1024 128 --expert_hidden_sizes 1024 128 --router_epochs 30 --expert_epochs 30 --checkpoints_dir checkpoints/cv_flat_binary --vocab_dir data/vocab_flat_binary --manifests_dir data/manifests_flat_binary --results_dir results/cv_flat_binary > patbinn_parallel_225_fold3.log 2>&1"
    "RUN python3 src/run_cv.py --cv_sample_manifest data/manifests/cv_sample_manifest.csv --k_folds 5 --only_fold 4 --classifier flat --precision binary --max_kmers 1024 --router_hidden_sizes 1024 128 --expert_hidden_sizes 1024 128 --router_epochs 30 --expert_epochs 30 --checkpoints_dir checkpoints/cv_flat_binary --vocab_dir data/vocab_flat_binary --manifests_dir data/manifests_flat_binary --results_dir results/cv_flat_binary > patbinn_parallel_225_fold4.log 2>&1"
    "RUN python3 src/run_encoding_experiment.py --label k6_1024 --k 6 --max_kmers 1024 --fold 1 --manifests_dir data/manifests_k6 --vocab_cache data/vocab/kmer_vocab_fold1_k6.json --save_model checkpoints/encoding/k6_1024.pth --out_csv results/encoding_variants.csv > patbinn_parallel_228_k6.log 2>&1"
    "RUN python3 src/run_encoding_experiment.py --label canonical_k5_512 --k 5 --max_kmers 512 --canonical --fold 1 --manifests_dir data/manifests_canon5 --vocab_cache data/vocab/kmer_vocab_fold1_k5canon.json --save_model checkpoints/encoding/canonical_k5_512.pth --out_csv results/encoding_variants.csv > patbinn_parallel_228_canon5.log 2>&1"
    "RUN python3 src/run_encoding_experiment.py --label k4_256 --k 4 --max_kmers 256 --fold 1 --manifests_dir data/manifests_k4 --vocab_cache data/vocab/kmer_vocab_fold1_k4.json --save_model checkpoints/encoding/k4_256.pth --out_csv results/encoding_variants.csv > patbinn_parallel_229_k4.log 2>&1"
    "RUN python3 src/train_final_model.py --cv_sample_manifest data/manifests/cv_sample_manifest.csv --val_fold 4 --precision binary --router_hidden_sizes 1024 128 --expert_hidden_sizes 1024 128 --router_epochs 30 --expert_epochs 30 --max_kmers 1024 --vocab_cache data/vocab/kmer_vocab_final.json --save_model checkpoints/final/moe_final_1024x128.pth > patbinn_parallel_231_final.log 2>&1"
)
run_batch 5 "${phase1[@]}"
echo "=== PHASE 1 done ==="

echo "=== PHASE 2: fold-1-checkpoint-dependent sweeps, concurrency=3 (partial retraining, lighter) ==="
phase2=(
    "RUN python3 src/sweep_router.py --cv_sample_manifest data/manifests/cv_sample_manifest.csv --k_folds 5 --fold 1 --manifests_dir data/manifests_router_sweep --base_checkpoint checkpoints/cv_hierarchical_binary/fold1.pth --vocab_cache data/vocab_hierarchical_binary/kmer_vocab_fold1.json --router_epochs 30 --max_kmers 1024 --checkpoints_dir checkpoints/router_sweep --out_csv results/router_sweep.csv > patbinn_parallel_226_router.log 2>&1"
    "RUN python3 src/sweep_experts.py --cv_sample_manifest data/manifests/cv_sample_manifest.csv --k_folds 5 --fold 1 --manifests_dir data/manifests_expert_sweep --base_checkpoint checkpoints/cv_hierarchical_binary/fold1.pth --vocab_cache data/vocab_hierarchical_binary/kmer_vocab_fold1.json --expert_epochs 30 --max_kmers 1024 --checkpoints_dir checkpoints/expert_sweep --out_csv results/expert_sweep.csv > patbinn_parallel_227_expert.log 2>&1"
    "bash scripts_parallel_kraken_task.sh > patbinn_parallel_230_kraken.log 2>&1"
)
run_batch 3 "${phase2[@]}"
echo "=== PHASE 2 done ==="

echo "=== PHASE 3: needs all-5-folds / deployed-model outputs from phase 1, concurrency=2 ==="
phase3=(
    "RUN python3 src/sweep_read_counts.py --checkpoints_dir checkpoints/cv_hierarchical_binary --manifests_dir data/manifests_hierarchical_binary --k_folds 5 --out_csv results/read_count_sweep.csv > patbinn_parallel_233_readcount.log 2>&1"
    "RUN python3 src/sweep_error_rates.py --checkpoint checkpoints/final/moe_final_1024x128.pth --sample_manifest data/manifests/final_train_manifest.csv --split val --device cpu --out_csv results/error_rate_sweep.csv > patbinn_parallel_232_errorrate.log 2>&1"
)
run_batch 2 "${phase3[@]}"
echo "=== PHASE 3 done ==="

echo "=== ALL PHASES COMPLETE ==="
