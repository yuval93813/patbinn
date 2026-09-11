# Scripts

## `make_genome_inventory.py`

Regenerates [docs/GENOME_INVENTORY.txt](../docs/GENOME_INVENTORY.txt) from
`data/manifests/genome_manifest_multi.csv`. The inventory is the human-readable
provenance record of every NCBI accession used, grouped by family and variant,
with each isolate's CV fold.

```bash
python3 scripts/make_genome_inventory.py
```

The inventory is derived entirely from the manifest, so it never needs editing
by hand. Regenerate it whenever the genome manifest changes.

## `slurm/`: cluster job scripts

These submit the full pipeline as Slurm jobs. They are the exact scripts that
produced the results in this repository, with cluster-specific details removed.

### Running them

**No partition or account is hardcoded.** Supply them at submit time. Slurm
honours the `SBATCH_PARTITION` and `SBATCH_ACCOUNT` environment variables:

```bash
SBATCH_PARTITION=gpu sbatch scripts/slurm/slurm_run_cv.sh
# or equivalently
sbatch --partition=gpu --account=myaccount scripts/slurm/slurm_run_cv.sh
```

**Submit from the repository root.** Every script resolves its working
directory from `$SLURM_SUBMIT_DIR`, which is where `sbatch` was invoked, not
where the script lives. (This is deliberate: Slurm copies the submitted script
to a spool directory before executing it, so `BASH_SOURCE`-based resolution
would silently point at the spool directory instead of the project.)

Each script builds `patbinn.sif` from `patbinn.def` if it is missing or out of
date, then runs the job inside it.

### The scripts

**Data preparation**

| Script | Runs |
|---|---|
| `slurm_data_prep.sh` | `download_genomes.py` → `simulate_reads.py`, single-genome-per-class variant |
| `slurm_cv_data_prep.sh` | `download_genomes.py` → `simulate_reads.py`, the multi-isolate CV dataset |
| `slurm_resimulate_reads.sh` | `simulate_reads.py` only, to re-simulate reads from already-downloaded genomes |

**Main results**

| Script | Runs |
|---|---|
| `slurm_run_cv.sh` | `src/run_cv.py` as a 5-task array, one per fold, then aggregates. Set `CLASSIFIER=flat` and/or `PRECISION=float32` to run the baselines. |
| `slurm_parallel_batch.sh` | The whole experiment matrix as one dependency-chained batch |
| `slurm_train_moe.sh` | A single hierarchical training run |
| `slurm_train_final.sh` | `src/train_final_model.py`: the deployable model, no holdout |

**Sweeps**

| Script | Runs |
|---|---|
| `slurm_router_sweep.sh` | `src/sweep_router.py`: router size/depth |
| `slurm_expert_sweep.sh`, `slurm_expert_sweep_retry.sh` | `src/sweep_experts.py`: expert size/depth |
| `slurm_sweep_architecture.sh`, `slurm_sweep_arch_extra.sh` | `src/sweep_architecture.py`: joint grid and extra shapes |
| `slurm_sweep_cross.sh` | `src/sweep_cross_combos.py`: router × expert interaction check |
| `slurm_encoding_experiments.sh`, `slurm_k4.sh` | `src/run_encoding_experiment.py`: encoding variants |
| `slurm_read_count_sweep.sh` | `src/sweep_read_counts.py` |
| `slurm_error_rate_sweep.sh` | `src/sweep_error_rates.py` |

**Kraken2 comparison**

| Script | Runs |
|---|---|
| `slurm_kraken_compare.sh` | `src/compare_kraken2.py --build` then `--classify` |
| `scripts_parallel_kraken_task.sh` | Helper invoked per task by the above |

### Typical order

```
slurm_cv_data_prep.sh
  -> slurm_run_cv.sh            (x3: hierarchical/binary, flat, float32)
  -> slurm_router_sweep.sh, slurm_expert_sweep.sh, slurm_encoding_experiments.sh
  -> slurm_read_count_sweep.sh, slurm_error_rate_sweep.sh
  -> slurm_kraken_compare.sh
  -> slurm_train_final.sh
```

### A note on runtime

These are long jobs, not minutes-long ones: a full CV fold trains a router plus
eight experts in sequence. Set Slurm time limits generously.
