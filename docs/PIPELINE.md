# Data pipeline

## Overview

```
manifest of (family, variant)               <- the classes we want, hand-curated
        |  data/download_genomes.py
        v
real NCBI genome isolates, fold-assigned    <- data/refs/, genome_manifest_multi.csv
        |  data/simulate_reads.py (ART Illumina)
        v
simulated short reads, fold-tagged          <- data/reads_multi/, cv_sample_manifest.csv
        |  run_cv.py (per fold: synthesize train/val/test split)
        v
k-mer vectors -> router + experts (or flat) <- train_moe.py / train_flat.py
        |
        v
per-fold results -> aggregate_cv_results.py -> results/cv_*/
```

The dataset this produces: **543 genomes, 58 classes, 8 families**, 3 to 10
independent isolates per class. Every accession actually used is listed in
[GENOME_INVENTORY.txt](GENOME_INVENTORY.txt), which is regenerated from the
manifest by `scripts/make_genome_inventory.py`.

## 1. Genome download (`data/download_genomes.py`)

Input: a manifest CSV of `(family, variant, ncbi_query_terms, length_band)`.
`data/manifests/generalization_manifest.csv` holds the 58 classes kept for this
experiment, see [EXPERIMENTS.md](EXPERIMENTS.md) for which classes were
dropped and why.

For each class the downloader queries NCBI E-utilities (`esearch`/`efetch`)
with a **field-tagged `[Organism]` query plus a genome-length band**. Free-text
search is not safe here: early iterations matched human HLA mRNA for a
"SARS-CoV-2" query and Camel alphacoronavirus for an "HCoV-229E" query.
Candidates are further validated by keyword presence in the title, matched on
word boundaries so that "coronavirus" does not spuriously match inside
"alphacoronavirus".

Up to `--num_isolates` (10 in the full run) independent isolates are collected
per class, deduplicated three ways (by sequence ID, by version-stripped base
accession, and by exact sequence hash), because the same genome can appear more
than once in NCBI results under all three.

**Fold assignment happens here, at isolate granularity:**
`fold = isolate_idx % K` (round-robin, K=5). This is the detail that prevents
leakage; see the section below.

Output: `data/refs/<family>/<variant>/isolate_{i}.fasta` and
`data/manifests/genome_manifest_multi.csv` (one row per isolate, carrying
`isolate_idx` and `fold`).

### Two retrieval bugs that were found and fixed

Both silently reduced isolate counts rather than erroring, so they are worth
knowing about before changing this file:

1. A bare organism query's result budget was exhausted by patent sequences and
   gene fragments before the length-band filter could admit a real genome.
2. A whole-word organism-name check rejected genuinely correct hits under
   several since-renamed ICTV species names (e.g. *Human herpesvirus 1* →
   *Human alphaherpesvirus 1*).

Fixing them recovered enough additional isolates that **no class is represented
by a single genome any more**: every one of the 58 classes now has at least 3
independent isolates and therefore receives a genuine held-out-genome test in
every fold.

## 2. Read simulation (`data/simulate_reads.py`)

For each isolate genome, ART (`art_illumina`, HiSeq2500 profile) simulates
paired-end 150 bp Illumina reads at 30× coverage, with **6 independent
replicates per isolate**, so that simulator noise is not a single fixed draw.

Every simulated read inherits its parent isolate's `fold`. The script writes
`data/manifests/cv_sample_manifest.csv`: one row per simulated
sample/replicate, with `family`, `family_idx`, `variant`, `variant_idx`,
`isolate_idx`, `replicate`, `r1_path`, `r2_path` and `fold`.

**A leakage check is asserted at write time:** every row sharing an
`isolate_idx` within a `(family, variant)` must share the same `fold`. This is
the actual guarantee that no genome's reads are split across train and test.
Without it, "held-out" could silently mean held-out *reads* from a genome the
model already trained on, a far weaker and misleadingly easy claim.

## 3. Cross-validation orchestration (`src/run_cv.py`)

For each held-out fold `i` in `0..K-1`:

- fold `i` → **test**
- fold `(i+1) % K` → **validation** (early stopping and model selection)
- the remaining `K-2` folds → **train**

Validation is disjoint from test, so early-stopping decisions never see the
fold being scored. `src/run_cv.py` synthesizes a `split` column per fold, writes a
per-fold manifest, and builds **a separate k-mer vocabulary per fold from that
fold's training rows only**: the vocabulary must not leak which k-mers are
common in held-out isolates.

Per fold, `train_moe.run_training()` (or the flat equivalent) trains and saves
`checkpoints/cv_<tag>/fold{i}.pth`; `evaluate_moe.evaluate_checkpoint()` then
scores it on that fold's test split and writes
`results/cv_<tag>/fold{i}_result.json`.

`src/aggregate_cv_results.py` combines the five per-fold JSONs into mean ± std per
metric, which is why every headline number in this repository is reported as a
mean over folds rather than a single accuracy.

`k_folds=2` is rejected with a clear error: test and validation would consume
both folds, leaving nothing to train on.

## Why isolate-level fold assignment matters

If *reads* rather than whole isolates were split into train and test, the model
could see 5 of an isolate's 6 replicate read sets during training and be scored
on the 6th. That yields near-guaranteed high accuracy by memorizing one
genome's k-mer fingerprint, and says nothing about generalizing to a genome
never seen before. Isolate-level assignment is what makes "held-out" mean *an
entirely different real virus isolate*, the whole point of this benchmark. See
[EXPERIMENTS.md](EXPERIMENTS.md) for the single-genome result this replaces.

## Index remapping: a subtlety worth preserving

`moe_dataset.remap_variant_idx(rows)` densifies both `family_idx` and
`variant_idx`, and **must be called on the complete manifest before any
fold-filtering**, never on a pre-filtered subset.

Remapping from a split-filtered subset produces a different mapping whenever
that subset happens to be missing a variant, yielding indices that do not match
what the checkpoint was trained with. `evaluate_moe.evaluate_checkpoint()`
therefore remaps from all rows and only then filters to the test split.

This also fixed a related bug: `src/train_moe.py` once computed one global variant
count for every expert while `src/evaluate_moe.py` derived it per family. The two
agreed only because every family originally had 8 variants; they stopped
agreeing as soon as family sizes became uneven, which they now are (3 to 8).

## Running on a cluster

The scripts in [`../scripts/slurm/`](../scripts/slurm/) submit this pipeline as
Slurm jobs. Three portability details are baked into them:

1. **No hardcoded partition or account.** Supply them at submit time, e.g.
   `SBATCH_PARTITION=gpu sbatch scripts/slurm/slurm_run_cv.sh`. Slurm honours
   the `SBATCH_PARTITION` and `SBATCH_ACCOUNT` environment variables.
2. **Working directory resolution uses `$SLURM_SUBMIT_DIR`**, never
   `$(dirname "${BASH_SOURCE[0]}")`. Slurm copies the submitted script to a
   spool directory before executing it, so `BASH_SOURCE`-based resolution
   silently points at the spool directory instead of the project. Submit from
   the repository root.
3. **Each `(classifier, precision)` combination gets its own vocabulary,
   checkpoint and results directories**, so concurrent jobs cannot race on a
   shared vocabulary cache file.

Note also that `moe_dataset.encode_sequences_parallel()` must use a **spawn**
multiprocessing context, never the default fork: both trainers move a model to
CUDA before constructing the dataset, and forking after CUDA initialization is
unsafe.
