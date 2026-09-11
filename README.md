# PatBiNN: Hierarchical Binarized Virus Read Classifier

A hardware-oriented **binarized neural network** that identifies which of 58
known virus variants a single short Illumina sequencing read came from, using a
two-stage **Mixture-of-Experts** design (a family router, then one of 8
per-family variant experts) instead of one flat 58-way classifier.

Every number in this repository comes from **genome-level 5-fold
cross-validation** on 543 real NCBI isolates: whole genomes are withheld from
training, never merely some of their reads. A model is only ever tested on
viruses whose genome it has never seen.

---

## Results

5-fold mean ± standard deviation. *Specimen-level* is the per-genome majority
vote over that specimen's reads; it is the number that matters operationally,
since in practice you classify a sample, not a single read.

| Condition | Per-read (family) | Per-read (variant, oracle-routed) | **Specimen-level** |
|---|---|---|---|
| **Hierarchical MoE, binary** (main result) | 97.76% ± 0.69 | 91.81% ± 1.39 | **99.55% ± 1.02** |
| Hierarchical MoE, float32 (precision control) | 99.57% ± 0.22 | 98.76% ± 0.55 | 99.73% ± 0.53 |
| Flat 58-way classifier (no hierarchy) | n/a | 43.53% ± 3.16 *(leaf)* | 88.78% ± 4.15 |

> **Reading the per-read columns.** For the hierarchical model, *family* is the
> router's own accuracy and *variant* is measured with oracle routing: the
> expert is handed the true family, so it is an easier task and is **not**
> comparable to the flat classifier's leaf accuracy. The specimen-level column
> is measured identically for all three and is the fair head-to-head.

**What this run was built to test, in four questions:**

1. **Does it generalize to unseen genomes?** Yes. 99.55% specimen-level on
   entirely held-out isolates, with every one of the 58 classes represented by
   at least 3 independent genomes.
2. **Does the hierarchy help?** Yes, substantially: 99.55% vs 88.78%
   specimen-level against the flat baseline, and the gap is far larger per read.
3. **Is binarization affordable?** Yes. The deployable binary network gives up
   only 0.18 points at the specimen level against its float32 twin
   (99.55% vs 99.73%), well inside fold-to-fold noise, despite every weight
   and activation being constrained to ±1.
4. **Are two hidden layers necessary?** Yes. No single-hidden-layer router
   reaches any two-layer router (64.2-84.6% vs 75.6-98.0% per read); the second
   layer is a requirement, not a refinement.

Full per-fold tables, every sweep, and the limitations that remain are in
**[docs/RESULTS.md](docs/RESULTS.md)**.

### Against Kraken2: comparable accuracy, far smaller computation

Benchmarked against [Kraken2](https://github.com/DerrickWood/kraken2) on
*identical* reads, with Kraken2's database built from exactly the genomes
PatBiNN was allowed to train on, so neither method has seen a test isolate.

**This is a parity result, and is deliberately not stated as a win.** On clean
reads the two are close (97.58% vs 96.67% specimen-level); as sequencing error
rises Kraken2 is clearly *more* robust per read (74.09% vs 60.76% at 5% error),
because Kraken2 needs one surviving exact k-mer whereas a bag-of-k-mers vector
degrades globally. Kraken2's failure mode is abstention: it leaves 8% of reads
unclassified on clean data and 58% at 10% error, while PatBiNN has no
abstention mechanism and expresses the same degradation as error instead.

The supportable claim is **comparable accuracy from a far smaller, wholly
binary computation**, not superior accuracy. See
[docs/RESULTS.md](docs/RESULTS.md#comparison-with-kraken2) for the full sweep
and both scoring conventions.

---

## Why this project exists

An earlier version of this work trained on exactly **one reference genome per
class** and reported ~99% accuracy. That number was real but meaningless:
"held-out" test reads differed from training reads only in the read simulator's
noise, not in real biological diversity. That is closer to memorizing a fixed
set of k-mer fingerprints than to learning anything that generalizes.

Everything here exists to fix that: a multi-isolate NCBI downloader,
isolate-level fold assignment, and genome-level cross-validation. **Any change
to the data pipeline that risks reintroducing train/test leakage at the genome
level is the single most important thing to get right in this codebase.**

---

## Repository layout

```
src/                       Model, training, evaluation and experiment drivers

  Library
    BNN_model.py           BinaryMLP (binarized, straight-through estimator)
                           + StandardMLP (float32 twin, identical interface)
    moe_model.py           build_moe(): router + 8 per-family experts
    moe_dataset.py         Manifest loading, k-mer vectorization, index remapping
    kmer_encoding.py       Bag-of-k-mers: extract, build vocabulary, vectorize
    training_utils.py      Shared training loop (train_epoch)

  Training and evaluation
    train_moe.py           Stage-wise hierarchical training (router, then experts)
    evaluate_moe.py        Per-read and per-specimen majority-vote evaluation
    train_flat.py          Flat 58-way baseline
    evaluate_flat.py         (no hierarchy; the control the MoE must beat)
    train_final_model.py   The deployable model: all isolates, no holdout

  Experiments
    run_cv.py              Genome-level K-fold CV orchestrator
    aggregate_cv_results.py  Combines per-fold JSON into mean ± std
    sweep_router.py        Router size/depth sweep (experts restored, ~9x cheaper)
    sweep_experts.py       Expert size/depth sweep
    sweep_architecture.py  Joint router/expert hidden-size grid
    sweep_cross_combos.py  Checks router x expert sizing has no interaction
    sweep_read_counts.py   How many reads a confident specimen call needs
    sweep_error_rates.py   Robustness to injected substitution error
    run_encoding_experiment.py   One input-encoding variant (k, canonical, width)
    compare_kraken2.py     Kraken2 baseline: build database, run matched sweep
    plot_results.py        Figures (colourblind-safe palette)

data/                      Dataset construction and the manifests
  download_genomes.py      NCBI multi-isolate downloader (+ fold assignment)
  simulate_reads.py        ART Illumina read simulator
  manifests/               The curated manifests, see data/README.md

results/                   Every experiment's output, see results/README.md

scripts/                   Cluster jobs and utilities, see scripts/README.md
  slurm/                   The job scripts that produced these results
  make_genome_inventory.py Regenerates the genome provenance record

docs/
  ARCHITECTURE.md          Network design: router, experts, binarization
  PIPELINE.md              Data flow: download -> simulate -> folds -> train
  EXPERIMENTS.md           Why each experiment is designed the way it is
  RESULTS.md               Full results, per-fold tables, honest limitations
  GENOME_INVENTORY.txt     Every accession used, by class and fold

tests/test_smoke.py        Repository integrity checks (see below)
```

**Running anything in `src/`**: invoke it by path from the repository root, e.g.
`python3 src/run_cv.py`. Python puts the script's own directory on `sys.path`,
so the modules find each other, while the working directory stays the repository
root so that relative paths like `data/manifests/...` resolve as documented.

## Tests

`tests/test_smoke.py` does not train anything. It checks the things that rot
silently when code moves or results are regenerated by hand: every module still
imports, every result CSV parses with no blank metric cells, the manifests still
satisfy the genome-level no-leakage invariant, the dataset is the size the docs
claim, and nothing references a file that has been removed.

```bash
# Full suite (the import check needs torch, so use the container)
apptainer exec patbinn.sif python3 -m pytest tests/ -q

# Or without pytest / without torch
python3 tests/test_smoke.py
```

The same checks run in CI on every push (`.github/workflows/smoke.yml`).

---

## Quickstart

Requires [Apptainer](https://apptainer.org/) and an NVIDIA GPU. The container
carries PyTorch, CUDA and the ART read simulator, so nothing else needs
installing.

```bash
# 0. Build the container (~15 min)
apptainer build --fakeroot patbinn.sif patbinn.def

# 1. Download real reference genomes (58 classes, up to 10 isolates each)
python3 data/download_genomes.py \
  --manifest data/manifests/generalization_manifest.csv \
  --out_dir data/refs \
  --genome_manifest_out data/manifests/genome_manifest_multi.csv \
  --num_isolates 10 --k_folds 5

# 2. Simulate Illumina reads from those genomes
apptainer exec --bind $(pwd):/workspace --pwd /workspace patbinn.sif \
  python3 data/simulate_reads.py \
  --genome_manifest data/manifests/genome_manifest_multi.csv \
  --reads_dir data/reads_multi \
  --sample_manifest_out data/manifests/cv_sample_manifest.csv \
  --num_replicates 6 --read_length 150 --coverage 30

# 3. Genome-level 5-fold cross-validation (the main result)
apptainer exec --nv --bind $(pwd):/workspace --pwd /workspace patbinn.sif \
  python3 src/run_cv.py \
  --cv_sample_manifest data/manifests/cv_sample_manifest.csv --k_folds 5

# 4. Aggregate and plot
python3 src/aggregate_cv_results.py \
  --results_dir results/cv_hierarchical_binary --out_csv results/cv_results.csv
apptainer exec --bind $(pwd):/workspace --pwd /workspace patbinn.sif \
  python3 src/plot_results.py

# 5. Train the deployable model (all isolates, no held-out test set)
apptainer exec --nv --bind $(pwd):/workspace --pwd /workspace patbinn.sif \
  python3 src/train_final_model.py \
  --cv_sample_manifest data/manifests/cv_sample_manifest.csv \
  --val_fold 4 --router_hidden_sizes 1024 128 --expert_hidden_sizes 1024 128 \
  --save_model checkpoints/final/moe_final_1024x128.pth
```

To reproduce the baselines and sweeps, set `CLASSIFIER=flat` or
`PRECISION=float32` on step 3, and see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
for each sweep's exact invocation. On a Slurm cluster, submit the equivalent
job chain from [scripts/slurm/](scripts/slurm/) instead. See
[scripts/README.md](scripts/README.md).

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{harary2026patbinn,
  title={PatBiNN: A 65 nm Processing-in-CAM Based BNN Implementation for Pathogen Genome Classification},
  author={Harary, Yuval and Sharoni, Almog and Garz{\'o}n, Esteban and Yavits, Leonid},
  booktitle={2026 Design, Automation \& Test in Europe Conference (DATE)},
  pages={1--7},
  year={2026},
  organization={IEEE}
}
```

## License

Released under the MIT License, see [LICENSE](LICENSE).
