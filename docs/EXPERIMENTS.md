# Experiment design

## The problem with the original result

The first version of this project trained a binary classifier over 64 leaf
classes (8 families × 8 variants) using exactly **one reference genome per
class**, and reported ~99% test accuracy.

That number is real but does not show what it appears to show. With one genome
per class, "test" reads and "train" reads come from the *same* genome, differing
only in which reads ART happened to simulate and what noise it added. The model
need only memorize ~64 k-mer fingerprints; it never has to learn which sequence
features generalize across independent viral isolates.

This experiment exists to answer the question that result could not: **does the
hierarchical design generalize to real, unseen viral diversity?** That requires
genuinely independent isolates per class and a held-out-*genome* protocol.

## Class selection: 58 kept, 6 dropped

Real isolate availability was checked on NCBI per `(family, variant)` before
committing. 58 of the original 64 classes have ample real diversity and are kept
in `data/manifests/generalization_manifest.csv`. **Six are dropped** for
insufficient discoverable independent complete genomes:

- `Bat-CoV-RaTG13`: one isolate exists on NCBI at all
- `HIV-1` subtypes A, D, F, G, H: 0-1 complete-genome-tagged records each.
  HIV subtype assignment mostly lives in curated databases such as LANL rather
  than in NCBI deflines, so this is a conservative lower bound rather than proof
  that no more exist.

This leaves Coronaviridae with 7 variants and Retroviridae with 3 (subtype B,
subtype C, HIV-2), so **families have non-uniform variant counts**, which
surfaced a real bug, described below.

## Why K = 5 folds

Fold count is bounded by the scarcest kept class, not chosen by default. Under
round-robin assignment (`fold = isolate_idx % K`), a class with fewer than `K`
isolates cannot appear in every fold, leaving a silent zero-support hole that
accuracy averaging will not catch.

At K = 5, **55 of 58 classes have at least 5 isolates** and are represented in
every fold. The three exceptions are Ebola-TaiForest (3), HBV-genotype-F (3) and
HIV-1-subtype-C (4), which appear in 3-4 of the 5 folds. A larger K would widen
that gap for no benefit; a smaller K would leave less data to train on. K = 5 is
the best available trade-off, and `k_folds=2` is rejected outright because test
and validation would consume both folds.

Each fold splits at genome level: fold `i` → test, fold `(i+1) % K` →
validation, the rest → train. See [PIPELINE.md](PIPELINE.md) for the
leakage-prevention mechanics.

## How much real data backs each class

*Source: `data/manifests/genome_manifest_multi.csv`; full accession list in
[GENOME_INVENTORY.txt](GENOME_INVENTORY.txt).*

| Family | Variants | Genomes | Isolates/class (min-max) | Scarcest variant |
|---|---|---|---|---|
| Coronaviridae | 7 | 70 | 10-10 | n/a |
| Flaviviridae | 8 | 80 | 10-10 | n/a |
| Picornaviridae | 8 | 80 | 10-10 | n/a |
| Paramyxoviridae | 8 | 80 | 10-10 | n/a |
| Herpesviridae | 8 | 76 | 6-10 | HHV-6B (6) |
| Hepadnaviridae | 8 | 71 | 3-10 | HBV-genotype-F (3) |
| Filoviridae | 8 | 65 | 3-10 | Ebola-TaiForest (3) |
| Retroviridae | 3 | 21 | 4-10 | HIV-1-subtype-C (4) |
| **Total** | **58** | **543** | **3-10** (mean 9.4) | |

**Every class is backed by at least 3 independent isolates**, so every class
receives a genuine held-out-genome test. This was not always true, see below.

### A resolved limitation, recorded because it shaped earlier results

An earlier version of this dataset had **12 of 58 classes represented by exactly
one isolate**, most severely 7 of Herpesviridae's 8 variants. Because
round-robin assignment always places `isolate 0` in fold 0, those classes were
permanently locked into fold 0: in the single run where fold 0 was the test
fold they had *zero* training exposure, and in every other run they were never
tested. Fold 0 was consequently the outlier in every result table, with expert
oracle accuracy around 50% against 93-98% elsewhere.

That was never a pipeline bug: it was a data-availability limit, and it was
fixed at the source by repairing the two retrieval bugs described in
[PIPELINE.md](PIPELINE.md). With 543 genomes and a minimum of 3 isolates per
class, **the effect is gone**: fold 0 now scores 100% at the specimen level,
and the spread across folds is ±1.02 points rather than ±6.7.

Any documentation, figure or analysis predating that fix and referring to a
"fold-0 problem" is describing the old dataset, not this one.

## The bug this scale-up surfaced

`src/train_moe.py` originally computed one global `num_variants` for every expert,
while `src/evaluate_moe.py` already derived the count per family. The two agreed
only because every family originally had exactly 8 variants. The moment
Retroviridae dropped to 3, this became a checkpoint shape mismatch at evaluation
time.

Fixed by densifying both `family_idx` and `variant_idx` from the full kept-class
row set (`remap_variant_idx` in `src/moe_dataset.py`) **before any fold-filtering**,
and by changing `build_moe()` to take a per-family variant count list rather
than a scalar.

## Conditions compared

| Axis | Values | Question it answers |
|---|---|---|
| Architecture | hierarchical MoE vs flat 58-way | Does decomposing into family-then-variant help? |
| Precision | binary (BNN) vs float32 | Does the hardware-deployable network cost accuracy? |
| Router size/depth | 8 shapes, 1-3 hidden layers | What is the smallest router that still works? |
| Expert size/depth | 8 shapes, 1-2 hidden layers | Same question for the fine-grained stage |
| Router × expert interaction | 2 opposite-corner combos | Does the independent-sweep assumption hold? |
| Input encoding | k ∈ {4,5,6}, canonical on/off | Can the input vector be made smaller for free? |
| Reads voted per specimen | 1 to 1000, post-hoc | How many reads does a confident call need? |
| Injected substitution error | 0-10% | How gracefully does it degrade? |

The first two axes run under **full 5-fold genome-level cross-validation** and
are reported as mean ± std across folds. The remaining sweeps run on **fold 1**
as a single representative split, because each point is a full retrain; fold 1
is used rather than fold 0 for historical reasons (fold 0 was compromised in the
earlier dataset) and `src/sweep_router.py` and `src/run_encoding_experiment.py` still
refuse fold 0 explicitly.

Single-fold sweeps are reliable for *ranking* shapes and encodings against one
another. Their absolute values carry single-fold variance and should not be
quoted as five-fold results.

### Why the router sweep is exact rather than approximate

Experts are trained on the reads of their own family, selected by the **true**
label, never on router output. The router is frozen during expert training only
so expert gradients cannot disturb it. An expert is therefore valid under any
router, and swapping routers requires no expert retraining.

That is what lets `src/sweep_router.py` restore experts from an existing checkpoint
and retrain only the router, roughly 9× cheaper than a full pipeline, with no
methodological compromise. The control: oracle-routed expert accuracy is
identical to the last decimal across all 8 rows of `results/router_sweep.csv`.

Two traps in the partial-retraining path are already handled and should not be
undone: saved shape metadata must describe the **restored** stage rather than
the CLI flag (otherwise `load_checkpoint` rebuilds the wrong shape and the
checkpoint will not load), and retraining a subset of experts at a width
differing from the restored ones raises an error rather than silently
mis-recording, because the single-tuple checkpoint schema cannot express it.

## Status

All planned experiments are complete. Outputs live in
[`../results/`](../results/), see [results/README.md](../results/README.md) for
what each file contains.

| Experiment | Folds | Output |
|---|---|---|
| Hierarchical MoE, binary | 5/5 | `cv_hierarchical_binary/`, `cv_results.csv` |
| Hierarchical MoE, float32 | 5/5 | `cv_hierarchical_float32/`, `cv_results_float32.csv` |
| Flat 58-way baseline, binary | 5/5 | `cv_flat_binary/`, `cv_results_flat.csv` |
| Router size/depth sweep | fold 1 | `router_sweep.csv` |
| Expert size/depth sweep | fold 1 | `expert_sweep.csv` |
| Input-encoding variants | fold 1 | `encoding_variants.csv` |
| Read-count-to-confidence sweep | all 5 | `read_count_sweep.csv` |
| Error-rate sweep | fold 4 | `error_rate_sweep.csv` |
| Error-rate sweep, fold-1 (for Kraken2) | fold 1 | `patbinn_fold1_error_sweep.csv` |
| Error-rate sweep, canonical encoding | fold 1 | `canonical_fold1_error_sweep.csv` |
| Kraken2 baseline | fold 1 | `kraken2_error_sweep.csv`, `kraken2_sample_acc_normalized.csv` |
| Final deployable model (543 isolates, no holdout) | n/a | `checkpoints/final/moe_final_1024x128.pth` |

Remaining open directions, in rough order of value:

- **An abstention mechanism.** The classifier is closed-world and must answer
  for every read; Kraken2's ability to decline is a genuine advantage under
  noise. This is the most obvious gap.
- **Alternative preprocessing**: spaced seeds or minimizers. Both are better
  motivated than a BWT, which is a reversible permutation whose value is
  substring search over a reference; taking k-mers of its output destroys the
  run structure that makes it useful.
- **Five-fold backing for the single-fold sweeps**, if any of those numbers are
  to be quoted as headline results.
- **Validation on real sequencing runs** rather than ART-simulated reads.
