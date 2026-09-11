# Results

Every number here is read directly from the CSV/JSON files in
[`../results/`](../results/). Nothing is rounded from memory; the file backing
each table is named above it.

Unless stated otherwise:

- **Per-read** accuracy is over individual 150 bp reads.
- **Specimen-level** accuracy is the per-genome majority vote over that
  specimen's reads: the operational number, since you classify a sample rather
  than a read.
- All cross-validation is **genome-level**: whole isolates are withheld, never
  merely some of their reads.

---

## 1. Main cross-validation result

*Source: `results/cv_results.csv`, `results/cv_results_float32.csv`,
`results/cv_results_flat.csv`*

5-fold mean ± standard deviation:

| Condition | Per-read, family (router) | Per-read, variant (oracle-routed) | Specimen-level |
|---|---|---|---|
| **Hierarchical MoE, binary** | **97.76% ± 0.69** | **91.81% ± 1.39** | **99.55% ± 1.02** |
| Hierarchical MoE, float32 | 99.57% ± 0.22 | 98.76% ± 0.55 | 99.73% ± 0.53 |
| Flat 58-way, binary | n/a | 43.53% ± 3.16 *(leaf)* | 88.78% ± 4.15 |

**The oracle-routed column is not comparable to the flat classifier's leaf
accuracy.** Oracle routing hands the expert the true family and therefore
measures an easier task; it is reported to separate expert quality from router
quality, not to flatter the hierarchy. The specimen-level column is measured
identically across all three rows and is the fair comparison.

Per-fold specimen-level accuracy, hierarchical binary:

| Fold | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Test specimens | 678 | 660 | 654 | 636 | 630 |
| Specimen accuracy | 100.00% | 97.73% | 100.00% | 100.00% | 100.00% |
| Family accuracy | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

Family routing is perfect at the specimen level in every fold: all residual
error is within-family variant confusion, concentrated in fold 1.

### Three findings

1. **The hierarchy is doing real work.** 99.55% vs 88.78% specimen-level, and
   the per-read gap is far wider. A flat 58-way softmax must separate a
   Flaviviridae variant from a Coronaviridae variant using the same weights
   that separate two Coronaviridae variants from each other; splitting those
   two very different similarity scales is what buys the improvement.
2. **Binarization is close to free.** 0.18 points at the specimen level
   (99.55% vs 99.73%), inside fold-to-fold noise, even though every weight
   and activation is constrained to ±1. Per read the binary penalty is real and
   visible (91.81% vs 98.76% oracle-routed); majority voting absorbs almost all
   of it.
3. **Majority voting is the mechanism.** The jump from ~92% per read to ~99.6%
   per specimen is the entire operational case for this design. Section 5
   quantifies how many reads that actually requires.

---

## 2. Router size and depth

*Source: `results/router_sweep.csv` (fold 1). Experts are restored from an
existing checkpoint, so expert quality is held fixed and only the router
varies.*

| Router shape | Layers | Parameters | Per-read family | Specimen-level |
|---|---|---|---|---|
| 64 | 1 | 66,305 | 70.01% | 80.30% |
| 128 | 1 | 132,609 | 64.18% | 92.58% |
| 256 | 1 | 265,217 | 67.41% | 95.91% |
| 512 | 1 | 530,433 | 84.64% | 92.58% |
| 128×64 | 2 | 140,546 | 75.55% | 97.73% |
| 256×128 | 2 | 297,474 | 82.21% | 97.73% |
| **1024×128** | 2 | 1,185,282 | **97.13%** | **97.73%** |
| 1024×512×128 | 3 | 1,646,083 | 97.99% | 97.73% |

**The second hidden layer is a requirement, not a refinement.** No
single-hidden-layer shape reaches any two-layer shape at the specimen level,
and the best one-layer router (512, at 530k parameters) is beaten by the
smallest two-layer router (128×64, at 141k), a quarter of the parameters.

**Beyond two layers buys nothing.** A third hidden layer adds 0.86 points per
read and exactly zero at the specimen level, for 460k additional parameters.

**Specimen-level accuracy saturates long before per-read accuracy does.**
Every two-layer router lands on the identical 97.73%, while per-read accuracy
still climbs from 75.55% to 97.99% across that range. Majority voting absorbs
the entire 22-point per-read deficit, so a router chosen on per-read accuracy
alone would be sized far larger than the task actually needs.

---

## 3. Expert size and depth

*Source: `results/expert_sweep.csv` (fold 1). The router is held fixed, so only
expert capacity varies.*

| Expert shape | Layers | Parameters (all 8) | Per-read variant (oracle) | Specimen-level |
|---|---|---|---|---|
| 128 | 1 | 1,060,104 | 62.83% | 95.76% |
| 256 | 1 | 2,120,200 | 73.05% | 96.97% |
| 512 | 1 | 4,240,392 | 69.58% | 95.45% |
| 1024 | 1 | 8,480,776 | 75.48% | 96.82% |
| 256×128 | 2 | 2,379,024 | 82.51% | 97.12% |
| 512×128 | 2 | 4,746,512 | 85.31% | 98.64% |
| **1024×128** | 2 | 9,481,488 | 89.38% | **98.94%** |
| 1024×256 | 2 | 10,541,584 | 91.43% | 98.18% |

Two layers beat one on the expert side as well, at every width. `1024×128` is
the best specimen-level shape; `1024×256` buys 2 points per read but *loses*
0.76 points per specimen for a further million parameters, so the extra
capacity is not converting into better specimen calls.

`512×128` is the interesting economy point: half the parameters of `1024×128`
for 0.30 points less at the specimen level.

---

## 4. Input encoding

*Source: `results/encoding_variants.csv` (fold 1). Each row is a full
retrain (router and experts) at that encoding.*

| Encoding | Vocabulary | Total parameters | Per-read family | Per-read variant (oracle) | Specimen-level |
|---|---|---|---|---|---|
| k=5, 1024 (baseline) | 1024 | 10,666,770 | 97.37% | 93.97% | 97.73% |
| k=6, 1024 | 1024 | 10,666,770 | 96.93% | 76.08% | 98.48% |
| **k=5 canonical, 512** | **512** | **5,948,178** | **97.97%** | 91.88% | **98.33%** |
| k=4, 256 | 256 | 3,588,882 | 75.47% | 74.36% | 94.55% |

**Canonical k=5 is the result worth acting on.** Collapsing every k-mer with
its reverse complement halves the vector to 512 dimensions, which removes
**44%** of the classifier's parameters (10.67M → 5.95M) while *improving*
specimen-level accuracy (97.73% → 98.33%) and per-read family accuracy
(97.37% → 97.97%).

Why it works: reads come off both strands (measured 211 forward / 177 reverse
in 400 SARS-CoV-2 reads), and the plain encoding is verifiably **not**
strand-invariant: the same locus read from opposite strands produces two
disjoint k-mer sets, and the network has to learn that equivalence from data
instead of getting it for free. Odd `k` admits no self-complementary k-mer, so
1024 5-mers map onto exactly 512 with no collisions.

**k=6 is not an improvement**: the same parameter count, worse per-read variant
accuracy (76.08% vs 93.97%), because a 1024-word vocabulary covers a far
smaller fraction of the 4096 possible 6-mers. **k=4 is too coarse**: 256
dimensions cannot separate 58 classes, and everything degrades together.

---

## 5. How many reads does a confident call need?

*Source: `results/read_count_sweep.csv` (fold 1). Pure inference; the reads
contributing to each specimen's majority vote are capped at N.*

| Reads voted | 1 | 5 | 10 | 25 | 50 | 100 | 200 | 500 | 1000 |
|---|---|---|---|---|---|---|---|---|---|
| Specimen accuracy | 89.55% | 94.70% | 96.67% | 97.12% | 97.88% | 97.58% | 97.58% | 97.73% | 97.58% |

**The specimen decision saturates at roughly 50 reads**, at ~97.6-97.9%. A
specimen at 30× coverage supplies thousands of reads, so well over 99% of the
available reads are surplus to the decision, which is the entire argument for
deploying this on a small, power-constrained device.

Accuracy is not perfectly monotonic past saturation (97.88% at 50 reads,
97.58% at 100-1000). That is a two-specimen difference out of 660, i.e. ±0.3
points of vote-order noise, not a real decline.

---

## 6. Robustness to sequencing error

*Source: `results/error_rate_sweep.csv` (fold 4, 630 specimens). Substitution
errors are injected into held-out reads at inference time; no retraining.*

| Injected error | 0% | 2% | 4% | 6% | 8% | 10% |
|---|---|---|---|---|---|---|
| Per-read leaf | 94.62% | 86.90% | 74.03% | 59.37% | 46.40% | 35.33% |
| Specimen-level | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 97.46% |

Per-read accuracy collapses roughly linearly with substitution rate, which is expected:
a single substitution corrupts up to `k` k-mers at once, and the
bag-of-k-mers vector degrades globally rather than losing one feature.

**Majority voting is remarkably effective at hiding this.** Specimen-level
accuracy is perfect through 8% injected error, where barely half of individual
reads are still classified correctly, and is still 97.46% at 10%, a
substitution rate far above any real Illumina run.

---

## 7. Comparison with Kraken2

*Source: `results/kraken2_error_sweep.csv`,
`results/kraken2_sample_acc_normalized.csv`,
`results/patbinn_fold1_error_sweep.csv`,
`results/canonical_fold1_error_sweep.csv`; all fold 1, 660 specimens.*

Kraken2's database is built from exactly the genomes PatBiNN was allowed to
train on (every genome outside the held-out test fold), so neither method has
seen a test isolate. Both are scored on byte-identical reads, corrupted by the
same function with the same seed.

| Error | Kraken2 per-read | PatBiNN per-read | Kraken2 unclassified | Kraken2 lenient | Kraken2 fair | PatBiNN specimen | Canonical specimen |
|---|---|---|---|---|---|---|---|
| 0% | 87.09% | **89.74%** | 8.12% | 96.67% | 90.30% | **97.58%** | 98.18% |
| 2% | **84.09%** | 80.86% | 10.46% | 97.42% | 90.76% | 97.58% | 97.88% |
| 4% | **78.62%** | 67.66% | 15.11% | 96.67% | 87.58% | 97.12% | 97.42% |
| 6% | **68.08%** | 54.04% | 25.39% | 96.52% | 80.76% | 96.82% | 97.12% |
| 8% | **53.22%** | 42.04% | 40.84% | 94.70% | 72.27% | 94.70% | 95.00% |
| 10% | **37.40%** | 32.49% | 57.96% | 94.24% | 32.58% | 93.94% | 76.06% |

Two scoring conventions are reported for Kraken2 because it can decline to
classify a read:

- **Lenient**: Kraken2's own convention: the specimen vote runs only over the
  reads it chose to classify. Unclassified reads are ignored.
- **Fair**: unclassified reads are counted as errors, the same standard
  PatBiNN is held to, since PatBiNN has no abstention mechanism and must answer
  for every read.

### This is a parity claim, and should not be upgraded to a win

- On **clean reads** the two are close, with PatBiNN slightly ahead per read
  (89.74% vs 87.09%) and at the specimen level (97.58% vs 96.67% lenient).
- **Under noise Kraken2 is clearly more robust per read**: 74.09% vs 60.76% at
  5% error, and the gap widens from 2% onward. Kraken2 needs a single surviving
  exact k-mer match; a bag-of-k-mers vector degrades globally. This is a real
  and material advantage for Kraken2.
- Kraken2's failure mode is **abstention** (8% → 58% unclassified), not
  misclassification. Under its own lenient convention it therefore looks nearly
  flat across the whole sweep; under the fair convention it falls off a cliff
  at 10%. Neither view alone is honest on its own, which is why both are
  reported.

The supportable claim is **comparable accuracy from a far smaller, wholly
binary computation** (a network of ~6-10M binary weights against a multi-GB
k-mer database), not superior accuracy.

Note the canonical 512-bit encoding tracks or beats the 1024-bit model
everywhere up to 8% error, then **collapses at 10%** (76.06%): halving the
activation vector leaves too little redundancy to absorb that much corruption.

---

## 8. Limitations

Stated plainly, because they bound what these numbers mean.

**Reads are simulated, not sequenced.** Every read comes from ART's Illumina
error model applied to a real NCBI genome. Real runs carry adapter
contamination, PCR duplicates, coverage bias and host background that ART does
not reproduce. The error-rate sweep in section 6 is a stress test for
substitutions only; it is not a substitute for real sequencing data.

**Class balance is uneven by nature.** Families range from 3 to 8 variants and
from 21 to 80 genomes (Retroviridae is the smallest at 3 variants / 21
genomes). Specimen-level accuracy is dominated by the larger families.

**Closed-world assumption.** The classifier always returns one of 58 classes.
There is no "none of the above" output, so a read from an unrepresented
organism is silently forced into the nearest class. Kraken2, by contrast, can
abstain, which is exactly why section 7 reports both scoring conventions. An
abstention mechanism is the most obvious next addition.

**Single-fold sweeps.** Sections 2, 3, 4, 5 and 7 are measured on fold 1
only, not across all five folds, because each sweep point is a full retrain.
They are reliable for ranking shapes and encodings against each other; their
absolute values carry single-fold variance and should not be quoted as
five-fold results.

**Specimen-level accuracy is a majority vote over many reads.** It is the right
operational metric, but it is not comparable to a per-read number from another
tool. Wherever an external classifier is compared, per-read and specimen-level
figures are reported side by side and the oracle-routed figure is excluded.
