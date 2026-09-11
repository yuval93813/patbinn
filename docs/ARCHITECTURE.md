# Architecture

## The classification problem

Given a single short Illumina read (150 bp, either mate of a pair), predict
which of **58 virus variants** it came from. The 58 variants are grouped into
**8 families**: Coronaviridae, Flaviviridae, Filoviridae, Hepadnaviridae,
Retroviridae, Herpesviridae, Picornaviridae and Paramyxoviridae, each
containing 3 to 8 variants (Coronaviridae holds SARS-CoV-2, MERS-CoV,
HCoV-229E and so on).

## Two designs, compared

### Flat classifier (`src/train_flat.py`)

One network, one softmax over all 58 leaf classes. The simplest possible
baseline; it exists so the hierarchical design has something to beat.

### Hierarchical Mixture-of-Experts (`src/train_moe.py`, `src/moe_model.py`)

Two stages:

1. **Router**: one network, softmax over the 8 families. *Which family is this
   read from?*
2. **Experts**: 8 independent networks, one per family, each a softmax over
   only that family's variants. *Given the family, which variant?*

Routing is **hard**: the router's argmax selects the single expert that runs on
that read. There is no soft mixing or weighted blend of expert outputs, so a
read costs one router pass plus one expert pass regardless of how many families
exist.

**Why this helps accuracy, not just interpretability.** A flat 58-way softmax
has to draw a boundary between, say, a Flaviviridae variant and a
Coronaviridae variant using the same weights it uses to separate two
Coronaviridae variants from each other, two very different scales of k-mer
similarity entangled in one classifier. Splitting the problem lets the router
specialize on coarse family-level signal and each expert on fine within-family
signal. The measured gap is large: 99.55% vs 88.78% specimen-level
(see [RESULTS.md](RESULTS.md#1-main-cross-validation-result)).

`src/moe_model.py`'s `build_moe()` constructs the router and the list of 8 experts;
`HierarchicalMoEClassifier` wraps both and provides `freeze_router()` for
stage-wise training (see [PIPELINE.md](PIPELINE.md)).

## Binarization

Both designs are tested at two precisions:

- **binary** (`BinaryMLP` in `src/BNN_model.py`): weights and activations are
  constrained to ±1 and trained with a straight-through estimator
  (`BinarizeLinear`, `BinaryActivation`). This is the point of the project: a
  network this constrained can in principle run on in-memory-compute hardware
  (crossbar or CAM arrays) rather than a GPU. Its accuracy is the headline
  result.
- **float32** (`StandardMLP`): a drop-in twin with an identical
  constructor/forward signature but ordinary `nn.Linear` + `BatchNorm1d` +
  `ReLU`. It exists purely as a control, isolating *"does binarization cost
  accuracy"* from *"does the hierarchy help"*. Without it, the two effects
  would be confounded.

Both expose the same `forward()`/`features()` interface, so
`build_moe(..., precision="binary"|"float32")` switches between them with no
other code change.

The straight-through estimator approximates the derivative of the binarization
function by the identity inside [-1, 1] and by zero outside it, so full-precision
shadow weights are updated by ordinary backpropagation while the forward pass
stays binarized. `training_utils.train_epoch` additionally clamps non-BatchNorm
parameters to [-1.5, 1.5] after each step; this is tuned for binarized training
and is switched off for the float32 control, where it would otherwise act as an
uncontrolled extra regularizer and confound the comparison.

**Measured cost of binarization: 0.18 points** at the specimen level
(99.55% vs 99.73%). Per read the penalty is real and much larger
(91.81% vs 98.76% oracle-routed); majority voting absorbs nearly all of it.

## Input representation: bag-of-k-mers

Reads are not fed to the network as sequence. Each read becomes a fixed-length
**bag-of-k-mers** vector (`src/kmer_encoding.py`):

1. Extract every overlapping length-`k` substring of the read (`k=5` by
   default).
2. Look each k-mer up in a **shared vocabulary** of the top 1024 most
   informative k-mers, ranked by **document frequency** across training samples.
   Document frequency rather than raw count is deliberate, so that families with
   larger genomes do not dominate the vocabulary merely by containing more total
   k-mer occurrences.
3. Emit a 1024-dimensional binary occurrence vector.

That vector, not the read, is what every network sees. It is cheap to compute
and its fixed width matches the fixed-input-size assumption of the target
hardware.

### Canonical k-mers

Reads come off both strands (measured 211 forward / 177 reverse in 400
SARS-CoV-2 reads), and the plain encoding is verifiably **not**
strand-invariant: the same locus read from opposite strands yields two disjoint
k-mer sets, and the network must learn that equivalence from data.

Passing `--canonical` emits `min(kmer, revcomp(kmer))` instead. Odd `k` admits
no self-complementary k-mer, so the 1024 5-mers map onto exactly 512 with no
collisions. This **halves the input width and removes 44% of the classifier's
parameters** (10.67M → 5.95M) while *improving* specimen-level accuracy
(97.73% → 98.33%). See
[RESULTS.md](RESULTS.md#4-input-encoding).

Two properties of the implementation matter and must not be undone:

- The **vocabulary cache keys on `canonical`** as well as on `k`/`max_kmers`.
  It previously keyed only on the latter two, which would have let a canonical
  run silently load a non-canonical cache.
- **Evaluation reads `canonical` from the checkpoint**, so the encoding cannot
  diverge between training and inference. Any new evaluation path must do the
  same.

## Network shapes

Router and experts are 2-hidden-layer MLPs:
`in_features(1024) → hidden1 → hidden2 → num_classes`, where `num_classes` is 8
for the router and that family's variant count for an expert.

**Two hidden layers, not one.** No single-hidden-layer router reaches any
two-layer router at the specimen level, and the best one-layer shape (512, 530k
parameters) is beaten by the smallest two-layer shape (128×64, 141k), a
quarter of the parameters. A third layer adds 0.86 points per read and nothing
per specimen.

Router and expert sizes were swept **independently** (`src/sweep_router.py`,
`src/sweep_experts.py`), which is valid because the two do not interact
meaningfully, confirmed by `src/sweep_cross_combos.py`, which jointly trains
opposite-corner shape combinations.

The independent sweep is also exact rather than approximate, for a structural
reason: **experts are trained on the reads of their own family selected by the
true label**, never on router output (`src/train_moe.py`: `r["family_idx"] ==
family_idx`). The router is frozen during expert training only so that expert
gradients cannot disturb it. An expert is therefore valid under any router, and
swapping routers requires no expert retraining, which is what makes
`src/sweep_router.py` about 9× cheaper than full retraining with no methodological
compromise. The control for this claim: oracle-routed expert accuracy is
identical to the last decimal across all 8 rows of `results/router_sweep.csv`.

The deployed shape is **1024×128 for both router and experts**: the best
specimen-level shape in both sweeps, and the shape validated across the full
5-fold cross-validation. `512×128` experts are the economy option: half the
parameters for 0.30 points less.

## Handling inputs wider than the array

The target hardware processes a fixed number of rows at a time. A 1024-bit
activation vector is therefore split into eight 128-row segments processed
serially, with a majority computed per segment and a majority of those
per-segment majorities producing the output activation. The canonical 512-bit
encoding needs only four such segments, halving the serial depth, which, on
top of the 44% parameter reduction, is the practical argument for it.
