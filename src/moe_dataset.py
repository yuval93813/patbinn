#!/usr/bin/env python3
"""
Dataset utilities for the hierarchical MoE virus classifier: loads simulated Illumina
reads (per data/simulate_reads.py's sample_manifest.csv), builds a shared k-mer
vocabulary, and exposes them as PyTorch Datasets for the router (family label) and
per-family experts (variant label), plus a full-metadata mode used by evaluate_moe.py
for per-sample majority voting.
"""

import csv
import json
import multiprocessing as mp
import os

import torch
from torch.utils.data import Dataset

from kmer_encoding import parse_fastq, build_kmer_vocabulary_by_document_frequency, encode_sequence_kmers, extract_kmers


def available_cpu_count():
    """CPUs actually usable by this process (respects Slurm/cgroup cpuset restrictions,
    unlike a plain cpu_count() which can report the whole node's core count)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# Module-level so worker processes (spawned/forked by multiprocessing.Pool) can see
# them without re-pickling the (possibly large) kmer_to_idx dict on every task.
_worker_kmer_to_idx = None
_worker_k = None
_worker_canonical = False


def _init_encode_worker(kmer_to_idx, k, canonical=False):
    global _worker_kmer_to_idx, _worker_k, _worker_canonical
    _worker_kmer_to_idx = kmer_to_idx
    _worker_k = k
    _worker_canonical = canonical


def _encode_worker(seq):
    """Returns the sparse list of hit vocab indices for one read (cheaper to ship
    across a process boundary than a mostly-zero 1000+-dim dense vector)."""
    hits = set()
    for kmer in extract_kmers(seq, _worker_k, canonical=_worker_canonical):
        idx = _worker_kmer_to_idx.get(kmer)
        if idx is not None:
            hits.add(idx)
    return list(hits)


def encode_sequences_parallel(sequences, k, kmer_to_idx, vocab_size, num_workers=None,
                               canonical=False):
    """
    Parallel counterpart to calling encode_sequence_kmers once per sequence in a
    Python loop - that loop is the dominant cost of building a FastqKmerDataset at
    real scale (millions of reads), and doesn't parallelize on its own since it runs
    in Dataset.__init__, before any DataLoader workers exist. Returns a
    (len(sequences), vocab_size) float32 tensor.
    """
    vectors = torch.zeros((len(sequences), vocab_size), dtype=torch.float32)
    if not sequences:
        return vectors

    if num_workers is None:
        num_workers = min(32, available_cpu_count())

    if num_workers <= 1 or len(sequences) < 10000:
        for i, seq in enumerate(sequences):
            vectors[i] = encode_sequence_kmers(seq, k, kmer_to_idx, vocab_size, canonical=canonical)
        return vectors

    # Explicit 'spawn' context: by the time this runs, the caller has typically
    # already done model.to('cuda'), initializing a CUDA context in this process.
    # Forking after that (the default start method on Linux) inherits a broken
    # duplicate of that context in the child and is a well-documented hang/crash
    # hazard - spawn starts clean interpreters instead, sidestepping it entirely.
    ctx = mp.get_context("spawn")
    with ctx.Pool(num_workers, initializer=_init_encode_worker,
                  initargs=(kmer_to_idx, k, canonical)) as pool:
        for i, hits in enumerate(pool.imap(_encode_worker, sequences, chunksize=512)):
            if hits:
                vectors[i, hits] = 1.0

    return vectors


def load_sample_manifest(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["family_idx"] = int(row["family_idx"])
        row["variant_idx"] = int(row["variant_idx"])
        row["replicate"] = int(row["replicate"])
    return rows


def remap_variant_idx(rows):
    """
    Rewrite family_idx to a dense 0..(n_families-1) range, and each (new) family's
    variant_idx values to a dense 0..(n_kept-1) range.

    Needed once any classes/families are dropped from the original 8x8 taxonomy
    (e.g. rare-isolate classes excluded from the generalization benchmark, or a
    manifest subset to only some families for a pilot run): the surviving indices
    are then a sparse subset of the original slots (e.g. family_idx={1} only, or
    variant_idx={1,2,7} within a family), which still "works" with cross-entropy/
    argmax but silently assumes dense 0..N-1 indices wherever a single global
    num_families/num_variants or a `range(num_families)` loop is used - a family_idx
    subset like {1} alone would otherwise leave a phantom family_idx=0 slot with no
    data. Mapping is computed from ALL rows passed in (not a fold-filtered subset)
    so it stays stable across CV folds - always call this once, immediately after
    load_sample_manifest, before splitting into folds/train/val/test.

    Mutates and returns `rows` (adds/overwrites family_idx/variant_idx in place).
    """
    family_remap = {old: new for new, old in enumerate(sorted({row["family_idx"] for row in rows}))}
    for row in rows:
        row["family_idx"] = family_remap[row["family_idx"]]

    variants_by_family = {}
    for row in rows:
        variants_by_family.setdefault(row["family_idx"], set()).add(row["variant_idx"])

    variant_remap = {
        family_idx: {old: new for new, old in enumerate(sorted(variants))}
        for family_idx, variants in variants_by_family.items()
    }

    for row in rows:
        row["variant_idx"] = variant_remap[row["family_idx"]][row["variant_idx"]]

    return rows


def _load_reads_for_sample(row):
    return parse_fastq(row["r1_path"]) + parse_fastq(row["r2_path"])


def build_global_kmer_vocab(train_rows, k, max_kmers, max_reads_per_sample=None, cache_path=None,
                             canonical=False):
    """
    Build (and optionally cache) a shared k-mer vocabulary using genome/sample-level
    document frequency, over the given (already train-split-filtered) rows, so the
    vocabulary itself never sees held-out data. Callers are responsible for
    filtering to their train split before calling this (a single global split=="train"
    filter doesn't generalize to CV, where "train" differs per fold).
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if (cached["k"] == k and cached["max_kmers"] == max_kmers
                and cached.get("canonical", False) == canonical):
            print(f"Loaded cached k-mer vocabulary from {cache_path} ({len(cached['kmer_to_idx'])} k-mers)")
            return cached["kmer_to_idx"], len(cached["kmer_to_idx"])

    def sequence_groups():
        for row in train_rows:
            seqs = _load_reads_for_sample(row)
            if max_reads_per_sample is not None:
                seqs = seqs[:max_reads_per_sample]
            yield seqs

    kmer_to_idx, vocab_size = build_kmer_vocabulary_by_document_frequency(
        sequence_groups(), k, max_kmers, canonical=canonical)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"k": k, "max_kmers": max_kmers, "canonical": canonical,
                        "kmer_to_idx": kmer_to_idx}, f)
        print(f"Cached k-mer vocabulary to {cache_path}")

    return kmer_to_idx, vocab_size


def build_leaf_index(rows):
    """
    Stable global (family_idx, variant_idx) -> leaf_idx map, sorted by
    (family_idx, variant_idx). Deliberately NOT a family_idx*num_variants+variant_idx
    formula - that breaks the moment variant counts are non-uniform across families
    (exactly what happens once rare classes are dropped from a family). Compute this
    once from the full kept-class manifest (not a fold-filtered subset) so leaf
    indices stay stable across folds/scripts even when a fold's rows don't cover
    every leaf.
    """
    keys = sorted({(row["family_idx"], row["variant_idx"]) for row in rows})
    return {key: i for i, key in enumerate(keys)}


class FastqKmerDataset(Dataset):
    """
    Flattens a set of sample_manifest rows into individual reads, each encoded as a
    k-mer presence/absence vector over a shared vocabulary.

    label_mode:
      - 'family':  __getitem__ returns (vector, family_label)          -- router training
      - 'variant': __getitem__ returns (vector, variant_label)         -- expert training
                   (sample_rows must already be filtered to one family)
      - 'leaf':    __getitem__ returns (vector, leaf_label)            -- flat baseline
                   training (requires `leaf_index` from build_leaf_index)
      - 'full':    __getitem__ returns (vector, family_label, variant_label, sample_id)
                   -- used for evaluation / per-sample majority voting
    """

    def __init__(self, sample_rows, kmer_to_idx, vocab_size, k, label_mode="family", leaf_index=None,
                 max_reads_per_sample=None, canonical=False):
        assert label_mode in ("family", "variant", "leaf", "full")
        if label_mode == "leaf":
            assert leaf_index is not None, "label_mode='leaf' requires leaf_index=build_leaf_index(...)"
        self.vocab_size = vocab_size
        self.label_mode = label_mode

        sequences = []
        family_labels = []
        variant_labels = []
        leaf_labels = []
        self.sample_ids = []

        for row in sample_rows:
            seqs = _load_reads_for_sample(row)
            # Cap reads per sample - without this, a handful of huge-genome samples
            # (e.g. Herpesviridae at ~200kb vs Hepadnaviridae at ~3kb, both simulated
            # at the same coverage) can dominate the pooled dataset by orders of
            # magnitude, both blowing up memory (every read's k-mer vector is
            # precomputed and held in RAM below) and skewing training toward
            # whichever class happened to have the biggest genome.
            if max_reads_per_sample is not None:
                seqs = seqs[:max_reads_per_sample]
            sequences.extend(seqs)
            family_labels.extend([row["family_idx"]] * len(seqs))
            variant_labels.extend([row["variant_idx"]] * len(seqs))
            self.sample_ids.extend([row["sample_id"]] * len(seqs))
            if label_mode == "leaf":
                leaf_labels.extend([leaf_index[(row["family_idx"], row["variant_idx"])]] * len(seqs))

        # Precompute every read's k-mer vector once here, instead of re-extracting
        # k-mers from scratch on every __getitem__ call - since a training run
        # re-visits (and, under the balanced sampler, often re-visits many times
        # over) the same reads across dozens of epochs, encoding lazily redid this
        # work every single access. This turns that into a one-time O(N) cost.
        # Parallelized across CPU cores - at real scale (millions of reads) a plain
        # Python loop here dominates wall-clock time.
        self.vectors = encode_sequences_parallel(sequences, k, kmer_to_idx, vocab_size,
                                                  canonical=canonical)

        self.family_labels = torch.tensor(family_labels, dtype=torch.long)
        self.variant_labels = torch.tensor(variant_labels, dtype=torch.long)
        self.leaf_labels = torch.tensor(leaf_labels, dtype=torch.long) if label_mode == "leaf" else None

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        vector = self.vectors[idx]

        if self.label_mode == "family":
            return vector, self.family_labels[idx]
        if self.label_mode == "variant":
            return vector, self.variant_labels[idx]
        if self.label_mode == "leaf":
            return vector, self.leaf_labels[idx]
        return vector, self.family_labels[idx], self.variant_labels[idx], self.sample_ids[idx]


def make_balanced_sampler(labels):
    """Generalizes the whole-genome pipeline's binary balanced sampler to N classes."""
    from collections import Counter
    if torch.is_tensor(labels):
        labels = labels.tolist()
    counts = Counter(labels)
    weight_per_class = {c: 1.0 / n for c, n in counts.items()}
    sample_weights = torch.DoubleTensor([weight_per_class[label] for label in labels])
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
