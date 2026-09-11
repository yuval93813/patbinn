#!/usr/bin/env python3
"""
Shared k-mer encoding utilities.

Originally extracted from the whole-genome FASTA dataset so that the
whole-genome and short-read (moe_dataset.py) pipelines encode sequences
identically, via one shared implementation.
"""

# torch is imported lazily inside encode_sequence_kmers: everything else in this
# module is pure sequence handling, and keeping the module importable without
# torch lets tooling that only needs parsing or error injection (e.g. the Kraken2
# baseline, which runs in its own container) reuse these functions rather than
# duplicating them.
from collections import Counter


def parse_fasta(file_path):
    """Parse a FASTA file and return a list of sequences (uppercased, concatenated per record)."""
    sequences = []
    current_seq = ""

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(current_seq.upper())
                    current_seq = ""
            else:
                current_seq += line

        if current_seq:
            sequences.append(current_seq.upper())

    return sequences


def parse_fastq(file_path):
    """Parse a FASTQ file and return a list of sequences (uppercased). Quality scores are discarded."""
    sequences = []

    with open(file_path, 'r') as f:
        while True:
            header = f.readline()
            if not header:
                break
            seq = f.readline().strip()
            plus = f.readline()
            qual = f.readline()
            if not (seq and plus and qual):
                break
            sequences.append(seq.upper())

    return sequences


def inject_read_errors(sequence, error_rate, rng):
    """
    Simulate additional sequencing errors on top of whatever ART already introduced:
    each base is independently replaced with a different random base with probability
    `error_rate` (uniform substitution model). Used to test how the trained network's
    accuracy degrades as read quality drops below what ART's own quality profile
    produces (real Illumina error rates are typically well under 1%; this sweeps
    error_rate up to 10% to find where the network's performance actually breaks).
    """
    if error_rate <= 0:
        return sequence
    bases = "ACGT"
    out = list(sequence)
    for i, base in enumerate(out):
        if base in bases and rng.random() < error_rate:
            out[i] = rng.choice([b for b in bases if b != base])
    return "".join(out)


_COMPLEMENT = str.maketrans('ACGT', 'TGCA')


def reverse_complement(sequence):
    """Reverse complement of an ACGT string."""
    return sequence.translate(_COMPLEMENT)[::-1]


def extract_kmers(sequence, k, canonical=False):
    """Extract all K-mers of length k from a sequence (unknown nucleotides dropped).

    canonical=True emits min(kmer, reverse_complement(kmer)) instead of the k-mer as
    read. A sequencing run yields reads from both strands of the template - measured
    at roughly half and half in this dataset's simulated reads - so without
    canonicalization the same genomic locus produces two disjoint k-mer sets
    depending on the strand it happened to be read from, and the network has to
    learn that equivalence from data instead of receiving it for free. Collapsing
    each k-mer with its reverse complement also halves the size of the k-mer space
    for odd k (no odd-length k-mer can equal its own reverse complement), e.g.
    1024 -> 512 distinct 5-mers.

    Default False, so every result produced before this option existed is
    reproducible unchanged.
    """
    clean_sequence = ''.join([c for c in sequence if c in 'ACGT'])

    kmers = []
    for i in range(len(clean_sequence) - k + 1):
        kmer = clean_sequence[i:i + k]
        if len(kmer) == k:
            if canonical:
                rc = kmer.translate(_COMPLEMENT)[::-1]
                if rc < kmer:
                    kmer = rc
            kmers.append(kmer)

    return kmers


def build_kmer_vocabulary(sequences, k, max_kmers, verbose=True):
    """
    Build a K-mer vocabulary from a list of sequences using raw per-read/per-sequence
    frequency counting (matches the original FASTADataset behavior).

    Returns (kmer_to_idx, vocab_size).
    """
    kmer_counts = Counter()

    for sequence in sequences:
        kmers = extract_kmers(sequence, k)
        kmer_counts.update(kmers)

    most_common_kmers = kmer_counts.most_common(max_kmers)

    kmer_to_idx = {kmer: i for i, (kmer, count) in enumerate(most_common_kmers)}

    if verbose:
        print(f"Built K-mer vocabulary with {len(kmer_to_idx)} unique {k}-mers")
        print(f"Most common K-mers: {[kmer for kmer, count in most_common_kmers[:10]]}")

    return kmer_to_idx, len(kmer_to_idx)


def build_kmer_vocabulary_by_document_frequency(sequence_groups, k, max_kmers, verbose=True,
                                                canonical=False):
    """
    Build a K-mer vocabulary using genome-level document frequency: a k-mer's score is
    the number of distinct sequence groups (e.g. genomes) it appears in at least once,
    rather than its raw count across all reads.

    This prevents families with much larger genomes (and therefore far more simulated
    reads at fixed coverage) from monopolizing the top-max_kmers vocabulary slots.

    Args:
        sequence_groups: iterable of iterables of sequences, one inner iterable per genome.

    Returns (kmer_to_idx, vocab_size).
    """
    doc_freq = Counter()

    for sequences in sequence_groups:
        kmers_in_group = set()
        for sequence in sequences:
            kmers_in_group.update(extract_kmers(sequence, k, canonical=canonical))
        doc_freq.update(kmers_in_group)

    most_common_kmers = doc_freq.most_common(max_kmers)

    kmer_to_idx = {kmer: i for i, (kmer, count) in enumerate(most_common_kmers)}

    if verbose:
        print(f"Built K-mer vocabulary (document-frequency) with {len(kmer_to_idx)} unique {k}-mers")
        print(f"Most common K-mers: {[kmer for kmer, count in most_common_kmers[:10]]}")

    return kmer_to_idx, len(kmer_to_idx)


def encode_sequence_kmers(sequence, k, kmer_to_idx, vocab_size, canonical=False):
    """
    Encode a DNA/RNA sequence as a binary K-mer presence/absence vector over the
    given vocabulary.
    """
    kmers = extract_kmers(sequence, k, canonical=canonical)

    import torch
    kmer_vector = torch.zeros(vocab_size, dtype=torch.float32)

    unique_kmers = set(kmers)
    for kmer in unique_kmers:
        if kmer in kmer_to_idx:
            idx = kmer_to_idx[kmer]
            kmer_vector[idx] = 1.0

    return kmer_vector
