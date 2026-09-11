#!/usr/bin/env python3
"""
Robustness sweep: how does the final deployable model's accuracy degrade as
reads get noisier? Injects additional synthetic sequencing errors (uniform base
substitution, on top of whatever ART's own quality profile already introduced)
at rates from 0% to 10%, and re-evaluates the final checkpoint at each rate.

Uses the final model's own validation split (fold 4) as the evaluation set -
data the model never trained on (used only for early-stopping), since the final
model (unlike the CV fold checkpoints) has no held-out test split by design -
its whole point is to train on every real isolate available. This measures
robustness to read noise, a different question from the CV's generalization
claim, and reuses that CV result rather than re-establishing it.
"""

import argparse
import csv
import random

import torch

from kmer_encoding import parse_fastq, encode_sequence_kmers, inject_read_errors
from moe_dataset import remap_variant_idx
from evaluate_moe import load_checkpoint, run_model_batched, majority_vote

DEFAULT_ERROR_RATES = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]


def evaluate_at_error_rate(rows, router, experts, kmer_to_idx, vocab_size, k, device,
                            error_rate, max_reads_per_sample, seed, canonical=False):
    rng = random.Random(seed)

    read_family_true, read_family_pred = [], []
    read_leaf_correct = []
    read_variant_true, read_variant_pred_oracle = [], []
    n_samples, family_correct, leaf_correct_pipeline = 0, 0, 0

    for row in rows:
        family_idx, variant_idx = row["family_idx"], row["variant_idx"]
        sequences = parse_fastq(row["r1_path"]) + parse_fastq(row["r2_path"])
        if max_reads_per_sample is not None:
            sequences = sequences[:max_reads_per_sample]
        if not sequences:
            continue

        noisy = [inject_read_errors(s, error_rate, rng) for s in sequences]
        vectors = torch.stack([encode_sequence_kmers(s, k, kmer_to_idx, vocab_size, canonical=canonical)
                                for s in noisy])

        family_preds = run_model_batched(router, vectors, device)
        pred_family_maj = majority_vote(family_preds)

        pipeline_variant_preds = run_model_batched(experts[pred_family_maj], vectors, device)
        pipeline_variant_maj = majority_vote(pipeline_variant_preds)

        oracle_variant_preds = run_model_batched(experts[family_idx], vectors, device)

        # Per-read leaf accuracy: route each read by its OWN router prediction and
        # require both the family and the variant to be right. This is the metric
        # directly comparable to a reference classifier's exact assignment, since
        # the oracle figure above is handed the true family and so measures an
        # easier task. Reads are grouped by predicted family so each expert runs
        # once over its own slice rather than every expert over every read.
        by_pred = {}
        for i, f in enumerate(family_preds):
            by_pred.setdefault(f, []).append(i)
        for f, idxs in by_pred.items():
            sub = run_model_batched(experts[f], vectors[idxs], device)
            for j, i in enumerate(idxs):
                read_leaf_correct.append(f == family_idx and sub[j] == variant_idx)

        read_family_true.extend([family_idx] * len(sequences))
        read_family_pred.extend(family_preds)
        read_variant_true.extend([variant_idx] * len(sequences))
        read_variant_pred_oracle.extend(oracle_variant_preds)

        n_samples += 1
        family_correct += int(pred_family_maj == family_idx)
        leaf_correct_pipeline += int(pred_family_maj == family_idx and pipeline_variant_maj == variant_idx)

    per_read_family_acc = 100.0 * sum(p == t for p, t in zip(read_family_pred, read_family_true)) / max(len(read_family_true), 1)
    per_read_variant_acc_oracle = 100.0 * sum(p == t for p, t in zip(read_variant_pred_oracle, read_variant_true)) / max(len(read_variant_true), 1)
    per_read_leaf_acc = 100.0 * sum(read_leaf_correct) / max(len(read_leaf_correct), 1)

    return {
        "error_rate": error_rate,
        "n_samples": n_samples,
        "per_read_family_acc": per_read_family_acc,
        "per_read_variant_acc_oracle": per_read_variant_acc_oracle,
        "per_read_leaf_acc": per_read_leaf_acc,
        "sample_family_acc": 100.0 * family_correct / max(n_samples, 1),
        "sample_leaf_acc_pipeline": 100.0 * leaf_correct_pipeline / max(n_samples, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep synthetic read error rate against the final model")
    parser.add_argument("--checkpoint", default="checkpoints/final/moe_final_1024x128.pth")
    parser.add_argument("--sample_manifest", default="data/manifests/final_train_manifest.csv")
    parser.add_argument("--split", default="val",
                         help="The final model has no held-out test split by design - 'val' (fold 4) "
                              "is the only data it didn't train on")
    parser.add_argument("--error_rates", type=float, nargs="+", default=DEFAULT_ERROR_RATES)
    parser.add_argument("--max_reads_per_sample", type=int, default=200,
                         help="Cap reads/votes per sample for tractability across 11 error rates x samples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out_csv", default="results/error_rate_sweep.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                           else args.device if args.device != "auto" else "cpu")

    ckpt, router, experts = load_checkpoint(args.checkpoint, device)
    kmer_to_idx, vocab_size, k = ckpt["kmer_vocab"], ckpt["vocab_size"], ckpt["k"]
    canonical = ckpt.get("canonical", False)

    with open(args.sample_manifest, newline="") as f:
        all_rows = list(csv.DictReader(f))
    for r in all_rows:
        r["family_idx"] = int(r["family_idx"])
        r["variant_idx"] = int(r["variant_idx"])
    remap_variant_idx(all_rows)
    rows = [r for r in all_rows if r["split"] == args.split]
    print(f"Evaluating on split={args.split!r}: {len(rows)} samples")

    results = []
    for rate in args.error_rates:
        r = evaluate_at_error_rate(rows, router, experts, kmer_to_idx, vocab_size, k, device,
                                    rate, args.max_reads_per_sample, args.seed, canonical=canonical)
        print(f"error_rate={rate:.2f}  per_read_family={r['per_read_family_acc']:.2f}%  "
              f"per_read_variant_oracle={r['per_read_variant_acc_oracle']:.2f}%  "
              f"per_read_leaf={r['per_read_leaf_acc']:.2f}%  "
              f"sample_family={r['sample_family_acc']:.2f}%  sample_leaf_pipeline={r['sample_leaf_acc_pipeline']:.2f}%")
        results.append(r)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
