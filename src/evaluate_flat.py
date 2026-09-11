#!/usr/bin/env python3
"""
Evaluate a trained flat-classifier checkpoint (train_flat.py): per-sample majority
vote directly over leaf predictions, no router/expert hierarchy. Directly comparable
to evaluate_moe.py's sample_leaf_acc_pipeline on the same manifest/split.
"""

import argparse
import csv

import torch
from sklearn.metrics import classification_report, confusion_matrix

from kmer_encoding import parse_fastq, encode_sequence_kmers
from moe_dataset import remap_variant_idx
from evaluate_moe import majority_vote, run_model_batched
from BNN_model import BinaryMLP, StandardMLP


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_cls = StandardMLP if ckpt["precision"] == "float32" else BinaryMLP
    model = model_cls(in_features=ckpt["vocab_size"], hidden_sizes=tuple(ckpt["hidden_sizes"]),
                       num_classes=len(ckpt["leaf_index"]))
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt, model.to(device).eval()


def evaluate_checkpoint(checkpoint_path, sample_manifest_path, split="test",
                         max_votes_per_sample=None, device="auto", verbose=True):
    device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                           else device if device != "auto" else "cpu")

    ckpt, model = load_checkpoint(checkpoint_path, device)
    leaf_index = ckpt["leaf_index"]
    leaf_names = ckpt["leaf_names"]
    kmer_to_idx, vocab_size, k = ckpt["kmer_vocab"], ckpt["vocab_size"], ckpt["k"]

    with open(sample_manifest_path, newline="") as f:
        all_rows = list(csv.DictReader(f))
    for r in all_rows:
        r["family_idx"] = int(r["family_idx"])
        r["variant_idx"] = int(r["variant_idx"])
    remap_variant_idx(all_rows)
    rows = [r for r in all_rows if r["split"] == split]

    if not rows:
        if verbose:
            print(f"No samples found for split={split!r}")
        return {"n_samples": 0}

    all_true, all_pred = [], []
    sample_results = []
    for row in rows:
        leaf_idx = leaf_index[(row["family_idx"], row["variant_idx"])]
        sequences = parse_fastq(row["r1_path"]) + parse_fastq(row["r2_path"])
        if max_votes_per_sample is not None:
            sequences = sequences[:max_votes_per_sample]
        if not sequences:
            continue

        vectors = torch.stack([encode_sequence_kmers(s, k, kmer_to_idx, vocab_size) for s in sequences])
        preds = run_model_batched(model, vectors, device)
        pred_maj = majority_vote(preds)

        all_true.extend([leaf_idx] * len(sequences))
        all_pred.extend(preds)
        sample_results.append({
            "sample_id": row["sample_id"], "true_leaf": leaf_idx, "pred_leaf": pred_maj,
            "num_reads": len(sequences),
        })

    n = len(sample_results)
    per_read_leaf_acc = 100.0 * sum(p == t for p, t in zip(all_pred, all_true)) / max(len(all_true), 1)
    sample_leaf_acc = 100.0 * sum(r["pred_leaf"] == r["true_leaf"] for r in sample_results) / n

    results = {
        "n_samples": n,
        "leaf_names": leaf_names,
        "per_read_leaf_acc": per_read_leaf_acc,
        "sample_leaf_acc": sample_leaf_acc,
        "sample_results": sample_results,
    }

    if verbose:
        print("=" * 70)
        print("PER-READ METRICS (flat classifier)")
        print("=" * 70)
        labels = list(range(len(leaf_names)))
        print(classification_report(all_true, all_pred, labels=labels, target_names=leaf_names, zero_division=0))

        print("=" * 70)
        print("PER-SAMPLE MAJORITY VOTE (flat classifier)")
        print("=" * 70)
        print(f"\nSamples evaluated: {n}")
        print(f"Leaf accuracy (majority vote): {sample_leaf_acc:.2f}%")
        print("\nConfusion matrix:")
        print(confusion_matrix(all_true, all_pred, labels=labels))

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained flat-classifier checkpoint")
    parser.add_argument("--checkpoint", default="flat_bnn_best.pth")
    parser.add_argument("--sample_manifest", default="data/manifests/sample_manifest.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_votes_per_sample", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint, args.sample_manifest, args.split,
                        args.max_votes_per_sample, args.device, verbose=True)


if __name__ == "__main__":
    main()
