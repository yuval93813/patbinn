#!/usr/bin/env python3
"""
Evaluate a trained hierarchical MoE checkpoint (train_moe.py) on held-out test samples.

Reports:
  - per-read router accuracy, and per-read expert accuracy (oracle-routed)
  - the main event: per-sample majority vote. Each held-out sample (one simulated
    replicate = many reads) gets ONE predicted family (majority vote of the router's
    per-read predictions) and, from that, ONE predicted variant - reported in both
    "pipeline mode" (run the ROUTER-PREDICTED family's expert, the real deployed
    behavior) and "oracle-routing mode" (always run the TRUE family's expert, to
    isolate expert quality from router quality).
"""

import argparse
import csv
from collections import Counter

import torch
from sklearn.metrics import classification_report, confusion_matrix

from kmer_encoding import parse_fastq, encode_sequence_kmers
from moe_dataset import remap_variant_idx
from moe_model import build_moe


def majority_vote(values):
    return Counter(values).most_common(1)[0][0]


def run_model_batched(model, vectors, device, batch_size=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size].to(device)
            preds.extend(model(batch).argmax(dim=1).cpu().tolist())
    return preds


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    family_names = ckpt["family_names"]
    variant_names_by_family = ckpt["variant_names_by_family"]
    vocab_size = ckpt["vocab_size"]

    # Back-compat: checkpoints saved before router/expert hidden sizes were split,
    # or before the precision baseline, fall back to the old shared key/default.
    legacy_hidden_sizes = ckpt.get("hidden_sizes", (1024, 128))
    router_hidden_sizes = ckpt.get("router_hidden_sizes", legacy_hidden_sizes)
    expert_hidden_sizes = ckpt.get("expert_hidden_sizes", legacy_hidden_sizes)
    precision = ckpt.get("precision", "binary")

    router, _ = build_moe(len(family_names), 1, vocab_size, ckpt["model_type"],
                           router_hidden_sizes, expert_hidden_sizes, precision)
    router.load_state_dict(ckpt["router_state_dict"])
    router = router.to(device).eval()

    experts = []
    for i, family in enumerate(family_names):
        num_variants = len(variant_names_by_family[family])
        _, expert_list = build_moe(1, num_variants, vocab_size, ckpt["model_type"],
                                    router_hidden_sizes, expert_hidden_sizes, precision)
        expert = expert_list[0]
        expert.load_state_dict(ckpt["expert_state_dicts"][i])
        experts.append(expert.to(device).eval())

    return ckpt, router, experts


def evaluate_checkpoint(checkpoint_path, sample_manifest_path, split="test",
                         max_votes_per_sample=None, device="auto", verbose=True):
    """
    Runs the full per-read + per-sample-majority-vote evaluation and returns every
    metric as a plain dict (in addition to printing it when verbose=True), so
    orchestration scripts (run_cv.py, sweep_architecture.py, sweep_read_counts.py)
    can call this directly instead of shelling out and re-parsing printed text.
    """
    device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                           else device if device != "auto" else "cpu")

    ckpt, router, experts = load_checkpoint(checkpoint_path, device)
    family_names = ckpt["family_names"]
    variant_names_by_family = ckpt["variant_names_by_family"]
    kmer_to_idx, vocab_size, k = ckpt["kmer_vocab"], ckpt["vocab_size"], ckpt["k"]
    # The encoding must match the one the checkpoint was trained with, otherwise the
    # activation vectors mean something different to the network than they did in training.
    canonical = ckpt.get("canonical", False)

    with open(sample_manifest_path, newline="") as f:
        all_rows = list(csv.DictReader(f))
    for r in all_rows:
        r["family_idx"] = int(r["family_idx"])
        r["variant_idx"] = int(r["variant_idx"])
    # Remap from ALL rows (not just this split) - must reproduce the exact same
    # dense variant_idx mapping training used, which was also computed from the
    # full manifest. Remapping from a split-filtered subset could silently produce
    # a different mapping if that subset happens to be missing a variant.
    remap_variant_idx(all_rows)
    rows = [r for r in all_rows if r["split"] == split]

    if not rows:
        if verbose:
            print(f"No samples found for split={split!r}")
        return {"n_samples": 0}

    all_read_family_true, all_read_family_pred = [], []
    all_read_variant_true, all_read_variant_pred_oracle = [], []

    sample_results = []
    for row in rows:
        family_idx, variant_idx = row["family_idx"], row["variant_idx"]
        sequences = parse_fastq(row["r1_path"]) + parse_fastq(row["r2_path"])
        if max_votes_per_sample is not None:
            sequences = sequences[:max_votes_per_sample]
        if not sequences:
            continue

        vectors = torch.stack([encode_sequence_kmers(s, k, kmer_to_idx, vocab_size, canonical=canonical)
                                for s in sequences])

        family_preds = run_model_batched(router, vectors, device)
        pred_family_maj = majority_vote(family_preds)

        pipeline_variant_preds = run_model_batched(experts[pred_family_maj], vectors, device)
        pipeline_variant_maj = majority_vote(pipeline_variant_preds)

        oracle_variant_preds = run_model_batched(experts[family_idx], vectors, device)
        oracle_variant_maj = majority_vote(oracle_variant_preds)

        all_read_family_true.extend([family_idx] * len(sequences))
        all_read_family_pred.extend(family_preds)
        all_read_variant_true.extend([variant_idx] * len(sequences))
        all_read_variant_pred_oracle.extend(oracle_variant_preds)

        sample_results.append({
            "sample_id": row["sample_id"],
            "true_family": family_idx, "true_variant": variant_idx,
            "pred_family": pred_family_maj,
            "pipeline_variant": pipeline_variant_maj,
            "oracle_variant": oracle_variant_maj,
            "num_reads": len(sequences),
        })

    n = len(sample_results)
    per_read_router_acc = (100.0 * sum(p == t for p, t in zip(all_read_family_pred, all_read_family_true))
                            / max(len(all_read_family_true), 1))
    per_read_expert_acc_oracle = (100.0 * sum(p == t for p, t in zip(all_read_variant_pred_oracle, all_read_variant_true))
                                  / max(len(all_read_variant_true), 1))

    family_correct = sum(r["pred_family"] == r["true_family"] for r in sample_results)
    pipeline_leaf_correct = sum(r["pred_family"] == r["true_family"] and r["pipeline_variant"] == r["true_variant"]
                                 for r in sample_results)
    oracle_variant_correct = sum(r["oracle_variant"] == r["true_variant"] for r in sample_results)
    routed_correctly = [r for r in sample_results if r["pred_family"] == r["true_family"]]
    variant_given_correct_routing = (
        100.0 * sum(r["pipeline_variant"] == r["true_variant"] for r in routed_correctly) / len(routed_correctly)
        if routed_correctly else float("nan")
    )

    results = {
        "n_samples": n,
        "family_names": family_names,
        "variant_names_by_family": variant_names_by_family,
        "per_read_router_acc": per_read_router_acc,
        "per_read_expert_acc_oracle": per_read_expert_acc_oracle,
        "sample_family_acc": 100.0 * family_correct / n,
        "sample_variant_acc_given_correct_routing": variant_given_correct_routing,
        "sample_leaf_acc_pipeline": 100.0 * pipeline_leaf_correct / n,
        "sample_variant_acc_oracle": 100.0 * oracle_variant_correct / n,
        "sample_results": sample_results,
    }

    if verbose:
        print("=" * 70)
        print("PER-READ METRICS")
        print("=" * 70)
        print("\nRouter (family), per read:")
        # Explicit labels: a thin test split can easily miss an entire family or
        # variant, and classification_report errors out (rather than reporting 0
        # support) if target_names doesn't match the inferred label set.
        print(classification_report(all_read_family_true, all_read_family_pred,
                                     labels=list(range(len(family_names))), target_names=family_names, zero_division=0))
        print("\nExperts (variant | true family, oracle-routed), per read:")
        print(classification_report(all_read_variant_true, all_read_variant_pred_oracle, zero_division=0))

        print("=" * 70)
        print("PER-SAMPLE MAJORITY VOTE")
        print("=" * 70)
        print(f"\nSamples evaluated: {n}")
        print(f"Family accuracy:                       {results['sample_family_acc']:.2f}%")
        print(f"Variant accuracy | correct routing:     {results['sample_variant_acc_given_correct_routing']:.2f}%")
        print(f"Full leaf accuracy (pipeline mode):      {results['sample_leaf_acc_pipeline']:.2f}%")
        print(f"Variant accuracy (oracle routing mode):  {results['sample_variant_acc_oracle']:.2f}%")

        print("\nFamily confusion matrix:")
        print(confusion_matrix([r["true_family"] for r in sample_results], [r["pred_family"] for r in sample_results]))

        print("\nPer-sample detail:")
        for r in sample_results:
            family_ok = "OK" if r["pred_family"] == r["true_family"] else "MISROUTED"
            print(f"  {r['sample_id']:40s} true=({family_names[r['true_family']]}, "
                  f"{variant_names_by_family[family_names[r['true_family']]][r['true_variant']]})  "
                  f"pred_family={family_names[r['pred_family']]} [{family_ok}]  n_reads={r['num_reads']}")

        if max_votes_per_sample is None:
            read_counts = [r["num_reads"] for r in sample_results]
            if max(read_counts) > 2 * min(read_counts):
                print(f"\nNote: read counts per sample range from {min(read_counts)} to {max(read_counts)} "
                      f"(genome-size/coverage driven) - majority-vote confidence isn't uniformly comparable "
                      f"across families. Use --max_votes_per_sample to cap for comparability.")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained hierarchical MoE checkpoint")
    parser.add_argument("--checkpoint", default="moe_bnn_best.pth")
    parser.add_argument("--sample_manifest", default="data/manifests/sample_manifest.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_votes_per_sample", type=int, default=None,
                         help="Subsample reads per sample to this cap before voting, "
                              "for cross-family vote-count comparability")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint, args.sample_manifest, args.split,
                        args.max_votes_per_sample, args.device, verbose=True)


if __name__ == "__main__":
    main()
