#!/usr/bin/env python3
"""
Train and evaluate one input-encoding variant on a single fold, appending a row
to a shared CSV so variants are directly comparable.

Used for the two encoding questions the width sweeps cannot answer, both of
which change the activation vector itself and therefore require the experts to
be retrained rather than restored:

  k=6           4096 possible 6-mers against a 1024-wide vector, so vocabulary
                selection by document frequency becomes operative for the first
                time (at k=5 the vector holds the complete 5-mer space and
                nothing is ever discarded).
  canonical k=5 each k-mer collapsed with its reverse complement, which makes
                the encoding strand-invariant and halves the space to exactly
                512, so the input layer halves.

Defaults to fold 1, never fold 0.
"""

import argparse
import csv
import os

from train_moe import build_arg_parser, run_training
from evaluate_moe import evaluate_checkpoint
from run_cv import load_cv_manifest, write_fold_manifest
from sweep_architecture import count_state_dict_params

FIELDS = ["label", "k", "canonical", "max_kmers", "vocab_size", "fold",
          "router_params", "expert_params_total", "router_val_acc",
          "per_read_router_acc", "per_read_expert_acc_oracle",
          "sample_family_acc", "sample_leaf_acc_pipeline", "sample_variant_acc_oracle"]


def main():
    parser = build_arg_parser()
    parser.add_argument("--label", required=True, help="Name for this variant in the results CSV")
    parser.add_argument("--cv_sample_manifest", default="data/manifests/cv_sample_manifest.csv")
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--manifests_dir", default="data/manifests")
    parser.add_argument("--out_csv", default="results/encoding_variants.csv")
    args = parser.parse_args()

    if args.fold == 0:
        raise ValueError("fold 0 bounds accuracy irrespective of encoding; pick another fold")

    cv_rows = load_cv_manifest(args.cv_sample_manifest)
    fold_manifest = os.path.join(args.manifests_dir, f"cv_fold{args.fold}_sample_manifest.csv")
    write_fold_manifest(cv_rows, args.fold, args.k_folds, fold_manifest)
    args.sample_manifest = fold_manifest

    print(f"=== {args.label}: k={args.k} canonical={args.canonical} "
          f"max_kmers={args.max_kmers} fold={args.fold} ===")

    checkpoint = run_training(args)
    ev = evaluate_checkpoint(args.save_model, fold_manifest, split="test", verbose=False)

    row = {
        "label": args.label,
        "k": args.k,
        "canonical": args.canonical,
        "max_kmers": args.max_kmers,
        "vocab_size": checkpoint["vocab_size"],
        "fold": args.fold,
        "router_params": count_state_dict_params(checkpoint["router_state_dict"]),
        "expert_params_total": sum(count_state_dict_params(sd) for sd in checkpoint["expert_state_dicts"]),
        "router_val_acc": checkpoint["router_val_acc"],
        "per_read_router_acc": ev.get("per_read_router_acc"),
        "per_read_expert_acc_oracle": ev.get("per_read_expert_acc_oracle"),
        "sample_family_acc": ev.get("sample_family_acc"),
        "sample_leaf_acc_pipeline": ev.get("sample_leaf_acc_pipeline"),
        "sample_variant_acc_oracle": ev.get("sample_variant_acc_oracle"),
    }

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    exists = os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"\n{args.label}: vocab {row['vocab_size']}, per-read router "
          f"{row['per_read_router_acc']:.2f}%, per-read expert {row['per_read_expert_acc_oracle']:.2f}%, "
          f"sample-leaf {row['sample_leaf_acc_pipeline']:.2f}%")
    print(f"Appended to {args.out_csv}")


if __name__ == "__main__":
    main()
