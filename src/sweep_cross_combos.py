#!/usr/bin/env python3
"""
sweep_architecture.py deliberately swept router and expert shapes independently
(fixing one while varying the other), on the assumption that cross-terms between
the two are negligible since they're trained in separate stages. This script
directly tests that assumption on the two "opposite corner" combos not covered by
that sweep: small router + big expert, and big router + small expert - jointly
trained and evaluated as one model each, not composed after the fact from two
separate independent-sweep checkpoints.
"""

import argparse
import csv
import os

from train_moe import build_arg_parser, run_training
from evaluate_moe import evaluate_checkpoint
from run_cv import load_cv_manifest, write_fold_manifest
from sweep_architecture import count_state_dict_params

COMBOS = [
    ("small_router_big_expert", (256, 128), (1024, 128)),
    ("big_router_small_expert", (1024, 128), (256, 128)),
]


def main():
    parser = build_arg_parser()
    parser.add_argument("--cv_sample_manifest", default="data/manifests/cv_sample_manifest.csv")
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--sweep_fold", type=int, default=0)
    parser.add_argument("--checkpoints_dir", default="checkpoints/sweep")
    parser.add_argument("--vocab_cache_sweep", default="data/vocab/kmer_vocab_sweep.json")
    parser.add_argument("--manifests_dir", default="data/manifests")
    parser.add_argument("--out_csv", default="results/size_sweep_cross.csv")
    args = parser.parse_args()

    cv_rows = load_cv_manifest(args.cv_sample_manifest)
    fold_manifest_path = os.path.join(args.manifests_dir, f"cv_fold{args.sweep_fold}_sample_manifest.csv")
    write_fold_manifest(cv_rows, args.sweep_fold, args.k_folds, fold_manifest_path)

    results = []
    for name, router_shape, expert_shape in COMBOS:
        print(f"\n{'#' * 70}\n# cross combo: router={router_shape} expert={expert_shape}\n{'#' * 70}")
        run_args = argparse.Namespace(**vars(args))
        run_args.sample_manifest = fold_manifest_path
        run_args.vocab_cache = args.vocab_cache_sweep
        run_args.router_hidden_sizes = list(router_shape)
        run_args.expert_hidden_sizes = list(expert_shape)
        run_args.save_model = os.path.join(args.checkpoints_dir, f"cross_{name}.pth")

        checkpoint = run_training(run_args)
        eval_results = evaluate_checkpoint(run_args.save_model, fold_manifest_path, split="test", verbose=False)

        router_params = count_state_dict_params(checkpoint["router_state_dict"])
        expert_params_total = sum(count_state_dict_params(sd) for sd in checkpoint["expert_state_dicts"])

        results.append({
            "combo": name,
            "router_hidden_sizes": "x".join(str(s) for s in router_shape),
            "expert_hidden_sizes": "x".join(str(s) for s in expert_shape),
            "router_params": router_params,
            "expert_params_total": expert_params_total,
            "per_read_router_acc": eval_results.get("per_read_router_acc"),
            "per_read_expert_acc_oracle": eval_results.get("per_read_expert_acc_oracle"),
            "sample_family_acc": eval_results.get("sample_family_acc"),
            "sample_leaf_acc_pipeline": eval_results.get("sample_leaf_acc_pipeline"),
            "sample_variant_acc_oracle": eval_results.get("sample_variant_acc_oracle"),
        })

    fieldnames = ["combo", "router_hidden_sizes", "expert_hidden_sizes", "router_params", "expert_params_total",
                  "per_read_router_acc", "per_read_expert_acc_oracle",
                  "sample_family_acc", "sample_leaf_acc_pipeline", "sample_variant_acc_oracle"]
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nCross-combo results written to {args.out_csv}")


if __name__ == "__main__":
    main()
