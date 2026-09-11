#!/usr/bin/env python3
"""
Router-only size and depth sweep.

Experts are trained on their family's TRUE label and never on router output
(see train_moe.run_training), so an expert trained under one router remains
valid under any other. This sweep therefore retrains only the router at each
shape and restores the eight experts from an existing checkpoint, which costs
roughly one ninth of a full pipeline per configuration while being equivalent
to retraining everything.

Runs on a fold chosen by --fold, defaulting to 1 rather than 0: fold 0 is the
fold in which twelve singly-represented classes contribute no training material
at all, which depresses accuracy irrespective of architecture.
"""

import argparse
import csv
import os

from train_moe import build_arg_parser, run_training
from evaluate_moe import evaluate_checkpoint
from run_cv import load_cv_manifest, write_fold_manifest
from sweep_architecture import count_state_dict_params

# Shapes to sweep. '128' alone is the single-hidden-layer arrangement of the
# original PatBiNN classifier; 1024x512x128 is the untested three-layer depth.
DEFAULT_SHAPES = ["64", "128", "256", "512", "128,64", "256,128", "1024,128", "1024,512,128"]


def main():
    parser = build_arg_parser()
    parser.add_argument("--cv_sample_manifest", default="data/manifests/cv_sample_manifest.csv")
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=1,
                         help="Fold to sweep on. Not 0 - see module docstring.")
    parser.add_argument("--shapes", type=str, nargs="+", default=DEFAULT_SHAPES,
                         help="Router shapes, comma-separated per shape, e.g. 128 256,128 1024,512,128")
    parser.add_argument("--base_checkpoint", default="checkpoints/cv_hierarchical_binary/fold1.pth",
                         help="Checkpoint supplying the trained experts that every configuration reuses")
    parser.add_argument("--checkpoints_dir", default="checkpoints/router_sweep")
    parser.add_argument("--manifests_dir", default="data/manifests")
    parser.add_argument("--out_csv", default="results/router_sweep.csv")
    args = parser.parse_args()

    if args.fold == 0:
        raise ValueError("fold 0 bounds accuracy irrespective of architecture; pick another fold")

    shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes]
    print(f"Router sweep over {len(shapes)} shapes on fold {args.fold}: {shapes}")

    cv_rows = load_cv_manifest(args.cv_sample_manifest)
    fold_manifest = os.path.join(args.manifests_dir, f"cv_fold{args.fold}_sample_manifest.csv")
    write_fold_manifest(cv_rows, args.fold, args.k_folds, fold_manifest)

    os.makedirs(args.checkpoints_dir, exist_ok=True)
    results = []
    for shape in shapes:
        tag = "x".join(str(s) for s in shape)
        print(f"\n{'#' * 70}\n# router {tag}  (experts restored from {args.base_checkpoint})\n{'#' * 70}")

        run_args = argparse.Namespace(**vars(args))
        run_args.sample_manifest = fold_manifest
        run_args.vocab_cache = args.vocab_cache
        run_args.router_hidden_sizes = list(shape)
        run_args.resume_from = args.base_checkpoint
        run_args.only_stages = "router"
        run_args.save_model = os.path.join(args.checkpoints_dir, f"router_{tag}.pth")

        checkpoint = run_training(run_args)
        ev = evaluate_checkpoint(run_args.save_model, fold_manifest, split="test", verbose=False)

        results.append({
            "router_hidden_sizes": tag,
            "n_hidden_layers": len(shape),
            "router_params": count_state_dict_params(checkpoint["router_state_dict"]),
            "router_val_acc": checkpoint["router_val_acc"],
            "per_read_router_acc": ev.get("per_read_router_acc"),
            "sample_family_acc": ev.get("sample_family_acc"),
            "sample_leaf_acc_pipeline": ev.get("sample_leaf_acc_pipeline"),
            # unchanged across the sweep by construction - reported as the control
            "per_read_expert_acc_oracle": ev.get("per_read_expert_acc_oracle"),
        })
        print(f"-> router {tag}: per-read {results[-1]['per_read_router_acc']:.2f}%  "
              f"sample-leaf {results[-1]['sample_leaf_acc_pipeline']:.2f}%")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nRouter sweep results written to {args.out_csv}")


if __name__ == "__main__":
    main()
