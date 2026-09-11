#!/usr/bin/env python3
"""
Read-count-to-confidence sweep: how many reads does the per-sample majority vote
need before it becomes a confident call? Pure inference, no retraining - runs
evaluate_moe.py's --max_votes_per_sample subsampling across a range of N, once per
CV fold checkpoint (so the resulting curve's error bars reflect genome-level
variance across folds, not just vote-count noise within one model).
"""

import argparse
import csv
import os

from evaluate_moe import evaluate_checkpoint

DEFAULT_VOTE_COUNTS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]


def main():
    parser = argparse.ArgumentParser(description="Sweep read-count-to-confidence across CV fold checkpoints")
    parser.add_argument("--checkpoints_dir", default="checkpoints/cv")
    parser.add_argument("--manifests_dir", default="data/manifests")
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--vote_counts", type=int, nargs="+", default=DEFAULT_VOTE_COUNTS)
    parser.add_argument("--out_csv", default="results/read_count_sweep.csv")
    args = parser.parse_args()

    rows_out = []
    for fold in range(args.k_folds):
        ckpt_path = os.path.join(args.checkpoints_dir, f"fold{fold}.pth")
        manifest_path = os.path.join(args.manifests_dir, f"cv_fold{fold}_sample_manifest.csv")
        if not (os.path.exists(ckpt_path) and os.path.exists(manifest_path)):
            print(f"Skipping fold {fold}: missing checkpoint or manifest")
            continue

        for n in args.vote_counts:
            results = evaluate_checkpoint(ckpt_path, manifest_path, split="test",
                                           max_votes_per_sample=n, verbose=False)
            if results.get("n_samples", 0) == 0:
                continue
            rows_out.append({
                "fold": fold,
                "max_votes_per_sample": n,
                "sample_family_acc": results["sample_family_acc"],
                "sample_leaf_acc_pipeline": results["sample_leaf_acc_pipeline"],
                "sample_variant_acc_oracle": results["sample_variant_acc_oracle"],
                "n_samples": results["n_samples"],
            })
            print(f"fold={fold} N={n:>5}: family_acc={results['sample_family_acc']:6.2f}%  "
                  f"leaf_acc_pipeline={results['sample_leaf_acc_pipeline']:6.2f}%")

    if not rows_out:
        print("No results produced - check --checkpoints_dir/--manifests_dir point to an existing run_cv.py output.")
        return

    fieldnames = ["fold", "max_votes_per_sample", "sample_family_acc",
                  "sample_leaf_acc_pipeline", "sample_variant_acc_oracle", "n_samples"]
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nRead-count sweep results written to {args.out_csv}")


if __name__ == "__main__":
    main()
