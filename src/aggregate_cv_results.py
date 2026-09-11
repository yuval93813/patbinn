#!/usr/bin/env python3
"""Combine run_cv.py's per-fold result JSONs into one results/cv_results.csv, plus mean+-std."""

import argparse
import csv
import glob
import json
import os
import statistics


HIERARCHICAL_METRIC_KEYS = [
    "router_val_acc", "per_read_router_acc", "per_read_expert_acc_oracle",
    "sample_family_acc", "sample_variant_acc_given_correct_routing",
    "sample_leaf_acc_pipeline", "sample_variant_acc_oracle",
]
FLAT_METRIC_KEYS = ["val_acc", "per_read_leaf_acc", "sample_leaf_acc"]


def main():
    parser = argparse.ArgumentParser(description="Aggregate run_cv.py per-fold JSON results into one CSV")
    # Defaults point at the main condition, which ships with the repository, so
    # a bare run reproduces results/cv_results.csv. run_cv.py writes one
    # directory per (classifier, precision) combination -- pass --results_dir
    # results/cv_flat_binary or results/cv_hierarchical_float32 for the others.
    parser.add_argument("--results_dir", default="results/cv_hierarchical_binary")
    parser.add_argument("--out_csv", default="results/cv_results.csv")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.results_dir, "fold*_result.json")))
    if not paths:
        print(f"No fold result JSONs found in {args.results_dir}")
        return

    results = []
    for path in paths:
        with open(path) as f:
            results.append(json.load(f))

    ok_results = [r for r in results if "error" not in r]
    if len(ok_results) < len(results):
        errored = [r["fold"] for r in results if "error" in r]
        print(f"Warning: folds with errors (excluded from aggregation): {errored}")
    if not ok_results:
        return

    is_flat = ok_results[0].get("classifier") == "flat"
    metric_keys = FLAT_METRIC_KEYS if is_flat else HIERARCHICAL_METRIC_KEYS
    id_fields = (["fold", "classifier", "precision", "hidden_sizes", "n_test_samples"] if is_flat
                 else ["fold", "precision", "router_hidden_sizes", "expert_hidden_sizes", "n_test_samples"])

    fieldnames = id_fields + metric_keys
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in ok_results:
            writer.writerow({k: r.get(k) for k in fieldnames})

    print(f"Wrote {len(ok_results)} fold rows to {args.out_csv}\n")
    print(f"{'metric':<45}{'mean':>10}{'std':>10}")
    for key in metric_keys:
        values = [r[key] for r in ok_results if r.get(key) is not None]
        if not values:
            continue
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{key:<45}{mean:>10.2f}{std:>10.2f}")


if __name__ == "__main__":
    main()
