#!/usr/bin/env python3
"""
Architecture-size ablation: sweep router and expert two-hidden-layer shapes
independently over a (first_layer, second_layer) grid, on a single representative
CV split (not full K-fold, to bound cost - router/expert training is already fully
decoupled by the stage-wise freeze, so cross-terms between the two sweeps are
expected to be negligible). Finds the smallest-param-count shape within
--tolerance percentage points of the best observed accuracy for each component -
the "how small can this go" curve that matters for CAM/crossbar hardware
feasibility.

Run the final chosen (smallest-still-good) shapes through the full run_cv.py
K-fold CV separately afterward for the headline, statistically-rigorous number.
"""

import argparse
import csv
import os

from train_moe import build_arg_parser, run_training
from evaluate_moe import evaluate_checkpoint
from run_cv import load_cv_manifest, write_fold_manifest

FIRST_LAYER_WIDTHS = [256, 512, 1024]
SECOND_LAYER_WIDTHS = [256, 128]


def count_state_dict_params(state_dict):
    return sum(t.numel() for t in state_dict.values())


def sweep_component(component, shape_grid, fixed_hidden_sizes, base_args, fold_manifest_path,
                     checkpoints_dir, vocab_cache):
    assert component in ("router", "expert")
    results = []
    for shape in shape_grid:
        print(f"\n{'#' * 70}\n# sweep {component}: hidden_sizes={shape} "
              f"(fixed {'expert' if component == 'router' else 'router'}={fixed_hidden_sizes})\n{'#' * 70}")
        args = argparse.Namespace(**vars(base_args))
        args.sample_manifest = fold_manifest_path
        args.vocab_cache = vocab_cache
        args.save_model = os.path.join(checkpoints_dir, f"{component}_{'x'.join(str(s) for s in shape)}.pth")
        if component == "router":
            args.router_hidden_sizes = list(shape)
            args.expert_hidden_sizes = list(fixed_hidden_sizes)
        else:
            args.router_hidden_sizes = list(fixed_hidden_sizes)
            args.expert_hidden_sizes = list(shape)

        checkpoint = run_training(args)
        eval_results = evaluate_checkpoint(args.save_model, fold_manifest_path, split="test", verbose=False)

        router_params = count_state_dict_params(checkpoint["router_state_dict"])
        expert_params_total = sum(count_state_dict_params(sd) for sd in checkpoint["expert_state_dicts"])

        results.append({
            "component": component,
            "hidden_sizes": "x".join(str(s) for s in shape),
            "router_params": router_params,
            "expert_params_total": expert_params_total,
            "per_read_router_acc": eval_results.get("per_read_router_acc"),
            "per_read_expert_acc_oracle": eval_results.get("per_read_expert_acc_oracle"),
            "sample_family_acc": eval_results.get("sample_family_acc"),
            "sample_leaf_acc_pipeline": eval_results.get("sample_leaf_acc_pipeline"),
            "sample_variant_acc_oracle": eval_results.get("sample_variant_acc_oracle"),
        })
    return results


def pick_smallest_good(results, acc_key, param_key, tolerance):
    best = max(r[acc_key] for r in results if r[acc_key] is not None)
    good = [r for r in results if r[acc_key] is not None and r[acc_key] >= best - tolerance]
    return min(good, key=lambda r: r[param_key])


def main():
    parser = build_arg_parser()
    parser.add_argument("--cv_sample_manifest", default="data/manifests/cv_sample_manifest.csv")
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--sweep_fold", type=int, default=0,
                         help="Which fold to use as the single representative split for the sweep")
    parser.add_argument("--first_layer_widths", type=int, nargs="+", default=FIRST_LAYER_WIDTHS)
    parser.add_argument("--second_layer_widths", type=int, nargs="+", default=SECOND_LAYER_WIDTHS)
    parser.add_argument("--skip_grid", action="store_true",
                         help="Skip the first_layer_widths x second_layer_widths cross-product entirely - "
                              "use only --extra_shapes. For supplementary one-off shapes without re-running "
                              "the whole grid.")
    parser.add_argument("--extra_shapes", type=str, nargs="+", default=[],
                         help="Additional shapes to test beyond the grid, any hidden-layer count, e.g. "
                              "'128' for a single-hidden-layer shape or '512,256,128' for three layers. "
                              "Comma-separated ints per shape.")
    parser.add_argument("--fixed_hidden_sizes", type=int, nargs="+", default=[1024, 128],
                         help="Fixed two-hidden-layer shape for the component not currently being swept")
    parser.add_argument("--router_shape_override", type=str, default=None,
                         help="Skip picking chosen_router from this run's router_results and use this exact "
                              "comma-separated shape (e.g. '1024,256') to fix the router for the expert sweep. "
                              "Useful when supplementing an already-completed sweep whose chosen router shape "
                              "is already known.")
    parser.add_argument("--tolerance", type=float, default=1.0,
                         help="Percentage points within the best observed accuracy still considered 'good enough' "
                              "when picking the smallest (by param count) shape")
    parser.add_argument("--checkpoints_dir", default="checkpoints/sweep")
    parser.add_argument("--vocab_cache_sweep", default="data/vocab/kmer_vocab_sweep.json")
    parser.add_argument("--manifests_dir", default="data/manifests")
    parser.add_argument("--out_csv", default="results/size_sweep.csv")
    args = parser.parse_args()

    shape_grid = [] if args.skip_grid else \
        [(first, second) for first in args.first_layer_widths for second in args.second_layer_widths]
    shape_grid += [tuple(int(x) for x in s.split(",")) for s in args.extra_shapes]
    print(f"Shape grid ({len(shape_grid)} combos): {shape_grid}")

    cv_rows = load_cv_manifest(args.cv_sample_manifest)
    fold_manifest_path = os.path.join(args.manifests_dir, f"cv_fold{args.sweep_fold}_sample_manifest.csv")
    write_fold_manifest(cv_rows, args.sweep_fold, args.k_folds, fold_manifest_path)

    router_results = sweep_component("router", shape_grid, args.fixed_hidden_sizes, args,
                                      fold_manifest_path, args.checkpoints_dir, args.vocab_cache_sweep)
    if args.router_shape_override:
        router_shape = tuple(int(x) for x in args.router_shape_override.split(","))
        print(f"\nUsing router_shape_override={router_shape} to fix the router for the expert sweep "
              f"(skipping chosen-shape selection from this run's router_results).")
    else:
        chosen_router = pick_smallest_good(router_results, "per_read_router_acc", "router_params", args.tolerance)
        print(f"\nChosen router shape (smallest params within {args.tolerance}pp of best): "
              f"{chosen_router['hidden_sizes']} ({chosen_router['router_params']} params)")
        router_shape = tuple(int(x) for x in chosen_router["hidden_sizes"].split("x"))
    expert_results = sweep_component("expert", shape_grid, router_shape, args,
                                      fold_manifest_path, args.checkpoints_dir, args.vocab_cache_sweep)
    chosen_expert = pick_smallest_good(expert_results, "per_read_expert_acc_oracle", "expert_params_total", args.tolerance)
    print(f"Chosen expert shape (smallest params within {args.tolerance}pp of best): "
          f"{chosen_expert['hidden_sizes']} ({chosen_expert['expert_params_total']} params)")

    all_results = router_results + expert_results
    fieldnames = ["component", "hidden_sizes", "router_params", "expert_params_total",
                  "per_read_router_acc", "per_read_expert_acc_oracle",
                  "sample_family_acc", "sample_leaf_acc_pipeline", "sample_variant_acc_oracle"]
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nSize sweep results written to {args.out_csv}")
    print(f"Recommended smallest-still-good config: router_hidden_sizes={router_shape}, "
          f"expert_hidden_sizes={tuple(int(x) for x in chosen_expert['hidden_sizes'].split('x'))}")
    print("Run these through run_cv.py's full K-fold CV for the final rigorous number.")


if __name__ == "__main__":
    main()
