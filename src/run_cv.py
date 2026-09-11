#!/usr/bin/env python3
"""
Genome-level K-fold cross-validation orchestrator for the hierarchical MoE.

Consumes data/simulate_reads.py's multi-isolate cv_sample_manifest.csv (one row per
simulated replicate, tagged with isolate_idx/fold - all replicates of a given
isolate share one fold, assigned back in download_genomes.py). For each held-out
fold: synthesizes a train/val/test split column on the fly (this fold -> test, the
next fold -> val, everything else -> train), writes a per-fold manifest, and calls
train_moe.run_training / evaluate_moe.evaluate_checkpoint on it - both already
handle everything else (vocab building, dense variant remapping, non-uniform
per-family variant counts) unchanged.

Run all folds serially, or `--only_fold i` for one fold per Slurm array task
(then combine with aggregate_cv_results.py).
"""

import argparse
import csv
import json
import os

import train_flat
import evaluate_flat
from train_moe import build_arg_parser, run_training
from evaluate_moe import evaluate_checkpoint


def load_cv_manifest(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["fold"] = int(r["fold"])
    return rows


def write_fold_manifest(rows, test_fold, k_folds, out_path):
    val_fold = (test_fold + 1) % k_folds
    out_rows = []
    for r in rows:
        if r["fold"] == test_fold:
            split = "test"
        elif r["fold"] == val_fold:
            split = "val"
        else:
            split = "train"
        out_rows.append({**r, "split": split})

    fieldnames = list(rows[0].keys()) + ["split"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    n_train = sum(1 for r in out_rows if r["split"] == "train")
    n_val = sum(1 for r in out_rows if r["split"] == "val")
    n_test = sum(1 for r in out_rows if r["split"] == "test")
    print(f"  fold {test_fold}: {n_train} train / {n_val} val / {n_test} test samples -> {out_path}")


def run_fold(fold, k_folds, cv_rows, base_args, checkpoints_dir, vocab_dir, manifests_dir):
    fold_manifest_path = os.path.join(manifests_dir, f"cv_fold{fold}_sample_manifest.csv")
    write_fold_manifest(cv_rows, fold, k_folds, fold_manifest_path)

    args = argparse.Namespace(**vars(base_args))
    args.sample_manifest = fold_manifest_path
    args.vocab_cache = os.path.join(vocab_dir, f"kmer_vocab_fold{fold}.json")
    args.save_model = os.path.join(checkpoints_dir, f"fold{fold}.pth")

    checkpoint = run_training(args)
    if checkpoint is None:
        return {"fold": fold, "error": "no training data for this fold"}

    eval_results = evaluate_checkpoint(args.save_model, fold_manifest_path, split="test", verbose=False)

    return {
        "fold": fold,
        "precision": args.precision,
        "router_hidden_sizes": list(args.router_hidden_sizes),
        "expert_hidden_sizes": list(args.expert_hidden_sizes),
        "router_val_acc": checkpoint["router_val_acc"],
        "expert_val_accs": checkpoint["expert_val_accs"],
        "per_read_router_acc": eval_results.get("per_read_router_acc"),
        "per_read_expert_acc_oracle": eval_results.get("per_read_expert_acc_oracle"),
        "sample_family_acc": eval_results.get("sample_family_acc"),
        "sample_variant_acc_given_correct_routing": eval_results.get("sample_variant_acc_given_correct_routing"),
        "sample_leaf_acc_pipeline": eval_results.get("sample_leaf_acc_pipeline"),
        "sample_variant_acc_oracle": eval_results.get("sample_variant_acc_oracle"),
        "n_test_samples": eval_results.get("n_samples"),
    }


def run_fold_flat(fold, k_folds, cv_rows, base_args, checkpoints_dir, vocab_dir, manifests_dir):
    """
    Flat-classifier counterpart to run_fold, for the "does the hierarchy actually
    help" comparison - same fold manifests, same precision, reuses base_args'
    router_epochs/expert_hidden_sizes as stand-ins for the flat model's single
    epoch-count/hidden-size knobs (there's no router/expert split to have two of).
    """
    fold_manifest_path = os.path.join(manifests_dir, f"cv_fold{fold}_sample_manifest.csv")
    write_fold_manifest(cv_rows, fold, k_folds, fold_manifest_path)

    flat_args = argparse.Namespace(
        sample_manifest=fold_manifest_path,
        precision=base_args.precision,
        k=base_args.k,
        max_kmers=base_args.max_kmers,
        vocab_cache=os.path.join(vocab_dir, f"kmer_vocab_flat_fold{fold}.json"),
        max_reads_per_sample_for_vocab=base_args.max_reads_per_sample_for_vocab,
        max_reads_per_sample_train=base_args.max_reads_per_sample_train,
        epochs=base_args.router_epochs,
        batch_size=base_args.batch_size,
        hidden_sizes=list(base_args.expert_hidden_sizes),
        lr=base_args.lr,
        patience=base_args.patience,
        device=base_args.device,
        num_workers=base_args.num_workers,
        save_model=os.path.join(checkpoints_dir, f"flat_fold{fold}.pth"),
    )

    checkpoint = train_flat.run_training(flat_args)
    if checkpoint is None:
        return {"fold": fold, "error": "no training data for this fold"}

    eval_results = evaluate_flat.evaluate_checkpoint(flat_args.save_model, fold_manifest_path,
                                                       split="test", verbose=False)

    return {
        "fold": fold,
        "classifier": "flat",
        "precision": flat_args.precision,
        "hidden_sizes": list(flat_args.hidden_sizes),
        "val_acc": checkpoint["val_acc"],
        "per_read_leaf_acc": eval_results.get("per_read_leaf_acc"),
        "sample_leaf_acc": eval_results.get("sample_leaf_acc"),
        "n_test_samples": eval_results.get("n_samples"),
    }


def main():
    parser = build_arg_parser()  # reuses train_moe's hyperparameter flags (router/expert
    # hidden sizes, epochs, batch size, precision, etc.) - note --sample_manifest,
    # --vocab_cache, --save_model are overridden per-fold below regardless of any
    # value passed on the CLI, since those paths are fold-managed by this script.
    parser.add_argument("--cv_sample_manifest", default="data/manifests/cv_sample_manifest.csv")
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--only_fold", type=int, default=None,
                         help="Run just this fold (0-indexed) - for one Slurm array task per fold")
    parser.add_argument("--classifier", choices=["hierarchical", "flat"], default="hierarchical",
                         help="hierarchical=the MoE (default), flat=the no-hierarchy baseline over all leaves")
    parser.add_argument("--results_dir", default="results/cv")
    parser.add_argument("--checkpoints_dir", default="checkpoints/cv")
    parser.add_argument("--vocab_dir", default="data/vocab")
    parser.add_argument("--manifests_dir", default="data/manifests")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    cv_rows = load_cv_manifest(args.cv_sample_manifest)

    folds = [args.only_fold] if args.only_fold is not None else list(range(args.k_folds))
    for fold in folds:
        print(f"\n{'=' * 70}\nCV FOLD {fold} (of {args.k_folds}) [{args.classifier}]\n{'=' * 70}")
        if args.classifier == "flat":
            result = run_fold_flat(fold, args.k_folds, cv_rows, args, args.checkpoints_dir,
                                    args.vocab_dir, args.manifests_dir)
        else:
            result = run_fold(fold, args.k_folds, cv_rows, args, args.checkpoints_dir,
                               args.vocab_dir, args.manifests_dir)
        result_path = os.path.join(args.results_dir, f"fold{fold}_result.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nFold {fold} result written to {result_path}")
        if "error" not in result:
            if args.classifier == "flat":
                print(f"  sample_leaf_acc={result['sample_leaf_acc']:.2f}%")
            else:
                print(f"  sample_family_acc={result['sample_family_acc']:.2f}%  "
                      f"sample_leaf_acc_pipeline={result['sample_leaf_acc_pipeline']:.2f}%")


if __name__ == "__main__":
    main()
