#!/usr/bin/env python3
"""
Flat (non-hierarchical) baseline: one classifier directly over all leaf
(family,variant) classes, no router/expert split. Exists to answer "does the
hierarchical MoE actually help" - compare against train_moe.py's result on the
same manifest/split.
"""

import argparse
import os

import torch

from moe_dataset import load_sample_manifest, remap_variant_idx, build_global_kmer_vocab, build_leaf_index
from BNN_model import BinaryMLP, StandardMLP
from train_moe import make_loader, train_stage


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Flat baseline classifier over all leaf classes")
    parser.add_argument("--sample_manifest", default="data/manifests/sample_manifest.csv")
    parser.add_argument("--precision", choices=["binary", "float32"], default="binary")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max_kmers", type=int, default=1024)
    parser.add_argument("--vocab_cache", default="data/vocab/kmer_vocab_flat.json")
    parser.add_argument("--max_reads_per_sample_for_vocab", type=int, default=500)
    parser.add_argument("--max_reads_per_sample_train", type=int, default=2000,
                         help="Cap reads per sample when building training datasets (see train_moe.py for why)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[1024, 128])
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0,
                         help="See train_moe.py: 0 is correct here, vectors are precomputed upfront")
    parser.add_argument("--save_model", default="flat_bnn_best.pth")
    return parser


def run_training(args):
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
                           if args.device != "auto" else "cpu")
    print(f"Using device: {device}")

    rows = load_sample_manifest(args.sample_manifest)
    if not rows:
        print("No samples found in manifest - run simulate_reads.py first.")
        return None
    remap_variant_idx(rows)
    leaf_index = build_leaf_index(rows)
    leaf_names = [""] * len(leaf_index)
    for row in rows:
        leaf_names[leaf_index[(row["family_idx"], row["variant_idx"])]] = f"{row['family']}/{row['variant']}"

    print(f"{len(leaf_index)} leaf classes, {len(rows)} samples")

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"] or train_rows
    if not train_rows:
        raise ValueError("No rows with split=='train' in the manifest - check your split/fold configuration")

    kmer_to_idx, vocab_size = build_global_kmer_vocab(
        train_rows, args.k, args.max_kmers,
        max_reads_per_sample=args.max_reads_per_sample_for_vocab,
        cache_path=args.vocab_cache,
    )
    print(f"Shared vocabulary size: {vocab_size}")

    model_cls = StandardMLP if args.precision == "float32" else BinaryMLP
    model = model_cls(in_features=vocab_size, hidden_sizes=tuple(args.hidden_sizes),
                       num_classes=len(leaf_index)).to(device)
    clamp_weights = (args.precision == "binary")
    max_reads_cap = args.max_reads_per_sample_train if args.max_reads_per_sample_train > 0 else None

    train_loader = make_loader(train_rows, kmer_to_idx, vocab_size, args.k, "leaf",
                                args.batch_size, balanced=True, shuffle=True,
                                num_workers=args.num_workers, leaf_index=leaf_index,
                                max_reads_per_sample=max_reads_cap)
    val_loader = make_loader(val_rows, kmer_to_idx, vocab_size, args.k, "leaf",
                              args.batch_size, balanced=False, shuffle=False,
                              num_workers=args.num_workers, leaf_index=leaf_index,
                              max_reads_per_sample=max_reads_cap)

    val_acc = train_stage(model, train_loader, val_loader, device, args.epochs, args.lr, args.patience,
                           leaf_names, "flat", clamp_weights=clamp_weights)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "leaf_index": leaf_index,
        "leaf_names": leaf_names,
        "kmer_vocab": kmer_to_idx,
        "vocab_size": vocab_size,
        "k": args.k,
        "max_kmers": args.max_kmers,
        "precision": args.precision,
        "hidden_sizes": tuple(args.hidden_sizes),
        "val_acc": val_acc,
        "args": vars(args),
    }
    os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
    torch.save(checkpoint, args.save_model)
    print(f"\nSaved flat checkpoint to {args.save_model}")
    print(f"Val acc: {val_acc:.2f}%")
    return checkpoint


def main():
    args = build_arg_parser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
