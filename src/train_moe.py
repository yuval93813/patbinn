#!/usr/bin/env python3
"""
Stage-wise training for the hierarchical MoE virus classifier:
  1. Train the family router on all training reads (family label).
  2. Freeze the router.
  3. Train each family's variant expert independently, on only that family's reads.

Follows the CLI and checkpoint conventions of the original whole-genome
classifier, generalized from 2 classes to N families / N variants per family
(families may have different variant counts).
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from moe_dataset import (
    load_sample_manifest, remap_variant_idx, build_global_kmer_vocab,
    FastqKmerDataset, make_balanced_sampler,
)
from moe_model import build_moe, HierarchicalMoEClassifier
from training_utils import train_epoch


def evaluate_multiclass(model, loader, device, class_names=None):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in tqdm(loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            all_preds.extend(pred.cpu().tolist())
            all_targets.extend(target.cpu().tolist())

    accuracy = 100.0 * sum(p == t for p, t in zip(all_preds, all_targets)) / max(len(all_targets), 1)
    print(f"Accuracy: {accuracy:.2f}%")
    # Explicit labels covering every expected class index - with as few as ~5
    # isolates for some classes spread across 5 CV folds, a given fold's val split
    # for a family can easily end up containing only one (or a handful of) actual
    # variants. Without `labels=`, sklearn infers the label set from whatever
    # happens to appear in this batch's y_true/y_pred and errors out the moment
    # that's narrower than class_names, instead of just reporting 0 support for
    # the classes that didn't happen to show up.
    labels = list(range(len(class_names))) if class_names is not None else None
    print(classification_report(all_targets, all_preds, labels=labels, target_names=class_names, zero_division=0))
    print(confusion_matrix(all_targets, all_preds, labels=labels))
    return accuracy


def make_loader(rows, kmer_to_idx, vocab_size, k, label_mode, batch_size, balanced, shuffle, num_workers,
                 leaf_index=None, max_reads_per_sample=None, canonical=False):
    if len(rows) == 0:
        return None
    dataset = FastqKmerDataset(rows, kmer_to_idx, vocab_size, k, label_mode=label_mode, leaf_index=leaf_index,
                                max_reads_per_sample=max_reads_per_sample, canonical=canonical)
    sampler = None
    if balanced and len(dataset) > 0:
        labels = {"family": dataset.family_labels, "variant": dataset.variant_labels,
                  "leaf": dataset.leaf_labels}[label_mode]
        sampler = make_balanced_sampler(labels)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                       shuffle=(shuffle and sampler is None), num_workers=num_workers,
                       pin_memory=True, persistent_workers=(num_workers > 0))


def train_stage(model, train_loader, val_loader, device, epochs, lr, patience, class_names, stage_name,
                 clamp_weights=True):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.1)
    criterion = nn.CrossEntropyLoss()

    best_acc, patience_counter = 0.0, 0
    best_state = None

    for epoch in range(epochs):
        print(f"\n[{stage_name}] Epoch {epoch + 1}/{epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device,
                                             clamp_weights=clamp_weights)
        val_acc = evaluate_multiclass(model, val_loader, device, class_names)
        scheduler.step(val_acc)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc, patience_counter = val_acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"[{stage_name}] Early stopping after {patience} epochs without improvement")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_acc


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Stage-wise training for the hierarchical MoE virus classifier")
    parser.add_argument("--sample_manifest", default="data/manifests/sample_manifest.csv")
    parser.add_argument("--model_type", choices=["deep", "shallow"], default="shallow")
    parser.add_argument("--precision", choices=["binary", "float32"], default="binary",
                         help="binary=BinaryMLP (hardware-deployable, default), float32=StandardMLP baseline")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max_kmers", type=int, default=1024)
    parser.add_argument("--canonical", action="store_true",
                         help="Collapse each k-mer with its reverse complement (min of the pair). Reads come\n"
                              "from both strands, so this makes the encoding strand-invariant and halves the\n"
                              "k-mer space for odd k (1024 -> 512 distinct 5-mers). Off by default so earlier\n"
                              "results reproduce unchanged.")
    parser.add_argument("--vocab_cache", default="data/vocab/kmer_vocab.json")
    parser.add_argument("--max_reads_per_sample_for_vocab", type=int, default=500)
    parser.add_argument("--max_reads_per_sample_train", type=int, default=2000,
                         help="Cap reads per sample when building training datasets - without this, huge-genome "
                              "families (e.g. Herpesviridae) can dominate memory and training data by orders of "
                              "magnitude over small-genome families (e.g. Hepadnaviridae) at the same coverage. "
                              "Set to a negative value or 0 to disable (uses all reads, only safe for small runs).")
    parser.add_argument("--router_epochs", type=int, default=30)
    parser.add_argument("--expert_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--router_hidden_sizes", type=int, nargs="+", default=[1024, 128],
                         help="Router hidden layer widths for model_type='shallow'")
    parser.add_argument("--expert_hidden_sizes", type=int, nargs="+", default=[1024, 128],
                         help="Per-expert hidden layer widths for model_type='shallow' (same width for every expert)")
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0,
                         help="DataLoader worker processes. Default 0 (in-process): k-mer vectors are already "
                              "precomputed upfront (see FastqKmerDataset), so __getitem__ is a trivial tensor "
                              "index and multi-worker DataLoaders add only risk here - persistent_workers=True "
                              "processes accumulating across this script's 9 sequential stages (router + 8 "
                              "experts) without being released was observed to exhaust host memory (500GB+) "
                              "well before any single stage's own data would justify it.")
    parser.add_argument("--save_model", default="moe_bnn_best.pth")
    parser.add_argument("--resume_from", default=None,
                         help="Existing MoE checkpoint whose trained stages are reused instead of being "
                              "retrained. Only meaningful together with --only_stages.")
    parser.add_argument("--only_stages", default=None,
                         help="Which stages to (re)train: 'router', 'experts', or 'experts:Fam1,Fam2'. "
                              "Every other stage is restored from --resume_from. Experts are trained on "
                              "their family's TRUE label and never on router output, so an expert stays "
                              "valid under a different router - which is what makes a router-only sweep "
                              "equivalent to full retraining at a fraction of the cost. Default (unset) "
                              "trains the router and every expert from scratch.")
    return parser


def parse_only_stages(spec, family_names):
    """Resolve --only_stages into (train_router, families_to_train).

    Returns (True, all_families) when spec is None, i.e. the historical
    train-everything behaviour.
    """
    if spec is None:
        return True, list(family_names)
    spec = spec.strip()
    if spec == "router":
        return True, []
    if spec == "experts":
        return False, list(family_names)
    if spec.startswith("experts:"):
        wanted = [n.strip() for n in spec[len("experts:"):].split(",") if n.strip()]
        unknown = [n for n in wanted if n not in family_names]
        if unknown:
            raise ValueError(f"--only_stages names unknown families {unknown}; known: {family_names}")
        return False, wanted
    raise ValueError(f"--only_stages must be 'router', 'experts' or 'experts:Fam1,Fam2', got {spec!r}")


def run_training(args):
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
                           if args.device != "auto" else "cpu")
    print(f"Using device: {device}")

    rows = load_sample_manifest(args.sample_manifest)
    if not rows:
        print("No samples found in manifest - run simulate_reads.py first.")
        return None
    remap_variant_idx(rows)

    family_names = sorted({r["family"] for r in rows}, key=lambda f: next(r["family_idx"] for r in rows if r["family"] == f))
    num_families = max(r["family_idx"] for r in rows) + 1
    variant_names_by_family = {}
    for r in rows:
        variant_names_by_family.setdefault(r["family_idx"], {})[r["variant_idx"]] = r["variant"]
    variants_per_family = [len(variant_names_by_family[i]) for i in range(num_families)]

    print(f"{num_families} families, variants/family={variants_per_family}, {len(rows)} samples")

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"] or train_rows
    if not train_rows:
        raise ValueError(
            "No rows with split=='train' in the manifest - check your split/fold configuration "
            "(e.g. run_cv.py needs k_folds>=3: one fold each for test/val leaves none for train at k_folds=2)"
        )

    kmer_to_idx, vocab_size = build_global_kmer_vocab(
        train_rows, args.k, args.max_kmers,
        max_reads_per_sample=args.max_reads_per_sample_for_vocab,
        cache_path=args.vocab_cache,
        canonical=getattr(args, "canonical", False),
    )
    print(f"Shared vocabulary size: {vocab_size}")

    router, experts = build_moe(num_families, variants_per_family, vocab_size, args.model_type,
                                 tuple(args.router_hidden_sizes), tuple(args.expert_hidden_sizes), args.precision)
    moe = HierarchicalMoEClassifier(router, experts).to(device)
    clamp_weights = (args.precision == "binary")
    max_reads_cap = args.max_reads_per_sample_train if args.max_reads_per_sample_train > 0 else None

    train_router, families_to_train = parse_only_stages(
        getattr(args, "only_stages", None), family_names)
    resume_from = getattr(args, "resume_from", None)

    # Stages that are not being retrained are restored from an existing checkpoint.
    # The router is deliberately NOT restored when it is about to be retrained, since
    # a new width makes the saved state dict the wrong shape.
    prior_router_acc, prior_expert_accs = None, {}
    if resume_from:
        prior = torch.load(resume_from, map_location=device, weights_only=False)
        if prior["family_names"] != family_names:
            raise ValueError(f"--resume_from has family_names {prior['family_names']} but this manifest "
                              f"yields {family_names}; the checkpoints are not compatible")
        prior_expert_accs = dict(prior.get("expert_val_accs", {}))
        prior_router_acc = prior.get("router_val_acc")
        if not train_router:
            moe.router.load_state_dict(prior["router_state_dict"])
            print(f"Restored router from {resume_from}")
        for family_idx, family in enumerate(family_names):
            if family not in families_to_train:
                moe.experts[family_idx].load_state_dict(prior["expert_state_dicts"][family_idx])
                print(f"Restored expert[{family}] from {resume_from}")
    elif getattr(args, "only_stages", None):
        raise ValueError("--only_stages requires --resume_from, otherwise the stages that are not "
                          "retrained would be left randomly initialized")

    # ---- Stage 1: router ----
    router_acc = prior_router_acc
    if train_router:
        router_train_loader = make_loader(train_rows, kmer_to_idx, vocab_size, args.k, "family",
                                           args.batch_size, balanced=True, shuffle=True,
                                           num_workers=args.num_workers, max_reads_per_sample=max_reads_cap,
                                       canonical=getattr(args, 'canonical', False))
        router_val_loader = make_loader(val_rows, kmer_to_idx, vocab_size, args.k, "family",
                                         args.batch_size, balanced=False, shuffle=False,
                                         num_workers=args.num_workers, max_reads_per_sample=max_reads_cap,
                                       canonical=getattr(args, 'canonical', False))

        router_acc = train_stage(moe.router, router_train_loader, router_val_loader, device,
                                  args.router_epochs, args.lr, args.patience, family_names, "router",
                                  clamp_weights=clamp_weights)

    # ---- Freeze router, train experts ----
    moe.freeze_router()

    expert_accs = dict(prior_expert_accs)
    for family_idx in range(num_families):
        if family_names[family_idx] not in families_to_train:
            continue
        family_train_rows = [r for r in train_rows if r["family_idx"] == family_idx]
        family_val_rows = [r for r in val_rows if r["family_idx"] == family_idx] or family_train_rows
        if not family_train_rows:
            print(f"No training reads for family_idx={family_idx}, skipping expert")
            continue

        expert_train_loader = make_loader(family_train_rows, kmer_to_idx, vocab_size, args.k, "variant",
                                           args.batch_size, balanced=True, shuffle=True,
                                           num_workers=args.num_workers, max_reads_per_sample=max_reads_cap,
                                       canonical=getattr(args, 'canonical', False))
        expert_val_loader = make_loader(family_val_rows, kmer_to_idx, vocab_size, args.k, "variant",
                                         args.batch_size, balanced=False, shuffle=False,
                                         num_workers=args.num_workers, max_reads_per_sample=max_reads_cap,
                                       canonical=getattr(args, 'canonical', False))

        variant_names = [variant_names_by_family[family_idx][i]
                          for i in sorted(variant_names_by_family[family_idx])]
        acc = train_stage(moe.experts[family_idx], expert_train_loader, expert_val_loader, device,
                           args.expert_epochs, args.lr, args.patience, variant_names,
                           f"expert[{family_names[family_idx]}]", clamp_weights=clamp_weights)
        expert_accs[family_names[family_idx]] = acc

    # ---- Save combined checkpoint ----
    # A restored stage keeps the shape it was actually trained with, not whatever
    # the corresponding CLI flag happens to hold: evaluate_moe.load_checkpoint
    # rebuilds each stage from these keys, so recording the flag instead of the
    # restored shape would make the checkpoint unloadable.
    saved_router_sizes = tuple(args.router_hidden_sizes)
    saved_expert_sizes = tuple(args.expert_hidden_sizes)
    if resume_from:
        if not train_router:
            saved_router_sizes = tuple(prior.get("router_hidden_sizes", saved_router_sizes))
        if not families_to_train:
            saved_expert_sizes = tuple(prior.get("expert_hidden_sizes", saved_expert_sizes))
        elif len(families_to_train) != len(family_names):
            # Mixed retrained/restored experts would need per-expert shapes, which the
            # single-tuple checkpoint schema cannot express; refuse rather than lie.
            if tuple(prior.get("expert_hidden_sizes", saved_expert_sizes)) != saved_expert_sizes:
                raise ValueError(
                    "Retraining a subset of experts at a different width than the restored ones "
                    "cannot be represented in the checkpoint, which stores one expert shape for "
                    "all families. Retrain every expert, or keep --expert_hidden_sizes equal to "
                    f"the checkpoint's {tuple(prior.get('expert_hidden_sizes'))}."
                )

    checkpoint = {
        "router_state_dict": moe.router.state_dict(),
        "expert_state_dicts": [expert.state_dict() for expert in moe.experts],
        "family_names": family_names,
        "variant_names_by_family": {family_names[i]: [variant_names_by_family[i][j] for j in sorted(variant_names_by_family[i])]
                                     for i in variant_names_by_family},
        "kmer_vocab": kmer_to_idx,
        "vocab_size": vocab_size,
        "k": args.k,
        "max_kmers": args.max_kmers,
        "canonical": getattr(args, "canonical", False),
        "model_type": args.model_type,
        "router_hidden_sizes": saved_router_sizes,
        "expert_hidden_sizes": saved_expert_sizes,
        "precision": args.precision,
        "router_val_acc": router_acc,
        "expert_val_accs": expert_accs,
        "args": vars(args),
    }
    os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
    torch.save(checkpoint, args.save_model)
    print(f"\nSaved MoE checkpoint to {args.save_model}")
    print(f"Router val acc: {router_acc:.2f}%" if router_acc is not None else "Router val acc: (restored)")
    for family, acc in expert_accs.items():
        print(f"  Expert[{family}] val acc: {acc:.2f}%")

    return checkpoint


def main():
    args = build_arg_parser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
