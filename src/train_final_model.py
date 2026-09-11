#!/usr/bin/env python3
"""
Train the one actual deployable model: no held-out test fold, since this isn't
trying to measure generalization (that's what run_cv.py's 5-fold CV is for) - it's
producing the classifier you'd actually ship, trained on every real isolate
available.

Split is just train/val (val purely for early stopping, not reported as a result).
VAL_FOLD defaults to 4, deliberately never fold 0: several classes (mostly
Herpesviridae) have exactly one real isolate on NCBI and it always lands in fold 0
(see docs/EXPERIMENTS.md) - if fold 0 were held out for validation, those classes
would get zero training exposure in the one model that matters. Keeping fold 0 in
train guarantees every class the CV proved out gets used, and every thin class
gets whatever training signal its one genome can provide.
"""

import argparse
import csv
import os

from run_cv import load_cv_manifest
from train_moe import build_arg_parser, run_training


def write_final_manifest(rows, val_fold, out_path):
    out_rows = []
    for r in rows:
        split = "val" if r["fold"] == val_fold else "train"
        out_rows.append({**r, "split": split})

    fieldnames = list(rows[0].keys()) + ["split"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    n_train = sum(1 for r in out_rows if r["split"] == "train")
    n_val = sum(1 for r in out_rows if r["split"] == "val")
    print(f"Final model manifest: {n_train} train / {n_val} val samples (no test split) -> {out_path}")


def main():
    parser = build_arg_parser()
    parser.add_argument("--cv_sample_manifest", default="data/manifests/cv_sample_manifest.csv")
    parser.add_argument("--val_fold", type=int, default=4,
                         help="Fold used for early-stopping validation only. Never 0 - that's where every "
                              "single-isolate class's only real genome lives (see module docstring).")
    parser.add_argument("--final_manifest_out", default="data/manifests/final_train_manifest.csv")
    args = parser.parse_args()

    if args.val_fold == 0:
        raise ValueError("val_fold=0 would starve every single-isolate class of training data - pick another fold.")

    cv_rows = load_cv_manifest(args.cv_sample_manifest)
    write_final_manifest(cv_rows, args.val_fold, args.final_manifest_out)

    args.sample_manifest = args.final_manifest_out
    run_training(args)


if __name__ == "__main__":
    main()
