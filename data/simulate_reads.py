#!/usr/bin/env python3
"""
Simulate paired-end Illumina reads from downloaded reference genomes using ART
(art_illumina), one or more independent replicates per genome.

Two modes, auto-detected from the input genome manifest's columns:

- Single-isolate (genome_manifest.csv, no isolate_idx/fold columns): each replicate
  is an independent simulation run (distinct --rndSeed) and becomes one "sample" for
  per-sample majority-vote evaluation. Splitting happens at the REPLICATE level
  (train/val/test via assign_split) - held-out samples differ from train samples
  only in simulated sequencing noise, since there's one genome per leaf.

- Multi-isolate (genome_manifest_multi.csv, has isolate_idx/fold columns from
  download_genomes.py --num_isolates>1): writes a separate cv_sample_manifest.csv
  carrying isolate_idx/fold through instead of a train/val/test split column (that's
  a different, orthogonal concept - assigned later per-fold by run_cv.py). All
  replicates of a given isolate share that isolate's fold, so genome-level held-out
  evaluation has zero read-level leakage across folds.
"""

import argparse
import csv
import os
import subprocess
import sys


def assign_split(replicate_idx, num_replicates):
    """Deterministic train/val/test split at the replicate level (single-isolate mode)."""
    if num_replicates < 3:
        return "train"
    if replicate_idx == num_replicates - 1:
        return "test"
    if replicate_idx == num_replicates - 2:
        return "val"
    return "train"


def simulate_one(fasta_path, out_prefix, read_length, coverage, seed, profile):
    cmd = [
        "art_illumina",
        "-ss", profile,
        "-i", fasta_path,
        "-p",                       # paired-end
        "-l", str(read_length),
        "-f", str(coverage),
        "-m", str(2 * read_length),  # mean fragment size
        "-s", "10",                  # fragment size std dev
        "-rs", str(seed),
        "-o", out_prefix,
        "-na",                       # no .aln alignment files (not needed downstream)
        "-q",                        # quiet
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Simulate Illumina reads from reference genomes")
    parser.add_argument("--genome_manifest", default="data/manifests/genome_manifest.csv",
                         help="Output of download_genomes.py (single- or multi-isolate)")
    parser.add_argument("--reads_dir", default="data/reads",
                         help="Output directory for simulated FASTQ reads")
    parser.add_argument("--sample_manifest_out", default="data/manifests/sample_manifest.csv")
    parser.add_argument("--num_replicates", type=int, default=6)
    parser.add_argument("--read_length", type=int, default=150)
    parser.add_argument("--coverage", type=float, default=30.0)
    parser.add_argument("--profile", default="HS25", help="ART Illumina error/quality profile")
    parser.add_argument("--seed_base", type=int, default=1000)
    args = parser.parse_args()

    with open(args.genome_manifest, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        genomes = [r for r in reader if r.get("fasta_path")]

    if not genomes:
        print("No genomes with a fasta_path found in the manifest - run download_genomes.py first.", file=sys.stderr)
        sys.exit(1)

    multi = "isolate_idx" in fieldnames

    samples = []
    for genome in genomes:
        family, variant = genome["family"], genome["variant"]
        if multi:
            isolate_idx = int(genome["isolate_idx"])
            out_dir = os.path.join(args.reads_dir, family, variant, f"isolate_{isolate_idx}")
        else:
            out_dir = os.path.join(args.reads_dir, family, variant)
        os.makedirs(out_dir, exist_ok=True)

        for rep in range(args.num_replicates):
            if multi:
                sample_id = f"{family}__{variant}__isolate{isolate_idx}__rep{rep}"
            else:
                sample_id = f"{family}__{variant}__rep{rep}"
            out_prefix = os.path.join(out_dir, f"rep{rep}_")
            seed = args.seed_base + hash(sample_id) % 1_000_000

            print(f"[{sample_id}] simulating {args.coverage}x coverage, {args.read_length}bp PE, seed={seed}")
            simulate_one(genome["fasta_path"], out_prefix, args.read_length, args.coverage, seed, args.profile)

            r1_path = out_prefix + "1.fq"
            r2_path = out_prefix + "2.fq"
            if not (os.path.exists(r1_path) and os.path.exists(r2_path)):
                print(f"  ! expected outputs not found ({r1_path}, {r2_path}), skipping", file=sys.stderr)
                continue

            sample = {
                "sample_id": sample_id,
                "family": family,
                "family_idx": genome["family_idx"],
                "variant": variant,
                "variant_idx": genome["variant_idx"],
                "replicate": rep,
                "r1_path": r1_path,
                "r2_path": r2_path,
            }
            if multi:
                sample["isolate_idx"] = isolate_idx
                sample["fold"] = int(genome["fold"])
            else:
                sample["split"] = assign_split(rep, args.num_replicates)
            samples.append(sample)

    if multi:
        # No-leakage invariant: every row sharing an isolate_idx within a
        # (family,variant) must share the same fold - cheap to check, catches an
        # entire class of read-level-leakage bugs immediately.
        fold_by_isolate = {}
        for s in samples:
            key = (s["family"], s["variant"], s["isolate_idx"])
            if key in fold_by_isolate:
                assert fold_by_isolate[key] == s["fold"], f"inconsistent fold for isolate {key}"
            else:
                fold_by_isolate[key] = s["fold"]

    os.makedirs(os.path.dirname(args.sample_manifest_out), exist_ok=True)
    if multi:
        fieldnames_out = ["sample_id", "family", "family_idx", "variant", "variant_idx",
                           "isolate_idx", "replicate", "r1_path", "r2_path", "fold"]
    else:
        fieldnames_out = ["sample_id", "family", "family_idx", "variant", "variant_idx",
                           "replicate", "r1_path", "r2_path", "split"]
    with open(args.sample_manifest_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_out)
        writer.writeheader()
        writer.writerows(samples)

    print(f"\nSample manifest written to {args.sample_manifest_out}")
    print(f"{len(samples)} samples simulated across {len(genomes)} genomes")
    if multi:
        n_folds = len({s["fold"] for s in samples})
        for fold in sorted({s["fold"] for s in samples}):
            n = sum(1 for s in samples if s["fold"] == fold)
            print(f"  fold {fold}: {n} samples")
    else:
        for split in ("train", "val", "test"):
            n = sum(1 for s in samples if s["split"] == split)
            print(f"  {split}: {n} samples")


if __name__ == "__main__":
    main()
