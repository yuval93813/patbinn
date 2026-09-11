#!/usr/bin/env python3
"""
Compare PatBiNN against Kraken2 on identical reads across a sequencing-error sweep.

Fairness of the comparison rests on three points:

  * Kraken2 indexes exactly the genomes PatBiNN was allowed to learn from - every
    genome outside the held-out test fold - so neither method has seen a test
    isolate.
  * Both are scored on the same held-out reads, corrupted by the same function
    (`kmer_encoding.inject_read_errors`) with the same seed, so the two methods
    see byte-identical input at every error rate.
  * Kraken2 assigns to the lowest common ancestor, so a read may be placed
    correctly but less specifically (at the family node rather than the variant).
    That is counted separately from an outright miss rather than being scored as
    an error, which would understate Kraken2.

A minimal two-level taxonomy is synthesised (root -> family -> variant) because
the classes here are our own label set rather than NCBI taxids.

Stage 1 (--build) constructs the database; stage 2 (--classify) runs the sweep.
"""

import argparse
import csv
import os
import random
import shutil
import subprocess
from collections import Counter, defaultdict

from kmer_encoding import parse_fasta, parse_fastq, inject_read_errors

ROOT_TAXID = 1
FAMILY_TAXID_BASE = 10
VARIANT_TAXID_BASE = 1000


def load_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def taxonomy_ids(genome_rows):
    """Stable (family -> taxid) and ((family,variant) -> taxid) maps."""
    families = sorted({r["family"] for r in genome_rows})
    fam_taxid = {f: FAMILY_TAXID_BASE + i for i, f in enumerate(families)}
    variants = sorted({(r["family"], r["variant"]) for r in genome_rows})
    var_taxid = {fv: VARIANT_TAXID_BASE + i for i, fv in enumerate(variants)}
    return fam_taxid, var_taxid


def build_database(args):
    genome_rows = load_manifest(args.genome_manifest)
    ref_rows = [r for r in genome_rows if int(r["fold"]) != args.test_fold]
    print(f"Reference genomes (folds != {args.test_fold}): {len(ref_rows)} of {len(genome_rows)}")

    fam_taxid, var_taxid = taxonomy_ids(genome_rows)
    db = args.db
    os.makedirs(os.path.join(db, "taxonomy"), exist_ok=True)
    os.makedirs(os.path.join(db, "library"), exist_ok=True)

    # --- synthetic taxonomy: root -> family -> variant ---
    with open(os.path.join(db, "taxonomy", "nodes.dmp"), "w") as f:
        f.write(f"{ROOT_TAXID}\t|\t{ROOT_TAXID}\t|\tno rank\t|\n")
        for fam, t in fam_taxid.items():
            f.write(f"{t}\t|\t{ROOT_TAXID}\t|\tfamily\t|\n")
        for (fam, var), t in var_taxid.items():
            f.write(f"{t}\t|\t{fam_taxid[fam]}\t|\tspecies\t|\n")
    with open(os.path.join(db, "taxonomy", "names.dmp"), "w") as f:
        f.write(f"{ROOT_TAXID}\t|\troot\t|\t\t|\tscientific name\t|\n")
        for fam, t in fam_taxid.items():
            f.write(f"{t}\t|\t{fam}\t|\t\t|\tscientific name\t|\n")
        for (fam, var), t in var_taxid.items():
            f.write(f"{t}\t|\t{fam}_{var}\t|\t\t|\tscientific name\t|\n")

    # --- library: one record per reference genome, tagged with its variant taxid ---
    # Written to a staging file and registered with --add-to-library rather than
    # dropped into library/ directly: kraken2-build derives its seqid->taxid
    # prelim_map from that step, and without it the build aborts.
    staged = os.path.join(args.workdir, "reference.fna")
    os.makedirs(args.workdir, exist_ok=True)
    n = 0
    with open(staged, "w") as out:
        for r in ref_rows:
            seqs = parse_fasta(r["fasta_path"])
            if not seqs:
                print(f"  WARNING: no sequence in {r['fasta_path']}")
                continue
            t = var_taxid[(r["family"], r["variant"])]
            out.write(f">{r['family']}_{r['variant']}_iso{r['isolate_idx']}|kraken:taxid|{t}\n")
            out.write("".join(seqs) + "\n")
            n += 1
    print(f"Staged {n} reference sequences at {staged}")

    subprocess.run(["kraken2-build", "--add-to-library", staged, "--db", db,
                    "--threads", str(args.threads)], check=True)
    subprocess.run(["kraken2-build", "--build", "--db", db,
                    "--threads", str(args.threads)], check=True)
    print(f"Database built at {db}")


def parse_kraken_output(path):
    """read_id -> assigned taxid (0 when unclassified)."""
    assigned = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            assigned[parts[1]] = int(parts[2])
    return assigned


def classify(args):
    genome_rows = load_manifest(args.genome_manifest)
    fam_taxid, var_taxid = taxonomy_ids(genome_rows)
    taxid_family = {t: f for f, t in fam_taxid.items()}
    taxid_variant = {t: fv for fv, t in var_taxid.items()}
    variant_family_taxid = {t: fam_taxid[fv[0]] for fv, t in var_taxid.items()}

    sample_rows = [r for r in load_manifest(args.sample_manifest) if r["split"] == args.split]
    print(f"Test samples (split={args.split}): {len(sample_rows)}")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    results = []

    for rate in args.error_rates:
        rng = random.Random(args.seed)
        tmp_fa = os.path.join(args.workdir, f"reads_{rate:.2f}.fa")
        os.makedirs(args.workdir, exist_ok=True)

        # read_id -> (true variant taxid, sample_id)
        truth, per_sample = {}, defaultdict(list)
        with open(tmp_fa, "w") as out:
            for row in sample_rows:
                seqs = parse_fastq(row["r1_path"]) + parse_fastq(row["r2_path"])
                if args.max_reads_per_sample:
                    seqs = seqs[:args.max_reads_per_sample]
                t = var_taxid[(row["family"], row["variant"])]
                for i, s in enumerate(seqs):
                    rid = f"{row['sample_id']}#{i}"
                    out.write(f">{rid}\n{inject_read_errors(s, rate, rng)}\n")
                    truth[rid] = t
                    per_sample[row["sample_id"]].append(rid)

        out_txt = os.path.join(args.workdir, f"k2_{rate:.2f}.txt")
        subprocess.run(["kraken2", "--db", args.db, "--threads", str(args.threads),
                        "--output", out_txt, "--report", out_txt + ".report", tmp_fa],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assigned = parse_kraken_output(out_txt)

        exact = family_only = unclassified = wrong = 0
        for rid, true_t in truth.items():
            a = assigned.get(rid, 0)
            if a == 0:
                unclassified += 1
            elif a == true_t:
                exact += 1
            elif (a in taxid_family and a == variant_family_taxid[true_t]) or \
                 (a in taxid_variant and variant_family_taxid.get(a) == variant_family_taxid[true_t]):
                family_only += 1          # right family, not the right variant
            else:
                wrong += 1
        total = max(len(truth), 1)

        # per-sample majority vote over classified reads only
        sample_exact = 0
        for sid, rids in per_sample.items():
            votes = [assigned.get(r, 0) for r in rids]
            votes = [v for v in votes if v != 0]
            if votes and Counter(votes).most_common(1)[0][0] == truth[rids[0]]:
                sample_exact += 1

        row = {
            "error_rate": rate,
            "n_reads": total,
            "per_read_variant_acc": 100.0 * exact / total,
            "per_read_family_acc": 100.0 * (exact + family_only) / total,
            "pct_unclassified": 100.0 * unclassified / total,
            "pct_wrong": 100.0 * wrong / total,
            "sample_variant_acc": 100.0 * sample_exact / max(len(per_sample), 1),
        }
        results.append(row)
        print(f"error_rate={rate:.2f}  per-read variant {row['per_read_variant_acc']:.2f}%  "
              f"family {row['per_read_family_acc']:.2f}%  unclassified {row['pct_unclassified']:.2f}%  "
              f"sample {row['sample_variant_acc']:.2f}%")
        os.remove(tmp_fa)

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {args.out_csv}")


def main():
    p = argparse.ArgumentParser(description="Kraken2 baseline across a sequencing-error sweep")
    p.add_argument("--build", action="store_true", help="Build the database, then exit")
    p.add_argument("--classify", action="store_true", help="Run the error sweep")
    p.add_argument("--db", default="data/kraken2_db_fold1")
    p.add_argument("--genome_manifest", default="data/manifests/genome_manifest_multi.csv")
    p.add_argument("--sample_manifest", default="data/manifests/cv_fold1_sample_manifest.csv")
    p.add_argument("--split", default="test")
    p.add_argument("--test_fold", type=int, default=1,
                    help="Fold excluded from the reference database - must match the fold PatBiNN was tested on")
    p.add_argument("--error_rates", type=float, nargs="+",
                    default=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
    p.add_argument("--max_reads_per_sample", type=int, default=200,
                    help="Must match the PatBiNN error sweep for the two to be comparable")
    p.add_argument("--seed", type=int, default=0, help="Must match the PatBiNN error sweep")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--workdir", default="/tmp/kraken2_work")
    p.add_argument("--out_csv", default="results/kraken2_error_sweep.csv")
    args = p.parse_args()

    if args.build:
        build_database(args)
    if args.classify:
        classify(args)
    if not (args.build or args.classify):
        p.error("pass --build and/or --classify")


if __name__ == "__main__":
    main()
