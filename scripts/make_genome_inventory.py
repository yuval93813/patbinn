#!/usr/bin/env python3
"""
Regenerate docs/GENOME_INVENTORY.txt from the genome manifest.

The inventory is the provenance record for the dataset: every NCBI accession
actually used, grouped by family and variant, with the CV fold each isolate was
assigned to. It is derived entirely from the manifest, so it can be rebuilt at
any time and never has to be edited by hand.

Run from the repository root:

    python3 scripts/make_genome_inventory.py
"""

import argparse
import collections
import csv
import datetime
import os

RULE = "=" * 70
THIN = "-" * 70


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--genome_manifest",
                   default="data/manifests/genome_manifest_multi.csv")
    p.add_argument("--out", default="docs/GENOME_INVENTORY.txt")
    args = p.parse_args()

    rows = load(args.genome_manifest)
    for r in rows:
        r["isolate_idx"] = int(r["isolate_idx"])

    # family -> variant -> [rows], each preserving manifest order
    by_family = collections.OrderedDict()
    for r in rows:
        by_family.setdefault(r["family"], collections.OrderedDict()) \
                 .setdefault(r["variant"], []).append(r)

    n_classes = sum(len(v) for v in by_family.values())
    per_class = collections.Counter((r["family"], r["variant"]) for r in rows)
    singles = sorted(k for k, v in per_class.items() if v == 1)

    out = []
    out.append("PatBiNN - Genome Inventory")
    out.append(RULE)
    out.append(f"Generated:  {datetime.date.today().isoformat()}")
    out.append(f"Source:     {args.genome_manifest}")
    out.append(f"Regenerate: python3 scripts/make_genome_inventory.py")
    out.append("")
    out.append(f"Total genomes:  {len(rows)}")
    out.append(f"Total classes:  {n_classes}  (family/variant pairs)")
    out.append(f"Families:       {len(by_family)}")
    out.append(f"Isolates/class: min {min(per_class.values())}, "
               f"max {max(per_class.values())}, "
               f"mean {len(rows)/n_classes:.1f}")
    out.append("")
    out.append("All genomes were retrieved from the NCBI nucleotide database via")
    out.append("data/download_genomes.py, using organism- and length-band-constrained")
    out.append("Entrez queries with title/defline validation (see that file's docstring")
    out.append("for the exact retrieval and deduplication logic).")
    out.append("")
    out.append("Folds are assigned at download time as fold = isolate_idx % K, so every")
    out.append("isolate of a class is spread across folds and no genome can appear in")
    out.append("both the training and the test side of a split.")
    out.append("")
    if singles:
        out.append(f"Classes represented by a single isolate ({len(singles)}):")
        for fam, var in singles:
            out.append(f"  - {fam} / {var}")
    else:
        out.append("Every class is represented by more than one independent isolate,")
        out.append("so each class receives a genuine held-out-genome test in every fold.")
    out.append("")
    out.append(RULE)
    out.append("")

    for family, variants in by_family.items():
        n_fam = sum(len(v) for v in variants.values())
        out.append(f"## {family}  ({len(variants)} variants, {n_fam} genomes)")
        out.append(THIN)
        for variant, entries in variants.items():
            out.append(f"  {variant}  ({len(entries)} isolates)")
            for r in sorted(entries, key=lambda x: x["isolate_idx"]):
                # The manifest's `accession` column holds the NCBI UID; the
                # citable accession.version is the first token of the defline.
                defline = r["defline"].strip()
                acc, _, title = defline.partition(" ")
                out.append(f"    [{r['isolate_idx']}] fold={r['fold']}  "
                           f"{acc:<14} {r['length']:>7} bp  uid={r['accession']}")
                out.append(f"          {title}")
            out.append("")
        out.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(out).rstrip() + "\n")

    print(f"Wrote {args.out}: {len(rows)} genomes, {n_classes} classes, "
          f"{len(by_family)} families")


if __name__ == "__main__":
    main()
