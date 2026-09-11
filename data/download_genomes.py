#!/usr/bin/env python3
"""
Download reference genome(s) per (family, variant) leaf from NCBI, driven by a
manifest CSV (see data/manifests/{pilot,full,generalization}_manifest.csv).

Uses NCBI E-utilities directly (esearch + efetch) over urllib - no extra dependency.
Resolves each variant's accession(s) dynamically via a field-tagged Entrez query
(organism + optional title keyword), rather than hardcoding accessions that can go
stale or be wrong. Each candidate hit is validated against a per-family genome-length
band and a keyword check against the fetched definition line before being accepted,
to reject unrelated matches (e.g. a human mRNA record that happens to share words
with the query). Prints a review table at the end so a human can sanity-check what
was actually fetched before trusting it for training.

--num_isolates 1 (default) reproduces the original single-genome-per-leaf behavior
and output layout exactly. --num_isolates N>1 downloads up to N independent isolates
per leaf (for genome-level generalization testing) and additionally assigns each
isolate a round-robin CV fold (fold = isolate_idx % --k_folds), so every isolate's
simulated reads later land in exactly one fold with no read-level leakage.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
RATE_LIMIT_SECONDS = 0.4  # NCBI allows ~3 req/s without an API key
RETMAX = 10

# Rough expected genome length band per virus family, used to reject obviously
# wrong hits (e.g. a human transcript matching on incidental keywords).
FAMILY_LENGTH_BANDS = {
    "Coronaviridae": (25000, 32000),
    "Flaviviridae": (9000, 12000),
    "Filoviridae": (15000, 20000),
    "Hepadnaviridae": (2900, 3500),
    "Retroviridae": (8500, 10500),
    "Herpesviridae": (100000, 250000),
    "Picornaviridae": (6500, 8500),
    "Paramyxoviridae": (13000, 19000),
}

STOPWORDS = {"virus", "human", "the", "a", "of"}


def _get(url, params):
    query = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{url}?{query}", timeout=30) as resp:
        return resp.read()


def esearch_nuccore(term, retmax=RETMAX):
    data = _get(f"{EUTILS}/esearch.fcgi", {
        "db": "nuccore",
        "term": term,
        "retmax": retmax,
        "retmode": "json",
    })
    result = json.loads(data)["esearchresult"]
    return result.get("idlist", [])


def efetch_fasta(seq_id):
    data = _get(f"{EUTILS}/efetch.fcgi", {
        "db": "nuccore",
        "id": seq_id,
        "rettype": "fasta",
        "retmode": "text",
    })
    return data.decode("utf-8")


def build_query(organism, title_keyword, restrict_refseq, require_complete_genome=False):
    term = f'"{organism}"[Organism]'
    if title_keyword:
        # " OR " in title_keyword means "any of these named strains/clones", used where
        # a class must stay pinned to a specific named lineage (so a bare organism query
        # would pull in other classes too) but a single clone name is too narrow to find
        # more than one real isolate (e.g. HIV-1 subtype B is only ever labeled by strain
        # name, never literally "subtype B", so it needs several known strain names ORed
        # together rather than one).
        if " OR " in title_keyword:
            alts = [a.strip() for a in title_keyword.split(" OR ")]
            term += " AND (" + " OR ".join(f'"{a}"[Title]' for a in alts) + ")"
        else:
            term += f' AND "{title_keyword}"[Title]'
    if require_complete_genome:
        term += ' AND "complete genome"[Title]'
    if restrict_refseq:
        term += " AND srcdb_refseq[prop]"
    return term


def significant_words(text):
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS]


def contains_word(haystack_lower, word):
    """Whole-word match so e.g. 'coronavirus' doesn't match inside 'alphacoronavirus'."""
    return re.search(rf"\b{re.escape(word)}\b", haystack_lower) is not None


# ICTV has renamed several human-pathogen species since these organism strings
# were chosen, and current GenBank submissions use the new name while older
# records and some depositors still use the old one. A plain whole-word match
# on the legacy word therefore misses every modern record. Each entry here is a
# confirmed 1:1 species rename (not a genus grouping spanning multiple species,
# which is why this is safe unlike accepting e.g. any "coronavirus" genus
# prefix): Herpesviridae species each gained a subfamily prefix (e.g. "Human
# herpesvirus 1" -> "Human alphaherpesvirus 1"), Respirovirus species dropped
# "parainfluenza virus" for "respirovirus" (e.g. "Human parainfluenza virus 1"
# -> "Human respirovirus 1"), and Nipah virus was renamed to the binomial
# "Henipavirus nipahense" - found by noticing a live query returned far more
# hits (89) than our own validated result count (4), almost all from a wave of
# 2023-2025 Bangladesh outbreak sequences using the new name.
ORGANISM_WORD_RENAMES = {
    "herpesvirus": [r"(alpha|beta|gamma)herpesvirus"],
    "parainfluenza": [r"respirovirus"],
    "nipah": [r"nipahense"],
}


def contains_organism_word(haystack_lower, word):
    if contains_word(haystack_lower, word):
        return True
    for pattern in ORGANISM_WORD_RENAMES.get(word, []):
        if re.search(rf"\b{pattern}\b", haystack_lower) is not None:
            return True
    return False


def passes_validation(defline, sequence, organism, title_keyword, length_band):
    # The [Organism] field tag in the query is NOT fully reliable on its own - NCBI's
    # taxonomy matching can pull in a related-but-wrong species (e.g. it returned a
    # camel coronavirus for a "Human coronavirus 229E" organism query). So:
    #  - if a title_keyword is given (e.g. a specific strain/isolate/genotype name),
    #    that's strong, specific evidence on its own - require it and skip the organism
    #    check (many correct records use an abbreviated common name like "HIV-1" or
    #    "SARS coronavirus" that wouldn't textually contain the formal organism name).
    #  - otherwise, fall back to requiring at least one significant organism word in
    #    the defline, as a safety net against unrelated-species contamination.
    if length_band is not None:
        lo, hi = length_band
        if not (lo <= len(sequence) <= hi):
            return False, f"length {len(sequence)} outside expected band [{lo}, {hi}]"

    defline_lower = defline.lower()

    if title_keyword:
        # " OR " means "matches any one of these named strains" - see build_query.
        # A record only needs to satisfy one alternative in full, not all of them.
        # Alternatives are matched whole-word but WITHOUT significant_words' length
        # filter: strain codes like "RF" or "MN" are exactly 2 characters, and
        # significant_words would silently drop them, making `all()` over an empty
        # list vacuously true - i.e. matching every record instead of none.
        if " OR " in title_keyword:
            alts = [a.strip() for a in title_keyword.split(" OR ")]
            def alt_matches(alt):
                words = re.findall(r"[A-Za-z0-9]+", alt.lower())
                return bool(words) and all(contains_word(defline_lower, w) for w in words)
            if not any(alt_matches(alt) for alt in alts):
                return False, f"defline matches none of the alternatives: {alts}"
            return True, ""
        missing_kw = [w for w in significant_words(title_keyword) if not contains_word(defline_lower, w)]
        if missing_kw:
            return False, f"defline missing title keyword(s): {missing_kw}"
        return True, ""

    required = significant_words(organism)
    if required and not any(contains_organism_word(defline_lower, w) for w in required):
        return False, f"defline contains none of organism keyword(s): {required}"

    return True, ""


def parse_single_record(fasta_text):
    """Split a FASTA text blob into (defline, sequence). Errors if not exactly one record."""
    records = fasta_text.strip().split(">")
    records = [r for r in records if r.strip()]
    if len(records) != 1:
        raise ValueError(f"expected exactly 1 FASTA record, got {len(records)}")
    lines = records[0].splitlines()
    defline = lines[0]
    sequence = "".join(lines[1:]).upper()
    return defline, sequence


def base_accession(defline):
    """First whitespace token of the defline, with any trailing version suffix stripped."""
    accession = defline.split()[0]
    return accession.rsplit(".", 1)[0] if "." in accession else accession


def resolve_and_fetch_multiple(organism, title_keyword, length_band, num_isolates, retmax=None):
    """
    Search NCBI (RefSeq-restricted first, then unrestricted GenBank), returning up
    to num_isolates independent validated hits: (seq_id, defline, sequence) tuples,
    deduped by seq_id, by version-stripped base accession (so a resubmission of the
    same genome doesn't consume a second isolate slot), and by exact sequence hash
    (belt-and-suspenders against duplicate submissions under different accessions).
    Returns (hits, rejected_notes).
    """
    if retmax is None:
        retmax = max(50, num_isolates * 5)

    hits = []
    rejected_notes = []
    seen_seq_ids, seen_base_accessions, seen_hashes = set(), set(), set()

    # Three stages, most selective first. The middle stage matters: a bare organism
    # query commonly returns thousands of hits sorted by recency, so the top
    # `retmax` IDs can be entirely patent sequences and short gene fragments that
    # exhaust the budget before the length-band filter admits a second genome.
    # Requiring "complete genome" in the title fixes that for the common case
    # without narrowing results for organisms whose real hits already carry a more
    # specific title_keyword.
    stages = [
        dict(restrict_refseq=True, require_complete_genome=False),
        dict(restrict_refseq=False, require_complete_genome=True),
        dict(restrict_refseq=False, require_complete_genome=False),
    ]
    for stage in stages:
        if len(hits) >= num_isolates:
            break
        term = build_query(organism, title_keyword, **stage)
        ids = esearch_nuccore(term, retmax=retmax)
        time.sleep(RATE_LIMIT_SECONDS)

        for seq_id in ids:
            if len(hits) >= num_isolates:
                break
            if seq_id in seen_seq_ids:
                continue
            seen_seq_ids.add(seq_id)

            fasta_text = efetch_fasta(seq_id)
            time.sleep(RATE_LIMIT_SECONDS)
            try:
                defline, sequence = parse_single_record(fasta_text)
            except ValueError as e:
                rejected_notes.append(f"{seq_id}: {e}")
                continue

            acc = base_accession(defline)
            seq_hash = hashlib.sha256(sequence.encode()).hexdigest()
            if acc in seen_base_accessions:
                rejected_notes.append(f"{seq_id}: duplicate isolate (accession {acc} already collected)")
                continue
            if seq_hash in seen_hashes:
                rejected_notes.append(f"{seq_id}: duplicate isolate (identical sequence already collected)")
                continue

            ok, reason = passes_validation(defline, sequence, organism, title_keyword, length_band)
            if not ok:
                rejected_notes.append(f"{seq_id}: {reason}")
                continue

            seen_base_accessions.add(acc)
            seen_hashes.add(seq_hash)
            hits.append((seq_id, defline, sequence))

    return hits, rejected_notes


def resolve_and_fetch(organism, title_keyword, length_band):
    """Single-isolate convenience wrapper (original behavior)."""
    hits, rejected_notes = resolve_and_fetch_multiple(organism, title_keyword, length_band, num_isolates=1)
    if hits:
        seq_id, defline, sequence = hits[0]
        return seq_id, defline, sequence, ""
    return None, None, None, "; ".join(rejected_notes) or "no hits found"


def main():
    parser = argparse.ArgumentParser(description="Download reference genomes for the MoE virus classifier")
    parser.add_argument("--manifest", default="data/manifests/full_manifest.csv",
                         help="CSV with columns family,family_idx,variant,variant_idx,organism,title_keyword")
    parser.add_argument("--out_dir", default="data/refs",
                         help="Output directory for per-(family,variant) FASTA files")
    parser.add_argument("--genome_manifest_out", default="data/manifests/genome_manifest.csv",
                         help="Where to write the resolved manifest (with accessions/lengths)")
    parser.add_argument("--num_isolates", type=int, default=1,
                         help="Independent isolates to download per leaf (1 = original single-genome behavior)")
    parser.add_argument("--k_folds", type=int, default=5,
                         help="CV fold count for isolate-level fold assignment (only used if num_isolates>1)")
    args = parser.parse_args()

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))

    multi = args.num_isolates > 1
    results = []
    for row in rows:
        family, variant = row["family"], row["variant"]
        organism, title_keyword = row["organism"], row.get("title_keyword", "").strip()
        length_band = FAMILY_LENGTH_BANDS.get(family)

        print(f"[{family}/{variant}] organism={organism!r} title_keyword={title_keyword!r} "
              f"num_isolates={args.num_isolates}")
        hits, rejected_notes = resolve_and_fetch_multiple(organism, title_keyword, length_band, args.num_isolates)

        if not hits:
            note = "; ".join(rejected_notes) or "no hits found"
            print(f"  ! no validated hits ({note}), skipping")
            base_row = {**row, "accession": "", "length": 0, "defline": "", "fasta_path": "", "note": note}
            results.append({**base_row, "isolate_idx": 0, "fold": 0} if multi else base_row)
            continue

        if len(hits) < args.num_isolates:
            print(f"  ! only found {len(hits)}/{args.num_isolates} validated isolates")

        for isolate_idx, (seq_id, defline, sequence) in enumerate(hits):
            if multi:
                out_path = os.path.join(args.out_dir, family, variant, f"isolate_{isolate_idx}.fasta")
            else:
                out_path = os.path.join(args.out_dir, family, f"{variant}.fasta")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            with open(out_path, "w") as out_f:
                out_f.write(f">{defline}\n{sequence}\n")

            print(f"  -> [{isolate_idx}] {seq_id}  ({len(sequence)} bp)  {defline[:80]}")
            result_row = {
                **row,
                "accession": seq_id,
                "length": len(sequence),
                "defline": defline,
                "fasta_path": out_path,
                "note": "",
            }
            if multi:
                result_row["isolate_idx"] = isolate_idx
                result_row["fold"] = isolate_idx % args.k_folds
            results.append(result_row)

    os.makedirs(os.path.dirname(args.genome_manifest_out), exist_ok=True)
    fieldnames = ["family", "family_idx", "variant", "variant_idx", "organism", "title_keyword",
                  "accession", "length", "defline", "fasta_path", "note"]
    if multi:
        fieldnames += ["isolate_idx", "fold"]
    with open(args.genome_manifest_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResolved genome manifest written to {args.genome_manifest_out}")
    n_ok = sum(1 for r in results if r["accession"])
    n_leaves = len(rows)
    print(f"{n_ok}/{len(results)} isolate downloads succeeded, across {n_leaves} leaves")
    if multi:
        per_leaf_counts = {}
        for r in results:
            if r["accession"]:
                per_leaf_counts[(r["family"], r["variant"])] = per_leaf_counts.get((r["family"], r["variant"]), 0) + 1
        thin = {k: v for k, v in per_leaf_counts.items() if v < args.k_folds}
        if thin:
            print(f"Leaves with fewer isolates than k_folds={args.k_folds} (some CV folds will have 0 test "
                  f"isolates for these): {thin}")
    if n_ok < len(results):
        print("Review rows with an empty 'accession' column above (see 'note' column) and fix the organism/title_keyword, then re-run.")


if __name__ == "__main__":
    main()
