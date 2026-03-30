#!/usr/bin/env python3
"""
Enrich chunks_cleaned.jsonl with year & journal from all_cmip6_metadata.csv.
Rebuilds text_with_prefix to include (year, journal) in the context header.

Input:  chunks_cleaned.jsonl  +  all_cmip6_metadata.csv
Output: chunks_enriched.jsonl
"""

import csv
import json
import re
from pathlib import Path

RAG_DIR = Path(__file__).parent
CSV_PATH = RAG_DIR.parent / "raw_papers" / "_metadata" / "all_cmip6_metadata.csv"
INPUT_JSONL = RAG_DIR / "chunks_cleaned.jsonl"
OUTPUT_JSONL = RAG_DIR / "chunks_enriched.jsonl"


def normalize_doi(raw_doi: str) -> str:
    """Normalize DOI to the underscore format used in paper_id."""
    return raw_doi.replace("https://doi.org/", "").replace("/", "_").replace(".", "_")


def load_csv_metadata(csv_path: Path) -> dict:
    """Load year & journal keyed by normalized DOI."""
    meta = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = normalize_doi(row.get("doi", ""))
            meta[key] = {
                "year": row.get("publication_year", ""),
                "journal": row.get("journal", ""),
            }
    return meta


def rebuild_prefix(chunk: dict) -> str:
    """Rebuild text_with_prefix with year & journal."""
    title = chunk.get("title", "")
    doi = chunk.get("doi", "")
    section = chunk.get("section_path", "")
    year = chunk.get("year", "")
    journal = chunk.get("journal", "")
    raw = chunk.get("text_raw", "")

    # Build header line: Paper: "Title" (year, journal)
    header = f'Paper: "{title}"'
    if year or journal:
        parts = []
        if year:
            parts.append(str(year))
        if journal:
            parts.append(journal)
        header += f" ({', '.join(parts)})"

    lines = [header, f"DOI: {doi}"]
    if section:
        lines.append(f"Section: {section}")
    lines.append("---")
    lines.append(raw)

    return "\n".join(lines)


def main():
    print(f"Loading CSV metadata from {CSV_PATH}...")
    csv_meta = load_csv_metadata(CSV_PATH)
    print(f"  Loaded {len(csv_meta)} DOI entries.")

    print(f"Processing {INPUT_JSONL}...")
    matched = 0
    unmatched = 0
    total = 0

    with open(INPUT_JSONL, "r", encoding="utf-8") as fin, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

        for line in fin:
            chunk = json.loads(line)
            pid = chunk.get("paper_id", "")
            total += 1

            if pid in csv_meta:
                chunk["year"] = csv_meta[pid]["year"]
                chunk["journal"] = csv_meta[pid]["journal"]
                matched += 1
            else:
                unmatched += 1

            # Rebuild prefix with enriched metadata
            chunk["text_with_prefix"] = rebuild_prefix(chunk)

            fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            if total % 20000 == 0:
                print(f"  {total:,} chunks processed...")

    print(f"\nDone! {total:,} chunks written to {OUTPUT_JSONL}")
    print(f"  Matched: {matched:,} ({100*matched/total:.1f}%)")
    print(f"  Unmatched: {unmatched:,}")


if __name__ == "__main__":
    main()
