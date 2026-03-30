#!/usr/bin/env python3
"""
License Audit for CMIP6 RAG Corpus
===================================
Checks every paper in the Qdrant index against OpenAlex API
to verify Open Access status and license type.

Produces:
  - rag/license_audit.jsonl  (full audit per paper)
  - rag/license_audit_report.md (summary report)
  - rag/closed_papers.tsv  (papers that are NOT open access)

Usage:
    python rag/audit_licenses.py
"""

import json
import time
import sys
from pathlib import Path
from collections import Counter
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

PAPERS_FILE = Path(__file__).parent / "papers_in_qdrant.jsonl"
OUTPUT_AUDIT = Path(__file__).parent / "license_audit.jsonl"
OUTPUT_CLOSED = Path(__file__).parent / "closed_papers.tsv"
OUTPUT_REPORT = Path(__file__).parent / "license_audit_report.md"

OPENALEX_API = "https://api.openalex.org/works"
BATCH_SIZE = 25  # Smaller batches to avoid connection resets
POLITE_EMAIL = "dmitrii.pantiu@awi.de"  # for polite pool (faster rate limits)
MAX_RETRIES = 3

# Known Open Access publishers/journals (fallback if API fails)
KNOWN_OA_JOURNALS = {
    "nature communications", "scientific reports", "communications earth & environment",
    "npj climate and atmospheric science", "earth system dynamics",
    "geoscientific model development", "atmospheric chemistry and physics",
    "biogeosciences", "earth system science data", "the cryosphere",
    "natural hazards and earth system sciences", "ocean science",
    "frontiers in marine science", "frontiers in earth science",
    "frontiers in environmental science", "frontiers in climate",
    "environmental research letters", "plos one", "plos climate",
    "science advances", "proceedings of the national academy of sciences",
    "journal of advances in modeling earth systems", "earth s future",
    "journal of water and climate change",
}


def load_papers():
    """Load papers from JSONL."""
    papers = []
    with open(PAPERS_FILE) as f:
        for line in f:
            papers.append(json.loads(line))
    return papers


def normalize_doi(doi_str: str) -> str:
    """Convert stored DOI (underscored) to standard format."""
    # Our DOIs are stored like: 10_1038_s41467-023-41105-7
    # Need to convert to: 10.1038/s41467-023-41105-7
    if not doi_str:
        return ""
    # First _ → dot, second _ → slash, rest stay
    parts = doi_str.split("_", 2)
    if len(parts) >= 3:
        return f"{parts[0]}.{parts[1]}/{parts[2]}"
    elif len(parts) == 2:
        return f"{parts[0]}.{parts[1]}"
    return doi_str


def batch_check_openalex(dois: list[str]) -> dict:
    """Query OpenAlex for OA status of multiple DOIs at once."""
    results = {}
    
    # Build filter: doi:doi1|doi2|doi3
    doi_filter = "|".join(dois)
    url = (
        f"{OPENALEX_API}?"
        f"filter=doi:{doi_filter}"
        f"&select=doi,open_access,primary_location,title"
        f"&per_page={len(dois)}"
        f"&mailto={POLITE_EMAIL}"
    )
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            
            for work in data.get("results", []):
                doi = work.get("doi", "").replace("https://doi.org/", "")
                oa = work.get("open_access", {})
                loc = work.get("primary_location", {}) or {}
                source = loc.get("source", {}) or {}
                
                results[doi.lower()] = {
                    "is_oa": oa.get("is_oa", False),
                    "oa_status": oa.get("oa_status", "unknown"),
                    "oa_url": oa.get("oa_url", ""),
                    "license": (loc.get("license") or "unknown"),
                    "publisher": source.get("host_organization_name", ""),
                    "source_type": source.get("type", ""),
                }
            return results
        except (URLError, HTTPError, json.JSONDecodeError, ConnectionError, OSError) as e:
            wait = 2 ** attempt
            print(f"  API error (attempt {attempt}/{MAX_RETRIES}): {e} — retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    
    return results


def main():
    papers = load_papers()
    print(f"Loaded {len(papers)} unique papers from Qdrant")
    
    # Normalize DOIs
    for p in papers:
        p["doi_normalized"] = normalize_doi(p["doi"])
    
    # Batch query OpenAlex
    audit_results = []
    all_oa_info = {}
    
    # Process in batches
    dois_to_check = [p["doi_normalized"] for p in papers if p["doi_normalized"]]
    total_batches = (len(dois_to_check) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"Checking {len(dois_to_check)} DOIs against OpenAlex ({total_batches} batches)...")
    
    for i in range(0, len(dois_to_check), BATCH_SIZE):
        batch = dois_to_check[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        
        if batch_num % 10 == 0 or batch_num == 1:
            print(f"  Batch {batch_num}/{total_batches}...")
        
        results = batch_check_openalex(batch)
        all_oa_info.update(results)
        
        # Polite rate limiting (OpenAlex polite pool: 10 req/s)
        time.sleep(0.15)
    
    print(f"Got OA info for {len(all_oa_info)}/{len(dois_to_check)} DOIs")
    
    # Merge results
    closed_papers = []
    stats = Counter()
    license_stats = Counter()
    
    with open(OUTPUT_AUDIT, "w") as f_audit:
        for p in papers:
            doi_norm = p["doi_normalized"].lower()
            oa_info = all_oa_info.get(doi_norm, {})
            
            # Determine status
            is_oa = oa_info.get("is_oa", None)
            oa_status = oa_info.get("oa_status", "not_found")
            license_type = oa_info.get("license", "unknown")
            
            # Fallback: check known OA journals
            journal_lower = p.get("journal", "").lower()
            is_known_oa = journal_lower in KNOWN_OA_JOURNALS
            
            if is_oa is None:
                # Not found in OpenAlex
                if is_known_oa:
                    final_status = "oa_by_journal"
                else:
                    final_status = "not_found"
            elif is_oa:
                final_status = oa_status  # gold, green, hybrid, bronze
            else:
                if is_known_oa:
                    final_status = "oa_by_journal_override"
                else:
                    final_status = "closed"
            
            result = {
                **p,
                "is_oa": is_oa,
                "oa_status": oa_status,
                "final_status": final_status,
                "license": license_type,
                "publisher": oa_info.get("publisher", ""),
                "is_known_oa_journal": is_known_oa,
            }
            
            f_audit.write(json.dumps(result, ensure_ascii=False) + "\n")
            stats[final_status] += 1
            license_stats[license_type] += 1
            
            if final_status == "closed":
                closed_papers.append(result)
    
    # Write closed papers
    with open(OUTPUT_CLOSED, "w") as f:
        f.write("DOI\tTitle\tJournal\tYear\tOA_Status\tLicense\tPublisher\n")
        for p in sorted(closed_papers, key=lambda x: x.get("journal", "")):
            f.write(f"{p['doi_normalized']}\t{p['title']}\t{p.get('journal','')}\t"
                    f"{p.get('year','')}\t{p['oa_status']}\t{p['license']}\t"
                    f"{p.get('publisher','')}\n")
    
    # Generate report
    total = len(papers)
    with open(OUTPUT_REPORT, "w") as f:
        f.write("# CMIP6 RAG License Audit Report\n\n")
        f.write(f"**Date**: 2026-03-29\n")
        f.write(f"**Total unique papers**: {total}\n\n")
        
        f.write("## OA Status Distribution\n\n")
        f.write("| Status | Count | % |\n")
        f.write("|---|---:|---:|\n")
        for status, count in stats.most_common():
            pct = count / total * 100
            f.write(f"| {status} | {count} | {pct:.1f}% |\n")
        
        f.write(f"\n## License Distribution\n\n")
        f.write("| License | Count | % |\n")
        f.write("|---|---:|---:|\n")
        for lic, count in license_stats.most_common():
            pct = count / total * 100
            f.write(f"| {lic} | {count} | {pct:.1f}% |\n")
        
        safe_count = sum(c for s, c in stats.items() 
                        if s in ("gold", "green", "hybrid", "bronze", 
                                "oa_by_journal", "oa_by_journal_override"))
        closed_count = stats.get("closed", 0)
        not_found = stats.get("not_found", 0)
        
        f.write(f"\n## Summary\n\n")
        f.write(f"- ✅ **Confirmed Open Access**: {safe_count} ({safe_count/total*100:.1f}%)\n")
        f.write(f"- ❌ **Closed Access**: {closed_count} ({closed_count/total*100:.1f}%)\n")
        f.write(f"- ❓ **Not found in OpenAlex**: {not_found} ({not_found/total*100:.1f}%)\n")
        
        if closed_count > 0:
            f.write(f"\n## ⚠️ Closed Papers ({closed_count})\n\n")
            f.write("These papers may need to be removed from the RAG or verified manually:\n\n")
            f.write("| DOI | Journal | Year | Publisher |\n")
            f.write("|---|---|---|---|\n")
            for p in closed_papers[:50]:
                f.write(f"| {p['doi_normalized'][:40]} | {p.get('journal','')} | "
                        f"{p.get('year','')} | {p.get('publisher','')} |\n")
            if len(closed_papers) > 50:
                f.write(f"\n*...and {len(closed_papers)-50} more (see closed_papers.tsv)*\n")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"AUDIT COMPLETE")
    print(f"{'='*60}")
    print(f"Total papers:        {total}")
    print(f"Confirmed OA:        {safe_count} ({safe_count/total*100:.1f}%)")
    print(f"Closed access:       {closed_count} ({closed_count/total*100:.1f}%)")
    print(f"Not found:           {not_found} ({not_found/total*100:.1f}%)")
    print(f"\nDetails: {OUTPUT_REPORT}")
    print(f"Closed:  {OUTPUT_CLOSED}")


if __name__ == "__main__":
    main()
