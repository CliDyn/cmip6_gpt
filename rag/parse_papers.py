#!/usr/bin/env python3
"""
Sprint 1, Step 1: Parse papers with Docling (MAX QUALITY).

Converts PDF/HTML papers → structured Markdown + lossless JSON.
Docling handles both formats natively.

Configuration:
  - TableFormerMode.ACCURATE — best table structure recognition
  - OCR enabled with cell matching
  - Lossless DoclingDocument JSON saved for downstream chunking

Usage:
    python3 rag/parse_papers.py --limit 100    # top 100 CORE papers
    python3 rag/parse_papers.py                # all papers
    python3 rag/parse_papers.py --tier CORE    # only CORE
"""

import argparse
import csv
import os
import sys
import time
import json
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
PAPERS_DIR = PROJECT_ROOT / "raw_papers" / "downloaded_papers"
CORE_DIR = PAPERS_DIR / "core"
SUPP_DIR = PAPERS_DIR / "supplementary"
CORE_TSV = PROJECT_ROOT / "raw_papers" / "_metadata" / "papers_core_rag.tsv"
EXT_TSV = PROJECT_ROOT / "raw_papers" / "_metadata" / "papers_extended_rag.tsv"
OUTPUT_DIR = PROJECT_ROOT / "rag" / "parsed"


def create_max_quality_converter():
    """Create Docling converter with maximum quality settings."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
        TableStructureOptions,
    )

    pdf_pipeline_options = PdfPipelineOptions(
        # ── Table extraction: ACCURATE mode ──
        do_table_structure=True,
        table_structure_options=TableStructureOptions(
            mode=TableFormerMode.ACCURATE,   # best quality, ~2x slower
            do_cell_matching=True,           # match cells to PDF coordinates
        ),
        # ── OCR for scanned pages / embedded images ──
        do_ocr=True,
        # ── Generate page images for potential VLM use later ──
        generate_page_images=False,
        # ── Code block detection ──
        do_code_enrichment=True,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_pipeline_options,
            ),
        }
    )
    return converter


def doi_to_filename(doi: str) -> str:
    """Convert DOI to safe filename (matching download convention)."""
    return doi.replace("https://doi.org/", "").replace("/", "_").replace(".", "_")


def find_paper_file(safe_name: str) -> Path | None:
    """Find the downloaded paper file (PDF or HTML) in core/ or supplementary/."""
    for search_dir in [CORE_DIR, SUPP_DIR, PAPERS_DIR]:
        for ext in [".pdf", ".html", ".htm", ".xml", ".txt"]:
            p = search_dir / f"{safe_name}{ext}"
            if p.exists() and p.stat().st_size > 1000:
                return p
    return None


def load_paper_list(tier: str = "ALL") -> list[dict]:
    """Load papers from TSV files, sorted by citations (descending)."""
    papers = []
    
    files_to_load = []
    if tier in ("ALL", "CORE"):
        files_to_load.append(("CORE", CORE_TSV))
    if tier in ("ALL", "EXT"):
        files_to_load.append(("EXT", EXT_TSV))
    
    for t, tsv_path in files_to_load:
        with open(tsv_path, "r") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                doi = row.get("doi", "").strip().replace("https://doi.org/", "")
                if not doi:
                    continue
                papers.append({
                    "doi": doi,
                    "title": row.get("title", "").strip(),
                    "year": row.get("year", "").strip(),
                    "journal": row.get("journal", "").strip(),
                    "cited_by_count": int(row.get("citations", "0") or "0"),
                    "tier": t,
                    "safe_name": doi_to_filename(doi),
                })
    
    # Sort by citations (highest first)
    papers.sort(key=lambda p: p["cited_by_count"], reverse=True)
    return papers


def parse_single_paper(paper: dict, converter) -> dict:
    """Parse a single paper with Docling MAX QUALITY. Returns result dict."""
    safe_name = paper["safe_name"]
    out_md = OUTPUT_DIR / f"{safe_name}.md"
    out_json = OUTPUT_DIR / f"{safe_name}.json"
    
    # Skip if already parsed
    if out_md.exists() and out_md.stat().st_size > 100:
        return {"doi": paper["doi"], "status": "skipped", "reason": "already parsed"}
    
    # Find source file
    source_file = find_paper_file(safe_name)
    if not source_file:
        return {"doi": paper["doi"], "status": "failed", "reason": "file not found"}
    
    try:
        t0 = time.time()
        result = converter.convert(str(source_file))
        elapsed = time.time() - t0
        
        # Export to Markdown (for human inspection + chunking)
        md_text = result.document.export_to_markdown()
        
        if len(md_text.strip()) < 100:
            return {"doi": paper["doi"], "status": "failed", "reason": f"empty output ({len(md_text)} chars)"}
        
        # Save Markdown
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(md_text)
        
        # Save lossless DoclingDocument JSON (preserves full structure for chunking)
        doc_dict = result.document.export_to_dict()
        
        # Save combined metadata + document structure
        meta = {
            "doi": paper["doi"],
            "title": paper["title"],
            "year": paper["year"],
            "journal": paper["journal"],
            "tier": paper["tier"],
            "cited_by_count": paper["cited_by_count"],
            "source_file": source_file.name,
            "source_format": source_file.suffix,
            "markdown_chars": len(md_text),
            "parse_time_sec": round(elapsed, 1),
            "quality_mode": "ACCURATE",
            "docling_version": "2.76",
            "n_tables": len(doc_dict.get("tables", [])),
            "n_figures": len(doc_dict.get("figures", [])),
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        
        # Save full DoclingDocument (lossless) for chunking pipeline
        out_docling = OUTPUT_DIR / f"{safe_name}.docling.json"
        with open(out_docling, "w", encoding="utf-8") as f:
            json.dump(doc_dict, f)
        
        return {
            "doi": paper["doi"],
            "status": "ok",
            "chars": len(md_text),
            "time": round(elapsed, 1),
            "format": source_file.suffix,
            "tables": meta["n_tables"],
            "figures": meta["n_figures"],
        }
    
    except Exception as e:
        return {
            "doi": paper["doi"],
            "status": "failed",
            "reason": str(e)[:200],
        }


def main():
    parser = argparse.ArgumentParser(description="Parse CMIP6 papers with Docling (MAX QUALITY)")
    parser.add_argument("--limit", type=int, default=None, help="Parse only top N papers (by citations)")
    parser.add_argument("--tier", choices=["ALL", "CORE", "EXT"], default="CORE", help="Which tier to parse")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load papers
    papers = load_paper_list(args.tier)
    if args.limit:
        papers = papers[:args.limit]
    
    print(f"🔬 DOCLING MAX QUALITY MODE")
    print(f"   TableFormer: ACCURATE | OCR: ON | Cell matching: ON")
    print(f"   Code enrichment: ON | Lossless JSON: ON")
    print(f"{'─' * 60}")
    print(f"Papers to parse: {len(papers)} ({args.tier})")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Workers: {args.workers}")
    print(f"{'─' * 60}")
    
    # Create converter ONCE (loads ML models once, reuses across papers)
    print("Loading Docling ML models (ACCURATE mode)...")
    converter = create_max_quality_converter()
    print("Models loaded ✅")
    print(f"{'─' * 60}")
    
    ok = 0
    skipped = 0
    failed = 0
    total_tables = 0
    total_figures = 0
    
    # Sequential only for ACCURATE mode (model state not safe for multiprocessing)
    for i, paper in enumerate(papers, 1):
        print(f"[{i}/{len(papers)}] {paper['doi'][:50]} ({paper['cited_by_count']} cites, {paper['tier']})")
        result = parse_single_paper(paper, converter)
        
        if result["status"] == "ok":
            t = result.get("tables", 0)
            f = result.get("figures", 0)
            total_tables += t
            total_figures += f
            print(f"  ✅ {result['chars']} chars, {result['time']}s ({result['format']}) | 📊{t} tables, 🖼️{f} figs")
            ok += 1
        elif result["status"] == "skipped":
            print(f"  ⏭️  {result['reason']}")
            skipped += 1
        else:
            print(f"  ❌ {result['reason']}")
            failed += 1
    
    print(f"\n{'═' * 60}")
    print(f"🔬 QUALITY MODE: ACCURATE")
    print(f"✅ Parsed:  {ok}")
    print(f"⏭️  Skipped: {skipped}")
    print(f"❌ Failed:  {failed}")
    print(f"📊 Total tables extracted: {total_tables}")
    print(f"🖼️  Total figures found: {total_figures}")
    print(f"Total MD files in {OUTPUT_DIR}: {len(list(OUTPUT_DIR.glob('*.md')))} files")
    print(f"Total Docling JSON in {OUTPUT_DIR}: {len(list(OUTPUT_DIR.glob('*.docling.json')))} files")


if __name__ == "__main__":
    main()
