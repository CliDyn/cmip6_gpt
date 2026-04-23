"""
CMIP6 Methodology RAG — Literature-backed analysis guidance.
==============================================================

Uses the same Qdrant vector store (6,800+ CMIP6 papers, 101K chunks) as the
literature search tool, but with a METHODOLOGY focus:

  ► "How should ERA5 total precipitation units be converted?"
  ► "What is the correct QDM bias correction procedure for temperature?"
  ► "How to compute Clausius-Clapeyron scaling from CMIP6 projections?"

The agent calls this BEFORE writing analysis code to ground its methodology
in peer-reviewed literature — preventing unit conversion errors, incorrect
statistical procedures, and metric cheating.

Design:
  • Flexible top_k and num_queries — agent decides how deep to search.
  • Defaults: 3 queries × 8 chunks = 24 chunks per call.
  • Can be called multiple times if first pass is insufficient.
  • Prioritizes methods/data sections via chunk_type filtering.
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Add rag/ to path so we can import search functions
_rag_dir = str(Path(__file__).parent.parent.parent / "rag")
if _rag_dir not in sys.path:
    sys.path.insert(0, _rag_dir)

# Load DOI lookup (shared with literature_service)
_doi_lookup_path = Path(__file__).parent.parent.parent / "rag" / "doi_lookup.json"
try:
    with open(_doi_lookup_path) as _f:
        _DOI_LOOKUP = json.load(_f)
except FileNotFoundError:
    _DOI_LOOKUP = {}


def _denormalize_doi(norm_doi: str) -> str:
    """Convert normalized DOI (underscores) → real DOI via lookup table."""
    if not norm_doi:
        return norm_doi
    if norm_doi in _DOI_LOOKUP:
        return _DOI_LOOKUP[norm_doi]
    if norm_doi.startswith("10_"):
        rest = norm_doi[3:]
        slash_pos = rest.find("_")
        if slash_pos >= 0:
            return f"10.{rest[:slash_pos]}/{rest[slash_pos+1:]}"
        return f"10.{rest}"
    return norm_doi


# ── Tool Schema ──────────────────────────────────────────

class MethodologyRAGArgs(BaseModel):
    """Schema for cmip6_methodology_check tool."""

    queries: List[str] = Field(
        description=(
            "List of methodology-focused search queries to run against the CMIP6 literature corpus. "
            "Each query should target a specific methodological question. "
            "Default: ~3 queries. Use more if the topic is complex or unfamiliar. "
            "Examples:\n"
            '  - "ERA5 total precipitation units conversion metres to mm/day monthly accumulated"\n'
            '  - "Quantile Delta Mapping QDM bias correction temperature precipitation"\n'
            '  - "Clausius-Clapeyron scaling rate 7% per Kelvin precipitation temperature"\n'
            '  - "CMIP6 model evaluation RMSE area-weighted cosine latitude"\n'
            '  - "lake-effect snow Great Lakes CMIP6 precipitation projections methodology"'
        ),
    )
    chunks_per_query: Optional[int] = Field(
        default=8,
        description=(
            "Number of literature chunks to return per query (default 8). "
            "Increase to 12-15 if queries return poor results. "
            "Decrease to 3-5 for quick lookups."
        ),
    )
    year_min: Optional[int] = Field(
        default=None,
        description="Filter: minimum publication year (inclusive). Useful for finding latest methods.",
    )
    year_max: Optional[int] = Field(
        default=None,
        description="Filter: maximum publication year (inclusive).",
    )
    chunk_type: Optional[str] = Field(
        default=None,
        description=(
            "Filter by chunk type: 'text' (default, full paragraphs), "
            "'table' (data tables with numbers), 'caption' (figure/table captions). "
            "Leave None to search all types."
        ),
    )


# ── Tool Implementation ──────────────────────────────────

@tool(args_schema=MethodologyRAGArgs)
def cmip6_methodology_check(
    queries: List[str],
    chunks_per_query: int = 8,
    year_min: int = None,
    year_max: int = None,
    chunk_type: str = None,
) -> str:
    """Search 6,800+ CMIP6 papers for METHODOLOGY guidance before writing analysis code.

    CALL THIS TOOL BEFORE any data analysis to verify:
    ✓ Unit conversions (ERA5 tp → mm/day, CMIP6 pr × 86400, etc.)
    ✓ Statistical methods (QDM bias correction, EOF, trend significance)
    ✓ Physical constraints (Clausius-Clapeyron scaling, area-weighting)
    ✓ Best practices (calendar harmonization, regridding, ensemble averaging)
    ✓ Known pitfalls for specific variables, models, or regions

    HOW TO USE:
    1. Formulate 2-4 targeted methodology queries
    2. Read returned paper excerpts for authoritative procedures
    3. Apply verified methodology in python_repl
    4. If results are insufficient, call again with refined queries or more chunks

    DO NOT USE FOR: Literature reviews for user questions (use cmip6_literature_search).
    This tool is for YOUR (the agent's) internal methodology verification.
    """
    from search import hybrid_search

    all_results = []
    all_timings = []
    seen_chunk_ids = set()  # deduplicate across queries

    for q in queries:
        results, timing = hybrid_search(
            query=q,
            top_k=chunks_per_query,
            prefetch_k=50,
            rerank="vertex",
            year_min=year_min,
            year_max=year_max,
            chunk_type=chunk_type,
        )
        all_timings.append(timing)

        for r in results:
            cid = r.get("chunk_id", r.get("point_id", ""))
            if cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                all_results.append({
                    "query": q,
                    "score": round(r["score"], 4),
                    "title": r["title"],
                    "year": r["year"],
                    "journal": r["journal"],
                    "doi": _denormalize_doi(r["doi"]),
                    "section": r.get("section", ""),
                    "chunk_type": r.get("chunk_type", ""),
                    # Longer text for methodology — agent needs detail
                    "text": r["text"][:1500],
                })

    total_ms = sum(t["total_ms"] for t in all_timings)

    return json.dumps({
        "tool": "cmip6_methodology_check",
        "num_queries": len(queries),
        "chunks_per_query": chunks_per_query,
        "total_unique_chunks": len(all_results),
        "total_latency_ms": total_ms,
        "guidance": (
            "Review the excerpts below for authoritative methodology. "
            "Pay special attention to: unit conversion procedures, statistical methods, "
            "physical constraints, and known pitfalls. If the excerpts don't cover your "
            "question, call this tool again with more specific queries or increase chunks_per_query."
        ),
        "results": all_results,
    }, ensure_ascii=False, indent=2)
