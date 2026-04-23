"""
CMIP6 Literature Search + Citation Graph tools for the LangGraph agent.

Wraps rag/search.py functions (hybrid search, reranking, citation graph)
as LangGraph @tool functions for use by the CMIP6 agent.
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from src.config import Config

# Add rag/ to path so we can import search functions
_rag_dir = str(Path(__file__).parent.parent.parent / "rag")
if _rag_dir not in sys.path:
    sys.path.insert(0, _rag_dir)

# Load DOI lookup table (normalized → real DOI, 6,465 entries)
_doi_lookup_path = Path(__file__).parent.parent.parent / "rag" / "doi_lookup.json"
try:
    with open(_doi_lookup_path) as _f:
        _DOI_LOOKUP = json.load(_f)
except FileNotFoundError:
    _DOI_LOOKUP = {}


def denormalize_doi(norm_doi: str) -> str:
    """Convert normalized DOI (underscores) → real DOI via lookup table.
    Covers 98.2% of corpus. Fallback: basic prefix/slash restoration.
    """
    if not norm_doi:
        return norm_doi
    if norm_doi in _DOI_LOOKUP:
        return _DOI_LOOKUP[norm_doi]
    # Fallback: restore 10. and first /
    if norm_doi.startswith("10_"):
        rest = norm_doi[3:]
        slash_pos = rest.find("_")
        if slash_pos >= 0:
            return f"10.{rest[:slash_pos]}/{rest[slash_pos+1:]}"
        return f"10.{rest}"
    return norm_doi


# ── Tool Schemas ──────────────────────────────────────────

class LiteratureSearchArgs(BaseModel):
    """Schema for cmip6_literature_search tool."""
    query: str = Field(description="Natural language search query about CMIP6 climate science.")
    year_min: Optional[int] = Field(default=None, description="Filter: minimum publication year (inclusive).")
    year_max: Optional[int] = Field(default=None, description="Filter: maximum publication year (inclusive).")
    journals: Optional[List[str]] = Field(default=None, description="Filter: only these journal names.")
    exclude_dois: Optional[List[str]] = Field(
        default=None,
        description="DOIs to exclude (use when a single paper dominates results). Format: 10_1234_abc"
    )
    top_k: Optional[int] = Field(default=10, description="Number of results to return (default 10).")


class CitationGraphArgs(BaseModel):
    """Schema for cmip6_citation_graph tool."""
    doi: str = Field(description="Paper DOI in normalized format (slashes/dots replaced with underscores, e.g. 10_5194_gmd-12-3991-2019).")
    direction: str = Field(
        default="cited_by",
        description="'cited_by' = papers that cite this DOI; 'cites' = papers this DOI references."
    )


# ── Tool Implementations ──────────────────────────────────

@tool(args_schema=LiteratureSearchArgs)
def cmip6_literature_search(
    query: str,
    year_min: int = None,
    year_max: int = None,
    journals: list = None,
    exclude_dois: list = None,
    top_k: int = 10,
) -> str:
    """Search 6,800+ CMIP6 scientific papers (101K text chunks) using hybrid vector + keyword search with AI reranking.

    Returns relevant paper chunks with titles, years, journals, DOIs, and text excerpts.

    USE THIS TOOL FOR:
    - Scientific methodology questions ("How does FESOM2 handle mesh refinement?")
    - Model descriptions and evaluations ("What is IPSL-CM6A?")
    - Research findings and results ("What are ECS estimates in CMIP6?")
    - Literature reviews ("Recent work on AMOC tipping points")

    DO NOT USE FOR: Finding CMIP6 datasets to download (use cmip6_datasets_search instead).

    TIPS:
    - If one paper dominates all results, re-query with exclude_dois to get diverse papers
    - Use year_min/year_max to focus on recent or historical work
    - Results include DOIs — use cmip6_citation_graph to explore citation chains
    """
    from search import hybrid_search

    # Use Config.rag_chunks_per_search as the effective top_k
    effective_top_k = top_k if top_k != 10 else Config.rag_chunks_per_search

    results, timing = hybrid_search(
        query=query,
        top_k=effective_top_k,
        prefetch_k=50,
        rerank="vertex",
        exclude_dois=exclude_dois,
        year_min=year_min,
        year_max=year_max,
        journals=journals,
    )

    # Format for LLM consumption
    formatted = []
    for i, r in enumerate(results):
        formatted.append({
            "rank": i + 1,
            "score": round(r["score"], 4),
            "title": r["title"],
            "year": r["year"],
            "journal": r["journal"],
            "doi": denormalize_doi(r["doi"]),
            "paper_id": r["paper_id"],
            "text": r["text"][:1200],  # truncate for context window
        })

    return json.dumps({
        "query": query,
        "num_results": len(formatted),
        "latency_ms": timing["total_ms"],
        "results": formatted,
    }, ensure_ascii=False, indent=2)


@tool(args_schema=CitationGraphArgs)
def cmip6_citation_graph(doi: str, direction: str = "cited_by") -> str:
    """Explore citation relationships between CMIP6 papers in the corpus.

    USE THIS TOOL AFTER finding a key paper via cmip6_literature_search:
    - "cited_by":  "What papers cite this work?" → find follow-up research
    - "cites":     "What does this paper reference?" → find foundational work

    TIPS:
    - Only shows citations WITHIN our 6,800-paper corpus (not all citations globally)
    - Results sorted by global citation count (most influential first)
    - Use the returned paper_ids with cmip6_literature_search(exclude_dois=...) for diversity
    """
    from search import graph_cited_by, graph_cites

    if direction == "cites":
        papers = graph_cites(doi, limit=20)
        label = f"Papers referenced by {doi}"
    else:
        papers = graph_cited_by(doi, limit=20)
        label = f"Papers citing {doi}"

    return json.dumps({
        "query": label,
        "direction": direction,
        "doi": doi,
        "num_results": len(papers),
        "results": papers,
    }, ensure_ascii=False, indent=2)
