#!/usr/bin/env python3
"""
CMIP6 Hybrid Search — Qdrant + mxbai Reranker
==============================================
Usage:
    python search.py "FESOM ocean model unstructured mesh"
    python search.py "AMOC weakening SSP5-8.5" --top-k 10
    python search.py "Bayesian methods climate sensitivity" --rerank --top-k 5
    python search.py "FESOM mesh" --exclude-dois 10_1002_2017ms001099
    python search.py "Arctic sea ice" --year-min 2020 --journal "Nature Climate Change"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")

from google import genai
from google.genai import types
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding

# ── Constants ──────────────────────────────────────────────
COLLECTION = "cmip6_papers"
DENSE_DIM = 768
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
RERANK_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"  # local fallback
RERANK_MAX_LENGTH = 512  # truncate chunks for faster CPU inference
VERTEX_PROJECT = os.getenv("GCP_PROJECT", "majestic-lodge-353610")
VERTEX_RANKING_MODEL = "semantic-ranker-512@latest"
GRAPH_PATH = Path(__file__).parent / "citation_graph.json"

# ── Lazy singletons ───────────────────────────────────────
_gemini = None
_qdrant = None
_bm25 = None
_reranker = None
_citation_graph = None


def get_gemini():
    global _gemini
    if _gemini is None:
        _gemini = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _gemini


def get_qdrant():
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=QDRANT_URL, check_compatibility=False)
    return _qdrant


def get_bm25():
    global _bm25
    if _bm25 is None:
        _bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _bm25


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def get_vertex_ranker():
    """Vertex AI Ranking API client (free up to 5M records/mo)."""
    from google.cloud import discoveryengine_v1 as discoveryengine
    return discoveryengine.RankServiceClient()


# ── Embedding ─────────────────────────────────────────────
def embed_query(query: str) -> list[float]:
    """Embed a query with Gemini (RETRIEVAL_QUERY task type)."""
    r = get_gemini().models.embed_content(
        model="gemini-embedding-2-preview",
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=DENSE_DIM,
        ),
    )
    return r.embeddings[0].values


def sparse_query(query: str) -> models.SparseVector:
    """BM25 sparse encoding for a query."""
    sp = list(get_bm25().query_embed(query))[0]
    return models.SparseVector(
        indices=sp.indices.tolist(), values=sp.values.tolist()
    )


# ── Filters ───────────────────────────────────────────────
def build_filter(
    exclude_dois: list[str] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    journals: list[str] | None = None,
    tier: str | None = None,
    chunk_type: str | None = None,
) -> models.Filter | None:
    """Build Qdrant filter from optional constraints."""
    conditions_must = []
    conditions_must_not = []

    if exclude_dois:
        for doi in exclude_dois:
            conditions_must_not.append(
                models.FieldCondition(
                    key="paper_id",
                    match=models.MatchValue(value=doi),
                )
            )

    # Year filter: year is stored as string in Qdrant, so we match string values
    if year_min is not None or year_max is not None:
        y_min = year_min or 2000
        y_max = year_max or 2030
        year_strs = [str(y) for y in range(y_min, y_max + 1)]
        conditions_must.append(
            models.FieldCondition(
                key="year", match=models.MatchAny(any=year_strs)
            )
        )

    if journals:
        conditions_must.append(
            models.FieldCondition(
                key="journal",
                match=models.MatchAny(any=journals),
            )
        )

    if tier:
        conditions_must.append(
            models.FieldCondition(
                key="tier", match=models.MatchValue(value=tier)
            )
        )

    if chunk_type:
        conditions_must.append(
            models.FieldCondition(
                key="chunk_type", match=models.MatchValue(value=chunk_type)
            )
        )

    if not conditions_must and not conditions_must_not:
        return None

    return models.Filter(
        must=conditions_must if conditions_must else None,
        must_not=conditions_must_not if conditions_must_not else None,
    )


# ── Core Search ───────────────────────────────────────────
def hybrid_search(
    query: str,
    top_k: int = 5,
    prefetch_k: int = 50,
    rerank: str | bool = False,  # False, "vertex", "local", or True (=vertex)
    exclude_dois: list[str] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    journals: list[str] | None = None,
    tier: str | None = None,
    chunk_type: str | None = None,
) -> list[dict]:
    """
    Hybrid search: Dense (Gemini) + Sparse (BM25) → RRF fusion.
    Optional reranking with mxbai-rerank.

    Returns list of dicts with keys:
        score, point_id, chunk_id, paper_id, title, year, journal,
        doi, tier, chunk_type, section, text
    """
    # 1. Embed query
    t0 = time.time()
    dense = embed_query(query)
    sparse = sparse_query(query)
    t_embed = time.time() - t0

    # 2. Build filter
    qfilter = build_filter(
        exclude_dois=exclude_dois,
        year_min=year_min,
        year_max=year_max,
        journals=journals,
        tier=tier,
        chunk_type=chunk_type,
    )

    # 3. Hybrid search with RRF
    fetch_k = prefetch_k if rerank else max(prefetch_k, top_k)
    t1 = time.time()
    res = get_qdrant().query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=dense, using="dense", limit=fetch_k),
            models.Prefetch(query=sparse, using="sparse", limit=fetch_k),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=qfilter,
        limit=fetch_k if rerank else top_k,
        with_payload=True,
    )
    t_search = time.time() - t1

    # 4. Format results
    results = []
    for p in res.points:
        results.append({
            "score": p.score,
            "point_id": str(p.id),
            "chunk_id": p.payload.get("chunk_id", ""),
            "paper_id": p.payload.get("paper_id", ""),
            "title": p.payload.get("title", ""),
            "year": p.payload.get("year"),
            "journal": p.payload.get("journal", ""),
            "doi": p.payload.get("doi", ""),
            "tier": p.payload.get("tier", ""),
            "chunk_type": p.payload.get("chunk_type", ""),
            "section": p.payload.get("section", ""),
            "text": p.payload.get("text_raw", ""),
        })

    # 5. Rerank (optional)
    t_rerank = 0
    if rerank and results:
        t2 = time.time()
        if rerank == "vertex":
            results = _vertex_rerank(query, results, top_k)
        else:
            results = _local_rerank(query, results, top_k)
        t_rerank = time.time() - t2

    # 6. Timing metadata
    timing = {
        "embed_ms": round(t_embed * 1000),
        "search_ms": round(t_search * 1000),
        "rerank_ms": round(t_rerank * 1000) if rerank else None,
        "total_ms": round((t_embed + t_search + t_rerank) * 1000),
    }

    return results, timing


def _vertex_rerank(query, results, top_k):
    """Rerank via Vertex AI Ranking API (free, ~85ms, best quality)."""
    from google.cloud import discoveryengine_v1 as discoveryengine
    client = get_vertex_ranker()
    records = [
        discoveryengine.RankingRecord(
            id=str(i), title=r["title"], content=r["text"][:500]
        )
        for i, r in enumerate(results)
    ]
    request = discoveryengine.RankRequest(
        ranking_config=f"projects/{VERTEX_PROJECT}/locations/global/rankingConfigs/default_ranking_config",
        model=VERTEX_RANKING_MODEL,
        query=query, records=records, top_n=top_k,
    )
    response = client.rank(request=request)
    reranked = []
    for r in response.records:
        idx = int(r.id)
        original = results[idx].copy()
        original["rerank_score"] = r.score
        original["rrf_score"] = original.pop("score")
        original["score"] = r.score
        reranked.append(original)
    return reranked


def _local_rerank(query, results, top_k):
    """Rerank via local mxbai model (CPU fallback, slower)."""
    reranker = get_reranker()
    docs = [r["text"][:RERANK_MAX_LENGTH] for r in results]
    ranked = reranker.rank(query, docs, return_documents=False, top_k=top_k)
    reranked = []
    for item in ranked:
        idx = item["corpus_id"]
        r = results[idx].copy()
        r["rerank_score"] = item["score"]
        r["rrf_score"] = r.pop("score")
        r["score"] = item["score"]
        reranked.append(r)
    return reranked


# ── Citation Graph ────────────────────────────────────────
def get_citation_graph():
    """Lazy-load the citation graph."""
    global _citation_graph
    if _citation_graph is None:
        import networkx as nx
        from networkx.readwrite import json_graph
        if not GRAPH_PATH.exists():
            raise FileNotFoundError(f"Citation graph not found at {GRAPH_PATH}. Run build_citation_graph.py first.")
        with open(GRAPH_PATH) as f:
            data = json.load(f)
        _citation_graph = json_graph.node_link_graph(data, edges="links")
    return _citation_graph


def graph_cited_by(doi: str, limit: int = 20) -> list[dict]:
    """Papers that cite this DOI (in-edges)."""
    G = get_citation_graph()
    if doi not in G:
        return []
    citers = list(G.predecessors(doi))
    results = []
    for d in citers[:limit]:
        node = G.nodes[d]
        results.append({
            "paper_id": d, "title": node.get("title", ""),
            "year": node.get("year", 0), "journal": node.get("journal", ""),
            "cited_by_count": node.get("cited_by_count", 0),
        })
    results.sort(key=lambda x: x["cited_by_count"], reverse=True)
    return results


def graph_cites(doi: str, limit: int = 20) -> list[dict]:
    """Papers that this DOI references (out-edges), within corpus only."""
    G = get_citation_graph()
    if doi not in G:
        return []
    refs = list(G.successors(doi))
    results = []
    for d in refs[:limit]:
        node = G.nodes[d]
        results.append({
            "paper_id": d, "title": node.get("title", ""),
            "year": node.get("year", 0), "journal": node.get("journal", ""),
            "cited_by_count": node.get("cited_by_count", 0),
        })
    results.sort(key=lambda x: x["cited_by_count"], reverse=True)
    return results


# ── CLI ───────────────────────────────────────────────────
def print_results(query, results, timing, verbose=False):
    """Pretty-print search results."""
    rerank_str = f" + rerank {timing['rerank_ms']}ms" if timing["rerank_ms"] else ""
    print(f"\n{'='*80}")
    print(f'QUERY: "{query}"')
    print(f"Embed: {timing['embed_ms']}ms | Search: {timing['search_ms']}ms{rerank_str} | Total: {timing['total_ms']}ms | Results: {len(results)}")
    print(f"{'='*80}\n")

    for i, r in enumerate(results):
        score_label = "rerank" if "rerank_score" in r else "rrf"
        rrf_extra = f"  (RRF: {r['rrf_score']:.4f})" if "rrf_score" in r else ""
        print(f"--- Chunk {i+1}/{len(results)} ---")
        print(f"  {score_label}_score: {r['score']:.4f}{rrf_extra}")
        print(f"  paper_id:  {r['paper_id']}")
        print(f"  title:     {r['title']}")
        print(f"  year:      {r['year']}")
        print(f"  journal:   {r['journal']}")
        print(f"  doi:       {r['doi']}")
        print(f"  chunk_type:{r['chunk_type']}")
        if verbose:
            print(f"\n  TEXT:\n{r['text'][:500]}")
        else:
            print(f"  text:      {r['text'][:150]}...")
        print()


def main():
    parser = argparse.ArgumentParser(description="CMIP6 Hybrid Search")
    parser.add_argument("query", nargs="?", default="", help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--prefetch-k", type=int, default=50, help="Prefetch pool size for reranking")
    parser.add_argument("--rerank", nargs="?", const="vertex", default=None,
                        choices=["vertex", "local"],
                        help="Enable reranking: 'vertex' (default, API) or 'local' (mxbai CPU)")
    parser.add_argument("--exclude-dois", nargs="+", help="DOIs to exclude from results")
    parser.add_argument("--year-min", type=int, help="Minimum publication year")
    parser.add_argument("--year-max", type=int, help="Maximum publication year")
    parser.add_argument("--journals", nargs="+", help="Filter by journal names")
    parser.add_argument("--tier", help="Filter by tier (CORE/SUPPLEMENTARY)")
    parser.add_argument("--chunk-type", help="Filter by chunk type (text/table/caption)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full chunk text")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--count", action="store_true", help="Just print collection point count")
    parser.add_argument("--cited-by", metavar="DOI", help="Show papers that cite this DOI")
    parser.add_argument("--cites", metavar="DOI", help="Show papers this DOI references")

    args = parser.parse_args()

    if args.count:
        info = get_qdrant().get_collection(COLLECTION)
        print(f"Collection '{COLLECTION}': {info.points_count} points")
        return

    # Citation graph queries
    if args.cited_by or args.cites:
        if args.cited_by:
            papers = graph_cited_by(args.cited_by)
            label = f'Papers citing {args.cited_by}'
        else:
            papers = graph_cites(args.cites)
            label = f'Papers referenced by {args.cites}'

        if args.json:
            print(json.dumps({"query": label, "results": papers}, indent=2, ensure_ascii=False))
        else:
            print(f"\n{label} ({len(papers)} found):\n")
            for i, p in enumerate(papers):
                print(f"  {i+1:3d}. [{p['cited_by_count']:5d} cit] {p['title'][:60]} ({p['year']}, {p['journal'][:25]})")
        return

    rerank = args.rerank if args.rerank else False

    results, timing = hybrid_search(
        query=args.query,
        top_k=args.top_k,
        prefetch_k=args.prefetch_k,
        rerank=rerank,
        exclude_dois=args.exclude_dois,
        year_min=args.year_min,
        year_max=args.year_max,
        journals=args.journals,
        tier=args.tier,
        chunk_type=args.chunk_type,
    )

    if args.json:
        output = {"query": args.query, "timing": timing, "results": results}
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print_results(args.query, results, timing, verbose=args.verbose)


if __name__ == "__main__":
    main()
