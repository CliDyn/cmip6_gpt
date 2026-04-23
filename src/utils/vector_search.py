# Standard Library Imports
import os
import json
import operator
import numpy as np
from typing import List, Dict, Any, TypedDict, Annotated, Sequence

from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, START
from qdrant_client import QdrantClient

from src.config import Config
from src.services.llm_service import create_llm, create_split_query_template
from src.utils.metrics import pipeline_logger


# ─── Qdrant Client Cache ────────────────────────────────────────────

_qdrant_client = None


def get_qdrant_client() -> QdrantClient:
    """Returns a cached Qdrant client (singleton)."""
    global _qdrant_client
    if _qdrant_client is None:
        qdrant_config = Config.get_qdrant_config()
        url = qdrant_config.get("url", "http://localhost:6333")
        pipeline_logger.info(f"Connecting to Qdrant at {url}")
        _qdrant_client = QdrantClient(url=url, check_compatibility=False)
    return _qdrant_client


# ─── Embedding Helper ───────────────────────────────────────────────

_embed_client = None


def _get_embed_client():
    """Returns a cached google-genai client for embedding queries."""
    global _embed_client
    if _embed_client is None:
        from google import genai
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        _embed_client = genai.Client(api_key=api_key)
    return _embed_client


def embed_query(text: str, _max_retries: int = 4, _base_delay: float = 2.0) -> List[float]:
    """Embed a single query text using gemini-embedding-2-preview.
    
    Includes retry with exponential backoff for 429 rate-limit errors.
    """
    import time
    from google.genai import types
    client = _get_embed_client()
    
    for attempt in range(_max_retries):
        try:
            result = client.models.embed_content(
                model="gemini-embedding-2-preview",
                contents=[text],
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=768,
                ),
            )
            vec = np.array(result.embeddings[0].values, dtype=np.float32)
            norm = np.linalg.norm(vec)
            return (vec / norm).tolist() if norm > 0 else vec.tolist()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                delay = _base_delay * (2 ** attempt)
                pipeline_logger.warning(
                    f"Embedding rate-limited (attempt {attempt+1}/{_max_retries}), "
                    f"retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                raise
    # Final attempt — let it raise if it fails
    result = client.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=[text],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
    vec = np.array(result.embeddings[0].values, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm > 0 else vec.tolist()


def prewarm_retrievers():
    """Pre-connect to Qdrant at app startup."""
    try:
        client = get_qdrant_client()
        qdrant_config = Config.get_qdrant_config()
        collections = qdrant_config.get("collections", {})
        for name, coll_name in collections.items():
            info = client.get_collection(coll_name)
            pipeline_logger.info(f"Pre-warmed Qdrant collection: {coll_name} ({info.points_count} points)")
    except Exception as e:
        pipeline_logger.warning(f"Failed to pre-warm Qdrant: {e}")


def clear_retriever_cache():
    """Reset Qdrant client (useful for testing)."""
    global _qdrant_client, _embed_client
    _qdrant_client = None
    _embed_client = None


# ─── Qdrant Search ──────────────────────────────────────────────────

def qdrant_similarity_search(collection_name: str, query_vector: List[float], top_k: int = 10):
    """
    Search a Qdrant collection and return results in the same format as
    the old ChromaDB similarity_search_with_score.
    """
    client = get_qdrant_client()
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    matches = []
    for point in results.points:
        payload = point.payload or {}
        matches.append({
            "content": payload.get("text", ""),
            "metadata": {
                "source": payload.get("source", payload.get("id", "")),
                "long_name": payload.get("long_name", ""),
                "realm": payload.get("realm", ""),
                "units": payload.get("units", ""),
                "type": payload.get("type", ""),
            },
            "score": point.score,  # cosine similarity (higher = better)
        })

    return matches


# ─── LangGraph State ────────────────────────────────────────────────

class State(TypedDict):
    aggregate: Annotated[list, operator.add]
    fanout_values: Annotated[list, operator.add]
    query: str


# ─── Graph Nodes ────────────────────────────────────────────────────

class SplitQueryNode:
    def __init__(self, chain):
        self.chain = chain

    def clean_query(self, query):
        cleaned = query.lower()
        cleaned = cleaned.replace("user want ", "").replace("user wants ", "").replace("user need ", "").replace(
            "user needs ", "")
        return cleaned.strip()

    def __call__(self, state: State) -> Any:
        pipeline_logger.info(f"Splitting query: '{state['query']}'")
        split_result = self.chain.invoke({"query": state["query"]})
        pipeline_logger.debug(f"Raw split result: {split_result}")

        try:
            split_result = json.loads(split_result)
        except json.JSONDecodeError:
            pipeline_logger.warning("Failed to parse JSON, attempting to extract JSON from string")
            try:
                split_result = json.loads(split_result.split('```json')[-1].split('```')[0].strip())
            except json.JSONDecodeError:
                pipeline_logger.warning("Failed to extract JSON, using fallback parsing")
                split_result = {
                    "variable_query": state["query"],
                    "source_query": state["query"],
                    "experiment_query": state["query"]
                }

        variable_query = self.clean_query(split_result.get("variable_query", ""))
        source_query = self.clean_query(split_result.get("source_query", ""))
        experiment_query = self.clean_query(split_result.get("experiment_query", ""))

        pipeline_logger.debug(f"Split queries: variable='{variable_query}', source='{source_query}', experiment='{experiment_query}'")

        return {"aggregate": [variable_query, source_query, experiment_query]}


class RetrieveComponentNode:
    def __init__(self, component_type: str, collection_name: str):
        self.component_type = component_type
        self.collection_name = collection_name

    def __call__(self, state: State) -> Any:
        query_index = {"variable": 0, "source": 1, "experiment": 2}[self.component_type]
        query = state['aggregate'][query_index]
        if query != '':
            top_k = Config.get_rag_top_k()
            pipeline_logger.info(f"Retrieving top {top_k} for {self.component_type}: '{query}'")

            # Embed query and search Qdrant
            query_vector = embed_query(query)
            results = qdrant_similarity_search(self.collection_name, query_vector, top_k)

            pipeline_logger.info(f"Retrieved {len(results)} results for {self.component_type}")

            return {
                "fanout_values": [
                    {
                        self.component_type: results
                    }
                ]
            }


# ─── Re-ranking Node ────────────────────────────────────────────────

class ReRankNode:
    """
    Re-ranks vector search results by boosting candidates whose metadata
    source name appears in the original query.
    """
    def __init__(self, original_query: str):
        self.original_query = original_query.lower()
        self.reranking_config = Config.get_reranking_config()

    def __call__(self, state: State) -> Any:
        if not self.reranking_config.get("enabled", True):
            return {"fanout_values": []}

        boost = self.reranking_config.get("query_match_boost", 0.3)
        reranked_values = []

        for component in state.get("fanout_values", []):
            for key, matches in component.items():
                reranked = []
                for match in matches:
                    source_name = match.get("metadata", {}).get("source", "").lower()
                    original_score = match["score"]

                    # If the source name appears in the query, boost it (higher similarity)
                    if source_name and source_name in self.original_query:
                        match["score"] = min(1.0, original_score + boost)
                        pipeline_logger.debug(
                            f"Re-rank boost: {source_name} {original_score:.4f} -> {match['score']:.4f}"
                        )
                    reranked.append(match)

                # Re-sort by score (higher = better for cosine similarity)
                reranked.sort(key=lambda r: r["score"], reverse=True)
                reranked_values.append({key: reranked})

        return {"fanout_values": reranked_values}


# ─── Compiled Graph Cache ───────────────────────────────────────────

_compiled_graphs: Dict[frozenset, Any] = {}


def _build_and_compile_graph(vector_search_fields: List[str], query: str):
    """Build and compile a LangGraph for the given field combination."""
    qdrant_config = Config.get_qdrant_config()
    collections = qdrant_config.get("collections", {})

    split_query_template = create_split_query_template()
    llm = create_llm(temperature=0)
    split_chain = split_query_template | llm | StrOutputParser()

    builder = StateGraph(State)
    builder.add_node("split", SplitQueryNode(chain=split_chain))
    builder.add_edge(START, "split")

    retriever_nodes = {
        "variable_id": RetrieveComponentNode("variable", collections.get("variable_id", "cmip6_variables")),
        "source_id": RetrieveComponentNode("source", collections.get("source_id", "cmip6_sources")),
        "experiment_id": RetrieveComponentNode("experiment", collections.get("experiment_id", "cmip6_experiments")),
    }

    for field in vector_search_fields:
        node_name = f"retrieve_{field}"
        builder.add_node(node_name, retriever_nodes[field])

    def route_all(state: State) -> Sequence[str]:
        return [f"retrieve_{field}" for field in vector_search_fields]

    builder.add_conditional_edges("split", route_all, [f"retrieve_{field}" for field in vector_search_fields])

    # Re-ranking node
    reranking_config = Config.get_reranking_config()
    if reranking_config.get("enabled", True):
        rerank_node = ReRankNode(original_query=query)
        builder.add_node("rerank", rerank_node)
        for field in vector_search_fields:
            builder.add_edge(f"retrieve_{field}", "rerank")
        builder.add_edge("rerank", END)
    else:
        for node in [f"retrieve_{field}" for field in vector_search_fields]:
            builder.add_edge(node, END)

    return builder.compile()


# ─── Main Search Function (legacy — used by adviser tool) ──────────

def perform_vector_search(query: str, vector_search_fields: List[str]) -> Dict[str, Any]:
    """
    Performs a vector-based similarity search for the given query across specified CMIP6 fields.

    Uses Qdrant, LangGraph fan-out for parallel retrieval, and optional re-ranking.

    NOTE: This is the legacy function that uses SplitQueryNode (extra LLM call).
    For the main search pipeline, use perform_direct_vector_search() instead.
    """
    pipeline_logger.info(f"Vector search: query='{query}', fields={vector_search_fields}")

    pipeline_logger.info("Compiling LangGraph for field combination")
    graph = _build_and_compile_graph(vector_search_fields, query)

    initial_state = {
        "query": query,
        "aggregate": [],
        "fanout_values": []
    }

    result = graph.invoke(initial_state)

    # Process results
    vector_search_results = {}
    split_queries = dict(zip(["variable_query", "source_query", "experiment_query"], result['aggregate']))

    pipeline_logger.info("Processing vector search results:")
    for component in result['fanout_values']:
        for key, matches in component.items():
            pipeline_logger.info(f"  {key}: {len(matches)} results, top score: {matches[0]['score']:.4f}" if matches else f"  {key}: 0 results")
            vector_search_results[key] = matches

    # Map keys to correct facet names
    facet_map = {
        "variable": "variable_id",
        "source": "source_id",
        "experiment": "experiment_id"
    }
    vector_search_results = {facet_map[k]: v for k, v in vector_search_results.items()}

    return {
        "vector_search_results": vector_search_results,
        "split_queries": split_queries,
    }


# ─── Direct Search Function (3-to-1 refactor) ──────────────────────

def _clean_query(query: str) -> str:
    """Clean a query string — same logic as SplitQueryNode.clean_query."""
    if not query:
        return ""
    cleaned = query.lower().strip()
    cleaned = cleaned.replace("user want ", "").replace("user wants ", "")
    cleaned = cleaned.replace("user need ", "").replace("user needs ", "")
    return cleaned.strip()


def perform_direct_vector_search(
    split_queries: Dict[str, str],
    original_query: str = "",
) -> Dict[str, Any]:
    """
    Performs vector search with PRE-SPLIT queries from the agent's tool call.

    Bypasses the SplitQueryNode LLM call entirely — the agent already split the
    query into variable/source/experiment components via its tool call arguments.

    Uses Qdrant for similarity search with gemini-embedding-2-preview embeddings.
    """
    qdrant_config = Config.get_qdrant_config()
    collections = qdrant_config.get("collections", {})
    reranking_config = Config.get_reranking_config()
    top_k = Config.get_rag_top_k()
    boost = reranking_config.get("query_match_boost", 0.3) if reranking_config.get("enabled", True) else 0.0

    # Map facet_id to collection name
    collection_map = {
        "variable_id": collections.get("variable_id", "cmip6_variables"),
        "source_id": collections.get("source_id", "cmip6_sources"),
        "experiment_id": collections.get("experiment_id", "cmip6_experiments"),
    }

    vector_search_results = {}

    for facet_id, query in split_queries.items():
        cleaned = _clean_query(query)
        if not cleaned:
            pipeline_logger.info(f"  Skipping {facet_id}: no query provided")
            continue

        # Get collection name
        if facet_id not in collection_map:
            pipeline_logger.warning(f"  Unknown facet {facet_id}, skipping")
            continue

        collection_name = collection_map[facet_id]

        # Embed query and search Qdrant
        pipeline_logger.info(f"  Retrieving top {top_k} for {facet_id}: '{cleaned}'")
        query_vector = embed_query(cleaned)
        matches = qdrant_similarity_search(collection_name, query_vector, top_k)
        pipeline_logger.info(f"  Retrieved {len(matches)} results for {facet_id}")

        # Exact-match injection: if the query contains a known source name that
        # didn't make it into the top-k vector results, inject it with score=1.0.
        # Fixes: "historical" embedding scoring lower than "historical-withism" etc.
        existing_sources = {m["metadata"]["source"].lower() for m in matches}
        query_words = cleaned.lower().split()
        for word in query_words:
            if word not in existing_sources:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                client = get_qdrant_client()
                exact_hits = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="source", match=MatchValue(value=word))
                    ]),
                    limit=1,
                    with_payload=True,
                )[0]
                if exact_hits:
                    payload = exact_hits[0].payload or {}
                    matches.append({
                        "content": payload.get("text", ""),
                        "metadata": {"source": payload.get("source", word)},
                        "score": 1.0,
                    })
                    pipeline_logger.info(f"  Exact-match injected: '{word}' → score=1.0")

        # Re-ranking: boost candidates whose source name appears in the combined query
        if boost > 0 and original_query:
            combined_lower = (original_query + " " + cleaned).lower()
            for match in matches:
                source_name = match.get("metadata", {}).get("source", "").lower()
                if source_name and source_name in combined_lower:
                    old_score = match["score"]
                    match["score"] = min(1.0, old_score + boost)
                    pipeline_logger.debug(f"  Re-rank boost: {source_name} {old_score:.4f} -> {match['score']:.4f}")
            matches.sort(key=lambda r: r["score"], reverse=True)

        if matches:
            pipeline_logger.info(f"  {facet_id}: top score={matches[0]['score']:.4f}")

        vector_search_results[facet_id] = matches

    return {"vector_search_results": vector_search_results}