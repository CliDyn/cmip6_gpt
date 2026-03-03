# Standard Library Imports
import os
import json
import operator
from typing import List, Dict, Any, TypedDict, Annotated, Sequence

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, START

from src.services.llm_service import create_embedding
from src.config import Config
from src.services.llm_service import create_llm, create_embedding, create_split_query_template
from src.utils.metrics import pipeline_logger


# ─── Retriever Cache (Item #6) ──────────────────────────────────────
# Singleton cache for ChromaDB retrievers — avoids re-loading on every query.

_retriever_cache: Dict[str, Chroma] = {}


def get_cached_retriever(chroma_path: str) -> Chroma:
    """
    Returns a cached ChromaDB retriever for the given path.
    Creates and caches the retriever on first access.
    """
    if chroma_path not in _retriever_cache:
        pipeline_logger.info(f"Loading retriever from: {chroma_path}")
        embeddings = create_embedding()
        _retriever_cache[chroma_path] = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory=chroma_path,
        )
    else:
        pipeline_logger.debug(f"Using cached retriever for: {chroma_path}")
    return _retriever_cache[chroma_path]


def clear_retriever_cache():
    """Clear the retriever cache (useful for testing)."""
    _retriever_cache.clear()


def prewarm_retrievers():
    """Pre-load all ChromaDB retrievers at app startup for faster first queries."""
    paths = Config.get_retriever_paths()
    for name, path in paths.items():
        if os.path.exists(path):
            get_cached_retriever(path)
            pipeline_logger.info(f"Pre-warmed retriever: {name}")
        else:
            pipeline_logger.warning(f"Retriever path not found (skipping): {path}")
    pipeline_logger.info(f"Pre-warmed {len(_retriever_cache)} retrievers")


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
    def __init__(self, component_type: str, db):
        self.component_type = component_type
        self.db = db

    def __call__(self, state: State) -> Any:
        query_index = {"variable": 0, "source": 1, "experiment": 2}[self.component_type]
        query = state['aggregate'][query_index]
        if query != '':
            top_k = Config.get_rag_top_k()
            pipeline_logger.info(f"Retrieving top {top_k} for {self.component_type}: '{query}'")
            results = self.db.similarity_search_with_score(query, k=top_k)
            pipeline_logger.info(f"Retrieved {len(results)} results for {self.component_type}")

            return {
                "fanout_values": [
                    {
                        self.component_type: [
                            {
                                "content": doc.page_content,
                                "metadata": doc.metadata,
                                "score": score
                            } for doc, score in results
                        ]
                    }
                ]
            }


# ─── Re-ranking Node (Item #9) ──────────────────────────────────────

class ReRankNode:
    """
    Re-ranks vector search results by boosting candidates whose metadata
    source name appears in the original query. This improves precision
    when users mention specific model/variable/experiment names.
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

                    # If the source name appears in the query, boost it (lower distance)
                    if source_name and source_name in self.original_query:
                        match["score"] = max(0.0, original_score - boost)
                        pipeline_logger.debug(
                            f"Re-rank boost: {source_name} {original_score:.4f} -> {match['score']:.4f}"
                        )
                    reranked.append(match)

                # Re-sort by boosted score
                reranked.sort(key=lambda r: r["score"])
                reranked_values.append({key: reranked})

        # Replace old fanout_values — we return the delta that gets added
        return {"fanout_values": reranked_values}


# ─── Compiled Graph Cache ───────────────────────────────────────────
# Cache compiled LangGraph instances by field combination to avoid
# rebuilding + recompiling the graph on every query.

_compiled_graphs: Dict[frozenset, Any] = {}


def _build_and_compile_graph(vector_search_fields: List[str], query: str):
    """Build and compile a LangGraph for the given field combination."""
    retriever_paths = Config.get_retriever_paths()
    retrievers = {
        "variable_id": get_cached_retriever(retriever_paths["variable_id"]),
        "source_id": get_cached_retriever(retriever_paths["source_id"]),
        "experiment_id": get_cached_retriever(retriever_paths["experiment_id"]),
    }

    split_query_template = create_split_query_template()
    llm = create_llm(temperature=0)
    # Use modern RunnableSequence instead of deprecated LLMChain
    split_chain = split_query_template | llm | StrOutputParser()

    builder = StateGraph(State)
    builder.add_node("split", SplitQueryNode(chain=split_chain))
    builder.add_edge(START, "split")

    retriever_nodes = {
        "variable_id": RetrieveComponentNode("variable", retrievers["variable_id"]),
        "source_id": RetrieveComponentNode("source", retrievers["source_id"]),
        "experiment_id": RetrieveComponentNode("experiment", retrievers["experiment_id"]),
    }

    for field in vector_search_fields:
        node_name = f"retrieve_{field}"
        builder.add_node(node_name, retriever_nodes[field])

    def route_all(state: State) -> Sequence[str]:
        return [f"retrieve_{field}" for field in vector_search_fields]

    builder.add_conditional_edges("split", route_all, [f"retrieve_{field}" for field in vector_search_fields])

    # Re-ranking node (Item #9)
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

    Uses cached retrievers, LangGraph fan-out for parallel retrieval, and optional re-ranking.
    The compiled graph is cached per field combination for efficiency.

    NOTE: This is the legacy function that uses SplitQueryNode (extra LLM call).
    For the main search pipeline, use perform_direct_vector_search() instead.

    Args:
        query (str): The user's input query for CMIP6 data.
        vector_search_fields (List[str]): A list of CMIP6 fields to search.

    Returns:
        dict: Contains vector_search_results and split_queries.
    """
    pipeline_logger.info(f"Vector search: query='{query}', fields={vector_search_fields}")

    # Always rebuild graph so the split LLM uses the currently selected model.
    # (Retrievers themselves are cached separately — that's the expensive part.)
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

    # Item #10: Do NOT create dynamic args here — let the caller do it once
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

    Only searches fields where a non-empty query was provided.

    Args:
        split_queries: Dict with keys 'variable_id', 'source_id', 'experiment_id'
                       and values being the natural language sub-queries.
        original_query: The full original query (used for re-ranking boost).

    Returns:
        dict: Contains vector_search_results with the same format as perform_vector_search.
    """
    retriever_paths = Config.get_retriever_paths()
    reranking_config = Config.get_reranking_config()
    top_k = Config.get_rag_top_k()
    boost = reranking_config.get("query_match_boost", 0.3) if reranking_config.get("enabled", True) else 0.0

    vector_search_results = {}

    for facet_id, query in split_queries.items():
        cleaned = _clean_query(query)
        if not cleaned:
            pipeline_logger.info(f"  Skipping {facet_id}: no query provided")
            continue

        # Get retriever
        if facet_id not in retriever_paths:
            pipeline_logger.warning(f"  Unknown facet {facet_id}, skipping")
            continue
        db = get_cached_retriever(retriever_paths[facet_id])

        # Search
        pipeline_logger.info(f"  Retrieving top {top_k} for {facet_id}: '{cleaned}'")
        results = db.similarity_search_with_score(cleaned, k=top_k)
        pipeline_logger.info(f"  Retrieved {len(results)} results for {facet_id}")

        matches = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]

        # Re-ranking: boost candidates whose source name appears in the combined query
        if boost > 0 and original_query:
            combined_lower = (original_query + " " + cleaned).lower()
            for match in matches:
                source_name = match.get("metadata", {}).get("source", "").lower()
                if source_name and source_name in combined_lower:
                    old_score = match["score"]
                    match["score"] = max(0.0, old_score - boost)
                    pipeline_logger.debug(f"  Re-rank boost: {source_name} {old_score:.4f} -> {match['score']:.4f}")
            matches.sort(key=lambda r: r["score"])

        if matches:
            pipeline_logger.info(f"  {facet_id}: top score={matches[0]['score']:.4f}")

        vector_search_results[facet_id] = matches

    return {"vector_search_results": vector_search_results}