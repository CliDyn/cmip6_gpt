# Standard Library Imports
import os
import json
import operator
from typing import List, Dict, Any, TypedDict, Annotated, Sequence

from langchain_chroma import Chroma
from langchain import LLMChain
from langgraph.graph import StateGraph, END, START

from src.services.llm_service import create_embedding
from src.config import Config
from src.models.cmip6_args import create_dynamic_cmip6_args  
from src.services.llm_service import create_llm,create_embedding,create_split_query_template
from src.utils.chat_utils import display_debug_info




class State(TypedDict):
    aggregate: Annotated[list, operator.add]
    fanout_values: Annotated[list, operator.add]
    query: str


class SplitQueryNode:
    def __init__(self, llm_chain):
        self.llm_chain = llm_chain

    def clean_query(self, query):
        cleaned = query.lower()
        cleaned = cleaned.replace("user want ", "").replace("user wants ", "").replace("user need ", "").replace(
            "user needs ", "")
        return cleaned.strip()

    def __call__(self, state: State) -> Any:
        print(f"Splitting query: '{state['query']}'")
        split_result = self.llm_chain.run(query=state["query"])
        print(f"Raw split result: {split_result}")  # Add this line for debugging

        try:
            split_result = json.loads(split_result)
        except json.JSONDecodeError:
            print("Failed to parse JSON, attempting to extract JSON from string")
            # Attempt to extract JSON from the string
            try:
                split_result = json.loads(split_result.split('```json')[-1].split('```')[0].strip())
            except json.JSONDecodeError:
                print("Failed to extract JSON, using fallback parsing")
                # Fallback to a simple parsing method
                split_result = {
                    "variable_query": state["query"],
                    "source_query": state["query"],
                    "experiment_query": state["query"]
                }

        variable_query = self.clean_query(split_result.get("variable_query", ""))
        source_query = self.clean_query(split_result.get("source_query", ""))
        experiment_query = self.clean_query(split_result.get("experiment_query", ""))

        print(f"Split and cleaned queries:")
        print(f"  Variable query: '{variable_query}'")
        print(f"  Source query: '{source_query}'")
        print(f"  Experiment query: '{experiment_query}'")

        return {"aggregate": [variable_query, source_query, experiment_query]}


class RetrieveComponentNode:
    def __init__(self, component_type: str, db):
        self.component_type = component_type
        self.db = db

    def __call__(self, state: State) -> Any:
        query_index = {"variable": 0, "source": 1, "experiment": 2}[self.component_type]
        query = state['aggregate'][query_index]
        if query != '':
            print(f"Retrieving documents for {self.component_type} with query: '{query}'")
            results = get_top_matches(query, self.db)

            print(f"Retrieved {len(results)} results for {self.component_type}")

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



def load_retriever(chroma_path):
    """
    Loads a retriever for vector search using a Chroma vector store and embeddings.

    The function initializes an embedding model and loads a Chroma vector store from the specified 
    path. This store is used for vector-based retrieval of documents or data.

    Args:
        chroma_path (str): The file path to the directory where the Chroma vector store is persisted.

    Returns:
        Chroma: A Chroma vector store initialized with the specified embeddings and path.
    """
    embeddings = create_embedding()
    vector_store = Chroma(collection_name="example_collection",
                            embedding_function=embeddings,
                            persist_directory=chroma_path)
    return vector_store

def get_top_matches(query: str, db, k: int = 20):
    """
    Find the top k matches for a given query from the dataset using vector similarity.
    Args:
        query (str): The search query
        db: The Chroma database to search in
        k (int): The number of top matches to return (default is 10)
    Returns:
        List of top k matches
    """

    print(f"  Searching for top {k} matches for query: '{query}'")
    results = db.similarity_search_with_score(query, k=k)
    print(f"  Found {len(results)} matches")
    return results

def perform_vector_search(query: str, vector_search_fields: List[str]) -> Dict[str, Any]:
    """
    Performs a vector-based similarity search for the given query across specified CMIP6 fields.

    This function splits the user's query using a language model, retrieves data from Chroma vector 
    stores for variables, sources, and experiments, and compiles the search results. It dynamically 
    constructs CMIP6 argument schemas based on the search results and returns the vector search results, 
    split queries, and schema.

    Args:
        query (str): The user's input query for CMIP6 data.
        vector_search_fields (List[str]): A list of CMIP6 fields (e.g., variable_id, source_id, experiment_id) to search.

    Returns:
        dict: A dictionary containing vector search results, split queries, and dynamically created CMIP6 arguments.
    """
    RETRIEVERS_DIR = Config.get_retrievers_dir()
    print(RETRIEVERS_DIR)
    variable_retriever = load_retriever(
        os.path.join(RETRIEVERS_DIR, 'chroma_langchain_db'))
    sources_retriever = load_retriever(
        os.path.join(RETRIEVERS_DIR, 'chroma_langchain_db_sources_new1'))
    experiment_retriever = load_retriever(
        os.path.join(RETRIEVERS_DIR, 'chroma_langchain_db_exp_new'))
    print(f"\n--- VECTOR SIMILARITY SEARCH ---")
    print(f"Original Query: '{query}', Fields: {vector_search_fields}")
    split_query_template = create_split_query_template()
    llm = create_llm(temperature=0)
    split_chain = LLMChain(llm=llm, prompt=split_query_template)

    builder = StateGraph(State)
    builder.add_node("split", SplitQueryNode(llm_chain=split_chain))
    builder.add_edge(START, "split")

    retriever_nodes = {
        "variable_id": RetrieveComponentNode("variable", variable_retriever),
        "source_id": RetrieveComponentNode("source", sources_retriever),
        "experiment_id": RetrieveComponentNode("experiment", experiment_retriever)
    }

    for field in vector_search_fields:
        node_name = f"retrieve_{field}"
        builder.add_node(node_name, retriever_nodes[field])

    def route_all(state: State) -> Sequence[str]:
        return [f"retrieve_{field}" for field in vector_search_fields]

    builder.add_conditional_edges("split", route_all, [f"retrieve_{field}" for field in vector_search_fields])

    for node in [f"retrieve_{field}" for field in vector_search_fields]:
        builder.add_edge(node, END)

    graph = builder.compile()

    initial_state = {
        "query": query,
        "aggregate": [],
        "fanout_values": []
    }

    result = graph.invoke(initial_state)

    vector_search_results = {}
    split_queries = dict(zip(["variable_query", "source_query", "experiment_query"], result['aggregate']))

    print("\nProcessing vector search results:")
    for component in result['fanout_values']:
        for key, matches in component.items():
            print(f"\nResults for {key}:")
            for i, match in enumerate(matches[:5], 1):
                print(f"  {i}. Score: {match['score']:.4f}")
                print(f"     Content: {match['content'][:100]}...")
            vector_search_results[key] = matches

    print("--- END VECTOR SIMILARITY SEARCH ---\n")

    print("Vector search results structure:")
    print(json.dumps(vector_search_results, indent=2))

    # Map the keys to the correct facet names
    facet_map = {
        "variable": "variable_id",
        "source": "source_id",
        "experiment": "experiment_id"
    }
    vector_search_results = {facet_map[k]: v for k, v in vector_search_results.items()}

    DynamicCMIP6DownloadArgs = create_dynamic_cmip6_args(vector_search_fields, vector_search_results)

    print("\nDynamic CMIP6DownloadArgs Schema (after vector search):")
    schema = DynamicCMIP6DownloadArgs.schema()
    print(json.dumps(schema, indent=2))

    # display_debug_info("Debug: Dynamic CMIP6DownloadArgs Schema (after vector search)", schema)

    return {
        "vector_search_results": vector_search_results,
        "vector_serach_full_results": schema,
        "split_queries": split_queries,
        "cmip6_args": DynamicCMIP6DownloadArgs
    }