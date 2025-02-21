from pyesgf.search import SearchConnection
from typing import  Dict, Any,List
from src.utils.chat_utils import format_chat_history, display_debug_info
import streamlit as st
from src.services.llm_service import create_llm
import urllib
import json

def download_cmip6_data(**kwargs):
    """
    Downloads CMIP6 climate data based on the specified search parameters.
    This function connects to the ESGF search service and returns both the original facet counts
    and an additional detailed model-specific breakdown, excluding empty or "none" parameters.
    
    Args:
        **kwargs: Search parameters used to filter CMIP6 datasets.
    
    Returns:
        tuple: (str, int) - JSON-formatted string of organized results and hit count
    """
    print("\n--- DOWNLOADING CMIP6 DATA ---")
    print(f"Search parameters: {kwargs}")
    
    try:
        conn = SearchConnection('https://esgf-node.llnl.gov/esg-search', distrib=True)
        facets = [
            'source_id', 'frequency', 'nominal_resolution', 'experiment_id',
            'variable_id', 'sub_experiment_id', 'activity_id', 'realm', 'institution_id'
        ]
        
        ctx = conn.new_context(
            project='CMIP6',
            facets=','.join(facets),
            **kwargs
        )
        
        # Original result structure
        result = {
            "hit_count": ctx.hit_count,
            "facet_counts": {}
        }
        
        # Store original facet counts
        for facet in facets:
            result["facet_counts"][facet] = ctx.facet_counts.get(facet, {})
            
        # Create detailed model-specific analysis
        final_facet_values = {
            "hit_count": ctx.hit_count,
            "models": {}
        }
        
        # Process each dataset for model-specific information
        datasets = ctx.search()
        param_counts = {}
        
        for dataset in datasets:
            source_id = dataset.json.get('source_id')[0]
            
            if source_id not in final_facet_values["models"]:
                final_facet_values["models"][source_id] = {
                    "dataset_count": 0
                }
                param_counts[source_id] = {}
            
            final_facet_values["models"][source_id]["dataset_count"] += 1
            
            for facet in facets[1:]:  # Skip source_id
                if facet in dataset.json and dataset.json[facet]:
                    values = dataset.json[facet]
                    # Skip if the only value is "none"
                    if values == ["none"]:
                        continue
                        
                    if facet not in final_facet_values["models"][source_id]:
                        final_facet_values["models"][source_id][facet] = {}
                    
                    if facet not in param_counts[source_id]:
                        param_counts[source_id][facet] = {}
                    
                    for value in values:
                        # Skip empty values or "none"
                        if not value or value.lower() == "none":
                            continue
                            
                        if value not in param_counts[source_id][facet]:
                            param_counts[source_id][facet][value] = 0
                        param_counts[source_id][facet][value] += 1
        
        # Convert counts to final format, excluding empty facets
        for model_id, model_data in final_facet_values["models"].items():
            for facet in facets[1:]:
                if facet in param_counts[model_id]:
                    if param_counts[model_id][facet]:  # Only include if there are non-empty values
                        model_data[facet] = {
                            value: count 
                            for value, count in param_counts[model_id][facet].items()
                            if count > 0
                        }
                        # Remove the facet if it ended up empty after filtering
                        if not model_data[facet]:
                            del model_data[facet]
        
        # Original summary display
        # print(f"Number of datasets found: {result['hit_count']}")
        # print("Facet counts:")
        summary = json.dumps(result, indent=2)
        # print(summary)
        # display_debug_info("Facet Counts", summary)
        
        # Display detailed model-specific summary
        print("\nDetailed model-specific breakdown:")
        detailed_summary = json.dumps(final_facet_values, indent=2)
        print(detailed_summary)
        # display_debug_info("Final Facet Values", detailed_summary)
        
        print("--- END DOWNLOADING CMIP6 DATA ---\n")
        
        return summary, result['hit_count'], detailed_summary
        
    except Exception as e:
        error_msg = f"Error downloading CMIP6 data: {str(e)}"
        print(error_msg)
        return json.dumps({"error": error_msg}), 0
def create_esgf_search_link(facet_values):
    """
    Creates a consistent ESGF search URL by encapsulating all facets within the activeFacets parameter.
    Args:
        facet_values (dict): A dictionary where keys are facet names and values are either single facet values or lists of facet values.
    Returns:
        link (str): A fully constructed ESGF search URL.
    """
    esgf_base_url = "https://aims2.llnl.gov/search/cmip6/?"
    active_facets = {}

    print('Creating ESGF search link...')
    for key, value in facet_values.items():
        if value:
            if isinstance(value, list):
                # Ensure all values are strings and stripped of whitespace
                cleaned_values = [str(v).strip() for v in value if v]
                if cleaned_values:
                    active_facets[key] = cleaned_values
            else:
                # Wrap single values in a list
                cleaned_value = str(value).strip()
                if cleaned_value:
                    active_facets[key] = [cleaned_value]

    if not active_facets:
        print("No active facets to include in the search link.")
        return esgf_base_url + "project=CMIP6"

    # JSON-encode the activeFacets dictionary
    active_facets_json = json.dumps(active_facets)

    # URL-encode the JSON string
    encoded_active_facets = urllib.parse.quote(active_facets_json)

    # Construct the full URL with activeFacets and project parameter
    esgf_params = f"activeFacets={encoded_active_facets}&project=CMIP6"
    link = esgf_base_url + esgf_params

    print(f'Final ESGF Search Link: {link}')
    return link


def select_facets(query: str) -> Dict[str, Any]:
    """
    Selects relevant CMIP6 data facets for a given query based on conversation history.

    This function analyzes the user's query and the chat history 
    to determine which facets (e.g., source_id, variable_id, experiment_id) are relevant for 
    searching CMIP6 data. If facets require vector search, it will indicate that in the results.

    Args:
        query (str): The user's input query to search for CMIP6 data.

    Returns:
        dict: A dictionary containing:
            - relevant_facets (list): List of facets relevant to the query.
            - requires_vector_search (bool): Whether vector search is needed.
            - vector_search_fields (list): List of fields requiring vector search.
    """
    print(f"\n--- SELECTING FACETS FOR QUERY: {query} ---")
    chat_history = st.session_state.get('messages', [])
    # formatted_history = format_chat_history(chat_history)
    llm = create_llm(temperature=0)
    prompt = f"""
    Based on the following user query about CMIP6 data, determine which facets are relevant for the search.
    User query: {query}
    The possible facets are:
    - source_id: The model name in CMIP6 (e.g., CMCC-CM2-SR5, MPI-ESM1-2-LR) *Always* use naming from vector search only
    - frequency: Time frequency of the data (e.g., daily, monthly, yearly)
    - nominal_resolution: Spatial resolution of the data (e.g., 100 km, 250 km)
    - experiment_id: Experiment identifier (e.g., historical, ssp585). *Always* use naming from vector search only
    - variable_id: Variable identifier (e.g., tas for surface air temperature, pr for precipitation). *Always* use naming from vector search only
    - activity_id: Activity identifier for the CMIP6 project (e.g., CMIP, ScenarioMIP).
    - institution_id: Institution identifier for the CMIP6 project (e.g, AWI, UHH)
    - realm: Realm of the climate system (e.g., atmos, ocean, land)
    - sub_experiment_id: Sub-experiment identifier for specific initializations
    - variant_label: Identifier for the ensemble member specifying realization, initialization method, physics, and forcing indices (e.g., r1i1p1f1). The default value is r1i1p1f1. ALWAYS select variant_label even if it is not relevant for the search.

    **Instructions:**
    1. **Vector Search Priority:** If `source_id`, `experiment_id`, or `variable_id` are relevant to the query, they should **always** be retrieved using vector search **before** considering other facets.
    2. **Facet Relevance:** Only include facets that are directly relevant to the user's query.
    3. **Comprehensive Analysis:** Consider the entire conversation history to accurately determine the user's intent and the relevance of each facet.
    4. When users mention some general concepts like 'ocean data', 'sea ice data', 'atmosphere data' and ect, you can consider 'realm' facet.
    5. Complex Queries:
      - Avoid breaking down complicated user requests into multiple separate facets.
      - Instead, identify one primary facet that best represents the user’s main requirement.
      - For example, rather than using 'nominal_resolution' and 'institution_id' independently, combine them into a single facet like 'source_id' if it more directly aligns with the user’s needs.
      - If a user asks for general data (e.g., “sea ice data,” “ocean data,” “atmospheric data”), use the 'realm' facet rather than searching for a specific 'variable_id'.
    **ALWAYS FOLLOW THE INSTRUCTIONS**
    Return your response as a JSON object with the following structure:
    {{
        "relevant_facets": [list of relevant facet names],
        "requires_vector_search": true/false (whether source_id, variable_id, or experiment_id are present),
        "vector_search_fields": [list of fields requiring vector search]
    }}
    """
    response = llm.invoke(prompt)
    print("Raw LLM Response:")
    print(response.content)

    try:
        cleaned_content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(cleaned_content)
        print("Parsed JSON Result:")
        print(json.dumps(result, indent=2))
        print(f"--- END SELECTING FACETS ---\n")
        return result
    except json.JSONDecodeError as e:
        error_msg = f"Error parsing JSON: {str(e)}"
        print(error_msg)
        print(f"Raw content: {response.content}")
        print(f"--- END SELECTING FACETS (WITH ERROR) ---\n")
        return {
            "relevant_facets": [],
            "requires_vector_search": False,
            "vector_search_fields": []
        }

def select_facet_values(query: str, relevant_facets: List[str], dynamic_args_class) -> Dict[str, Any]:
    """
    Selects appropriate values for the relevant CMIP6 facets based on the user's query.

    This function analyzes the user's query and conversation history, 
    and then selects values for the relevant facets (e.g., variable_id, source_id) from the provided schema. 
    If multiple values are relevant for a facet, it returns them in a list.

    Args:
        query (str): The user's input query to search for CMIP6 data.
        relevant_facets (List[str]): List of facets that are relevant to the search.
        dynamic_args_class: A dynamically created argument class that provides the schema for the facets.

    Returns:
        dict: A dictionary where the keys are facet names and the values are the selected facet values.
    """
    chat_history = st.session_state.get('messages', [])
    formatted_history = format_chat_history(chat_history)
    llm = create_llm(temperature=0)

    prompt = f"""
    Based on the following user query about CMIP6 data and the relevant facets, determine appropriate values for each facet.
    Use the provided information about each facet to guide your selection.
    Strictly select only what the user wants; if you think that multiple values could fit the user's prompt, return them all in a list format.
    If a higher-priority facet already contains all necessary information (e.g., includes institution or resolution details), do not include additional, lower-priority facets that overlap or become redundant.
    ALWAYS keep 'variant_label' in facet_values ​​unless user request specifies otherwise (for example what to have all variant_label)
    Conversation: {formatted_history}
    User query: {query}

    Relevant facets and their descriptions:
    {dynamic_args_class.schema_json()}

    Return your response as a JSON object where the keys are the facet names and the values are the selected values for each facet.
    Only include values for the facets listed in the relevant_facets.
    Do not include any explanatory text before or after the JSON object.
    """
    response = llm.invoke(prompt)
    try:
        json_str = response.content
        start = json_str.find('{')
        end = json_str.rfind('}') + 1
        if start != -1 and end != -1:
            json_str = json_str[start:end]

        facet_values = json.loads(json_str)
        display_debug_info("Initial Facet Values", facet_values)
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON for facet values: {str(e)}")
        st.error(f"Raw content: {response.content}")
        facet_values = {}

    return facet_values