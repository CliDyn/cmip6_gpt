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

    This function connects to the ESGF (Earth System Grid Federation) search service, retrieves CMIP6 
    datasets based on the provided facets, and returns a summary of the search results, including the 
    hit count and facet breakdown.

    Args:
        **kwargs: Search parameters used to filter CMIP6 datasets (e.g., source_id, variable_id, etc.).

    Returns:
        str: A JSON-formatted string containing the search results or an error message if the download fails.
    """
    print("\n--- DOWNLOADING CMIP6 DATA ---")
    print(f"Search parameters: {kwargs}")
    try:
        #conn = SearchConnection('https://esgf-data.dkrz.de/esg-search', distrib=True)
        conn = SearchConnection('https://esgf-data.dkrz.de/esg-search', distrib=True)
        facets = [
            'source_id', 'frequency', 'nominal_resolution', 'experiment_id',
            'variable_id', 'sub_experiment_id', 'activity_id', 'realm'
        ]
        ctx = conn.new_context(
            project='CMIP6',
            facets=','.join(facets),
            **kwargs
        )
        result = {
            "hit_count": ctx.hit_count,
            "facet_counts": {}
        }
        for facet in facets:
            result["facet_counts"][facet] = ctx.facet_counts.get(facet, {})
        print(f"Number of datasets found: {result['hit_count']}")
        print("Facet counts:")
        print(json.dumps(result, indent=2))
        summary = json.dumps(result, indent=2)
        print("--- END DOWNLOADING CMIP6 DATA ---\n")
        return summary
    except Exception as e:
        error_msg = f"Error in CMIP6 data search: {str(e)}"
        print(error_msg)
        print("--- END DOWNLOADING CMIP6 DATA (WITH ERROR) ---\n")
        return error_msg
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
    formatted_history = format_chat_history(chat_history)
    llm = create_llm(temperature=0)
    prompt = f"""
    Based on the following user query about CMIP6 data, determine which facets are relevant for the search.
    The possible facets are:
    - source_id: The model name in CMIP6 (e.g., CMCC-CM2-SR5, MPI-ESM1-2-LR)
    - frequency: Time frequency of the data (e.g., daily, monthly, yearly)
    - nominal_resolution: Spatial resolution of the data (e.g., 100 km, 250 km)
    - experiment_id: Experiment identifier (e.g., historical, ssp585)
    - variable_id: Variable identifier (e.g., tas for surface air temperature, pr for precipitation)
    - activity_id: Activity identifier for the CMIP6 project (e.g., CMIP, ScenarioMIP)
    - realm: Realm of the climate system (e.g., atmos, ocean, land)
    - sub_experiment_id: Sub-experiment identifier for specific initializations
    - variant_label: Identifier for the ensemble member specifying realization, initialization method, physics, and forcing indices (e.g., r1i1p1f1). The default value is r1i1p1f1.

    **Instructions:**
    1. **Vector Search Priority:** If `source_id`, `experiment_id`, or `variable_id` are relevant to the query, they should **always** be retrieved using vector search **before** considering other facets.
    2. **Facet Relevance:** Only include facets that are directly relevant to the user's query.
    3. **Comprehensive Analysis:** Consider the entire conversation history to accurately determine the user's intent and the relevance of each facet.
        
    Conversation:{formatted_history}

    User query: {query}

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
        display_debug_info("Parsed Facet Values", facet_values)
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON for facet values: {str(e)}")
        st.error(f"Raw content: {response.content}")
        facet_values = {}

    return facet_values