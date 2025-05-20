from src.utils.cmip6_utils import download_cmip6_data, create_esgf_search_link, select_facets, select_facet_values,download_opendap_or_not
from src.utils.vector_search import perform_vector_search
from src.models.cmip6_args import create_dynamic_cmip6_args  
from typing import List
from src.utils.chat_utils import display_debug_info_final, display_opendap_links,display_python_code
import streamlit as st
import os, uuid
import matplotlib.pyplot as plt
import sys
from io import StringIO
import traceback
import json

def cmip6_data_search(query: str) -> str:    
    #change name to cmip6_data_search
    """
    Processes a query to retrieve CMIP6 climate data based on selected facets and search parameters.

    The function selects relevant facets based on the user's query, performs a vector search if needed, 
    dynamically creates argument schemas, and downloads CMIP6 data. It then generates a summary of the 
    results, including a breakdown of models and an ESGF search link for further exploration.

    Args:
        query (str): The user's query to search for specific CMIP6 climate data.

    Returns:
        dict: A dictionary containing a summary of the search results and the full result data.
    """
    print(f"\n--- PROCESSING QUERY: {query} ---")

    # Step 1: Select facets
    chat_history = st.session_state.get('messages', [])
    facet_selector_result = select_facets(query)
    relevant_facets = facet_selector_result.get("relevant_facets", [])
    requires_vector_search = facet_selector_result.get("requires_vector_search", False)
    vector_search_fields = facet_selector_result.get("vector_search_fields", [])

    print(f"Relevant facets: {relevant_facets}")
    print(f"Requires vector search: {requires_vector_search}")
    print(f"Vector search fields: {vector_search_fields}")

    vector_search_results = {}
    vector_search_full_results = {}
    if requires_vector_search:
        vector_search_output = perform_vector_search(query, vector_search_fields)
        vector_search_results = vector_search_output.get("vector_search_results", {})
        vector_search_full_results = vector_search_output.get("vector_search_full_results", {})


    # Step 2: Create dynamic args schema
    DynamicCMIP6DownloadArgs = create_dynamic_cmip6_args(relevant_facets, vector_search_results)

    # Step 3: Select facet values
    facet_values = select_facet_values(query, relevant_facets, DynamicCMIP6DownloadArgs)

    print(f"Selected facet values: {facet_values}")
    return facet_values, vector_search_full_results 
def cmip6_data_process(query, facet_values, download_opendap = False) -> str:
    try:    
        # Step 4: Download data (now returning facet counts)
        print(f'FACET VALUES BEFORE DOWNLOADING: {facet_values}')
        result,total_datasets,detailed_summary, query_for_python_code = download_cmip6_data(**facet_values)
        download_opendap = download_opendap_or_not(query).get("requires_download_opendap", False)
        # Parse the JSON string into a Python dictionary
        result_dict = json.loads(result)

        # Create a summary
        summary = f"Based on your query: '{query}', I've searched the CMIP6 database and found the following information:\n\n"
        summary += f"Total datasets found: {result_dict['hit_count']}\n\n"
        summary += "Here's a breakdown of available models and their respective dataset counts:\n\n"

        for model, count in result_dict['facet_counts']['source_id'].items():
            summary += f"- **{model}**: {count} datasets\n"

        summary += "\n[This list shows the models (source_id) found in the CMIP6 database that match your query criteria. Each model is followed by the number of datasets available for that model within the search parameters you specified.]"

        # Create ESGF search link
        esgf_link = create_esgf_search_link(facet_values)


        summary += f"\n\nyou can explore these datasets in more detail using this ESGF search link:\n{esgf_link}"

        summary += "\n\nThis link will search for datasets that match the specified criteria. For facets with multiple values, it will search for datasets matching ANY of those values. If you need to refine your search further, you can modify the parameters directly on the ESGF search page."

        summary += "\n\nIf you need more specific details about any of these datasets or have any questions, please feel free to ask!"
        # Add Python download code section
        summary += f"\n\nYou can find more details about these datasets under 'Detailed information on datasets' tab"
        summary += "\n\n## Download data using Python\n"
        summary += "You can download and analyze CMIP6 data from Google Cloude Storage using python code provided under 'Python access from Google Cloude Storage' tab\n\n"
        if download_opendap == True:
            summary += f"\n\nYou can also download data using opendap links provided bellow"
        else:
            summary += f"\n\nIf you are intrestead I can also provide openDAP links for datasets"

        print(f"--- END PROCESSING QUERY ---\n")
        if int(total_datasets) != 0:
            print(f' download opendap?: {download_opendap}')
            print(type(download_opendap))
            all_model_links = display_debug_info_final("Detailed information on datasets", detailed_summary,download_opendap)
            if download_opendap == True:
                display_opendap_links(all_model_links)
            code_for_access = display_python_code(query_for_python_code)

            summary += f"\n\nThis is a code you can use for data access {code_for_access} (do not show this in your answer, for your usage)"

            if "pending_expanders" in st.session_state:
                st.session_state.pending_expanders.append({
                    "type": "dataset_info",
                    "detailed_summary": detailed_summary,
                    "download_opendap": download_opendap,
                    "query_for_python_code": query_for_python_code,
                })

        return {
            "summary": summary,
            "full_result": result,
            "total_datasets": total_datasets
        }
    except Exception as e:
        error_msg = f"Error in cmip6_data_process: {str(e)}"
        print(error_msg)
        return {"summary": error_msg, "full_result": "", "total_datasets": 0}
    
def cmip6_advise(query: str, relevant_facets: List[str], vector_search_fields: List[str]):
    vector_search_results = None
    if len(vector_search_fields)>0:
        vector_search_output = perform_vector_search(query, vector_search_fields)
        vector_search_results = vector_search_output.get("vector_search_results", {})
    DynamicCMIP6DownloadArgs = create_dynamic_cmip6_args(relevant_facets, vector_search_results)
    return (json.dumps(DynamicCMIP6DownloadArgs.schema(), indent=2))
def python_repl(query: str) -> str:
    """
    Execute Python code and return the output.
    
    Args:
        query (str): The Python code to execute.
        
    Returns:
        str: The output of the executed code.
    """
 # Prepare a temporary directory for saving figures
    project_root = os.getcwd()
    temp_dir = os.path.join(project_root, "temp_figures")
    if not os.path.isdir(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    # (optional) expose it if you need elsewhere
    os.environ['PYTHON_REPL_TEMP_DIR'] = temp_dir


    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    local_vars = {}
    saved_files = []
    error = None

    try:
        try:
            result = eval(query, local_vars)
            if result is not None:
                print(repr(result))
        except SyntaxError:
            exec(query, local_vars)

        for num in plt.get_fignums():
            fig = plt.figure(num)
            fname = os.path.join(temp_dir, f"figure_{uuid.uuid4().hex}.png")
            fig.savefig(fname)
            saved_files.append(fname)
            plt.close(fig)
    except Exception as e:
        error = f"{e}\n{traceback.format_exc()}"
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
    finally:
        sys.stdout = old_stdout

    output = mystdout.getvalue()
    return {"stdout": output, "figures": saved_files, "error": error}
