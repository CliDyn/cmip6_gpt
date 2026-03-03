from src.utils.cmip6_utils import download_cmip6_data, create_esgf_search_link, select_facet_values, download_opendap_or_not
from src.utils.vector_search import perform_vector_search, perform_direct_vector_search
from src.models.cmip6_args import create_dynamic_cmip6_args  
from src.utils.metrics import PipelineMetrics, pipeline_logger
from src.utils.chat_utils import generate_python_code
from typing import List, Optional
import os, uuid
import matplotlib.pyplot as plt
import sys
from io import StringIO
import traceback
import json

def cmip6_data_search(
    variable_query: str = None,
    source_query: str = None,
    experiment_query: str = None,
    frequency: str = None,
    realm: str = None,
    nominal_resolution: str = None,
    activity_id: str = None,
    chat_history: list = None,
) -> str:
    """
    Processes a CMIP6 search using pre-split query arguments from the agent.
    Total LLM calls: 1 (select_facet_values only).
    """
    parts = [p for p in [variable_query, source_query, experiment_query] if p]
    original_query = ", ".join(parts) if parts else "CMIP6 data search"

    pipeline_logger.info(f"Processing search: variable='{variable_query}', source='{source_query}', "
                         f"experiment='{experiment_query}', freq='{frequency}', realm='{realm}'")
    metrics = PipelineMetrics()
    metrics.set_query(original_query)

    # Step 1: Derive relevant facets
    relevant_facets = []
    vector_search_queries = {}

    if variable_query:
        relevant_facets.append("variable_id")
        vector_search_queries["variable_id"] = variable_query
    if source_query:
        relevant_facets.append("source_id")
        vector_search_queries["source_id"] = source_query
    if experiment_query:
        relevant_facets.append("experiment_id")
        vector_search_queries["experiment_id"] = experiment_query
    if frequency:
        relevant_facets.append("frequency")
    if realm:
        relevant_facets.append("realm")
    if nominal_resolution:
        relevant_facets.append("nominal_resolution")
    if activity_id:
        relevant_facets.append("activity_id")

    relevant_facets.append("variant_label")

    pipeline_logger.info(f"Derived relevant facets: {relevant_facets}")

    if metrics:
        metrics.record_facets_selected(relevant_facets)

    # Step 2: Direct vector search
    vector_search_results = {}
    if vector_search_queries:
        with metrics.step("vector_search"):
            search_output = perform_direct_vector_search(
                split_queries=vector_search_queries,
                original_query=original_query,
            )
        vector_search_results = search_output.get("vector_search_results", {})

    # Step 3: Create dynamic args schema
    with metrics.step("create_dynamic_schema"):
        DynamicCMIP6DownloadArgs = create_dynamic_cmip6_args(relevant_facets, vector_search_results)
    vector_search_full_results = DynamicCMIP6DownloadArgs.model_json_schema()

    # Step 4: Select facet values (ONLY LLM call)
    with metrics.step("select_facet_values"):
        facet_values = select_facet_values(
            original_query, relevant_facets, DynamicCMIP6DownloadArgs,
            chat_history=chat_history or [], metrics=metrics
        )

    pipeline_logger.info(f"Selected facet values: {facet_values}")
    metrics.log_summary()
    return facet_values, vector_search_full_results


def cmip6_data_process(query, facet_values, download_opendap=False, chat_history=None) -> dict:
    """Process CMIP6 data request and return structured result (no Streamlit)."""
    try:
        print(f'FACET VALUES BEFORE DOWNLOADING: {facet_values}')
        result, total_datasets, detailed_summary, query_for_python_code = download_cmip6_data(**facet_values)
        download_opendap = download_opendap_or_not(query, chat_history=chat_history or []).get("requires_download_opendap", False)
        result_dict = json.loads(result)

        summary = f"Based on your query: '{query}', I've searched the CMIP6 database and found the following information:\n\n"
        summary += f"Total datasets found: {result_dict['hit_count']}\n\n"
        summary += "Here's a breakdown of available models and their respective dataset counts:\n\n"

        for model, count in result_dict['facet_counts']['source_id'].items():
            summary += f"- **{model}**: {count} datasets\n"

        esgf_link = create_esgf_search_link(facet_values)
        summary += f"\n\nYou can explore these datasets using this ESGF search link:\n{esgf_link}"

        python_code = generate_python_code(query_for_python_code)
        summary += "\n\nYou can find more details about these datasets under 'Detailed information on datasets' tab"
        summary += "\n\n## Download data using Python\n"
        summary += "You can download and analyze CMIP6 data from Google Cloud Storage using the python code provided under 'Python access from Google Cloud Storage' tab\n\n"

        if download_opendap:
            summary += "\n\nYou can also download data using OpenDAP links provided below"
        else:
            summary += "\n\nIf you are interested I can also provide OpenDAP links for datasets"

        print(f"--- END PROCESSING QUERY ---\n")

        return {
            "summary": summary,
            "full_result": result,
            "total_datasets": total_datasets,
            "detailed_summary": detailed_summary,
            "python_code": python_code,
            "query_for_python_code": query_for_python_code,
            "esgf_link": esgf_link,
            "download_opendap": download_opendap,
        }
    except Exception as e:
        error_msg = f"Error in cmip6_data_process: {str(e)}"
        print(error_msg)
        return {"summary": error_msg, "full_result": "", "total_datasets": 0}


def cmip6_advise(query: str, relevant_facets: List[str], vector_search_fields: List[str]):
    vector_search_results = None
    if len(vector_search_fields) > 0:
        vector_search_output = perform_vector_search(query, vector_search_fields)
        vector_search_results = vector_search_output.get("vector_search_results", {})
    DynamicCMIP6DownloadArgs = create_dynamic_cmip6_args(relevant_facets, vector_search_results)
    return json.dumps(DynamicCMIP6DownloadArgs.model_json_schema(), indent=2)


def python_repl(query: str) -> str:
    """Execute Python code and return the output."""
    project_root = os.getcwd()
    temp_dir = os.path.join(project_root, "temp_figures")
    if not os.path.isdir(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    os.environ['PYTHON_REPL_TEMP_DIR'] = temp_dir

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
