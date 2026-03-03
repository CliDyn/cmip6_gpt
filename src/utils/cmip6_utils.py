from pyesgf.search import SearchConnection
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from src.utils.chat_utils import format_chat_history
from src.utils.metrics import PipelineMetrics, pipeline_logger
from src.services.llm_service import create_llm
import urllib
import json


# --- Pydantic Response Models for Structured LLM Output ---

class FacetSelectionResponse(BaseModel):
    """Structured response for facet selection."""
    relevant_facets: List[str] = Field(
        description="List of CMIP6 facet names relevant to the user query"
    )
    requires_vector_search: bool = Field(
        description="Whether source_id, variable_id, or experiment_id are present and need vector search"
    )
    vector_search_fields: List[str] = Field(
        default_factory=list,
        description="List of fields requiring vector search (source_id, variable_id, experiment_id)"
    )


class OpenDAPDecision(BaseModel):
    """Structured response for OpenDAP download decision."""
    requires_download_opendap: bool = Field(
        default=False,
        description="Whether downloading OpenDAP links is required based on user request"
    )

def download_cmip6_data(**kwargs):
    """
    Downloads CMIP6 climate data based on the specified search parameters.
    """
    print("\n--- DOWNLOADING CMIP6 DATA ---")
    print(f"Search parameters: {kwargs}")

    try:
        conn = SearchConnection('https://esgf-data.dkrz.de/esg-search', distrib=True)
        facets = [
            'source_id', 'frequency', 'nominal_resolution', 'experiment_id',
            'variable_id', 'sub_experiment_id', 'activity_id', 'realm', 'institution_id',
            'table_id', 'member_id','grid_label'
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

        final_facet_values = {
            "hit_count": ctx.hit_count,
            "models": {}
        }

        datasets = ctx.search()
        param_counts = {}

        for dataset in datasets:
            source_ids = dataset.json.get('source_id')
            if not source_ids:
                continue
            source_id = source_ids[0]

            if source_id not in final_facet_values["models"]:
                final_facet_values["models"][source_id] = {
                    "dataset_count": 0
                }
                param_counts[source_id] = {}

            final_facet_values["models"][source_id]["dataset_count"] += 1

            for facet in facets[1:]:
                if facet in dataset.json and dataset.json[facet]:
                    values = dataset.json[facet]
                    if values == ["none"]:
                        continue

                    if facet not in final_facet_values["models"][source_id]:
                        final_facet_values["models"][source_id][facet] = {}

                    if facet not in param_counts[source_id]:
                        param_counts[source_id][facet] = {}

                    for value in values:
                        if not value or value.lower() == "none":
                            continue

                        if value not in param_counts[source_id][facet]:
                            param_counts[source_id][facet][value] = 0
                        param_counts[source_id][facet][value] += 1

        for model_id, model_data in final_facet_values["models"].items():
            for facet in facets[1:]:
                if facet in param_counts[model_id]:
                    if param_counts[model_id][facet]:
                        model_data[facet] = {
                            value: count
                            for value, count in param_counts[model_id][facet].items()
                            if count > 0
                        }
                        if not model_data[facet]:
                            del model_data[facet]

        summary = json.dumps(result, indent=2)
        query_for_python = dict_to_query_string(param_counts)
        detailed_summary = json.dumps(final_facet_values, indent=2)
        print(detailed_summary)
        print("--- END DOWNLOADING CMIP6 DATA ---\n")

        return summary, result['hit_count'], detailed_summary, query_for_python

    except Exception as e:
        error_msg = f"Error downloading CMIP6 data: {str(e)}"
        print(error_msg)
        return json.dumps({"error": error_msg}), 0, json.dumps({"hit_count": 0, "models": {}}), ""

def create_esgf_search_link(facet_values):
    """Creates a consistent ESGF search URL."""
    esgf_base_url = "https://esgf-data.dkrz.de/search/cmip6/?"
    active_facets = {}

    for key, value in facet_values.items():
        if value:
            if isinstance(value, list):
                cleaned_values = [str(v).strip() for v in value if v]
                if cleaned_values:
                    active_facets[key] = cleaned_values
            else:
                cleaned_value = str(value).strip()
                if cleaned_value:
                    active_facets[key] = [cleaned_value]

    if not active_facets:
        return esgf_base_url + "project=CMIP6"

    active_facets_json = json.dumps(active_facets)
    encoded_active_facets = urllib.parse.quote(active_facets_json)
    esgf_params = f"activeFacets={encoded_active_facets}&project=CMIP6"
    link = esgf_base_url + esgf_params
    return link


def select_facets(query: str, chat_history: List = None, metrics: Optional[PipelineMetrics] = None) -> Dict[str, Any]:
    """Selects relevant CMIP6 data facets for a given query."""
    pipeline_logger.info(f"Selecting facets for query: {query}")
    if chat_history is None:
        chat_history = []
    llm = create_llm(temperature=0)
    structured_llm = llm.with_structured_output(FacetSelectionResponse)

    prompt = f"""
    Based on the following user query about CMIP6 data, determine which facets are relevant for the search.
    User query: {query}
    The possible facets are:
    - source_id, frequency, nominal_resolution, experiment_id, variable_id,
      activity_id, institution_id, realm, sub_experiment_id, variant_label

    Instructions:
    1. Vector Search Priority: source_id, experiment_id, variable_id always via vector search.
    2. Only include directly relevant facets.
    3. For 'ocean data' etc → use realm.
    4. Prefer one primary facet over many weak ones.
    5. ALWAYS include variant_label.
    """

    try:
        result = structured_llm.invoke(prompt)
        result_dict = result.model_dump()
        pipeline_logger.info(f"Facets selected: {result_dict}")
        if metrics:
            metrics.record_facets_selected(result_dict["relevant_facets"])
        return result_dict
    except Exception as e:
        pipeline_logger.error(f"Structured output failed for select_facets: {e}")
        if metrics:
            metrics.record_error(f"select_facets structured output failed: {e}")
        return {
            "relevant_facets": [],
            "requires_vector_search": False,
            "vector_search_fields": []
        }

def download_opendap_or_not(query, chat_history: List = None):
    """Determine whether OpenDAP download is needed."""
    if chat_history is None:
        chat_history = []
    llm = create_llm(temperature=0)
    structured_llm = llm.with_structured_output(OpenDAPDecision)
    formatted_history = format_chat_history(chat_history)
    prompt = f"""
    Based on the following user query {query} and conversation history {formatted_history}, determine whether downloading OpenDAP links is required.
    Default: false. Set to true only if user explicitly requests OpenDAP links.
    """
    try:
        result = structured_llm.invoke(prompt)
        result_dict = result.model_dump()
        pipeline_logger.info(f"OpenDAP decision: {result_dict}")
        return result_dict
    except Exception as e:
        pipeline_logger.error(f"Structured output failed for download_opendap_or_not: {e}")
        return {"requires_download_opendap": False}

def select_facet_values(
    query: str,
    relevant_facets: List[str],
    dynamic_args_class,
    chat_history: List = None,
    metrics: Optional[PipelineMetrics] = None,
) -> Dict[str, Any]:
    """Selects appropriate values for the relevant CMIP6 facets."""
    if chat_history is None:
        chat_history = []
    formatted_history = format_chat_history(chat_history)
    llm = create_llm(temperature=0)

    prompt = f"""
    Based on the following user query about CMIP6 data and the relevant facets, determine appropriate values for each facet.
    Strictly select only what the user wants.
    ALWAYS keep 'variant_label' in facet_values unless user request specifies otherwise.
    Conversation: {formatted_history}
    User query: {query}

    Relevant facets and their descriptions:
    {json.dumps(dynamic_args_class.model_json_schema())}

    Return your response as a JSON object. Only include values for the listed facets.
    """

    try:
        structured_llm = llm.with_structured_output(dynamic_args_class)
        result = structured_llm.invoke(prompt)
        facet_values = {k: v for k, v in result.model_dump().items() if v is not None}
        pipeline_logger.info(f"Structured output facet values: {facet_values}")
    except Exception as e:
        pipeline_logger.warning(f"Structured output failed, falling back to raw JSON: {e}")
        if metrics:
            metrics.record_error(f"select_facet_values structured output failed: {e}")
        response = llm.invoke(prompt)
        try:
            json_str = response.content
            start = json_str.find('{')
            end = json_str.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = json_str[start:end]
            facet_values = json.loads(json_str)
        except json.JSONDecodeError as parse_err:
            pipeline_logger.error(f"Raw JSON fallback also failed: {parse_err}")
            facet_values = {}

    pipeline_logger.info(f"Initial Facet Values: {facet_values}")

    facet_values = _validate_facet_values(facet_values, dynamic_args_class, metrics)

    if metrics:
        metrics.record_facet_values(facet_values)

    return facet_values


def _validate_facet_values(
    facet_values: Dict[str, Any],
    dynamic_args_class,
    metrics: Optional[PipelineMetrics] = None,
) -> Dict[str, Any]:
    """Validates that LLM-selected facet values are within allowed options."""
    schema = dynamic_args_class.model_json_schema()
    properties = schema.get("properties", {})
    validated = {}

    for facet, value in facet_values.items():
        if value == "UNMATCHED":
            pipeline_logger.warning(f"Validation: LLM selected UNMATCHED for '{facet}' — dropping.")
            if metrics:
                metrics.record_validation(facet, value, accepted=False)
            continue

        if facet not in properties:
            pipeline_logger.warning(f"Validation: facet '{facet}' not in dynamic schema, skipping")
            if metrics:
                metrics.record_validation(facet, value, accepted=False)
            continue

        prop = properties[facet]
        allowed_values = None

        if "enum" in prop:
            allowed_values = prop["enum"]
        elif "anyOf" in prop:
            for option in prop["anyOf"]:
                if "enum" in option:
                    allowed_values = option["enum"]
                    break
        elif "allOf" in prop:
            for option in prop["allOf"]:
                if "enum" in option:
                    allowed_values = option["enum"]
                    break

        if allowed_values is not None:
            real_allowed = [v for v in allowed_values if v != "UNMATCHED"]

            if isinstance(value, list):
                valid_items = []
                for item in value:
                    if item == "UNMATCHED":
                        continue
                    if item in real_allowed:
                        if metrics:
                            metrics.record_validation(facet, item, accepted=True)
                        valid_items.append(item)
                    else:
                        pipeline_logger.warning(f"Validation HALLUCINATION: {facet}='{item}'")
                        if metrics:
                            metrics.record_validation(facet, item, accepted=False)
                if valid_items:
                    validated[facet] = valid_items if len(valid_items) > 1 else valid_items[0]
                else:
                    pipeline_logger.warning(f"Validation: all values for '{facet}' were hallucinated")
            else:
                if value in real_allowed:
                    validated[facet] = value
                    if metrics:
                        metrics.record_validation(facet, value, accepted=True)
                else:
                    pipeline_logger.warning(f"Validation HALLUCINATION: {facet}='{value}'")
                    if metrics:
                        metrics.record_validation(facet, value, accepted=False)
        else:
            validated[facet] = value
            if metrics:
                metrics.record_validation(facet, value, accepted=True)

    removed = set(facet_values.keys()) - set(validated.keys())
    if removed:
        pipeline_logger.warning(f"Validation removed facets: {removed}")

    return validated

def dict_to_query_string(data):
    """Convert a nested dictionary into a query string format."""
    attributes = {
        'activity_id': set(), 'institution_id': set(), 'source_id': set(),
        'experiment_id': set(), 'member_id': set(), 'table_id': set(),
        'variable_id': set(), 'grid_label': set()
    }

    attributes['source_id'] = set(data.keys())

    for model_name, model_data in data.items():
        for attr, values in model_data.items():
            if attr in attributes:
                attributes[attr].update(values.keys())

    member_ids = set()
    for model_data in data.values():
        if 'member_id' in model_data:
            member_ids.update(model_data['member_id'].keys())
    if len(member_ids) == 1:
        attributes['member_id'] = member_ids.pop()

    table_ids = set()
    for model_data in data.values():
        if 'table_id' in model_data:
            table_ids.update(model_data['table_id'].keys())
    if len(table_ids) == 1:
        attributes['table_id'] = table_ids.pop()

    variable_ids = set()
    for model_data in data.values():
        if 'variable_id' in model_data:
            variable_ids.update(model_data['variable_id'].keys())
    if len(variable_ids) == 1:
        attributes['variable_id'] = variable_ids.pop()

    parts = []
    for attr, values in attributes.items():
        if isinstance(values, set):
            values_list = sorted(list(values))
            if len(values_list) == 1:
                parts.append(f"{attr} == '{values_list[0]}'")
            else:
                formatted_values = "', '".join(values_list)
                parts.append(f"{attr} == ['{formatted_values}']")
        else:
            parts.append(f"{attr} == '{values}'")

    return " & ".join(parts)