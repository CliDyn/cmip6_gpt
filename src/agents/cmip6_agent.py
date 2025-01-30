from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import StructuredTool
from src.services.cmip6_service import cmip6_data_process, cmip6_data_search
from src.services.llm_service import create_llm, create_prompt_template
from pydantic import BaseModel
from langchain.callbacks.base import BaseCallbackHandler
from src.config import Config
from src.utils.vector_search import perform_vector_search
from typing import Dict, Any, List
import json

class CMIP6DataSearchArgsSchema(BaseModel):
    query: str
class CMIP6AdviseArgsSchema(BaseModel):
    query: str
    vector_search_fields: List[str]
class CMIP6DataProcessArgsSchema(BaseModel):
    query: str
    facet_values: Dict[str, Any]
class FacetValuesCaptureHandler(BaseCallbackHandler):
    def __init__(self):
        self.facet_values = None

    def on_tool_end(self, output, **kwargs):
        # Attempt to capture facet_values from the first tool output
        # If output is already a dict with 'facet_values':
        if isinstance(output, dict) and "facet_values" in output:
            self.facet_values = output["facet_values"]
        else:
            # If it's a string, try to parse it as JSON
            try:
                data = json.loads(output)
                if "facet_values" in data:
                    self.facet_values = data["facet_values"]
            except:
                pass

def create_cmip6_search_tool():
    """
    Creates a structured tool for processing CMIP6 data requests.

    This tool is designed to handle user queries related to downloading or accessing CMIP6 climate data.
    It uses the `cmip6_data_process` function to execute the necessary data retrieval and processing.
    The tool is invoked when the user asks for specific climate data from the CMIP6 dataset, and its
    usage depends on the context of the conversation.

    Returns:
        cmip6_tool: A StructuredTool object that processes CMIP6 data requests.
    """
    cmip6_serach_tool = StructuredTool.from_function(
    func=cmip6_data_search,
    name="cmip6_datasets_search",
    description=(
        "Use this tool to determine the relevant CMIP6 facets from the user’s request and perform a dataset search accordingly."
        "This tool is intended solely for identifying and retrieving dataset information based on facet criteria"
    ),
    args_schema=CMIP6DataSearchArgsSchema)
    return cmip6_serach_tool

def create_cmip6_access_tool():
    """
    Creates a structured tool for processing CMIP6 data requests.

    This tool is designed to handle user queries related to downloading or accessing CMIP6 climate data.
    It uses the `cmip6_data_process` function to execute the necessary data retrieval and processing.
    The tool is invoked when the user asks for specific climate data from the CMIP6 dataset, and its
    usage depends on the context of the conversation.

    Returns:
        cmip6_tool: A StructuredTool object that processes CMIP6 data requests.
    """
    cmip6_access_tool = StructuredTool.from_function(
    func=cmip6_data_process,
    name="cmip6_datasets_access",
    description=(
        "Use this tool when you already have all the necessary facet_values to fulfill the user’s request." 
        "You can adjust facet_values if needed"
        "If the current request does not require modifying or obtaining new facet_values, you should call this tool directly. \n" 
        "For instance, if the user’s question can be answered with previously identified facet_values or facet_values directly specified by the user, proceed with this tool."
        "Arguments must be a dictionary containing: \n"
	"•	query (string): The user’s request"
	"•	facet_values (dict): The required facet parameters for the CMIP6 data \n"

    "If you lack the required facet_values or the user’s request has changed in a way that necessitates re-evaluating them or if total_datasets = 0, use cmip6_datasets_search tool."
    ),
    args_schema=CMIP6DataProcessArgsSchema)
    return cmip6_access_tool
def create_cmip6_adviser_tool():
    cmip6_adviser_tool = StructuredTool.from_function(
    func=perform_vector_search,
    name="cmip6_adviser",
    description=(
        "Use this tool to answer user questions regarding ONLY variables, source_id (models), or experiments. "
        "Always add to the query what the user is looking for. For example, if the question is 'tos', you need to write in the query 'variable tos'."
    "Arguments must be a dictionary containing: \n"
	"•	query (string): The user’s request"
	"•	vector_search_fields (list): [list of fields requiring vector search]:  source_id, variable_id, or experiment_id \n"
    ),
    args_schema=CMIP6AdviseArgsSchema)
    return cmip6_adviser_tool
def create_cmip6_agent():
    """
    Creates an agent to handle CMIP6 data processing requests using an LLM and a structured tool.
    
    Returns:
        agent_executor: An AgentExecutor object that can be used to process CMIP6 data queries interactively.
    """
    # Retrieve model name and API key from Config
    llm = create_llm()
    prompt_template = create_prompt_template()
    cmip6_serach_tool = create_cmip6_search_tool()
    cmip6_access_tool = create_cmip6_access_tool()
    cmip6_adviser_tool = create_cmip6_adviser_tool()
    # Create the agent with LLM and CMIP6 tool
    facet_capture_handler = FacetValuesCaptureHandler()
    agent = create_openai_tools_agent(llm, [cmip6_serach_tool,cmip6_access_tool,cmip6_adviser_tool], prompt_template)
    agent_executor = AgentExecutor(agent=agent,
                                    tools=[cmip6_serach_tool,cmip6_access_tool,cmip6_adviser_tool],
                                    callbacks=[facet_capture_handler],
                                    max_iterations=20,
                                      verbose=True)
    return agent_executor