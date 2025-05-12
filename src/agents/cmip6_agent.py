# Updated implementation for src/agents/cmip6_agent.py

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import StructuredTool
from src.services.cmip6_service import cmip6_data_process, cmip6_data_search, cmip6_advise
from src.services.llm_service import create_llm, create_prompt_template
from pydantic import BaseModel, Field
from langchain.callbacks.base import BaseCallbackHandler
from src.config import Config
from typing import Dict, Any, List
import json
import sys
from io import StringIO
import traceback

class CMIP6DataSearchArgsSchema(BaseModel):
    query: str
class CMIP6AdviseArgsSchema(BaseModel):
    query: str
    relevant_facets: List[str]
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
        "Use this tool to determine the relevant CMIP6 facets from the user's request and perform a dataset search accordingly."
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
        "Use this tool when you already have all the necessary facet_values to fulfill the user's request." 
        "You can adjust facet_values if needed"
        "If the current request does not require modifying or obtaining new facet_values, you should call this tool directly. \n" 
        "For instance, if the user's question can be answered with previously identified facet_values or facet_values directly specified by the user, proceed with this tool."
        "Arguments must be a dictionary containing: \n"
	"•	query (string): The user's request"
	"•	facet_values (dict): The required facet parameters for the CMIP6 data \n"
    "•	download_opendap (boolean): True/False boolean variable to downoload/not-download the openDAP links. Ask user if the user wants to download openDAP links if the value is False \n"
    "If you lack the required facet_values or the user's request has changed in a way that necessitates re-evaluating them or if total_datasets = 0, use cmip6_datasets_search tool."
    ),
    args_schema=CMIP6DataProcessArgsSchema)
    return cmip6_access_tool

def create_cmip6_adviser_tool():
    cmip6_adviser_tool = StructuredTool.from_function(
    func=cmip6_advise,
    name="cmip6_adviser",
    description=(
        "Use this tool to answer user questions about CMIP6 parameters, including general inquiries or specific details about variables, source_id (models), or experiments.\n"
        "Only apply vector search when the question specifically involves variable_id, source_id, or experiment_id, and always include what the user is looking for in the query (e.g., if the user asks 'tos', adjust the query to 'variable tos').\n"
         "Do not use the tool for topics unrelated to CMIP6 parameters.\n"
         "Always include relevant_facets (e.g., if they ask 'tos', use '['variable_id']' for the relevant_facets or if they explicitly mention facet put it here)\n"

        "Arguments must be a dictionary containing:\n"

        "• query (string): The user's request, adjusted to clarify their intent\n"

        "• relevant_facets (list): List of relevant facets needed to answer the user's question\n"

        "• vector_search_fields (list): List of fields requiring vector search (source_id, variable_id, or experiment_id) — leave empty unless the question specifically involves these"
    ),
    args_schema=CMIP6AdviseArgsSchema)
    return cmip6_adviser_tool

# Define a function-based python REPL tool with a __name__ attribute
def python_repl(query: str) -> str:
    """
    Execute Python code and return the output.
    
    Args:
        query (str): The Python code to execute.
        
    Returns:
        str: The output of the executed code.
    """
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    
    # Create a dictionary for local variables
    local_vars = {}
    
    try:
        # Try to execute as an expression first
        try:
            result = eval(query, {}, local_vars)
            if result is not None:
                print(repr(result))
        except SyntaxError:
            # If fails as an expression, execute as a statement
            exec(query, {}, local_vars)
    except Exception as e:
        # Capture and return any errors
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
    
    # Restore stdout and get the output
    sys.stdout = old_stdout
    output = mystdout.getvalue()
    
    return output

# Arguments schema for the Python REPL
class PythonREPLSchema(BaseModel):
    query: str = Field(
        description="The Python code to execute. Input should be a valid Python command."
    )

def create_python_repl_tool():
    """
    Creates a Python REPL tool for executing Python code.
    
    Returns:
        StructuredTool: A tool that can execute Python code.
    """
    return StructuredTool.from_function(
        func=python_repl,
        name="python_repl",
        description=(
            "A Python shell. Use this to execute Python commands. Input should be a valid Python command. "
            "If you want to see the output of a value, you should print it out with `print(...)`. "
            "This tool is useful for data analysis, calculations, and creating visualizations."
        ),
        args_schema=PythonREPLSchema
    )

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
    # Create the Python REPL tool
    python_repl_tool = create_python_repl_tool()
    
    # Create the agent with LLM and all tools
    facet_capture_handler = FacetValuesCaptureHandler()
    all_tools = [cmip6_serach_tool, cmip6_access_tool, cmip6_adviser_tool, python_repl_tool]
    
    agent = create_openai_tools_agent(llm, all_tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent,
                                  tools=all_tools,
                                  callbacks=[facet_capture_handler],
                                  max_iterations=20,
                                  verbose=True)
    return agent_executor