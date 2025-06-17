# Optimized implementation for src/agents/cmip6_agent.py
# This version saves plots by path instead of converting to base64 to reduce token usage

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.agents import AgentAction
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import LLMResult

import streamlit as st
from langchain.tools import StructuredTool
from src.services.cmip6_service import cmip6_data_process, cmip6_data_search, cmip6_advise, python_repl
from src.services.llm_service import create_llm, create_prompt_template
from pydantic import BaseModel, Field
from langchain.callbacks.base import BaseCallbackHandler
from src.config import Config
import os, uuid
import traceback
import matplotlib.pyplot as plt
import sys
from io import StringIO
from typing import Dict, Any, List, Union
import json


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
        if isinstance(output, dict) and "facet_values" in output:
            self.facet_values = output["facet_values"]
        else:
            try:
                data = json.loads(output)
                if "facet_values" in data:
                    self.facet_values = data["facet_values"]
            except:
                pass

class PythonREPLSchema(BaseModel):
    query: str = Field(
        description="The Python code to execute. Input should be a valid Python command."
    )

class OptimizedPersistentPythonREPL:
    """
    Optimized Python REPL that saves plots by path instead of converting to base64.
    This reduces token usage significantly.
    Automatically creates a new figure for each code execution to ensure all plots are preserved.
    """
    def __init__(self):
        self.locals = {}
        self.temp_dir = os.path.join(os.getcwd(), "temp_figures")
        os.makedirs(self.temp_dir, exist_ok=True)
        os.environ['PYTHON_REPL_TEMP_DIR'] = self.temp_dir
        
        # Pre-import common data science libraries
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import xarray as xr
        self.locals.update({
            'pd': pd,
            'np': np,
            'plt': plt,
            'xr': xr
        })
        
    def run(self, query: str):
        import matplotlib.pyplot as plt
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        saved_file_paths = []  # Store just the paths, not the binary data
        error = None
        try:
            # Always create a new figure before running user code
            # plt.figure()
            exec(query, self.locals)
            # Save all open figures after code execution
            for num in plt.get_fignums():
                fig = plt.figure(num)
                fname = os.path.join(self.temp_dir, f"figure_{uuid.uuid4().hex}.png")
                fig.savefig(fname, dpi=300, bbox_inches='tight')
                saved_file_paths.append(fname)
                plt.close(fig)
        except Exception as e:
            error = f"Error: {str(e)}\n\n"
            if isinstance(e, TypeError) and "No numeric data to plot" in str(e):
                error += "The data you're trying to plot doesn't contain numeric values. Please check your data and make sure it contains numeric values before plotting."
            elif isinstance(e, SyntaxError):
                error += "There was a syntax error in your code. Please check your Python syntax."
            else:
                error += f"Traceback:\n{traceback.format_exc()}"
            print(error)
        finally:
            sys.stdout = old_stdout
            output = mystdout.getvalue()
        return {
            "stdout": output,
            "figure_paths": saved_file_paths,  # Store paths instead of binary data
            "error": error
        }

class OptimizedHistoryAppendingToolCallbackHandler(BaseCallbackHandler):
    """
    Optimized callback handler that stores plot paths instead of binary data.
    """
    def __init__(self, python_repl_instance: OptimizedPersistentPythonREPL = None):
        super().__init__()
        self.pending_tool_calls: List[Dict[str, Any]] = []
        self.python_repl_instance = python_repl_instance

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if response.generations:
            for generation_list in response.generations:
                for generation in generation_list:
                    if hasattr(generation, 'message') and hasattr(generation.message, 'tool_calls') and generation.message.tool_calls:
                        ai_message_dict = {
                            "role": "assistant",
                            "content": generation.message.content,
                            "tool_calls": generation.message.tool_calls
                        }
                        self.pending_tool_calls.extend(ai_message_dict["tool_calls"])

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        if action.tool == "python_repl":
            matching_pending_call = None
            for call_info in self.pending_tool_calls:
                if call_info.get("function", {}).get("name") == action.tool:
                    matching_pending_call = call_info
                    break

            if matching_pending_call:
                self.current_tool_call_id_for_on_tool_end = matching_pending_call["id"]
            else:
                self.current_tool_call_id_for_on_tool_end = f"call_fallback_{uuid.uuid4().hex}"

    def on_tool_end(self, output: str, name: str, **kwargs: Any) -> None:
        tool_call_id_to_use = None
        found_pending_call_idx = -1

        for i, call_info in enumerate(self.pending_tool_calls):
            if call_info.get("function", {}).get("name") == name:
                tool_call_id_to_use = call_info["id"]
                found_pending_call_idx = i
                break

        if tool_call_id_to_use and "messages" in st.session_state:
            if name == "python_repl":
                try:
                    output_dict = json.loads(output)
                    
                    # Create the tool message with paths only
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call_id_to_use,
                        "content": output_dict.get("stdout", ""),
                        "figure_paths": output_dict.get("figure_paths", []),  # Store paths only
                        "error": output_dict.get("error")
                    }
                    
                    # Display any error if present
                    if tool_message["error"]:
                        st.error(tool_message["error"])
                    
                    # Display plots using optimized handler
                    if tool_message["figure_paths"]:
                        from src.utils.chat_utils import OptimizedStreamlitPlotHandler
                        OptimizedStreamlitPlotHandler.display_plots_from_paths(tool_message["figure_paths"])
                    
                    # Append the message to chat history
                    st.session_state.messages.append(tool_message)
                    
                    # Update the last assistant message to include the figure paths
                    for msg in reversed(st.session_state.messages):
                        if msg["role"] == "assistant" and "tool_calls" in msg:
                            if "figure_paths" not in msg:
                                msg["figure_paths"] = []
                            msg["figure_paths"].extend(tool_message["figure_paths"])
                            break
                            
                except json.JSONDecodeError:
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call_id_to_use,
                        "content": output
                    }
                    st.session_state.messages.append(tool_message)
            else:
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id_to_use,
                    "content": output
                }
                st.session_state.messages.append(tool_message)

            if found_pending_call_idx != -1:
                self.pending_tool_calls.pop(found_pending_call_idx)

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], name: str, **kwargs: Any) -> None:
        tool_call_id_to_use = None
        found_pending_call_idx = -1

        for i, call_info in enumerate(self.pending_tool_calls):
            if call_info.get("function", {}).get("name") == name:
                tool_call_id_to_use = call_info["id"]
                found_pending_call_idx = i
                break

        if tool_call_id_to_use and "messages" in st.session_state:
            error_content = f"Error executing tool '{name}': {str(error)}\n{traceback.format_exc()}"
            tool_message_with_error = {
                "role": "tool",
                "tool_call_id": tool_call_id_to_use,
                "content": json.dumps({"error": error_content, "stdout": "", "figure_paths": []}),
            }
            st.session_state.messages.append(tool_message_with_error)
            
            if found_pending_call_idx != -1:
                self.pending_tool_calls.pop(found_pending_call_idx)

def create_cmip6_search_tool():
    cmip6_search_tool = StructuredTool.from_function(
        func=cmip6_data_search,
        name="cmip6_datasets_search",
        description=(
            "Use this tool to determine the relevant CMIP6 facets from the user's request and perform a dataset search accordingly."
            "This tool is intended solely for identifying and retrieving dataset information based on facet criteria"
        ),
        args_schema=CMIP6DataSearchArgsSchema
    )
    return cmip6_search_tool

def create_cmip6_access_tool():
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
        args_schema=CMIP6DataProcessArgsSchema
    )
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
        args_schema=CMIP6AdviseArgsSchema
    )
    return cmip6_adviser_tool

def create_optimized_python_repl_tool(repl_instance: OptimizedPersistentPythonREPL):
    """
    Creates an optimized Python REPL tool that saves plots by path instead of base64.
    This significantly reduces token usage.
    """
    def python_repl_wrapper(query: str):
        result = repl_instance.run(query)
        
        # Display plots immediately using paths
        if result.get("figure_paths"):
            import streamlit as st
            from src.utils.chat_utils import OptimizedStreamlitPlotHandler
            OptimizedStreamlitPlotHandler.display_plots_from_paths(result["figure_paths"])
        
        # Return the result as JSON with paths only (no base64 conversion)
        return json.dumps({
            "stdout": result.get("stdout", ""),
            "figure_paths": result.get("figure_paths", []),
            "error": result.get("error")
        })
    
    return StructuredTool.from_function(
        func=python_repl_wrapper,
        name="python_repl",
        description=(
            "A Python shell. Use this to execute Python commands. Input should be a valid Python command. "
            "If you want to see the output of a value, you should print it out with `print(...)`. "
            "Any matplotlib figures you create will be automatically displayed in the chat interface. "
            "This tool is useful for data analysis, calculations, and creating visualizations."
        ),
        args_schema=PythonREPLSchema
    )

def create_cmip6_agent():
    """
    Creates an optimized agent to handle CMIP6 data processing requests.
    This version uses path-based plot storage to reduce token usage.
    """
    llm = create_llm()
    prompt_template = create_prompt_template()
    cmip6_search_tool = create_cmip6_search_tool()
    cmip6_access_tool = create_cmip6_access_tool()
    cmip6_adviser_tool = create_cmip6_adviser_tool()
    
    # Create the optimized Python REPL tool
    persistent_repl_instance = OptimizedPersistentPythonREPL()
    python_repl_tool = create_optimized_python_repl_tool(persistent_repl_instance)
    
    # Create the agent with optimized callbacks
    facet_capture_handler = FacetValuesCaptureHandler()
    history_appender_callback = OptimizedHistoryAppendingToolCallbackHandler(persistent_repl_instance)
    all_tools = [cmip6_search_tool, cmip6_access_tool, cmip6_adviser_tool, python_repl_tool]
    
    agent = create_openai_tools_agent(llm, all_tools, prompt_template)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=all_tools,
        callbacks=[history_appender_callback, facet_capture_handler],
        max_iterations=100,
        verbose=True
    )
    return agent_executor