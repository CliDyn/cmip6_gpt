# Updated implementation for src/agents/cmip6_agent.py

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.agents import AgentAction
from langchain_core.messages import AIMessage, ToolMessage # Import these
from langchain_core.outputs import LLMResult # For on_llm_end if needed

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

class HistoryAppendingToolCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        # Use a list to handle potential (though less common for simple agents) multiple tool calls
        # requested by the LLM in a single step.
        self.pending_tool_calls: List[Dict[str, Any]] = []

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running. The response contains the AI's decision, including tool calls."""
        # This is often where the AIMessage with tool_calls is finalized by Langchain.
        # We want to ensure our history reflects this.
        # The response.generations[0][0].message often is an AIMessage object here.
        if response.generations:
            for generation_list in response.generations:
                for generation in generation_list:
                    if hasattr(generation, 'message') and hasattr(generation.message, 'tool_calls') and generation.message.tool_calls:
                        # This is an AIMessage from the LLM that includes tool calls
                        # Langchain's agent executor will typically handle adding this AIMessage to its internal state.
                        # We are trying to ensure st.session_state.messages mirrors this.

                        # Convert AIMessage to dict if it's not already
                        ai_message_dict = {
                            "role": "assistant",
                            "content": generation.message.content, # May be None
                            "tool_calls": generation.message.tool_calls # This is already in the correct format
                        }

                        # Store the tool_call_ids that are pending a response
                        self.pending_tool_calls.extend(ai_message_dict["tool_calls"])

                        # Append this AI message (with tool call) to our st.session_state.messages
                        # Ensure no duplicates if Langchain itself also modifies a shared history object
                        # For now, let's assume we are the primary mechanism for updating st.session_state.messages
                        # with these specific tool call/response dicts.
                        if "messages" in st.session_state:
                            # Check if this exact message (or one very similar) was just added by on_agent_action
                            # This is a bit tricky. on_agent_action is more about the *parsed* action.
                            # on_llm_end gives the raw LLM message.
                            # For now, let's assume on_agent_action is our primary trigger for the AI's tool_call message.
                            pass # We will rely on on_agent_action for this.
                        print(f"HISTORY_CALLBACK (on_llm_end): LLM decided to make tool calls: {ai_message_dict['tool_calls']}")


    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action. This is where the agent decides to call a tool."""
 
    
        if action.tool == "python_repl":
             # We need to find the 'id' of this specific tool call from self.pending_tool_calls
             # This matching is tricky if the LLM requests multiple calls to the *same* tool.
             # Let's assume for now it's the first pending python_repl call.
            matching_pending_call = None
            for call_info in self.pending_tool_calls:
                if call_info.get("function", {}).get("name") == action.tool:
                    # Crude match: first one for this tool name.
                    # A better match would use arguments if available and distinct.
                    # Or, if the agent framework provides the ID in `kwargs` or `action`.
                    matching_pending_call = call_info
                    break

            if matching_pending_call:
                self.current_tool_call_id_for_on_tool_end = matching_pending_call["id"]
                print(f"HISTORY_CALLBACK (on_agent_action): Matched tool call for python_repl. ID: {self.current_tool_call_id_for_on_tool_end}")
            else:
                # If no match, generate a new one and hope it aligns or the system is robust.
                # This is a fallback and indicates a potential issue in tracking.
                self.current_tool_call_id_for_on_tool_end = f"call_fallback_{uuid.uuid4().hex}"
                print(f"HISTORY_CALLBACK (on_agent_action): Could not find exact pending call for python_repl. Using fallback ID: {self.current_tool_call_id_for_on_tool_end}")
            # No, let's not add the AIMessage here. The AgentExecutor should do that.
            # The callback handler should only be concerned with adding the ToolMessage.


    def on_tool_end(
        self,
        output: str, # String representation of the tool's output dictionary
        name: str, # Name of the tool that just ran
        **kwargs: Any,
    ) -> None:
        """Run when tool ends running."""
        # Try to find the tool_call_id from the pending calls.
        # `name` argument here is the name of the tool that just finished.
        tool_call_id_to_use = None
        found_pending_call_idx = -1

        for i, call_info in enumerate(self.pending_tool_calls):
            if call_info.get("function", {}).get("name") == name:
                tool_call_id_to_use = call_info["id"]
                found_pending_call_idx = i
                break # Assume one call to this tool for now, or first one.

        if tool_call_id_to_use and "messages" in st.session_state:
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call_id_to_use,
                "content": output, # The raw string output from the tool
            }
            st.session_state.messages.append(tool_message)
            print(f"HISTORY_CALLBACK (on_tool_end): Appended ToolMessage for ID {tool_call_id_to_use}, tool '{name}'.")

            # Remove this call from pending_tool_calls
            if found_pending_call_idx != -1:
                self.pending_tool_calls.pop(found_pending_call_idx)
        else:
            if not tool_call_id_to_use:
                print(f"HISTORY_CALLBACK (on_tool_end): Could not find a pending tool_call_id for tool '{name}'. Tool output not added to history in tool format.")
            # else: (messages not in session_state, less likely)


    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], name: str, **kwargs: Any
    ) -> None:
        """Run when tool errors."""
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
                "content": json.dumps({"error": error_content, "stdout": "", "figures": []}), # Ensure content is string
            }
            st.session_state.messages.append(tool_message_with_error)
            print(f"HISTORY_CALLBACK (on_tool_error): Appended ToolMessage (error) for ID {tool_call_id_to_use}, tool '{name}'.")
            if found_pending_call_idx != -1:
                self.pending_tool_calls.pop(found_pending_call_idx)
        else:
            if not tool_call_id_to_use:
                 print(f"HISTORY_CALLBACK (on_tool_error): Could not find a pending tool_call_id for errored tool '{name}'. Error not added to history in tool format.")

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

# Arguments schema for the Python REPL
class PythonREPLSchema(BaseModel):
    query: str = Field(
        description="The Python code to execute. Input should be a valid Python command."
    )
class PersistentPythonREPL:
    def __init__(self):
        self.locals = {}
        self.temp_dir = os.path.join(os.getcwd(), "temp_figures")
        os.makedirs(self.temp_dir, exist_ok=True)
        os.environ['PYTHON_REPL_TEMP_DIR'] = self.temp_dir

    def run(self, query: str):
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        saved_files = []
        error = None

        try:
            try:
                result = eval(query, self.locals)
                if result is not None:
                    print(repr(result))
            except SyntaxError:
                exec(query, self.locals)

            for num in plt.get_fignums():
                fig = plt.figure(num)
                fname = os.path.join(self.temp_dir, f"figure_{uuid.uuid4().hex}.png")
                fig.savefig(fname, dpi=300)
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

def create_python_repl_tool(repl_instance: PersistentPythonREPL):
    """
    Creates a Python REPL tool for executing Python code.
    
    Returns:
        StructuredTool: A tool that can execute Python code.
    """
    return StructuredTool.from_function(
        func=repl_instance.run,
        name="python_repl",
        description=(
            "A Python shell. Use this to execute Python commands. Input should be a valid Python command. "
            "If you want to see the output of a value, you should print it out with `print(...)`. "
            "Any matplotlib figures you create will be saved with unique filenames in your system temp folder; "
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
    persistent_repl_instance = PersistentPythonREPL()
    python_repl_tool = create_python_repl_tool(persistent_repl_instance)
    
    # Create the agent with LLM and all tools
    facet_capture_handler = FacetValuesCaptureHandler()
    history_appender_callback = HistoryAppendingToolCallbackHandler()
    all_tools = [cmip6_serach_tool, cmip6_access_tool, cmip6_adviser_tool, python_repl_tool]
    
    agent = create_openai_tools_agent(llm, all_tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent,
                                  tools=all_tools,
                                  callbacks=[history_appender_callback, facet_capture_handler],
                                  max_iterations=20,
                                  verbose=True)
    return agent_executor