from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import StructuredTool
from src.services.cmip6_service import cmip6_data_process
from src.services.llm_service import create_llm, create_prompt_template
from pydantic import BaseModel
from src.config import Config

class CMIP6DataProcessArgsSchema(BaseModel):
    query: str

def create_cmip6_tool():
    """
    Creates a structured tool for processing CMIP6 data requests.

    This tool is designed to handle user queries related to downloading or accessing CMIP6 climate data.
    It uses the `cmip6_data_process` function to execute the necessary data retrieval and processing.
    The tool is invoked when the user asks for specific climate data from the CMIP6 dataset, and its
    usage depends on the context of the conversation.

    Returns:
        cmip6_tool: A StructuredTool object that processes CMIP6 data requests.
    """
    cmip6_tool = StructuredTool.from_function(
    func=cmip6_data_process,
    name="cmip6_data_process",
    description=(
        "Use this tool to process CMIP6 data requests. "
        "Invoke it when the user asks for downloading or accessing specific CMIP6 climate data. "
        "Ensure to use the tool appropriately based on the conversation context."
    ),
    args_schema=CMIP6DataProcessArgsSchema)
    return cmip6_tool

def create_cmip6_agent():
    """
    Creates an agent to handle CMIP6 data processing requests using an LLM and a structured tool.
    
    Returns:
        agent_executor: An AgentExecutor object that can be used to process CMIP6 data queries interactively.
    """
    # Retrieve model name and API key from Config
    llm = create_llm()
    prompt_template = create_prompt_template()
    cmip6_tool = create_cmip6_tool()
    # Create the agent with LLM and CMIP6 tool
    agent = create_openai_tools_agent(llm, [cmip6_tool], prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=[cmip6_tool], verbose=True)
    return agent_executor