# Agent implementation using langgraph.prebuilt (compatible with langchain ≥1.0)
# No more AgentExecutor or StructuredTool — uses @tool decorator + create_react_agent

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from src.services.cmip6_service import cmip6_data_process, cmip6_data_search, cmip6_advise
from src.services.llm_service import create_llm, create_prompt_template
from src.services.analysis_guide import analysis_guide_tool
from src.services.literature_service import cmip6_literature_search, cmip6_citation_graph
from src.tools.era5_monthly_tool import era5_monthly_tool
from src.config import Config
import os, uuid
import traceback
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server
import matplotlib.pyplot as plt
import sys
from io import StringIO
from typing import Dict, Any, List, Optional
import json


# ─── Tool Schemas ────────────────────────────────────────────────────

class CMIP6DataSearchArgs(BaseModel):
    """Schema for cmip6_datasets_search tool."""
    variable_query: Optional[str] = Field(
        default=None,
        description=(
            "Natural language description of the CLIMATE VARIABLE the user wants. "
            "Use the EXACT wording from the user's query — do NOT convert to CMIP6 IDs. "
            "Leave empty/null if the user asks for general categories like 'ocean data'."
        )
    )
    source_query: Optional[str] = Field(
        default=None,
        description="Natural language description of the CLIMATE MODEL the user wants."
    )
    experiment_query: Optional[str] = Field(
        default=None,
        description="Natural language description of the EXPERIMENT the user wants."
    )
    frequency: Optional[str] = Field(default=None, description="CMIP6 frequency code if mentioned (e.g., 'mon', 'day').")
    realm: Optional[str] = Field(default=None, description="Climate system realm ('ocean', 'atmos', 'seaIce', 'land').")
    nominal_resolution: Optional[str] = Field(default=None, description="Spatial resolution if mentioned.")
    activity_id: Optional[str] = Field(default=None, description="CMIP6 activity/MIP identifier if mentioned.")


class CMIP6AdviseArgs(BaseModel):
    query: str
    relevant_facets: List[str]
    vector_search_fields: List[str]


class CMIP6DataProcessArgs(BaseModel):
    query: str = Field(description="The user's original natural language query, e.g. 'monthly SST from MPI historical'.")
    facet_values: Dict[str, Any] = Field(
        description=(
            "A dict of CMIP6 facet values from the search results. "
            "Example: {'variable_id': 'tos', 'source_id': 'MPI-ESM1-2-LR', "
            "'experiment_id': 'historical', 'variant_label': 'r1i1p1f1'}. "
            "You MUST extract these from the cmip6_datasets_search output."
        )
    )


class PythonREPLArgs(BaseModel):
    query: str = Field(description="The Python code to execute.")


# ─── Python REPL ─────────────────────────────────────────────────────

class OptimizedPersistentPythonREPL:
    """Python REPL that saves plots by path instead of base64.
    Each session gets its own isolated instance to prevent cross-session leakage."""
    EXEC_TIMEOUT = 120  # seconds

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.locals = {}
        self.temp_dir = os.path.join(os.getcwd(), "temp_figures", session_id)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.environ['PYTHON_REPL_TEMP_DIR'] = self.temp_dir

        import pandas as pd
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import xarray as xr
        self.locals.update({
            'pd': pd, 'np': np, 'plt': plt, 'xr': xr
        })

    def run(self, query: str):
        import matplotlib.pyplot as plt
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

        # Lock protects global sys.stdout and matplotlib from concurrent access
        with _repl_lock:
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            saved_file_paths = []
            error = None

            def _exec_code():
                exec(query, self.locals)

            try:
                # Use ThreadPoolExecutor for timeout (thread-safe, unlike signal.alarm)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_exec_code)
                    future.result(timeout=self.EXEC_TIMEOUT)

                for num in plt.get_fignums():
                    fig = plt.figure(num)
                    fname = os.path.join(self.temp_dir, f"figure_{uuid.uuid4().hex}.png")
                    fig.savefig(fname, dpi=300, bbox_inches='tight')
                    saved_file_paths.append(fname)
                    plt.close(fig)
            except FuturesTimeout:
                error = f"Error: Code execution timed out after {self.EXEC_TIMEOUT}s"
                print(error)
            except Exception as e:
                error = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                print(error)
            finally:
                sys.stdout = old_stdout
                output = mystdout.getvalue()
            return {
                "stdout": output,
                "figure_paths": saved_file_paths,
                "error": error
            }


# ─── Session-scoped REPL registry ───────────────────────────────────
import threading

_repl_sessions: dict = {}
_repl_lock = threading.Lock()  # Protects global sys.stdout and matplotlib

def _get_repl(session_id: str = "default") -> OptimizedPersistentPythonREPL:
    """Returns a session-scoped REPL instance. Each session is isolated."""
    if session_id not in _repl_sessions:
        _repl_sessions[session_id] = OptimizedPersistentPythonREPL(session_id=session_id)
    return _repl_sessions[session_id]




# ─── Tool Definitions ───────────────────────────────────────────────

@tool(args_schema=CMIP6DataSearchArgs)
def cmip6_datasets_search(
    variable_query: str = None,
    source_query: str = None,
    experiment_query: str = None,
    frequency: str = None,
    realm: str = None,
    nominal_resolution: str = None,
    activity_id: str = None,
) -> str:
    """Search for CMIP6 datasets by splitting the user's request into component arguments.
    Extract the VARIABLE, MODEL, and EXPERIMENT from the user's query and pass them as
    separate natural-language arguments. The tool will resolve them to exact CMIP6 IDs via RAG,
    then automatically check data availability and return download info.
    IMPORTANT: Use the user's EXACT wording — do NOT convert to CMIP6 IDs yourself.
    Example: 'monthly SST from MPI model, historical' →
    variable_query='sea surface temperature', source_query='MPI model',
    experiment_query='historical', frequency='mon'
    For general categories like 'ocean data', use realm='ocean' instead of variable_query.
    """
    # Step 1: Search — resolve natural language to CMIP6 facet values
    facet_values, vector_search_results = cmip6_data_search(
        variable_query=variable_query,
        source_query=source_query,
        experiment_query=experiment_query,
        frequency=frequency,
        realm=realm,
        nominal_resolution=nominal_resolution,
        activity_id=activity_id,
    )

    # Step 2: Auto-chain into access — get availability + download info
    parts = [p for p in [variable_query, source_query, experiment_query] if p]
    query_text = ", ".join(parts) if parts else "CMIP6 data search"
    try:
        access_result = cmip6_data_process(query=query_text, facet_values=facet_values)
        # Combine search metadata + access results
        return json.dumps({
            "resolved_facets": facet_values,
            "access_info": access_result,
        }, default=str)
    except Exception as e:
        # If access fails, still return the resolved facets
        return json.dumps({
            "resolved_facets": facet_values,
            "search_schema": vector_search_results,
            "access_error": str(e),
        }, default=str)


@tool(args_schema=CMIP6DataProcessArgs)
def cmip6_datasets_access(query: str, facet_values: Dict[str, Any]) -> str:
    """Check data availability and get download info for CMIP6 datasets.
    You MUST pass facet_values as a dict extracted from cmip6_datasets_search results.
    Example call: cmip6_datasets_access(
        query="monthly SST from MPI",
        facet_values={"variable_id": "tos", "source_id": "MPI-ESM1-2-LR",
                      "experiment_id": "historical", "variant_label": "r1i1p1f1"}
    )
    If you don't have facet_values yet, use cmip6_datasets_search first.
    """
    result = cmip6_data_process(query=query, facet_values=facet_values)
    return json.dumps(result, default=str)


@tool(args_schema=CMIP6AdviseArgs)
def cmip6_adviser(query: str, relevant_facets: List[str], vector_search_fields: List[str]) -> str:
    """Answer questions about CMIP6 parameters (variables, models, experiments).
    Always include relevant_facets. Only include vector_search_fields when the
    question specifically involves variable_id, source_id, or experiment_id.
    """
    return cmip6_advise(query=query, relevant_facets=relevant_facets, vector_search_fields=vector_search_fields)


@tool(args_schema=PythonREPLArgs)
def python_repl(query: str, config: RunnableConfig = None) -> str:
    """A Python shell. Use this to execute Python commands. Input should be valid Python code.
    If you want to see the output of a value, print it with `print(...)`.
    Any matplotlib figures will be automatically saved and returned as file paths.
    """
    session_id = "default"
    if config and isinstance(config, dict):
        session_id = config.get("configurable", {}).get("session_id", "default")
    repl = _get_repl(session_id)
    result = repl.run(query)
    return json.dumps({
        "stdout": result.get("stdout", ""),
        "figure_paths": result.get("figure_paths", []),
        "error": result.get("error")
    })


# ─── Agent Factory ───────────────────────────────────────────────────

def create_cmip6_agent():
    """Creates a CMIP6 agent using langgraph's create_react_agent."""
    llm = create_llm()
    prompt_template = create_prompt_template()

    all_tools = [
        cmip6_datasets_search, cmip6_datasets_access, cmip6_adviser,
        cmip6_literature_search, cmip6_citation_graph,
        python_repl, analysis_guide_tool,
        era5_monthly_tool,
    ]

    # create_react_agent returns a compiled LangGraph
    # The prompt_template's system message is passed as the system prompt
    system_message = prompt_template.messages[0].content if prompt_template.messages else ""

    agent = create_react_agent(
        model=llm,
        tools=all_tools,
        prompt=system_message,
    )

    return agent