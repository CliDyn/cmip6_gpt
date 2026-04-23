# Agent implementation using langgraph.prebuilt (compatible with langchain ≥1.0)
# No more AgentExecutor or StructuredTool — uses @tool decorator + create_react_agent

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from src.services.cmip6_service import cmip6_data_process, cmip6_data_search, cmip6_data_search_batch, cmip6_advise
from src.services.llm_service import create_llm, create_prompt_template, create_reviewer_llm
from src.services.analysis_guide import analysis_guide_tool
from src.services.literature_service import cmip6_literature_search, cmip6_citation_graph
from src.services.methodology_rag_service import cmip6_methodology_check
from src.tools.era5_monthly_tool import era5_monthly_tool
from src.config import Config
import os, uuid, base64
import traceback
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server
import matplotlib.pyplot as plt
import sys
from io import StringIO
from typing import Dict, Any, List, Optional, Literal
import json


# ─── Tool Schemas ────────────────────────────────────────────────────

class CMIP6SearchItem(BaseModel):
    """A single CMIP6 dataset search query. Use natural language for variable/source/experiment — resolved via RAG vector search."""
    variable_query: Optional[str] = Field(
        default=None,
        description="Variable in natural language, e.g. 'sea surface temperature', 'uo', 'eastward wind'. Resolved to CMIP6 variable_id via vector search."
    )
    source_query: Optional[str] = Field(
        default=None,
        description="Model name in natural language, e.g. 'AWI', 'MPI', 'HadGEM3'. Resolved to CMIP6 source_id via vector search."
    )
    experiment_query: Optional[str] = Field(
        default=None,
        description="Experiment in natural language, e.g. 'historical', 'ssp585', 'piControl'. Resolved to CMIP6 experiment_id via vector search."
    )
    frequency: Optional[Literal[
        "1hr", "1hrCM", "1hrPt", "3hr", "3hrPt", "6hr", "6hrPt", "day", "dec", "fx",
        "mon", "monC", "monPt", "subhrPt", "yr", "yrPt"
    ]] = Field(
        default=None,
        description="Time frequency: 'mon' (monthly), 'day' (daily), '1hr'/'3hr'/'6hr' (hourly), 'yr' (annual), 'fx' (fixed/time-invariant), 'dec' (decadal). Pt suffix = instantaneous."
    )
    realm: Optional[Literal[
        "aerosol", "atmos", "atmosChem", "land", "landIce", "ocean", "ocnBgchem", "seaIce"
    ]] = Field(
        default=None,
        description="Model realm: 'ocean', 'atmos', 'land', 'seaIce', 'aerosol', 'landIce', 'atmosChem', 'ocnBgchem'."
    )
    nominal_resolution: Optional[Literal[
        "10 km", "25 km", "50 km", "100 km", "200 km", "250 km", "500 km",
        "1x1 degree", "2x2 degree", "10000 km"
    ]] = Field(
        default=None,
        description="Grid resolution from high (10km, 25km) to medium (50km, 100km) to low (250km, 500km)."
    )
    activity_id: Optional[Literal[
        "AerChemMIP", "C4MIP", "CDRMIP", "CFMIP", "CMIP", "CORDEX", "DAMIP",
        "DCPP", "DynVarMIP", "FAFMIP", "GMMIP", "GeoMIP", "HighResMIP", "ISMIP6",
        "LS3MIP", "LUMIP", "OMIP", "PAMIP", "PMIP", "RFMIP", "SIMIP", "ScenarioMIP",
        "VIACSAB", "VolMIP"
    ]] = Field(
        default=None,
        description="MIP activity: 'CMIP' (DECK), 'ScenarioMIP' (SSPs), 'HighResMIP' (high-res), 'OMIP' (ocean), 'PMIP' (paleo), 'DAMIP' (detection/attribution), etc."
    )


class CMIP6DataSearchArgs(BaseModel):
    """Schema for cmip6_datasets_search tool — accepts one or many searches."""
    searches: List[CMIP6SearchItem] = Field(
        description=(
            "List of search items. ALWAYS pass a list, even for a single search. "
            "Each item specifies what data to find. Use natural language for "
            "variable/source/experiment — the tool resolves to CMIP6 IDs via RAG, then checks ESGF.\n"
            "For MULTIPLE variables/experiments, put ALL in one list — they are batched into ONE LLM call.\n"
            "Examples:\n"
            "  Single: [{variable_query: 'sea surface temperature', source_query: 'MPI', "
            "experiment_query: 'historical', frequency: 'mon'}]\n"
            "  Batch:  [{variable_query: 'uo', source_query: 'AWI', experiment_query: 'historical', frequency: 'mon'}, "
            "{variable_query: 'vo', source_query: 'AWI', experiment_query: 'historical', frequency: 'mon'}, ...]"
        )
    )


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


class ReviewFigureArgs(BaseModel):
    figure_path: str = Field(
        description="Absolute path to the PNG figure to review. Use a path from python_repl's figure_paths output."
    )
    mode: Literal["describe", "correct", "qa"] = Field(
        default="qa",
        description=(
            "Review mode:\n"
            "• 'describe' — Narrate what the figure shows: data patterns, trends, key features. "
            "Use when the user asks 'what does this show?' or you need to interpret results.\n"
            "• 'correct' — Find visual issues and list SPECIFIC fixes to apply. "
            "Use after the first plot to catch problems before the user sees it.\n"
            "• 'qa' — Quick pass/fail quality check. "
            "Use for final confirmation after applying fixes."
        )
    )


class ReviewerArgs(BaseModel):
    """Submission package for peer-review. Provide EXACTLY these 5 fields.
    DO NOT include chat history, intermediate outputs, or your reasoning."""
    task: str = Field(
        description=(
            "The user's original question or task — VERBATIM or as a tight 1-2 sentence summary. "
            "This tells the reviewer WHAT was asked. "
            "Good: 'Analyze JJA precipitation trends over Texas (25-37N, 93-107W) using CMIP6 multi-model ensemble for 2070-2099 under SSP2-4.5 and SSP5-8.5.' "
            "Bad: 'The user wanted some climate analysis' (too vague)."
        )
    )
    background: str = Field(
        description=(
            "Concise methodology summary: data sources, models, variables, time periods, "
            "processing steps. The reviewer uses this to check if the approach is sound. "
            "MUST include: (a) which CMIP6 models, (b) which variable(s) + units, "
            "(c) experiment_id(s), (d) time periods, (e) key processing (regridding, masking, bias-correction). "
            "Good: 'Used 7 CMIP6 models (CESM2, MPI-ESM1-2-HR, EC-Earth3, MIROC6, GFDL-ESM4, NorESM2-MM, ACCESS-CM2). "
            "Variable: pr (kg/m2/s, converted to mm/day). Experiments: historical (1985-2014), SSP2-4.5 and SSP5-8.5 (2070-2099). "
            "ERA5 monthly precipitation as observational reference. All data regridded to 1x1 deg, masked to Texas boundaries using shapely.' "
            "Bad: 'Used some models and processed the data' (useless)."
        )
    )
    code: str = Field(
        description=(
            "The COMPLETE Python code from your LAST python_repl call that produced the final result. "
            "Copy-paste EVERY LINE — the reviewer audits it line-by-line for bugs, unit errors, "
            "and methodology issues. Do NOT summarize, truncate, or paraphrase the code. "
            "If the code was 200 lines, paste all 200 lines."
        )
    )
    stdout: str = Field(
        default="",
        description=(
            "CRITICAL: The console output / printed statistics (min, max, mean, shape, units) "
            "of the final arrays from python_repl. Copy the STDOUT section from python_repl output. "
            "Reviewers MUST use this to verify physical plausibility before suggesting changes. "
            "Without this, reviewers will hallucinate unit conversion errors."
        )
    )
    figure_path: Optional[str] = Field(
        default=None,
        description=(
            "Absolute path to the FINAL figure PNG, taken from python_repl's figure_paths output. "
            "Example: '/Users/.../figures/analysis_abc123.png'. "
            "The reviewer will visually inspect it for IPCC compliance, colorbar, axes, layout."
        )
    )



# ─── Python REPL ─────────────────────────────────────────────────────

class OptimizedPersistentPythonREPL:
    """Python REPL that saves plots by path instead of base64.
    Each session gets its own isolated workspace directory to prevent
    data files from accumulating in the project root."""
    EXEC_TIMEOUT = 600  # seconds (cloud data downloads + FESOM grids can be huge)
    PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.locals = {}

        # Create isolated workspace: results/{session_id}/
        self.workspace = os.path.join(self.PROJECT_ROOT, "results", session_id)
        self.figures_dir = os.path.join(self.workspace, "figures")
        self.data_dir = os.path.join(self.workspace, "data")
        self.temp_dir = os.path.join(self.workspace, "temp_figures")

        os.makedirs(self.workspace, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        os.environ['PYTHON_REPL_TEMP_DIR'] = self.temp_dir
        os.environ['SESSION_WORKSPACE'] = self.workspace
        os.environ['SESSION_FIGURES_DIR'] = self.figures_dir

        import pandas as pd
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import xarray as xr

        # Import audited scientific helpers
        from src.utils.geo_helpers import (
            aligned_weighted_mean, lonlat_gradient_magnitude,
            add_season_year, concat_historical_and_scenario,
            seasonal_mean_continuous, extract_figure_metadata, hidden_lines_qa
        )

        self.locals.update({
            'pd': pd, 'np': np, 'plt': plt, 'xr': xr,
            # Expose workspace paths so agent code can use them
            'WORKSPACE': self.workspace,
            'FIGURES_DIR': self.figures_dir,
            'DATA_DIR': self.data_dir,
            # Audited scientific helpers — use these instead of ad-hoc reimplementation
            'aligned_weighted_mean': aligned_weighted_mean,
            'lonlat_gradient_magnitude': lonlat_gradient_magnitude,
            'add_season_year': add_season_year,
            'concat_hist_ssp': concat_historical_and_scenario,
            'seasonal_mean_continuous': seasonal_mean_continuous,
            'figure_metadata_qa': extract_figure_metadata,
            'hidden_lines_qa': hidden_lines_qa,
        })

    def _rss_mb(self):
        """Current RSS in MB (macOS: ru_maxrss is in bytes)."""
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    def run(self, query: str):
        import matplotlib.pyplot as plt
        import matplotlib.figure as mpl_figure
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
        import gc

        # Inject gc and memory helper into REPL namespace so agent code can use them
        self.locals['gc'] = gc
        self.locals['__rss_mb'] = self._rss_mb

        rss_before = self._rss_mb()
        print(f"[REPL] RSS before: {rss_before:.0f}MB | workspace: {self.workspace}")

        # Change to session workspace so relative saves go there, not project root
        original_cwd = os.getcwd()
        os.chdir(self.workspace)

        # Lock protects sys.stdout and matplotlib from concurrent access within session
        with _get_repl_lock(self.session_id):
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            saved_file_paths = []
            manual_saves = []  # track ALL savefig() calls
            error = None

            # Intercept BOTH plt.savefig AND Figure.savefig
            # This catches: plt.savefig(), fig.savefig(), ax.figure.savefig()
            _orig_fig_savefig = mpl_figure.Figure.savefig

            _seen_paths = set()  # deduplicate

            def _intercepted_fig_savefig(self_fig, *args, **kwargs):
                _orig_fig_savefig(self_fig, *args, **kwargs)
                if args:
                    fpath = str(args[0])
                    if not os.path.isabs(fpath):
                        fpath = os.path.join(os.getcwd(), fpath)
                    if fpath not in _seen_paths:
                        _seen_paths.add(fpath)
                        manual_saves.append(fpath)

            mpl_figure.Figure.savefig = _intercepted_fig_savefig

            def _exec_code():
                # Run climate code linter before execution
                from src.utils.code_linter import lint_climate_code, format_linter_warnings
                issues = lint_climate_code(query)

                # HARD GATE: critical issues block execution entirely
                critical = [i for i in issues if i["severity"] == "critical"]
                if critical:
                    block_text = format_linter_warnings(critical)
                    raise ValueError(
                        f"EXECUTION BLOCKED BY PHYSICS LINTER:\n{block_text}\n"
                        "You MUST fix these epistemic errors before execution. "
                        "Do NOT retry the same code."
                    )

                # Non-critical warnings are printed but execution proceeds
                non_critical = [i for i in issues if i["severity"] != "critical"]
                warnings_text = format_linter_warnings(non_critical)
                if warnings_text:
                    print(warnings_text)

                exec(query, self.locals)

            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_exec_code)
                    future.result(timeout=self.EXEC_TIMEOUT)

                # Strategy: prefer manual saves if agent called savefig();
                # only auto-capture if NO manual saves happened
                if manual_saves:
                    # Agent explicitly saved — use those, don't duplicate
                    for mp in manual_saves:
                        if os.path.exists(mp):
                            dest = os.path.join(self.temp_dir, f"figure_{uuid.uuid4().hex}.png")
                            import shutil
                            shutil.copy2(mp, dest)
                            saved_file_paths.append(dest)
                    # Close any remaining open figures to prevent leakage
                    plt.close('all')
                else:
                    # No manual saves — auto-capture any open figures
                    for num in plt.get_fignums():
                        fig = plt.figure(num)
                        fname = os.path.join(self.temp_dir, f"figure_{uuid.uuid4().hex}.png")
                        _orig_fig_savefig(fig, fname, dpi=300, bbox_inches='tight')
                        saved_file_paths.append(fname)
                        plt.close(fig)
            except FuturesTimeout:
                error = (
                    f"TIMEOUT: Code execution exceeded {self.EXEC_TIMEOUT}s limit. "
                    f"The data may be too large to compute in memory. "
                    f"Try: (1) subset to a smaller region/time range, "
                    f"(2) use .isel() to sample, or (3) use dask lazy computation."
                )
            except Exception as e:
                error = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            finally:
                sys.stdout = old_stdout
                mpl_figure.Figure.savefig = _orig_fig_savefig
                output = mystdout.getvalue()
                # Restore original cwd so other server code isn't affected
                os.chdir(original_cwd)
            
            # Post-execution memory cleanup
            gc.collect()
            rss_after = self._rss_mb()
            print(f"[REPL] RSS after: {rss_after:.0f}MB (delta: +{rss_after - rss_before:.0f}MB)")
            if rss_after > 3000:  # > 3GB warning
                print(f"[REPL] ⚠️ HIGH MEMORY ({rss_after:.0f}MB) — clearing unused locals")
                # Aggressively clean up xarray datasets left in namespace
                to_del = []
                for k, v in self.locals.items():
                    if hasattr(v, 'close') and hasattr(v, 'dims') and k not in ('xr', 'pd', 'np', 'plt', 'gc'):
                        try:
                            v.close()
                        except:
                            pass
                        to_del.append(k)
                for k in to_del:
                    del self.locals[k]
                gc.collect()
                print(f"[REPL] Cleaned {len(to_del)} datasets, RSS now: {self._rss_mb():.0f}MB")

            # Always return structured result — never empty
            return {
                "stdout": output if output else (error or ""),
                "figure_paths": saved_file_paths,
                "error": error
            }



# ─── Session-scoped REPL registry ───────────────────────────────────
import threading

_repl_sessions: dict = {}
_repl_session_locks: dict = {}  # per-session locks (Phase 12: replaces global lock)
_registry_lock = threading.Lock()  # protects _repl_sessions dict itself

def _get_repl(session_id: str = "default") -> OptimizedPersistentPythonREPL:
    """Returns a session-scoped REPL instance. Each session is isolated."""
    with _registry_lock:
        if session_id not in _repl_sessions:
            _repl_sessions[session_id] = OptimizedPersistentPythonREPL(session_id=session_id)
            _repl_session_locks[session_id] = threading.Lock()
    return _repl_sessions[session_id]

def _get_repl_lock(session_id: str = "default") -> threading.Lock:
    """Returns the per-session lock for the given session."""
    with _registry_lock:
        if session_id not in _repl_session_locks:
            _repl_session_locks[session_id] = threading.Lock()
    return _repl_session_locks[session_id]


# ─── Tool Definitions ───────────────────────────────────────────────

@tool(args_schema=CMIP6DataSearchArgs)
def cmip6_datasets_search(searches: List[CMIP6SearchItem]) -> str:
    """Search for CMIP6 datasets. ALWAYS pass a list of search items, even for one query.
    Each item has fields: variable_query, source_query, experiment_query, frequency, realm, etc.
    Use natural language — the tool resolves to CMIP6 IDs via RAG, then checks ESGF.
    For MULTIPLE variables/experiments, put ALL in one list — they are processed together.
    This batches the internal LLM clarification into ONE call for efficiency.
    Example — single: searches=[{variable_query: 'SST', source_query: 'MPI', experiment_query: 'historical'}]
    Example — batch:  searches=[{variable_query: 'uo', ...}, {variable_query: 'vo', ...}, ...]
    """
    # Convert Pydantic models to dicts (exclude None values) for the batch pipeline
    search_dicts = [s.model_dump(exclude_none=True) if hasattr(s, 'model_dump') else s for s in searches]
    
    # Use batch path: vector searches individually, then ONE LLM call for all facet selections
    try:
        batch_results = cmip6_data_search_batch(search_dicts)
    except Exception as e:
        # If batch pipeline itself fails, HARD STOP — do not let agent hallucinate datasets
        print(f"Batch search pipeline failed: {e}")
        error_results = [{
            "search": sd,
            "status": "system_error",
            "error": (
                f"CRITICAL SYSTEM ERROR: ESGF Pipeline failed ({str(e)}). "
                f"DO NOT PROCEED. DO NOT HALLUCINATE DATASETS OR MODELS. "
                f"Inform the user you must refine the search query or retry."
            )
        } for sd in search_dicts]
        return json.dumps(error_results, default=str)
    
    results = []
    import resource, gc
    def _rss_mb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)  # macOS: bytes→MB
    
    print(f"[datasets_search] Starting batch of {len(search_dicts)} queries | RSS={_rss_mb():.0f}MB")
    
    for i, ((facet_values, schema), sd) in enumerate(zip(batch_results, search_dicts)):
        label = f"[{i+1}/{len(search_dicts)}]"
        parts = [v for v in [sd.get('variable_query'), sd.get('source_query'), sd.get('experiment_query')] if v]
        query_text = ", ".join(parts) if parts else "CMIP6 data search"
        try:
            access_result = cmip6_data_process(query=query_text, facet_values=facet_values)
            # Keep only essential fields to reduce memory (full_result can be huge)
            slim_result = {
                "summary": access_result.get("summary", ""),
                "total_datasets": access_result.get("total_datasets", 0),
                "python_code": access_result.get("python_code", ""),
                "esgf_link": access_result.get("esgf_link", ""),
            }
            del access_result  # free full_result, detailed_summary immediately
            results.append({"search": sd, "resolved_facets": facet_values, "access_info": slim_result})
            print(f"{label} ✓ {query_text} | RSS={_rss_mb():.0f}MB")
        except Exception as e:
            results.append({"search": sd, "resolved_facets": facet_values, "access_error": str(e)})
            print(f"{label} ⚠ {query_text}: {e}")
        # Free ESGF connections and intermediate objects after each query
        gc.collect()
    
    print(f"[datasets_search] Batch complete: {len(results)} results | RSS={_rss_mb():.0f}MB")
    return json.dumps(results, default=str)


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
    
    SESSION WORKSPACE:
    Each session runs in its own isolated directory: results/{session_id}/
    Three path variables are pre-loaded in the REPL namespace:
      - WORKSPACE  → root of this session's directory (cwd is set here)
      - FIGURES_DIR → results/{session_id}/figures/ — save publication figures here
      - DATA_DIR    → results/{session_id}/data/ — save intermediate data here
    Relative paths (e.g. 'output.nc', 'plot.png') resolve to WORKSPACE.
    Use os.path.join(FIGURES_DIR, 'my_figure.png') for organized figure storage.
    
    ⚠️ CRITICAL FOR CLOUD DATA (Pangeo zarr, FESOM, etc.):
    - NEVER write a single massive code block. SPLIT into multiple calls:
      Call 1: Open datasets + inspect shapes
      Call 2: Compute results (lazy → .compute() only final arrays)
      Call 3: Plot
    - NEVER .compute() on full cloud datasets (they can be 100+ GB)
    - For unstructured grids (ncells dim): subsample for scatter plots
    - Timeout is 600s. If data is huge, process year-by-year.
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


@tool(args_schema=ReviewFigureArgs)
def review_figure(figure_path: str, mode: str = "qa") -> list:
    """Visually inspect a generated figure. Use AFTER python_repl produces a plot.
    Returns the rendered image so you can check correctness and decide whether to fix it.
    
    Three modes:
    • 'correct' — Find issues AND suggest improvements. Use right after first plot.
    • 'describe' — Narrate what the figure shows (patterns, trends, features).
    • 'qa' — Quick pass/fail quality check. Use after applying fixes.
    
    Typical workflow:
      python_repl (plot) → review_figure(mode='correct') → python_repl (fix) → review_figure(mode='qa')
    """
    if not os.path.exists(figure_path):
        return [{"type": "text", "text": f"ERROR: File not found: {figure_path}"}]

    # Mode-specific prompts
    prompts = {
        "correct": (
            "You are a senior climate scientist reviewing this figure for publication in "
            "Nature Climate Change. The figure must be polished, informative, and visually "
            "striking — not merely correct. Examine the image carefully.\n\n"
            "## A. DATA CORRECTNESS\n"
            "□ Are data values physically reasonable? (no negative precipitation, SST not 1000°C)\n"
            "□ Any blank patches, NaN gaps, or white holes in the data that should not be there?\n"
            "□ Is the colormap scientifically appropriate? (diverging for anomalies, sequential "
            "for absolute values — never rainbow/jet)\n"
            "□ Does the colorbar range match the actual data distribution? If the data spans "
            "1–3 mm/day but the colorbar goes 0–6, most of the colour space is wasted and "
            "spatial differences become invisible.\n\n"
            "## B. LABELS & TEXT\n"
            "□ Every axis must have a label with variable name and units. Any missing?\n"
            "□ Title must be descriptive: variable, domain, time period, scenario.\n"
            "□ Legend must be present if multiple series, correctly identifying all lines.\n"
            "□ Colorbar must have a label with units and readable tick values.\n"
            "□ Is any text overlapping other text or data? Any text cut off at edges?\n"
            "□ Font sizes: all text must be readable at journal column width (~8 cm). "
            "Axis labels ≥10pt, tick labels ≥8pt, titles ≥11pt.\n\n"
            "## C. LAYOUT & AESTHETICS\n"
            "□ Panel spacing — are panels cramped or overlapping? Is there enough whitespace?\n"
            "□ Figure aspect ratio — does the shape suit the data? (maps ~2:1, time series ~3:1)\n"
            "□ Visual hierarchy — are the most important elements (ERA5, multi-model mean) "
            "visually dominant? They should be thicker and bolder than individual model lines.\n"
            "□ Are overlapping lines distinguishable? Individual models should be semi-transparent.\n"
            "□ Does the legend obscure data? It should be in an empty region.\n"
            "□ Map projections — are coastlines, borders, and geographic features clearly visible?\n"
            "□ If the prompt requested specific geographic features (e.g., lake polygons, state "
            "borders, snowbelt rectangles, country outlines), are they ACTUALLY PRESENT and "
            "visually prominent in the figure?\n\n"
            "## D. COMMON VISUAL PROBLEMS\n"
            "□ Y-axis stretched by outlier — most data compressed into a narrow band?\n"
            "□ Tick label collisions — overlapping or unreadable?\n"
            "□ Spaghetti chaos — too many lines without visual hierarchy?\n"
            "□ Wasted whitespace — large empty margins that could be trimmed?\n"
            "□ Colorbars too large or too small relative to the map panels?\n\n"
            "## E. PROJECTION FIGURE CHECKS (if showing historical vs SSP scenarios)\n"
            "□ Do SSP scenarios fork from ~2015, not start at 1950 with duplicated historical?\n"
            "□ Does the colorbar range capture real differences between scenarios?\n"
            "□ If multi-scenario maps all look identical, the visualisation has failed — "
            "consider anomaly maps instead.\n\n"
            "## F. IMPROVEMENT SUGGESTIONS (go beyond error-checking)\n"
            "Even if no errors are found, suggest concrete improvements:\n"
            "□ Would the maps look better with filled geographic features (lakes as blue "
            "polygons, land/ocean masking) instead of just outlines?\n"
            "□ Would a different panel layout tell the story more effectively? (e.g., wider "
            "time series panel, larger maps, different row arrangement)\n"
            "□ Could you add annotations that aid interpretation? (peak labels, threshold "
            "lines, domain-mean values on map panels)\n"
            "□ Would the colormap benefit from being tighter to the data percentiles?\n"
            "□ Would difference/anomaly maps be more informative than absolute values?\n"
            "□ Is there a way to make the key scientific message visually obvious at first "
            "glance, without needing to read the caption?\n\n"
            "For EACH issue or suggestion, write:\n"
            "  ISSUE: [what's wrong or could be better — specify which panel/element]\n"
            "  FIX: [exact code to implement the fix]\n\n"
            "If the figure is genuinely publication-ready with no improvements possible, "
            "respond with: '✅ PASS — figure is publication-ready.'"
        ),
        "describe": (
            "You are a climate scientist interpreting this figure.\n"
            "Describe what the figure shows in detail:\n\n"
            "1. WHAT is plotted? (variable, units, domain, time period)\n"
            "2. KEY PATTERNS — What are the main spatial/temporal features?\n"
            "3. TRENDS — Any clear trends, cycles, or regime changes?\n"
            "4. ANOMALIES — Anything unexpected or noteworthy?\n"
            "5. COMPARISON — If multiple datasets/models shown, how do they compare?\n"
            "6. PHYSICAL INTERPRETATION — What do these patterns mean climatologically?\n\n"
            "Be quantitative — cite approximate values, ranges, and magnitudes from the plot."
        ),
        "qa": (
            "Final quality check — is this figure publication-ready?\n"
            "✅ PASS or ❌ FAIL?\n\n"
            "Check QUICKLY but THOROUGHLY:\n"
            "• Labels present with units? Titles descriptive?\n"
            "• Legend present and NOT overlapping data?\n"
            "• Data range physically reasonable?\n"
            "• Text readable (not too small, not cut off, not overlapping)?\n"
            "• Panel spacing adequate (not cramped)?\n"
            "• Visual hierarchy clear (ERA5/means bold, individual models subtle)?\n"
            "• No ugly artifacts (stretched axes, collision of annotations, wasted space)?\n\n"
            "If PASS — say '✅ PASS — publication-ready' and move on.\n"
            "If FAIL — list the specific issue(s) with exact matplotlib fix code."
        ),
    }

    prompt_text = prompts.get(mode, prompts["qa"])

    try:
        with open(figure_path, "rb") as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        mime = "image/png"
        if figure_path.lower().endswith((".jpg", ".jpeg")):
            mime = "image/jpeg"

        size_kb = len(img_bytes) / 1024

        text_part = (
            f"[review_figure | mode={mode} | {size_kb:.0f} KB]\n"
            f"Path: {figure_path}\n\n"
            f"{prompt_text}"
        )

        # Vertex AI cannot handle multimodal list[dict] in tool returns —
        # return text-only so the agent still gets the review prompt
        current_model = Config.get_model_name()
        if current_model.endswith("-vertex"):
            return text_part

        return [
            {"type": "text", "text": text_part},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{img_b64}"},
            },
        ]
    except Exception as e:
        return f"ERROR reading figure: {str(e)}"


# ─── Reviewer Tools ──────────────────────────────────────────────────

_REVIEWER_SYSTEM_PROMPT = (
    "You are a Ruthless, Data-Driven Climate Physics Peer Reviewer for a Nature-class journal. "
    "Your job is to identify fatal methodological, physical, and visual flaws. You DO NOT have "
    "execution access, but you are provided with the STDOUT (printed empirical statistics of "
    "the data arrays).\n\n"

    "=== 1. THE EMPIRICAL CONSTRAINT (CRITICAL — READ FIRST) ===\n"
    "Look at the STDOUT data BEFORE reviewing the code.\n"
    "- If precipitation means are ~1-10 mm/day for a non-arid region, the unit conversions "
    "ARE CORRECT. DO NOT hallucinate that they need to be divided by days_in_month or "
    "multiplied by 86400 again.\n"
    "- If temperature anomalies are between -15 and +15, they are correct.\n"
    "- If pressure values are ~1000 hPa, the Pa→hPa conversion is correct.\n"
    "TRUST THE EMPIRICAL OUTPUT over your assumptions about dataset metadata. "
    "NEVER suggest a mathematical fix that would result in physically impossible values.\n"
    "HISTORICAL INCIDENT: A reviewer once incorrectly flagged ERA5 'tp * 1000' as wrong, "
    "claiming it needed division by days_in_month. The agent blindly obeyed, DESTROYING "
    "a correct pipeline and producing 0.07 mm/day for Texas summer precipitation. "
    "DO NOT REPEAT THIS.\n\n"

    "=== 2. HUNT FOR SILENT MATH ERRORS ===\n"
    "Search the code ruthlessly for these specific epistemic failures:\n\n"
    "MASKING DENOMINATORS: Did the author mask a subset of the globe (e.g., Texas), but "
    "use a global weight array for the denominator in np.nansum()? This artificially "
    "crushes regional RMSE/means.\n\n"
    "STATIONARY QDM: If Quantile Delta Mapping (QDM) is used, did the author compute "
    "future quantiles over the entire 150-year trended future array? This destroys "
    "non-stationarity. They must use rolling windows or specific future time slices.\n\n"
    "CHRONOLOGICAL MATCHING: Did they compute chronological RMSE between free-running "
    "CMIP6 historical models and ERA5? Scientifically invalid due to out-of-phase "
    "internal variability. Compare climatologies, variances, or spectra instead.\n\n"
    "GUARDRAIL HACKING: Reject any code that uses arbitrary scalar shifts (e.g. "
    "data -= bias or data += 0.63) to artificially mask bias or force baselines to match.\n\n"
    "INDEX-SPACE GRADIENTS: Did they apply scipy.ndimage.sobel or np.gradient directly "
    "on lat/lon grids? Image filters calculate gradients per-pixel, ignoring Earth "
    "curvature. Flag as CRITICAL.\n\n"
    "UNWEIGHTED SPATIAL MEANS: Did they compute .mean(dim=['lat','lon']) without "
    ".weighted(cos(lat))? Grid cells shrink at the poles.\n\n"

    "=== 3. DATA INTEGRITY AND METHODOLOGY ===\n"
    "Verify correct variables, unit conversions (ONLY if STDOUT suggests implausible "
    "values), temporal subsetting (historical ≤ 2014, SSP ≥ 2015), calendar handling, "
    "area weighting, ensemble design, and bias correction methodology.\n\n"

    "=== 4. FIGURE QUALITY AND PRESENTATION ===\n"
    "- Hidden baselines: Did set_ylim clip the ERA5/observational line off screen?\n"
    "- Blank map patches: Does set_extent match the data extent?\n"
    "- Scenario fork: Do SSP pathways diverge from ~2015, not duplicated back to 1950?\n"
    "- Colour scale: Tight to data distribution? Diverging for anomalies?\n"
    "- All axes labelled with units? Legends present?\n"
    "- Maps: coastlines + borders visible? Projection appropriate?\n\n"

    "=== REPORTING FORMAT ===\n"
    "For each issue identified, state:\n"
    "  [CRITICAL / MAJOR / MINOR] Issue: <concise description>\n"
    "  Evidence: <what in the code/STDOUT/figure proves this>\n"
    "  Confidence: <HIGH if proven by STDOUT/metadata, LOW if hypothesis>\n"
    "  Recommendation: <specific, actionable fix>\n\n"
    "CRITICAL — produces incorrect scientific results\n"
    "MAJOR — methodological concern weakening conclusions\n"
    "MINOR — stylistic or presentational improvement\n\n"
    "RULE: If confidence is LOW (hypothesis only, not backed by STDOUT or metadata), "
    "severity CANNOT be CRITICAL. Emit as MAJOR with a note to verify.\n\n"
    "If the code is sound and the STDOUT stats are physically realistic, output "
    "'✅ PASS'. Do not invent flaws."
)


def _build_reviewer_content(task: str, background: str, code: str, stdout: str = "", figure_path: Optional[str] = None, model_name: str = "") -> list:
    """Build multimodal content for a reviewer: structured text + optional figure.
    
    Image format differs by provider:
    - OpenAI (gpt-*): image_url with data URI
    - Gemini (gemini-*): image_url with data URI (same as OpenAI)
    - Anthropic (claude-*): native image format with base64 source
    """
    submission = (
        f"=== SUBMISSION FOR REVIEW ===\n\n"
        f"## TASK\n{task}\n\n"
        f"## BACKGROUND & METHODOLOGY\n{background}\n\n"
        f"## FINAL CODE\n```python\n{code}\n```\n\n"
        f"## CODE STDOUT (EMPIRICAL DATA)\n```text\n{stdout}\n```\n"
        f"Use the STDOUT above to empirically verify whether values are physically realistic "
        f"before proposing mathematical changes. If precipitation means are ~1-10 mm/day, "
        f"the conversions ARE correct. Trust empirical output over your assumptions."
    )
    if figure_path and os.path.exists(figure_path):
        submission += f"\n\n## FIGURE\nThe figure produced by this code is attached below as an image. Review it carefully."
    
    content_parts = [{"type": "text", "text": submission}]
    
    if figure_path and os.path.exists(figure_path):
        with open(figure_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        if model_name.startswith("claude"):
            # Anthropic native format
            content_parts.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_b64,
                },
            })
        else:
            # OpenAI and Gemini both use image_url with data URI
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            })
    
    return content_parts


@tool(args_schema=ReviewerArgs)
def reviewer_1(task: str, background: str, code: str, stdout: str = "", figure_path: Optional[str] = None) -> str:
    """Reviewer #1 — independent peer-review by a DIFFERENT LLM model.
    Call AFTER you have a FINAL analysis with code and figure. Pass ONLY:
      - task: the original user question/task
      - background: data sources, models used, processing steps
      - code: the complete final Python code
      - stdout: the printed output / statistics from python_repl (CRITICAL for empirical verification)
      - figure_path: path to the final figure (optional)

    DO NOT pass the full chat history. The reviewer sees only the submission package.
    After receiving feedback from BOTH reviewers, synthesize their suggestions into an improved final version.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    model_name = Config.reviewer_model_1
    reviewer_llm = create_reviewer_llm(model_name)
    system_msg = SystemMessage(content=_REVIEWER_SYSTEM_PROMPT)
    content_parts = _build_reviewer_content(task, background, code, stdout=stdout, figure_path=figure_path, model_name=model_name)
    human_msg = HumanMessage(content=content_parts)

    try:
        response = reviewer_llm.invoke([system_msg, human_msg])
        review_text = response.content if isinstance(response.content, str) else str(response.content)
        return f"[Reviewer #1 -- {model_name}]\n\n{review_text}"
    except Exception as e:
        return f"[Reviewer #1 -- ERROR] {str(e)}"


@tool(args_schema=ReviewerArgs)
def reviewer_2(task: str, background: str, code: str, stdout: str = "", figure_path: Optional[str] = None) -> str:
    """Reviewer #2 — independent peer-review by a DIFFERENT LLM model.
    Call AFTER you have a FINAL analysis with code and figure. Pass ONLY:
      - task: the original user question/task
      - background: data sources, models used, processing steps
      - code: the complete final Python code
      - stdout: the printed output / statistics from python_repl (CRITICAL for empirical verification)
      - figure_path: path to the final figure (optional)

    DO NOT pass the full chat history. The reviewer sees only the submission package.
    After receiving feedback from BOTH reviewers, synthesize their suggestions into an improved final version.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    model_name = Config.reviewer_model_2
    reviewer_llm = create_reviewer_llm(model_name)
    system_msg = SystemMessage(content=_REVIEWER_SYSTEM_PROMPT)
    content_parts = _build_reviewer_content(task, background, code, stdout=stdout, figure_path=figure_path, model_name=model_name)
    human_msg = HumanMessage(content=content_parts)

    try:
        response = reviewer_llm.invoke([system_msg, human_msg])
        review_text = response.content if isinstance(response.content, str) else str(response.content)
        return f"[Reviewer #2 -- {model_name}]\n\n{review_text}"
    except Exception as e:
        return f"[Reviewer #2 -- ERROR] {str(e)}"


# ─── Agent Factory ───────────────────────────────────────────────────

def create_cmip6_agent():
    """Creates a CMIP6 agent using langgraph's create_react_agent."""
    llm = create_llm()
    prompt_template = create_prompt_template()

    all_tools = [
        cmip6_datasets_search, cmip6_datasets_access,
        cmip6_adviser,
        cmip6_literature_search, cmip6_citation_graph,
        cmip6_methodology_check,
        python_repl, review_figure, analysis_guide_tool,
        era5_monthly_tool,
        reviewer_1, reviewer_2,
    ]

    system_message = prompt_template.messages[0].content if prompt_template.messages else ""

    agent = create_react_agent(
        model=llm,
        tools=all_tools,
        prompt=system_message,
    )

    return agent