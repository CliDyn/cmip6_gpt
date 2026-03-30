# CMIP6-GPT Agent — Deep Architecture & Data Pipeline Audit

> **Instructions**: Attached is the full source code of the CMIP6-GPT project (`project_export.txt`). You are a senior AI/ML systems architect. Perform a thorough audit of the agent's architecture, data retrieval pipeline, and React frontend. Answer every question below with specific code references, concrete recommendations, and refactored snippets where appropriate.

---

## 1. Agent Architecture & Tool Routing

The agent is built with `langgraph.prebuilt.create_react_agent` and has 5 tools: `cmip6_datasets_search`, `cmip6_datasets_access`, `cmip6_adviser`, `python_repl`, `analysis_guide_tool`.

1. The system prompt uses a simple rule-based routing section ("Route every user message to exactly ONE tool"). Is this the optimal approach for a ReAct agent, or would a more structured router (e.g. a dedicated classification step before the agent loop) reduce mis-routing? What are the trade-offs?
2. `cmip6_datasets_search` internally auto-chains into `cmip6_datasets_access` (lines 160–176 of `cmip6_agent.py`). This means the agent also has a **separate** `cmip6_datasets_access` tool. Is this redundant? Should search always auto-chain, or should the agent decide when to call access separately? What's the cleaner design?
3. The `cmip6_adviser` tool returns `DynamicCMIP6DownloadArgs.model_json_schema()` — raw JSON schema. Is this useful to the agent/user, or should it return a human-readable summary instead?

---

## 2. RAG / Vector Search Pipeline

The pipeline: **user query → agent splits into variable/source/experiment args → ChromaDB similarity search → re-ranking → dynamic Pydantic schema → LLM selects facet values → ESGF API call**.

4. There are **two** vector search paths: `perform_vector_search()` (legacy, uses LangGraph + SplitQueryNode = extra LLM call) and `perform_direct_vector_search()` (new, no extra LLM call). The legacy path is still used by `cmip6_adviser`. Should it be fully deprecated? What's the migration path?
5. The re-ranking logic boosts candidates whose `metadata.source` appears in the original query (simple string containment). Is this sufficient, or should a cross-encoder or more sophisticated re-ranker be used? What's the cost/benefit?
6. `create_dynamic_cmip6_args()` builds a Pydantic model with `Literal` enums from RAG results, then asks the LLM to pick via `structured_output`. This is clever but complex. Are there failure modes? What happens when:
   - All candidates score above threshold → adaptive fallback takes top 3 → but none are correct?
   - The `UNMATCHED` escape hatch is selected → what recovery mechanism exists?
7. Score threshold and max_candidates come from `config.yaml`. Are the current defaults likely optimal for `gemini-embedding-001`? What calibration approach would you recommend?

---

## 3. Data Retrieval & Processing

8. `download_cmip6_data()` in `cmip6_utils.py` calls the ESGF search API (`pyesgf`), iterates **all** datasets (`ctx.search()`), and builds a nested summary. For queries with thousands of results, this could be very slow. Is pagination or result-capping implemented? What optimizations are needed?
9. `cmip6_data_process()` calls `download_opendap_or_not()` — **another LLM call** just to decide whether to show OpenDAP links. Is this justified, or could it be a simple heuristic (e.g. keyword detection)?
10. `generate_python_code()` is called to produce download code. Is this generated per-query by an LLM, or is it template-based? What's the risk of generating incorrect/insecure code?
11. The `select_facet_values()` function has a fallback chain: structured output → raw JSON parsing → empty dict. Is the error handling granular enough? Should failed facet selection abort the pipeline or attempt recovery?

---

## 4. Session & State Management

12. `session_manager.py` stores chat history in-memory. What happens on server restart? Is there a persistence layer? Should there be one?
13. The REPL (`OptimizedPersistentPythonREPL`) uses a **global singleton** with shared `self.locals`. This means variables persist across queries within a session — is this intentional and safe? What about cross-session leakage since it's a single global instance?
14. There's a **duplicate** `python_repl()` function in `cmip6_service.py` (lines 147–184) that seems unused. Confirm whether it's dead code and should be removed.

---

## 5. Server & API Design

15. The FastAPI server has both `/api/chat` (synchronous) and `/api/chat/stream` (SSE). The sync endpoint doesn't use conversation history (`[HumanMessage(content=req.message)]`), while the stream endpoint builds full history. Is this a bug?
16. Agent instances are cached per `model_name` in `_agents` dict. But if the user changes the model mid-session, do the tools and their internal state (REPL, retrievers) reset correctly?
17. The SSE stream sends raw figure paths from the local filesystem. What happens in a Docker/cloud deployment — are paths still valid?
18. CORS is hardcoded to `localhost:5173` and `localhost:3000`. What's the production CORS strategy?

---

## 6. Frontend (React + Vite)

19. Review `App.jsx`, `ChatMessage.jsx`, `PlotViewer.jsx`, and `Sidebar.jsx`. Is the component decomposition clean? What would you refactor?
20. The `api.js` module handles SSE streaming via `fetch` + `ReadableStream`. Is error handling and reconnection logic robust?
21. Are there any performance concerns (unnecessary re-renders, missing `useMemo`/`useCallback`, large state objects)?

---

## 7. Overall Assessment

22. **LLM call budget per query**: Map out every LLM invocation in a typical "find me monthly SST from MPI historical" flow. How many LLM calls are made? Is this optimal?
23. **Latency breakdown**: Where are the biggest latency bottlenecks? Rank them.
24. **Security**: The Python REPL executes arbitrary code via `exec()`. What sandboxing exists? What are the risks?
25. **Scalability**: What breaks first if this goes from 1 user to 100 concurrent users?
26. **Top 5 improvements** you'd prioritize, ranked by impact/effort ratio.

---

**Format your response as a structured report with numbered answers matching the questions above. Include code snippets for any concrete refactoring suggestions.**
