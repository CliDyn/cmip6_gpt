"""
FastAPI backend for CMIP6 GPT.
Uses langgraph agent (create_react_agent) with SSE streaming.
"""
import os

# ─── macOS fork-safety: prevent SIGSEGV in libproj/grpc atfork handlers ──
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")

import logging
import json
from dotenv import load_dotenv

load_dotenv()

from src.utils.token_tracker import TokenTracker, get_usage_summary

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from src.agents.cmip6_agent import create_cmip6_agent
from src.utils.vector_search import prewarm_retrievers
from src.config import Config
from session_manager import session_manager
from agent_logger import AgentStepLogger

logger = logging.getLogger("cmip6_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# ─── App ─────────────────────────────────────────────────────────────

app = FastAPI(title="CMIP6 GPT API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Agent Cache ─────────────────────────────────────────────────────

_agents: Dict[str, Any] = {}

# Cancel flags per session — set by /api/sessions/{sid}/cancel, checked in
# the event_generator between agent steps. Cancellation can only happen at
# step boundaries (i.e., between tool calls / LLM calls), not mid-tool.
_cancel_flags: Dict[str, bool] = {}


def _get_agent(model_name: str = None):
    model_name = model_name or Config.get_model_name()
    if model_name not in _agents:
        Config.set_model_name(model_name)
        print(f"Creating agent for model: {model_name}")
        _agents[model_name] = create_cmip6_agent()
    return _agents[model_name]


# ─── Startup ─────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    try:
        prewarm_retrievers()
        print("Retrievers pre-warmed. Server ready.")
    except Exception as e:
        print(f"Warning: retriever prewarm failed (will load lazily): {e}")
        print("Server ready (retrievers will load on first query).")


# ─── Models ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    message: str
    session_id: str = Field(default="default")
    model_name: str = Field(default=None)
    google_api_key_slot: int = Field(default=1, ge=1, le=2)
    rag_chunks: int = Field(default=10, ge=3, le=25)
    rag_searches: int = Field(default=5, ge=1, le=12)
    reviewer_model_1: str = Field(default="gemini-3.1-pro-preview")
    reviewer_model_2: str = Field(default="gemini-3.1-pro-preview")
    reviewers_enabled: bool = Field(default=True)


class SessionResponse(BaseModel):
    session_id: str


class ConfigResponse(BaseModel):
    models: List[str]
    current_model: str
    reviewer_models: List[str]
    google_api_key_slots: List[str]


# ─── Google API Key Slot Switching ───────────────────────────────────

_current_key_slot = 1

def _apply_google_api_key_slot(slot: int):
    """Switch GOOGLE_API_KEY env var to the requested slot (1 or 2).
    Also clears cached embedding/LLM singletons so they pick up the new key.
    No-op when GOOGLE_API_KEY_* slots are unset (Vertex-only deployments)."""
    global _current_key_slot
    key = os.environ.get(f"GOOGLE_API_KEY_{slot}", "")
    if not key:
        # Vertex-only deployments have no AI-Studio slots configured.
        # Stay silent — `vertex_api_key` covers all Gemini routing.
        return
    if slot != _current_key_slot:
        os.environ["GOOGLE_API_KEY"] = key
        _current_key_slot = slot
        # Reset cached singletons that hold the old key
        from src.services.llm_service import _reset_embedding_cache
        from src.utils.vector_search import clear_retriever_cache
        _reset_embedding_cache()
        clear_retriever_cache()
        # Clear agent cache so LLM is recreated with new key
        _agents.clear()
        logger.info(f"Switched to GOOGLE_API_KEY_{slot}")


# ─── Routes ──────────────────────────────────────────────────────────

@app.get("/api/config")
async def get_config():
    return ConfigResponse(
        models=Config.get_available_models(),
        current_model=Config.get_model_name(),
        reviewer_models=Config.REVIEWER_MODELS,
        google_api_key_slots=["Key 1 (primary)", "Key 2 (backup)"],
    )


@app.post("/api/sessions/create")
async def create_session():
    sid = session_manager.create_session()
    return SessionResponse(session_id=sid)


@app.post("/api/sessions/clear")
async def clear_session(req: SessionResponse):
    session_manager.clear_session(req.session_id)
    return {"status": "cleared"}


@app.get("/api/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    msgs = session_manager.get_messages(session_id)
    return {"messages": msgs}


@app.post("/api/sessions/{session_id}/cancel")
async def cancel_session(session_id: str):
    """Signal the streaming generator to stop at its next step boundary.

    Cancellation is cooperative: the running agent.stream() loop checks the
    flag between events. A long-running tool call (download, REPL exec) will
    finish before the cancel takes effect.
    """
    _cancel_flags[session_id] = True
    logger.info(f"[cancel] flag set for session {session_id}")
    return {"status": "cancel_requested", "session_id": session_id}


def _extract_text_from_event(event: dict) -> str:
    """Extract text content from a langgraph stream event."""
    # langgraph events come as {"agent": {"messages": [...]}} or {"tools": {"messages": [...]}}
    for key in ("agent", "tools", "__end__"):
        if key in event:
            messages = event[key].get("messages", [])
            for msg in messages:
                if hasattr(msg, 'content') and isinstance(msg.content, str):
                    return msg.content
    return ""


def _extract_figure_paths(text: str) -> List[str]:
    """Extract figure file paths from agent text output."""
    import re
    paths = []
    # Look for figure paths in the text
    md_figures = re.findall(r'!\[.*?\]\((.*?)\)', text)
    for fig_path in md_figures:
        abs_path = os.path.join(os.getcwd(), "temp_figures", os.path.basename(fig_path))
        if os.path.exists(abs_path):
            paths.append(f"/api/figures/{os.path.basename(abs_path)}")
    # Also check for figure_paths in JSON tool outputs
    json_matches = re.findall(r'"figure_paths"\s*:\s*\[(.*?)\]', text)
    for match in json_matches:
        for path in re.findall(r'"([^"]+\.png)"', match):
            if os.path.exists(path):
                paths.append(f"/api/figures/{os.path.basename(path)}")
    return paths


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Process a chat message and return the full response."""
    session = session_manager.get_session(req.session_id)
    session.messages.append({"role": "user", "content": req.message})

    # Apply Google API key slot
    _apply_google_api_key_slot(req.google_api_key_slot)

    agent = _get_agent(req.model_name)
    # Apply per-request RAG settings
    Config.rag_chunks_per_search = req.rag_chunks
    Config.rag_num_searches = req.rag_searches
    Config.reviewer_model_1 = req.reviewer_model_1
    Config.reviewer_model_2 = req.reviewer_model_2
    Config.reviewers_enabled = req.reviewers_enabled
    tracker = TokenTracker()
    model_used = req.model_name or Config.get_model_name()
    tracker.start_request(req.session_id, model_used, req.message)
    slog = AgentStepLogger(req.session_id, model_used, req.message, req.rag_searches, req.rag_chunks)

    try:
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        history = []
        history.append(SystemMessage(content=(
            f"[RAG DEPTH OVERRIDE] For this session: perform exactly {req.rag_searches} parallel "
            f"cmip6_literature_search calls (not fewer), each returning {req.rag_chunks} chunks. "
            f"Total expected coverage: ~{req.rag_searches * req.rag_chunks} chunks."
        )))
        for m in session.messages:
            if m["role"] == "user":
                history.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant" and m.get("content"):
                history.append(AIMessage(content=m["content"]))

        result = agent.invoke(
            {"messages": history},
            config={"configurable": {"session_id": req.session_id}, "recursion_limit": 80},
        )

        final_messages = result.get("messages", [])
        full_response = ""
        for msg in reversed(final_messages):
            if hasattr(msg, 'content') and msg.type == "ai" and msg.content:
                content = msg.content
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                full_response = content
                break

        # Log every message in the trace
        for msg in final_messages:
            if hasattr(msg, 'type'):
                if msg.type == "ai":
                    tracker.record(msg)
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tc in msg.tool_calls:
                            slog.tool_call(tc.get('name', '?'), tc.get('args', {}))
                    elif msg.content:
                        slog.ai_response(msg.content if isinstance(msg.content, str) else str(msg.content))
                elif msg.type == "tool" and msg.content:
                    tool_name = getattr(msg, 'name', '?')
                    slog.tool_result(tool_name, str(msg.content))
                    try:
                        tool_data = json.loads(msg.content)
                        if isinstance(tool_data, dict):
                            if tool_name == 'cmip6_literature_search' and tool_data.get('results'):
                                slog.rag_sources(tool_data.get('query', ''), tool_data['results'])
                            if tool_data.get('figure_paths'):
                                slog.figure(tool_data['figure_paths'])
                    except (json.JSONDecodeError, TypeError):
                        pass

        usage_summary = tracker.end_request()
        slog.set_usage(usage_summary)

        figure_paths = _extract_figure_paths(full_response)
        for msg in final_messages:
            if hasattr(msg, 'content') and msg.type == "tool":
                try:
                    tool_data = json.loads(msg.content)
                    if isinstance(tool_data, dict) and "figure_paths" in tool_data:
                        for p in tool_data["figure_paths"]:
                            url = f"/api/figures/{os.path.basename(p)}"
                            if url not in figure_paths:
                                figure_paths.append(url)
                except (json.JSONDecodeError, TypeError):
                    pass

        slog.flush()

        assistant_msg = {
            "role": "assistant",
            "content": full_response,
            "figure_paths": figure_paths,
        }
        session.messages.append(assistant_msg)

        return {
            "response": full_response,
            "figure_paths": figure_paths,
            "session_id": req.session_id,
            "usage": {
                "total_tokens": usage_summary["total_tokens"],
                "cost_eur": usage_summary["total_cost_eur"],
            },
        }

    except Exception as e:
        import traceback
        tracker.end_request()
        slog.error(str(e))
        slog.flush()
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        session.messages.append({
            "role": "assistant",
            "content": f"An error occurred: {str(e)}. Please try rephrasing.",
        })
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """Process a chat message and stream via SSE using langgraph's stream."""
    session = session_manager.get_session(req.session_id)
    session.messages.append({"role": "user", "content": req.message})
    print(f"[stream] model_name from request: '{req.model_name}', key_slot: {req.google_api_key_slot}")

    # Apply Google API key slot
    _apply_google_api_key_slot(req.google_api_key_slot)

    agent = _get_agent(req.model_name)
    # Apply per-request RAG settings
    Config.rag_chunks_per_search = req.rag_chunks
    Config.rag_num_searches = req.rag_searches
    Config.reviewer_model_1 = req.reviewer_model_1
    Config.reviewer_model_2 = req.reviewer_model_2
    Config.reviewers_enabled = req.reviewers_enabled

    async def event_generator():
        full_response = ""
        figure_paths = []
        last_python_code = ""  # track code for 'View code' button
        tracker = TokenTracker()
        model_used = req.model_name or Config.get_model_name()
        tracker.start_request(req.session_id, model_used, req.message)
        slog = AgentStepLogger(req.session_id, model_used, req.message, req.rag_searches, req.rag_chunks)
        # Reset cancel flag for this session at the start of a new request
        _cancel_flags[req.session_id] = False
        try:
            from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

            history = []
            history.append(SystemMessage(content=(
                f"[RAG DEPTH OVERRIDE] For this session: perform exactly {req.rag_searches} parallel "
                f"cmip6_literature_search calls (not fewer), each returning {req.rag_chunks} chunks. "
                f"Total expected coverage: ~{req.rag_searches * req.rag_chunks} chunks."
            )))
            for m in session.messages:
                if m["role"] == "user":
                    history.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant" and m.get("content"):
                    content = m["content"]
                    # Trim very long assistant messages to save memory
                    # Keep last 2000 chars (model names, conclusions) + first 500 chars (intro)
                    if len(content) > 3000:
                        content = content[:500] + "\n\n[... trimmed for memory ...]\n\n" + content[-2000:]
                    history.append(AIMessage(content=content))

            yield f"data: {json.dumps({'type': 'status', 'content': 'Thinking...'})}\n\n"

            # User explicitly opted into "wait as long as needed" — generous
            # retry budget for transient cloud errors (Vertex 429, GCS hiccups).
            # Bumped 2026-05-15 after preview-model quota volatility — agent
            # retries up to 30× with 30s–180s waits so even a multi-minute
            # Vertex cool-down period gets ridden out instead of failing.
            MAX_RETRIES = 30
            for attempt in range(MAX_RETRIES + 1):
                full_response = ""
                figure_paths_attempt = []
                cancelled = False
                try:
                    for event in agent.stream(
                        {"messages": history},
                        stream_mode="updates",
                        config={"configurable": {"session_id": req.session_id}, "recursion_limit": 80},
                    ):
                        # Cooperative cancel check between agent steps
                        if _cancel_flags.get(req.session_id):
                            logger.info(f"[cancel] honoring cancel for session {req.session_id}")
                            yield f"data: {json.dumps({'type': 'status', 'content': '🛑 Stopped by user'})}\n\n"
                            full_response = full_response or "🛑 Run stopped by user."
                            cancelled = True
                            break
                        for node_name, node_output in event.items():
                            messages = node_output.get("messages", [])
                            for msg in messages:
                                if hasattr(msg, 'type'):
                                    if msg.type == "ai":
                                        tracker.record(msg)

                                    # ── AI decides to call tool(s) ──
                                    if msg.type == "ai" and hasattr(msg, 'tool_calls') and msg.tool_calls:
                                        for tc in msg.tool_calls:
                                            tc_name = tc.get('name', '?')
                                            tc_args = tc.get('args', {})
                                            args_preview = {k: (str(v)[:120] + '…' if len(str(v)) > 120 else v) for k, v in tc_args.items()}
                                            print(f"[agent] 🔧 TOOL CALL: {tc_name}({args_preview})")
                                            slog.tool_call(tc_name, tc_args)
                                            if tc_name in ('python_repl', 'Python_REPL'):
                                                last_python_code = tc_args.get('query', '')
                                        tool_names = [tc.get('name', '') for tc in msg.tool_calls]
                                        status_map = {
                                            'cmip6_datasets_search': '🔍 Searching datasets...',
                                            'cmip6_datasets_access': '📦 Checking data access...',
                                            'cmip6_adviser': '📖 Looking up information...',
                                            'cmip6_literature_search': '📚 Searching scientific papers...',
                                            'cmip6_citation_graph': '🔗 Exploring citation graph...',
                                            'Python_REPL': '🐍 Running analysis...',
                                            'python_repl': '🐍 Running analysis...',
                                            'retrieve_era5_data': '🌍 Fetching ERA5 data...',
                                            'retrieve_era5_monthly': '🌍 Fetching ERA5 data...',
                                            'review_figure': '👁️ Inspecting figure...',
                                            'reviewer_1': '📋 Reviewer #1 analyzing...',
                                            'reviewer_2': '🎨 Reviewer #2 analyzing...',
                                        }
                                        for tn in tool_names:
                                            label = status_map.get(tn, f'⚙️ Using {tn}...')
                                            yield f"data: {json.dumps({'type': 'status', 'content': label})}\n\n"

                                    elif msg.type == "ai" and not msg.content and not getattr(msg, 'tool_calls', None):
                                        logger.warning(f"[agent] ⚠️ Empty AI response (0 tokens) on attempt {attempt+1}/{MAX_RETRIES+1}")

                                    elif msg.type == "ai" and msg.content:
                                        content = msg.content
                                        if isinstance(content, list):
                                            content = "".join(
                                                part.get("text", "") if isinstance(part, dict) else str(part)
                                                for part in content
                                            )
                                        print(f"[agent] 💬 AI: {content[:200]}…")
                                        full_response = content
                                        slog.ai_response(content)
                                        yield f"data: {json.dumps({'type': 'status', 'content': ''})}\n\n"
                                        yield f"data: {json.dumps({'type': 'text', 'content': content})}\n\n"

                                    elif msg.type == "tool" and msg.content:
                                        tool_name = getattr(msg, 'name', '?')
                                        print(f"[agent] 📋 TOOL RESULT ({tool_name}): {str(msg.content)[:300]}")
                                        slog.tool_result(tool_name, str(msg.content))
                                        try:
                                            tool_data = json.loads(msg.content)
                                            if isinstance(tool_data, dict) and tool_name == 'cmip6_literature_search' and tool_data.get('results'):
                                                sources = [{
                                                    'title': r.get('title', ''),
                                                    'year': r.get('year', ''),
                                                    'journal': r.get('journal', ''),
                                                    'doi': r.get('doi', ''),
                                                    'score': r.get('score', 0),
                                                    'text': r.get('text', '')[:400],
                                                } for r in tool_data['results']]
                                                slog.rag_sources(tool_data.get('query', ''), tool_data['results'])
                                                yield f"data: {json.dumps({'type': 'sources', 'query': tool_data.get('query', ''), 'results': sources})}\n\n"
                                            if isinstance(tool_data, dict) and tool_data.get("figure_paths"):
                                                urls = [f"/api/figures/{os.path.basename(p)}" for p in tool_data["figure_paths"]]
                                                figure_paths_attempt.extend(urls)
                                                slog.figure(tool_data['figure_paths'])
                                                payload = {'type': 'figures', 'paths': urls}
                                                if last_python_code:
                                                    payload['code'] = last_python_code
                                                payload['stdout'] = tool_data.get('stdout', '')
                                                yield f"data: {json.dumps(payload)}\n\n"
                                                last_python_code = ""
                                        except (json.JSONDecodeError, TypeError):
                                            pass
                except Exception as stream_err:
                    import traceback as tb
                    err_str = str(stream_err)
                    full_tb = tb.format_exc()
                    logger.error(f"[agent] ⚡ Stream error on attempt {attempt+1}/{MAX_RETRIES+1}: {err_str}\n{full_tb}")
                    print(f"[STREAM ERROR] {full_tb}")
                    if attempt < MAX_RETRIES:
                        import time
                        # Exponential backoff capped at 180s (3 min) — gives
                        # Vertex sliding-window quotas time to fully reset on
                        # preview models that hard-cap below 5 RPM.
                        wait = min(30 * (1.5 ** attempt), 180)
                        logger.warning(f"[agent] 🔄 Retrying ({attempt+1}/{MAX_RETRIES}) in {wait:.0f}s after: {err_str[:100]}")
                        yield f"data: {json.dumps({'type': 'status', 'content': f'⚡ Transient error (attempt {attempt+1}/{MAX_RETRIES}), retrying in {wait:.0f}s...'})}\n\n"
                        time.sleep(wait)
                        continue
                    else:
                        full_response = f"⚠️ The model connection failed after {MAX_RETRIES+1} attempts: {err_str[:200]}. Please try again."
                        yield f"data: {json.dumps({'type': 'text', 'content': full_response})}\n\n"
                        break

                if cancelled:
                    figure_paths = figure_paths_attempt
                    break
                if full_response:
                    figure_paths = figure_paths_attempt
                    break
                if attempt < MAX_RETRIES:
                    logger.warning(f"[agent] 🔄 Retrying ({attempt+2}/{MAX_RETRIES+1}) due to empty model response...")
                    yield f"data: {json.dumps({'type': 'status', 'content': '🔄 Model returned empty response, retrying...'})}\n\n"
                else:
                    logger.error("[agent] ❌ All retries exhausted — model returned empty response")
                    full_response = "⚠️ The model returned an empty response after multiple attempts. This can happen with certain Gemini preview models. Please try rephrasing your question or switching models."
                    yield f"data: {json.dumps({'type': 'text', 'content': full_response})}\n\n"

            usage_summary = tracker.end_request()
            slog.set_usage(usage_summary)
            slog.flush()

            if full_response and not full_response.startswith("⚠️ The model returned an empty"):
                session.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "figure_paths": figure_paths,
                })
            else:
                logger.warning(f"[session] Skipping empty/error response from session history")
            yield f"data: {json.dumps({'type': 'usage', 'total_tokens': usage_summary['total_tokens'], 'cost_eur': usage_summary['total_cost_eur'], 'llm_calls': usage_summary['llm_calls']})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            import traceback as tb
            tracker.end_request()
            slog.error(str(e))
            slog.flush()
            error_msg = str(e)
            full_tb = tb.format_exc()
            logger.error(f"[agent] ❌ Top-level stream error: {error_msg}\n{full_tb}")
            print(f"[TOP-LEVEL STREAM ERROR] {full_tb}")
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
            session.messages.append({
                "role": "assistant",
                "content": f"Error: {error_msg}",
            })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.get("/api/usage")
async def get_usage():
    """Return aggregate token usage statistics."""
    return get_usage_summary()


@app.get("/api/figures/{filename}")
async def get_figure(filename: str):
    """Serve a figure file from temp_figures or any session results directory."""
    search_dirs = [
        os.path.join(os.getcwd(), "temp_figures"),
        os.path.join(os.getcwd(), "results"),
    ]
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for root, dirs, files in os.walk(search_dir):
            if filename in files:
                return FileResponse(os.path.join(root, filename), media_type="image/png")
    raise HTTPException(status_code=404, detail="Figure not found")


# ─── Entry Point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
