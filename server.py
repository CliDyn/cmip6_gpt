"""
FastAPI backend for CMIP6 GPT.
Uses langgraph agent (create_react_agent) with SSE streaming.
"""
import os
import json
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from src.agents.cmip6_agent import create_cmip6_agent
from src.utils.vector_search import prewarm_retrievers
from src.config import Config
from session_manager import session_manager

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


class SessionResponse(BaseModel):
    session_id: str


class ConfigResponse(BaseModel):
    models: List[str]
    current_model: str


# ─── Routes ──────────────────────────────────────────────────────────

@app.get("/api/config")
async def get_config():
    return ConfigResponse(
        models=Config.get_available_models(),
        current_model=Config.get_model_name()
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

    agent = _get_agent(req.model_name)

    try:
        # langgraph agent uses invoke with {"messages": [...]}
        from langchain_core.messages import HumanMessage, AIMessage

        # Build full conversation history for the agent (same as stream endpoint)
        history = []
        for m in session.messages:
            if m["role"] == "user":
                history.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant" and m.get("content"):
                history.append(AIMessage(content=m["content"]))

        result = agent.invoke(
            {"messages": history},
            config={"configurable": {"session_id": req.session_id}, "recursion_limit": 30},
        )

        # Extract the final response
        final_messages = result.get("messages", [])
        full_response = ""
        for msg in reversed(final_messages):
            if hasattr(msg, 'content') and msg.type == "ai" and msg.content:
                full_response = msg.content
                break

        figure_paths = _extract_figure_paths(full_response)

        # Also scan tool outputs for figure paths
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
        }

    except Exception as e:
        import traceback
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
    print(f"[stream] model_name from request: '{req.model_name}'")
    agent = _get_agent(req.model_name)

    async def event_generator():
        full_response = ""
        figure_paths = []
        last_python_code = ""  # track code for 'View code' button
        try:
            from langchain_core.messages import HumanMessage, AIMessage

            # Build full conversation history for the agent
            history = []
            for m in session.messages:
                if m["role"] == "user":
                    history.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant" and m.get("content"):
                    history.append(AIMessage(content=m["content"]))

            yield f"data: {json.dumps({'type': 'status', 'content': 'Thinking...'})}\n\n"

            for event in agent.stream(
                {"messages": history},
                stream_mode="updates",
                config={"configurable": {"session_id": req.session_id}, "recursion_limit": 30},
            ):
                # langgraph stream events: {"agent": {"messages": [AIMessage(...)]}}
                for node_name, node_output in event.items():
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        if hasattr(msg, 'type'):
                            # ── AI decides to call tool(s) ──
                            if msg.type == "ai" and hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    tc_name = tc.get('name', '?')
                                    tc_args = tc.get('args', {})
                                    args_preview = {k: (str(v)[:120] + '…' if len(str(v)) > 120 else v) for k, v in tc_args.items()}
                                    print(f"[agent] 🔧 TOOL CALL: {tc_name}({args_preview})")
                                    # Capture Python code for 'View code'
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
                                }
                                for tn in tool_names:
                                    label = status_map.get(tn, f'⚙️ Using {tn}...')
                                    yield f"data: {json.dumps({'type': 'status', 'content': label})}\n\n"

                            # ── AI final text response ──
                            elif msg.type == "ai" and msg.content:
                                content = msg.content
                                if isinstance(content, list):
                                    content = "".join(
                                        part.get("text", "") if isinstance(part, dict) else str(part)
                                        for part in content
                                    )
                                print(f"[agent] 💬 AI: {content[:200]}…")
                                full_response = content
                                yield f"data: {json.dumps({'type': 'status', 'content': ''})}\n\n"
                                yield f"data: {json.dumps({'type': 'text', 'content': content})}\n\n"

                            # ── Tool result ──
                            elif msg.type == "tool" and msg.content:
                                tool_name = getattr(msg, 'name', '?')
                                print(f"[agent] 📋 TOOL RESULT ({tool_name}): {str(msg.content)[:300]}")
                                try:
                                    tool_data = json.loads(msg.content)
                                    # Emit RAG sources to frontend
                                    if isinstance(tool_data, dict) and tool_name == 'cmip6_literature_search' and tool_data.get('results'):
                                        sources = [{
                                            'title': r.get('title', ''),
                                            'year': r.get('year', ''),
                                            'journal': r.get('journal', ''),
                                            'doi': r.get('doi', ''),
                                            'score': r.get('score', 0),
                                            'text': r.get('text', '')[:400],
                                        } for r in tool_data['results']]
                                        yield f"data: {json.dumps({'type': 'sources', 'query': tool_data.get('query', ''), 'results': sources})}\n\n"
                                    # Emit figures
                                    if isinstance(tool_data, dict) and tool_data.get("figure_paths"):
                                        urls = [f"/api/figures/{os.path.basename(p)}" for p in tool_data["figure_paths"]]
                                        figure_paths.extend(urls)
                                        payload = {'type': 'figures', 'paths': urls}
                                        if last_python_code:
                                            payload['code'] = last_python_code
                                        payload['stdout'] = tool_data.get('stdout', '')
                                        yield f"data: {json.dumps(payload)}\n\n"
                                        last_python_code = ""
                                except (json.JSONDecodeError, TypeError):
                                    pass

            session.messages.append({
                "role": "assistant",
                "content": full_response,
                "figure_paths": figure_paths,
            })
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            error_msg = str(e)
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


@app.get("/api/figures/{filename}")
async def get_figure(filename: str):
    """Serve a figure file from any session subdirectory."""
    base = os.path.join(os.getcwd(), "temp_figures")
    # Search in all session subdirectories
    for root, dirs, files in os.walk(base):
        if filename in files:
            return FileResponse(os.path.join(root, filename), media_type="image/png")
    raise HTTPException(status_code=404, detail="Figure not found")


# ─── Entry Point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
