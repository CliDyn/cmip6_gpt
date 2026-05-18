#!/usr/bin/env python3
"""
CMIP6 GPT — CLI version.
Same agent, tools, and conversation history as the web UI.

Usage:
    python cli.py                                      # in-process agent
    python cli.py --model gpt-5.5                      # specify model
    python cli.py --session run-2026-05-06             # reuse a session
    python cli.py --remote http://localhost:8000       # talk to running server
    python cli.py --remote http://localhost:8000 \
                  --session <id from browser>          # MIRROR with browser
    python cli.py --ask "your question"                # one-shot, then exit
"""

import os
import sys
import json
import argparse
import subprocess
import platform
import uuid

import httpx
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from src.config import Config


def _load_local_agent():
    """Lazy-import heavy in-process agent dependencies. Only called in local mode."""
    from src.agents.cmip6_agent import create_cmip6_agent
    from src.utils.vector_search import prewarm_retrievers
    from src.utils.token_tracker import TokenTracker, get_last_input_tokens
    return create_cmip6_agent, prewarm_retrievers, TokenTracker, get_last_input_tokens


# ─── Colors ──────────────────────────────────────────────────────────

class C:
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    CYAN    = "\033[36m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    RED     = "\033[31m"
    MAGENTA = "\033[35m"
    RESET   = "\033[0m"


def styled(text, *styles):
    return "".join(styles) + text + C.RESET


# ─── Figure viewer ───────────────────────────────────────────────────

def open_figure(path: str):
    """Open a figure in the OS default viewer."""
    if not os.path.exists(path):
        return
    try:
        if platform.system() == "Darwin":
            subprocess.Popen(["open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif platform.system() == "Linux":
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.startfile(path)
    except Exception:
        pass


# ─── Stream event processing ────────────────────────────────────────

STATUS_MAP = {
    "cmip6_datasets_search":   "🔍 Searching datasets...",
    "cmip6_datasets_access":   "📦 Checking data access...",
    "cmip6_adviser":           "📖 Looking up information...",
    "cmip6_literature_search": "📚 Searching literature...",
    "cmip6_citation_graph":    "🕸️  Citation graph...",
    "cmip6_methodology_check": "🧪 Methodology RAG...",
    "python_repl":             "🐍 Running analysis...",
    "Python_REPL":             "🐍 Running analysis...",
    "get_analysis_guide":      "📘 Loading analysis guide...",
    "analysis_guide_tool":     "📘 Loading analysis guide...",
    "review_figure":           "🖼️  Reviewing figure...",
    "reviewer_1":              "👨‍🔬 Reviewer #1...",
    "reviewer_2":              "👨‍🔬 Reviewer #2...",
    "save_to_memory":          "📝 Pinning to blackboard...",
    "forget":                  "🗑️  Removing blackboard entry...",
    "era5_monthly_tool":       "🌍 Fetching ERA5 monthly...",
}


def process_stream_local(agent, messages, session_id, tracker, prior_blackboard=None):
    """Stream agent events, print status/text, return final response + figures + bb.

    `prior_blackboard` is the accumulated blackboard from earlier turns; we
    re-inject it as initial state so the agent sees facts saved on Turn 1
    when running Turn 2, 3, ... — same behaviour as the message history.
    """
    full_response = ""
    figure_paths = []
    blackboard = dict(prior_blackboard or {})

    initial_state = {"messages": messages}
    if prior_blackboard:
        initial_state["blackboard"] = dict(prior_blackboard)

    stream_iter = agent.stream(
        initial_state,
        stream_mode="updates",
        config={"configurable": {"session_id": session_id}, "recursion_limit": 80},
    )

    for event in stream_iter:
        for node_name, node_output in event.items():
            if not isinstance(node_output, dict):
                continue
            # Track blackboard updates as they propagate through the graph
            if "blackboard" in node_output and isinstance(node_output["blackboard"], dict):
                blackboard.update(node_output["blackboard"])
                # Drop tombstones (None) — those are deletes
                blackboard = {k: v for k, v in blackboard.items() if v is not None}

            for msg in node_output.get("messages", []) or []:
                if not hasattr(msg, "type"):
                    continue

                # AI decides to call tool(s)
                if msg.type == "ai" and getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        name = tc.get("name", "?")
                        label = STATUS_MAP.get(name, f"⚙️  Using {name}...")
                        print(f"\r{styled(label, C.DIM)}", end="", flush=True)
                    # Record token usage for this AI step
                    tracker.record(msg)

                # AI final text response
                elif msg.type == "ai" and msg.content:
                    content = msg.content
                    if isinstance(content, list):
                        content = "".join(
                            p.get("text", "") if isinstance(p, dict) else str(p)
                            for p in content
                        )
                    print(f"\r{' ' * 60}\r", end="")
                    full_response = content
                    tracker.record(msg)

                # Tool result — extract figure paths from python_repl payloads
                elif msg.type == "tool" and msg.content:
                    try:
                        data = json.loads(msg.content)
                        if isinstance(data, dict) and data.get("figure_paths"):
                            figure_paths.extend(data["figure_paths"])
                    except (json.JSONDecodeError, TypeError):
                        pass

    return full_response, figure_paths, blackboard


# ─── Remote mode (talk to running server.py) ─────────────────────────

def fetch_remote_messages(base_url: str, session_id: str) -> list:
    """Pull existing session history from the server (so CLI shows browser turns).
    Endpoint returns {"messages": [...]}; unwrap to a list."""
    try:
        r = httpx.get(f"{base_url}/api/sessions/{session_id}/messages", timeout=60.0)
        r.raise_for_status()
        body = r.json()
        if isinstance(body, dict):
            return body.get("messages") or []
        return body or []
    except httpx.HTTPError:
        return []


def clear_remote_session(base_url: str, session_id: str) -> bool:
    try:
        r = httpx.post(
            f"{base_url}/api/sessions/clear",
            json={"session_id": session_id},
            timeout=60.0,
        )
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def process_stream_remote(base_url: str, message: str, session_id: str,
                          model_name: str, rag_chunks: int = 10, rag_searches: int = 5):
    """POST to the running server and consume its SSE event stream.

    Same endpoint the browser uses, so the server's session_manager treats this
    request identically — message gets appended to the same in-memory session,
    workspace under results/{session_id}/ is shared, REPL state is shared.
    """
    payload = {
        "message": message,
        "session_id": session_id,
        "model_name": model_name,
        "rag_chunks": rag_chunks,
        "rag_searches": rag_searches,
        "google_api_key_slot": 1,
        "reviewer_model_1": Config.reviewer_model_1,
        "reviewer_model_2": Config.reviewer_model_2,
        "reviewers_enabled": Config.reviewers_enabled,
    }

    full_response = ""
    figure_paths = []
    sources_seen = []

    with httpx.stream(
        "POST",
        f"{base_url}/api/chat/stream",
        json=payload,
        timeout=httpx.Timeout(None, read=None),  # no timeout for streaming
        headers={"Accept": "text/event-stream"},
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8", "replace")
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if not data_str:
                continue
            try:
                evt = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            etype = evt.get("type")
            if etype == "status":
                content = evt.get("content", "")
                if content:
                    print(f"\r{styled(content, C.DIM)}", end="", flush=True)
                else:
                    print(f"\r{' ' * 60}\r", end="")
            elif etype == "text":
                full_response = evt.get("content", "")
            elif etype == "figures":
                paths = evt.get("paths", []) or []
                # Server returns API-relative paths like /api/figures/abc.png — turn into URL
                figure_paths.extend(f"{base_url}{p}" if p.startswith("/") else p for p in paths)
            elif etype == "sources":
                sources_seen.append(evt)
            # Ignore other event types silently

    return full_response, figure_paths, sources_seen


# ─── Format response ────────────────────────────────────────────────

def print_response(text: str, figures: list, usage: dict | None = None, bb_count: int = 0):
    print(f"\n{styled('🤖 CMIP6 GPT', C.BOLD, C.CYAN)}")
    print(text)

    if figures:
        print(f"\n{styled(f'📊 {len(figures)} figure(s) generated:', C.GREEN)}")
        for i, path in enumerate(figures, 1):
            print(f"   {i}. {path}")
            open_figure(path)

    footer_parts = []
    if usage:
        footer_parts.append(
            f"💰 {usage['total_input_tokens']:,} in + {usage['total_output_tokens']:,} out "
            f"= {usage['total_tokens']:,} tok | €{usage['total_cost_eur']:.4f} | "
            f"{usage['llm_calls']} call(s) | {usage['elapsed_seconds']}s"
        )
    if bb_count:
        footer_parts.append(f"🧠 blackboard: {bb_count} entries")
    if footer_parts:
        print(styled("  ·  ".join(footer_parts), C.DIM))
    print()


def print_blackboard(blackboard: dict):
    if not blackboard:
        print(styled("🧠 Blackboard is empty.\n", C.DIM))
        return
    print(styled(f"🧠 Blackboard ({len(blackboard)} entries):", C.BOLD, C.CYAN))
    grouped = {}
    for k, v in blackboard.items():
        cat, _, sub = k.partition(".")
        grouped.setdefault(cat or "misc", []).append((sub or k, v))
    for cat in sorted(grouped):
        print(f"  {styled(cat, C.YELLOW)}:")
        for sub, v in grouped[cat]:
            preview = v if len(v) <= 90 else v[:87] + "..."
            print(f"    - {sub}: {styled(preview, C.DIM)}")
    print()


# ─── Main loop ───────────────────────────────────────────────────────

def _run_local(args, session_id: str) -> None:
    """In-process agent loop (default). Runs the agent directly in this process.
    Faster startup than --remote but does not share state with a running server."""
    create_cmip6_agent, prewarm_retrievers, TokenTracker, get_last_input_tokens = _load_local_agent()

    print(styled("\nPre-warming retrievers...", C.DIM), end=" ", flush=True)
    prewarm_retrievers()
    print(styled("done ✓", C.GREEN))

    print(styled("Creating agent...", C.DIM), end=" ", flush=True)
    agent = create_cmip6_agent()
    print(styled("done ✓\n", C.GREEN))

    history: list = []
    last_usage: dict | None = None
    blackboard: dict = {}

    def run_turn(user_text: str) -> None:
        nonlocal history, last_usage, blackboard, agent
        history.append(HumanMessage(content=user_text))
        tracker = TokenTracker()
        tracker.start_request(session_id, Config.get_model_name(), user_text)
        try:
            response_text, figures, new_blackboard = process_stream_local(
                agent, history, session_id, tracker, prior_blackboard=blackboard,
            )
            history.append(AIMessage(content=response_text))
            blackboard = {k: v for k, v in (new_blackboard or {}).items() if v is not None}
            last_usage = tracker.end_request()
            print_response(response_text, figures, usage=last_usage, bb_count=len(blackboard))
        except KeyboardInterrupt:
            print(styled("\n⏹ Interrupted\n", C.RED))
        except Exception as e:
            print(f"\r{' ' * 60}\r", end="")
            print(styled(f"\n❌ Error: {e}\n", C.RED))

    if args.ask:
        run_turn(args.ask)
        return

    while True:
        try:
            user_input = input(styled("👤 You: ", C.BOLD, C.GREEN)).strip()
        except (KeyboardInterrupt, EOFError):
            print(styled("\n\nBye! 👋", C.DIM))
            break
        if not user_input:
            continue
        cmd = user_input.lower()

        if cmd == "/quit":
            print(styled("Bye! 👋", C.DIM)); break
        if cmd == "/clear":
            history.clear(); blackboard.clear()
            print(styled("Chat cleared ✓ (blackboard wiped)\n", C.YELLOW)); continue
        if cmd in ("/bb", "/blackboard"):
            print_blackboard(blackboard); continue
        if cmd == "/usage":
            if last_usage:
                u = last_usage
                print(styled(
                    f"Last turn: {u['total_input_tokens']:,} in + {u['total_output_tokens']:,} out "
                    f"= {u['total_tokens']:,} tok | €{u['total_cost_eur']:.4f} | "
                    f"{u['llm_calls']} call(s) | {u['elapsed_seconds']}s", C.DIM,
                ))
            else:
                print(styled("No turns yet.", C.DIM))
            print(styled(
                f"Provider's last reported input_tokens for this session: "
                f"{get_last_input_tokens(session_id):,}", C.DIM,
            ))
            print(); continue
        if cmd.startswith("/model"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print(styled(f"Current model: {Config.get_model_name()}", C.DIM))
                print(styled(f"Available: {', '.join(Config.get_available_models())}", C.DIM))
            else:
                Config.set_model_name(parts[1].strip())
                agent = create_cmip6_agent()
                print(styled(f"Switched to {parts[1].strip()} ✓", C.YELLOW))
            print(); continue
        run_turn(user_input)


def _run_remote(args, session_id: str, base_url: str) -> None:
    """Thin SSE client of a running server.py. Shares session_id + workspace +
    message history with the browser — open the same session_id in a browser
    tab and refresh after each turn to see CLI activity, and vice versa."""

    # Probe server
    try:
        r = httpx.get(f"{base_url}/api/sessions/{session_id}/messages", timeout=60.0)
        r.raise_for_status()
        body = r.json()
        prior = (body.get("messages") if isinstance(body, dict) else body) or []
    except Exception as e:
        print(styled(f"❌ Cannot reach server at {base_url} — {e}\n", C.RED))
        sys.exit(1)

    if prior:
        print(styled(f"📜 Loaded {len(prior)} prior message(s) from session "
                     f"{session_id}:", C.DIM))
        for m in prior[-6:]:
            role = m.get("role", "?")
            preview = (m.get("content") or "")[:140].replace("\n", " ")
            tag = "👤" if role == "user" else "🤖"
            print(f"   {tag} {styled(preview, C.DIM)}")
        if len(prior) > 6:
            print(styled(f"   ... ({len(prior) - 6} earlier)\n", C.DIM))
        print()

    def run_turn(user_text: str) -> None:
        try:
            response, figures, _ = process_stream_remote(
                base_url, user_text, session_id, Config.get_model_name(),
            )
            print_response(response, figures)
        except KeyboardInterrupt:
            print(styled("\n⏹ Interrupted\n", C.RED))
        except Exception as e:
            print(f"\r{' ' * 60}\r", end="")
            print(styled(f"\n❌ Remote error: {e}\n", C.RED))

    if args.ask:
        run_turn(args.ask); return

    while True:
        try:
            user_input = input(styled("👤 You: ", C.BOLD, C.GREEN)).strip()
        except (KeyboardInterrupt, EOFError):
            print(styled("\n\nBye! 👋", C.DIM)); break
        if not user_input:
            continue
        cmd = user_input.lower()

        if cmd == "/quit":
            print(styled("Bye! 👋", C.DIM)); break
        if cmd == "/clear":
            ok = clear_remote_session(base_url, session_id)
            print(styled("Server session cleared ✓\n" if ok else "Clear failed.\n",
                         C.YELLOW if ok else C.RED)); continue
        if cmd in ("/bb", "/blackboard"):
            print(styled("(blackboard inspection not exposed by server in --remote mode)\n",
                         C.DIM)); continue
        if cmd == "/usage":
            print(styled("(usage live in server logs/token_usage.jsonl)\n", C.DIM)); continue
        if cmd.startswith("/model"):
            parts = user_input.split(maxsplit=1)
            if len(parts) >= 2:
                Config.set_model_name(parts[1].strip())
                print(styled(f"Switched to {parts[1].strip()} (remote will use it next turn) ✓\n",
                             C.YELLOW))
            else:
                print(styled(f"Current model: {Config.get_model_name()}\n", C.DIM))
            continue
        run_turn(user_input)


def main():
    parser = argparse.ArgumentParser(description="CMIP6 GPT CLI")
    parser.add_argument("--model", type=str, default=None, help="LLM model name")
    parser.add_argument("--session", type=str, default=None,
                        help="Session id (defaults to a fresh UUID; reuse to keep workspace).")
    parser.add_argument("--ask", type=str, default=None,
                        help="Run a single question non-interactively and exit.")
    parser.add_argument("--remote", type=str, default=None,
                        help="Talk to a running server (e.g. http://localhost:8000) "
                             "instead of running the agent in-process. "
                             "Same --session as the browser → shared history + workspace.")
    args = parser.parse_args()

    if args.model:
        Config.set_model_name(args.model)
    model_name = Config.get_model_name()
    session_id = args.session or f"cli-{uuid.uuid4().hex[:8]}"

    print(styled("━" * 64, C.DIM))
    print(styled("  CMIP6 GPT — CLI", C.BOLD, C.CYAN))
    print(styled(f"  Model:   {model_name}", C.DIM))
    print(styled(f"  Session: {session_id}", C.DIM))
    if args.remote:
        print(styled(f"  Remote:  {args.remote}  (mirrored with browser)", C.MAGENTA))
    else:
        print(styled(f"  Mode:    in-process (workspace results/{session_id}/)", C.DIM))
    print(styled("━" * 64, C.DIM))
    print(styled("  Commands: /clear  /model <name>  /bb  /usage  /quit", C.DIM))
    print(styled("━" * 64, C.DIM))

    if args.remote:
        _run_remote(args, session_id, args.remote.rstrip("/"))
    else:
        _run_local(args, session_id)


if __name__ == "__main__":
    main()
