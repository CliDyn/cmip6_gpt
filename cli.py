#!/usr/bin/env python3
"""
CMIP6 GPT — CLI version.
Same agent, tools, and conversation history as the web UI.

Usage:
    python cli.py                     # default model from config
    python cli.py --model gpt-5.2    # specify model
"""

import os
import sys
import json
import argparse
import subprocess
import platform

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from src.agents.cmip6_agent import create_cmip6_agent
from src.utils.vector_search import prewarm_retrievers
from src.config import Config


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
    "cmip6_datasets_search": "🔍 Searching datasets...",
    "cmip6_datasets_access": "📦 Checking data access...",
    "cmip6_adviser":         "📖 Looking up information...",
    "python_repl":           "🐍 Running analysis...",
    "Python_REPL":           "🐍 Running analysis...",
    "get_analysis_guide":    "📘 Loading analysis guide...",
}


def process_stream(agent, messages):
    """Stream agent events, print status/text, return final response + figure paths."""
    full_response = ""
    figure_paths = []

    for event in agent.stream({"messages": messages}, stream_mode="updates"):
        for node_name, node_output in event.items():
            for msg in node_output.get("messages", []):
                if not hasattr(msg, "type"):
                    continue

                # Agent decides to call tool(s)
                if msg.type == "ai" and hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        name = tc.get("name", "?")
                        label = STATUS_MAP.get(name, f"⚙️  Using {name}...")
                        print(f"\r{styled(label, C.DIM)}", end="", flush=True)

                # Agent final text response
                elif msg.type == "ai" and msg.content:
                    content = msg.content
                    if isinstance(content, list):
                        content = "".join(
                            p.get("text", "") if isinstance(p, dict) else str(p)
                            for p in content
                        )
                    # Clear status line
                    print(f"\r{' ' * 60}\r", end="")
                    full_response = content

                # Tool result — check for figures
                elif msg.type == "tool" and msg.content:
                    try:
                        data = json.loads(msg.content)
                        if isinstance(data, dict) and data.get("figure_paths"):
                            figure_paths.extend(data["figure_paths"])
                    except (json.JSONDecodeError, TypeError):
                        pass

    return full_response, figure_paths


# ─── Format response ────────────────────────────────────────────────

def print_response(text: str, figures: list):
    print(f"\n{styled('🤖 CMIP6 GPT', C.BOLD, C.CYAN)}")
    print(text)

    if figures:
        print(f"\n{styled(f'📊 {len(figures)} figure(s) generated:', C.GREEN)}")
        for i, path in enumerate(figures, 1):
            print(f"   {i}. {path}")
            open_figure(path)
    print()


# ─── Main loop ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CMIP6 GPT CLI")
    parser.add_argument("--model", type=str, default=None, help="LLM model name")
    args = parser.parse_args()

    if args.model:
        Config.set_model_name(args.model)

    model_name = Config.get_model_name()

    # Startup
    print(styled("━" * 60, C.DIM))
    print(styled("  CMIP6 GPT — CLI", C.BOLD, C.CYAN))
    print(styled(f"  Model: {model_name}", C.DIM))
    print(styled("━" * 60, C.DIM))
    print(styled("  Commands: /clear  /model <name>  /quit", C.DIM))
    print(styled("━" * 60, C.DIM))

    print(styled("\nPre-warming retrievers...", C.DIM), end=" ", flush=True)
    prewarm_retrievers()
    print(styled("done ✓", C.GREEN))

    print(styled("Creating agent...", C.DIM), end=" ", flush=True)
    agent = create_cmip6_agent()
    print(styled("done ✓\n", C.GREEN))

    history = []  # conversation history as LangChain messages

    while True:
        try:
            user_input = input(styled("👤 You: ", C.BOLD, C.GREEN)).strip()
        except (KeyboardInterrupt, EOFError):
            print(styled("\n\nBye! 👋", C.DIM))
            break

        if not user_input:
            continue

        # ─── Commands ────────────────────────────────────
        if user_input.lower() == "/quit":
            print(styled("Bye! 👋", C.DIM))
            break

        if user_input.lower() == "/clear":
            history.clear()
            print(styled("Chat cleared ✓\n", C.YELLOW))
            continue

        if user_input.lower().startswith("/model"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print(styled(f"Current model: {Config.get_model_name()}", C.DIM))
                print(styled(f"Available: {', '.join(Config.get_available_models())}", C.DIM))
            else:
                new_model = parts[1].strip()
                Config.set_model_name(new_model)
                agent = create_cmip6_agent()
                print(styled(f"Switched to {new_model} ✓", C.YELLOW))
            print()
            continue

        # ─── Agent call ──────────────────────────────────
        history.append(HumanMessage(content=user_input))

        try:
            response_text, figures = process_stream(agent, history)
            history.append(AIMessage(content=response_text))
            print_response(response_text, figures)
        except KeyboardInterrupt:
            print(styled("\n⏹ Interrupted\n", C.RED))
        except Exception as e:
            print(f"\r{' ' * 60}\r", end="")
            print(styled(f"\n❌ Error: {e}\n", C.RED))


if __name__ == "__main__":
    main()
