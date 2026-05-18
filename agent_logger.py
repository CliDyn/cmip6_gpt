"""
Structured agent step logger.
Saves a JSON log per request to logs/ with every tool call, tool result,
RAG source, AI response, figure, and token usage.
"""
import os
import json
import time
from datetime import datetime, timezone


LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


class AgentStepLogger:
    """Accumulates agent steps during a single request and flushes to disk."""

    def __init__(self, session_id: str, model_name: str, user_prompt: str,
                 rag_searches: int = 5, rag_chunks: int = 10):
        self.session_id = session_id
        self.model_name = model_name
        self.user_prompt = user_prompt
        self.rag_searches = rag_searches
        self.rag_chunks = rag_chunks
        self.steps = []
        self.sources = []        # accumulated RAG sources
        self.figures = []        # accumulated figure paths
        self.final_response = ""
        self.usage = {}
        self._t0 = time.time()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._filename = f"{ts}_{session_id[:8]}.json"
        self._step_counter = 0

    def _step(self, kind: str, **kwargs):
        self._step_counter += 1
        entry = {
            "step": self._step_counter,
            "t_sec": round(time.time() - self._t0, 2),
            "kind": kind,
            **kwargs,
        }
        self.steps.append(entry)

    # ── Events ──────────────────────────────────────────────

    def tool_call(self, name: str, args: dict):
        """Record an agent tool call."""
        # For Python REPL, store full code; for others, truncate large args
        args_log = {}
        for k, v in args.items():
            sv = str(v)
            if name in ("Python_REPL", "python_repl") and k == "query":
                args_log[k] = sv  # full code
            elif len(sv) > 500:
                args_log[k] = sv[:500] + "… [truncated]"
            else:
                args_log[k] = v
        self._step("tool_call", tool=name, args=args_log)

    def tool_result(self, name: str, content: str):
        """Record a tool result."""
        # Keep manageable size but preserve enough for reproducibility
        truncated = content[:2000] + ("… [truncated]" if len(content) > 2000 else "")
        self._step("tool_result", tool=name, content=truncated)

    def rag_sources(self, query: str, results: list):
        """Record RAG sources from a literature search."""
        compact = [{
            "title": r.get("title", ""),
            "doi": r.get("doi", ""),
            "year": r.get("year", ""),
            "score": r.get("score", 0),
        } for r in results]
        self._step("rag_sources", query=query, count=len(results), top_sources=compact[:10])
        self.sources.extend(compact)

    def ai_response(self, content: str):
        """Record a final AI text response."""
        self.final_response = content
        self._step("ai_response", length=len(content), preview=content[:500])

    def ai_reasoning(self, content: str):
        """Record AI reasoning / intermediate text (with tool calls)."""
        self._step("ai_reasoning", preview=content[:300] if content else "")

    def figure(self, paths: list):
        """Record generated figure paths."""
        self.figures.extend(paths)
        self._step("figure", paths=paths)

    def error(self, message: str):
        """Record an error."""
        self._step("error", message=message[:1000])

    def set_usage(self, usage: dict):
        """Record token usage summary."""
        self.usage = usage

    # ── Flush ───────────────────────────────────────────────

    def flush(self):
        """Write the full log to disk."""
        elapsed = round(time.time() - self._t0, 2)
        doc = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "model": self.model_name,
            "rag_config": {
                "searches": self.rag_searches,
                "chunks_per_search": self.rag_chunks,
            },
            "user_prompt": self.user_prompt,
            "elapsed_sec": elapsed,
            "total_steps": self._step_counter,
            "total_rag_sources": len(self.sources),
            "total_figures": len(self.figures),
            "figures": self.figures,
            "usage": self.usage,
            "steps": self.steps,
            "final_response": self.final_response,
        }
        path = os.path.join(LOGS_DIR, self._filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False, default=str)
        print(f"[logger] 📝 Agent log saved: {path} ({self._step_counter} steps, {elapsed}s)")
        return path
