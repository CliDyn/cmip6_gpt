"""
Token Usage Tracker — logs every LLM call to a JSONL file.

Each line in the log contains:
  - timestamp, session_id, model, request_id
  - input_tokens, output_tokens, total_tokens
  - estimated cost (EUR)
  - cumulative totals for the request

Usage:
    tracker = TokenTracker()
    tracker.start_request(session_id, model_name, user_message)
    # ... inside stream loop, for each AI message:
    tracker.record(msg)
    # ... after request completes:
    summary = tracker.end_request()
"""

import os
import json
import time
import threading
from datetime import datetime, timezone
from typing import Optional

# ─── Pricing (EUR per 1M tokens, as of April 2026) ──────────────────
# Source: Google Cloud billing SKUs from the project
PRICING = {
    # Gemini 3 Pro (Gemini API — "short" context)
    "gemini-3-pro": {"input": 1.70, "output": 10.20},
    "gemini-3.1-pro-preview": {"input": 1.70, "output": 10.20},
    # Gemini 3 Pro (Vertex AI — same pricing)
    "gemini-3.0-pro": {"input": 1.70, "output": 10.20},
    # Gemini 3 Flash
    "gemini-3-flash": {"input": 0.10, "output": 0.40},
    "gemini-3-flash-preview": {"input": 0.10, "output": 0.40},
    "gemini-3.1-flash-lite-preview": {"input": 0.02, "output": 0.08},
    # Gemini 2.5
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    # Embeddings
    "gemini-embedding-2-preview": {"input": 0.006, "output": 0.0},
    "gemini-embedding-001": {"input": 0.006, "output": 0.0},
    # OpenAI (approximate EUR)
    "gpt-5.2": {"input": 2.50, "output": 10.00},
    "gpt-5.5": {"input": 2.50, "output": 10.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}

# Default pricing for unknown models
DEFAULT_PRICING = {"input": 2.00, "output": 10.00}

LOG_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')), "logs")
LOG_FILE = os.path.join(LOG_DIR, "token_usage.jsonl")

_lock = threading.Lock()

# ─── Per-session "last seen" telemetry ────────────────────────────────
# Used by the dynamic compression hook to know how big the previous LLM
# input was, so it can decide whether the next call needs compression.
# Updated from TokenTracker.record() on every AI response.
_session_last_input_tokens: dict[str, int] = {}
_session_telemetry_lock = threading.Lock()


def get_last_input_tokens(session_id: str) -> int:
    """Return the most recent input_tokens count reported by the model for
    this session, or 0 if no LLM call has run yet."""
    with _session_telemetry_lock:
        return _session_last_input_tokens.get(session_id, 0)


def set_last_input_tokens(session_id: str, n: int) -> None:
    if n <= 0:
        return
    with _session_telemetry_lock:
        _session_last_input_tokens[session_id] = int(n)


def _get_pricing(model_name: str) -> dict:
    """Find pricing for a model, falling back to prefix matching."""
    if model_name in PRICING:
        return PRICING[model_name]
    # Try prefix match (e.g. "gemini-3.1-pro-preview" → "gemini-3-pro")
    for key in PRICING:
        if model_name.startswith(key):
            return PRICING[key]
    return DEFAULT_PRICING


def _estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in EUR for a single LLM call."""
    pricing = _get_pricing(model_name)
    cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
    return round(cost, 6)


class TokenTracker:
    """Tracks token usage across a single user request (which may involve multiple LLM calls)."""

    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self._session_id = "default"
        self._model_name = "unknown"
        self._user_message = ""
        self._request_start = 0.0
        self._calls = []  # list of per-call records
        self._total_input = 0
        self._total_output = 0
        self._total_cost = 0.0

    def start_request(self, session_id: str, model_name: str, user_message: str):
        """Call at the beginning of a chat request."""
        self._session_id = session_id
        self._model_name = model_name
        self._user_message = user_message[:200]  # truncate for log readability
        self._request_start = time.time()
        self._calls = []
        self._total_input = 0
        self._total_output = 0
        self._total_cost = 0.0

    def record(self, msg) -> Optional[dict]:
        """
        Record token usage from a LangChain message.
        LangChain messages from Google/OpenAI include `usage_metadata`:
          {input_tokens: int, output_tokens: int, total_tokens: int}
        Returns the call record if tokens were found, else None.
        """
        usage = getattr(msg, 'usage_metadata', None) or getattr(msg, 'response_metadata', {}).get('usage', None)
        if not usage:
            # Try response_metadata.token_usage (OpenAI style)
            resp_meta = getattr(msg, 'response_metadata', {})
            if isinstance(resp_meta, dict):
                usage = resp_meta.get('token_usage') or resp_meta.get('usage_metadata')
        if not usage:
            return None

        # Normalize — usage can be a dict or an object
        if isinstance(usage, dict):
            input_tokens = usage.get('input_tokens', 0) or usage.get('prompt_tokens', 0)
            output_tokens = usage.get('output_tokens', 0) or usage.get('completion_tokens', 0)
        else:
            input_tokens = getattr(usage, 'input_tokens', 0) or getattr(usage, 'prompt_tokens', 0)
            output_tokens = getattr(usage, 'output_tokens', 0) or getattr(usage, 'completion_tokens', 0)

        if input_tokens == 0 and output_tokens == 0:
            return None

        model = self._model_name
        # Try to get actual model from response metadata
        resp_meta = getattr(msg, 'response_metadata', {})
        if isinstance(resp_meta, dict) and resp_meta.get('model_name'):
            model = resp_meta['model_name']

        cost = _estimate_cost(model, input_tokens, output_tokens)

        self._total_input += input_tokens
        self._total_output += output_tokens
        self._total_cost += cost

        # Publish for the dynamic compression hook (last value wins).
        set_last_input_tokens(self._session_id, input_tokens)

        call_record = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_eur": cost,
            "has_tool_calls": bool(getattr(msg, 'tool_calls', None)),
        }
        self._calls.append(call_record)
        return call_record

    def end_request(self) -> dict:
        """
        Finalize the request and write a summary line to the JSONL log.
        Returns the summary dict.
        """
        elapsed = round(time.time() - self._request_start, 2) if self._request_start else 0

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self._session_id,
            "model": self._model_name,
            "user_message": self._user_message,
            "llm_calls": len(self._calls),
            "total_input_tokens": self._total_input,
            "total_output_tokens": self._total_output,
            "total_tokens": self._total_input + self._total_output,
            "total_cost_eur": round(self._total_cost, 6),
            "elapsed_seconds": elapsed,
            "calls": self._calls,
        }

        # Write to JSONL log (thread-safe)
        with _lock:
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(summary) + "\n")

        # Also print a console summary
        print(
            f"[tokens] 💰 Request done: {self._total_input:,} in + "
            f"{self._total_output:,} out = {self._total_input + self._total_output:,} total | "
            f"€{self._total_cost:.4f} | {len(self._calls)} LLM calls | {elapsed}s"
        )

        return summary


# ─── Convenience: read log stats ────────────────────────────────────

def get_usage_summary() -> dict:
    """Read the JSONL log and return aggregate statistics."""
    if not os.path.exists(LOG_FILE):
        return {"total_requests": 0, "total_cost_eur": 0, "total_tokens": 0}

    total_requests = 0
    total_cost = 0.0
    total_tokens = 0
    by_model = {}

    with open(LOG_FILE) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                total_requests += 1
                total_cost += entry.get("total_cost_eur", 0)
                total_tokens += entry.get("total_tokens", 0)
                model = entry.get("model", "unknown")
                if model not in by_model:
                    by_model[model] = {"requests": 0, "tokens": 0, "cost_eur": 0.0}
                by_model[model]["requests"] += 1
                by_model[model]["tokens"] += entry.get("total_tokens", 0)
                by_model[model]["cost_eur"] += entry.get("total_cost_eur", 0)
            except json.JSONDecodeError:
                continue

    return {
        "total_requests": total_requests,
        "total_cost_eur": round(total_cost, 4),
        "total_tokens": total_tokens,
        "avg_cost_per_request": round(total_cost / max(total_requests, 1), 4),
        "avg_tokens_per_request": total_tokens // max(total_requests, 1),
        "by_model": by_model,
    }
