"""
Citation Integrity Retry Loop
==============================
Wraps a LangGraph `create_react_agent.invoke()` call with a regeneration
loop that fires when the agent's final response cites DOIs not present in
the session blackboard (i.e. parametric fabrication of academic provenance).

Drop-in usage in `server.py`:

    from src.utils.citation_retry import invoke_with_citation_guard

    result = invoke_with_citation_guard(
        agent,
        history,                       # list[HumanMessage | AIMessage | SystemMessage]
        config={"configurable": {"session_id": req.session_id}, "recursion_limit": 80},
        max_retries=2,                 # how many regeneration passes before giving up
    )

    # result is the same shape that agent.invoke() returns
    # plus result["citation_audit"] = list[CitationValidationResult] (one per pass)

Behaviour
---------
- After each invoke, extracts the last AI message's text and validates it
  against `result.get("blackboard")` via `citation_validator`.
- If fabricated DOIs found AND retries remaining: appends the offending
  AIMessage + a corrective SystemMessage + a fresh HumanMessage telling the
  agent to regenerate, then re-invokes.
- Stops on first clean pass, or when `max_retries` is exhausted.
- Each pass's `CitationValidationResult` is collected in `result["citation_audit"]`
  for downstream logging / metrics.

Why a loop and not a single-shot guard
--------------------------------------
Gemini's first regeneration sometimes re-uses the same fabricated DOI on a
slightly different sentence (parametric drift is sticky). A 2-3 retry budget
catches the common case where the model needs an explicit second nudge,
without burning unbounded tokens on a model that refuses to converge.

Failure modes the loop does NOT fix
-----------------------------------
- Wrong attribution: a DOI that IS in RAG but the cited paper doesn't
  actually support the claim. This module only checks DOI provenance, not
  factual fidelity to the cited content.
- Missing citations: claims without any DOI. That's a separate audit
  dimension (under-citation) requiring a different detector.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.utils.citation_validator import (
    CitationValidationResult,
    format_violation,
    validate_citations,
)

logger = logging.getLogger("cmip6_pipeline")


def _extract_final_ai_text(messages: List[Any]) -> str:
    """Return the most recent assistant text from a LangGraph message list."""
    for msg in reversed(messages):
        if not hasattr(msg, "type") or msg.type != "ai":
            continue
        content = getattr(msg, "content", None)
        if not content:
            continue
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        if content:
            return content
    return ""


def invoke_with_citation_guard(
    agent: Any,
    history: List[Any],
    *,
    config: Optional[Dict[str, Any]] = None,
    max_retries: int = 2,
    on_violation: Optional[Callable[[CitationValidationResult, int], None]] = None,
) -> Dict[str, Any]:
    """Run the agent and retry on citation hallucination up to `max_retries` times.

    Parameters
    ----------
    agent
        LangGraph `create_react_agent` (or anything with a compatible
        `.invoke(state, config=...)` signature returning a state dict).
    history
        Initial message list. Mutated in-place across retries to preserve
        the conversational record (each retry appends the offending AIMessage
        + correction SystemMessage + retry HumanMessage).
    config
        Standard LangGraph config dict, passed through unchanged.
    max_retries
        Number of regeneration passes if violations are found. Default 2.
        Set to 0 to validate-only without retrying.
    on_violation
        Optional callback `(result, attempt_idx) -> None` invoked each time
        a violation is detected. Use it to log/track retries upstream.

    Returns
    -------
    dict
        The final agent state (last successful or last attempted invocation),
        with an extra key `citation_audit: list[CitationValidationResult]`
        containing one entry per pass.
    """
    audit_trail: List[CitationValidationResult] = []
    result: Dict[str, Any] = {}

    for attempt in range(max_retries + 1):
        result = agent.invoke({"messages": history}, config=config or {})
        messages = result.get("messages", []) or []
        final_text = _extract_final_ai_text(messages)
        blackboard = result.get("blackboard") or {}

        check = validate_citations(final_text, blackboard)
        audit_trail.append(check)

        if check.ok:
            logger.info(
                "citation guard: pass %d OK (%d DOIs cited, all grounded)",
                attempt, check.n_cited,
            )
            break

        logger.warning(
            "citation guard: pass %d violation — %d/%d DOIs fabricated: %s",
            attempt, check.n_fabricated, check.n_cited,
            sorted(check.fabricated)[:5],
        )
        if on_violation is not None:
            try:
                on_violation(check, attempt)
            except Exception:  # never let a callback failure break the loop
                logger.exception("citation guard: on_violation callback raised")

        if attempt >= max_retries:
            logger.warning(
                "citation guard: retry budget (%d) exhausted, returning final response with violations",
                max_retries,
            )
            break

        # Append offending response + corrective system message + retry trigger.
        # The agent gets to see its own previous draft (so it knows what to
        # rewrite) and the explicit list of bad DOIs.
        history.append(AIMessage(content=final_text))
        history.append(SystemMessage(content=format_violation(check)))
        history.append(HumanMessage(content=(
            "Regenerate the previous response per the citation integrity "
            "violation above. Do not invent any new DOIs."
        )))

    result["citation_audit"] = audit_trail
    return result


__all__ = ["invoke_with_citation_guard"]
